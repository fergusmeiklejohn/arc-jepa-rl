"""Vector quantization modules for discrete latent bottlenecks."""

from __future__ import annotations

from dataclasses import dataclass

try:  # pragma: no cover - exercised via tests when torch is available
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = object  # type: ignore
    F = object  # type: ignore


class VectorQuantizerUnavailable(RuntimeError):
    """Raised when the vector quantizer is instantiated without PyTorch."""


@dataclass
class VectorQuantizerOutput:
    quantized: "torch.Tensor"
    loss: "torch.Tensor"
    indices: "torch.Tensor"
    perplexity: "torch.Tensor"


if torch is not None:  # pragma: no branch

    class VectorQuantizer(nn.Module):
        """Standard VQ-VAE style bottleneck with optional EMA updates."""

        def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            *,
            commitment_cost: float = 0.25,
            ema_decay: float | None = None,
            epsilon: float = 1e-5,
            refresh_unused_codes: bool = False,
            refresh_interval: int = 100,
            refresh_usage_threshold: float = 1e-3,
        ) -> None:
            super().__init__()

            if num_embeddings <= 0:
                raise ValueError("num_embeddings must be positive")
            if embedding_dim <= 0:
                raise ValueError("embedding_dim must be positive")
            if commitment_cost <= 0:
                raise ValueError("commitment_cost must be positive")
            if ema_decay is not None and not 0.0 < ema_decay < 1.0:
                raise ValueError("ema_decay must be in (0, 1)")
            if epsilon <= 0:
                raise ValueError("epsilon must be positive")
            if refresh_unused_codes and refresh_interval <= 0:
                raise ValueError("refresh_interval must be positive when refresh is enabled")
            if refresh_usage_threshold < 0:
                raise ValueError("refresh_usage_threshold must be non-negative")

            self.num_embeddings = num_embeddings
            self.embedding_dim = embedding_dim
            self.commitment_cost = commitment_cost
            self.ema_decay = ema_decay
            self.epsilon = epsilon
            self.refresh_unused_codes = bool(refresh_unused_codes)
            self.refresh_interval = int(refresh_interval)
            self.refresh_usage_threshold = float(refresh_usage_threshold)
            self._refresh_step = 0

            self.embedding = nn.Embedding(num_embeddings, embedding_dim)
            self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

            if ema_decay is not None:
                self.register_buffer("_ema_cluster_size", torch.zeros(num_embeddings))
                self.register_buffer("_ema_w", self.embedding.weight.detach().clone())

        def forward(self, inputs: "torch.Tensor") -> VectorQuantizerOutput:
            if inputs.dim() < 2:
                raise ValueError("inputs must have shape (..., embedding_dim)")
            if inputs.size(-1) != self.embedding_dim:
                raise ValueError("last dimension must equal embedding_dim")

            flat_input = inputs.reshape(-1, self.embedding_dim)

            distances = (
                torch.sum(flat_input**2, dim=1, keepdim=True)
                - 2 * torch.matmul(flat_input, self.embedding.weight.t())
                + torch.sum(self.embedding.weight**2, dim=1)
            )

            encoding_indices = torch.argmin(distances, dim=1)
            encodings = F.one_hot(encoding_indices, self.num_embeddings).type(flat_input.dtype)

            quantized = self.embedding(encoding_indices).view_as(inputs)

            if self.training and self.ema_decay is not None:
                decay = self.ema_decay
                assert decay is not None

                cluster_size = encodings.sum(dim=0)
                self._ema_cluster_size.mul_(decay).add_(cluster_size, alpha=1 - decay)

                dw = torch.matmul(encodings.t(), flat_input)
                self._ema_w.mul_(decay).add_(dw, alpha=1 - decay)

                n = self._ema_cluster_size.sum()
                cluster_size = (
                    (self._ema_cluster_size + self.epsilon)
                    / (n + self.num_embeddings * self.epsilon)
                    * n
                )
                self.embedding.weight.data.copy_(self._ema_w / cluster_size.unsqueeze(1))
            else:
                cluster_size = encodings.sum(dim=0)

            self._maybe_refresh_codebook(
                flat_input.detach(),
                distances.detach(),
                encoding_indices.detach(),
                cluster_size.detach(),
            )

            e_latent_loss = torch.mean((quantized.detach() - inputs) ** 2)
            q_latent_loss = torch.mean((quantized - inputs.detach()) ** 2)
            loss = e_latent_loss + self.commitment_cost * q_latent_loss

            quantized = inputs + (quantized - inputs).detach()

            avg_probs = encodings.mean(dim=0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

            indices = encoding_indices.view(*inputs.shape[:-1])
            return VectorQuantizerOutput(quantized=quantized, loss=loss, indices=indices, perplexity=perplexity)

        def _maybe_refresh_codebook(
            self,
            flat_input: "torch.Tensor",
            distances: "torch.Tensor",
            encoding_indices: "torch.Tensor",
            cluster_size: "torch.Tensor",
        ) -> None:
            if (
                not self.training
                or not self.refresh_unused_codes
                or flat_input.numel() == 0
                or distances.numel() == 0
            ):
                return

            self._refresh_step += 1
            if self._refresh_step % self.refresh_interval != 0:
                return

            usage_source: "torch.Tensor"
            if self.ema_decay is not None:
                usage_source = self._ema_cluster_size
            else:
                usage_source = cluster_size

            total_mass = torch.sum(usage_source).clamp(min=1.0)
            usage = usage_source / total_mass
            dead_mask = usage <= self.refresh_usage_threshold
            if not torch.any(dead_mask):
                return

            self._refresh_step = 0
            dead_indices = torch.nonzero(dead_mask, as_tuple=False).flatten()
            if dead_indices.numel() == 0:
                return

            reconstruction_error = torch.gather(distances, 1, encoding_indices.unsqueeze(1)).squeeze(1)
            if reconstruction_error.numel() == 0:
                return

            candidate_order = torch.argsort(reconstruction_error, descending=True)
            if candidate_order.numel() == 0:
                return

            num_candidates = candidate_order.numel()
            for offset, embed_idx in enumerate(dead_indices):
                sample_idx = candidate_order[offset % num_candidates]
                new_vector = flat_input[sample_idx]
                self.embedding.weight.data[embed_idx] = new_vector
                if self.ema_decay is not None:
                    self._ema_w[embed_idx] = new_vector
                    mean_usage = torch.clamp(self._ema_cluster_size.mean(), min=1.0)
                    self._ema_cluster_size[embed_idx] = mean_usage

else:  # pragma: no cover - executed only when torch is missing at import time

    class VectorQuantizer:  # type: ignore[override]
        def __init__(self, *args, **kwargs) -> None:  # noqa: D401
            raise VectorQuantizerUnavailable(
                "PyTorch is required to use training.modules.vq.VectorQuantizer"
            )
