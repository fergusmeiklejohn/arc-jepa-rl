"""Vector quantization modules for discrete latent bottlenecks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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


# Type alias for VQ mode
VQMode = Literal["hard", "gumbel"]


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

            with torch.no_grad():
                # Distance + assignment are kept out of autograd to avoid large graph retention.
                distances = (
                    torch.sum(flat_input**2, dim=1, keepdim=True)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t())
                    + torch.sum(self.embedding.weight**2, dim=1)
                )
                encoding_indices = torch.argmin(distances, dim=1)
                total_elems = max(1, flat_input.shape[0])

                if self.training and self.ema_decay is not None:
                    decay = float(self.ema_decay)
                    cluster_size = torch.bincount(
                        encoding_indices, minlength=self.num_embeddings
                    ).float()
                    self._ema_cluster_size.mul_(decay).add_(cluster_size, alpha=1 - decay)

                    dw = torch.zeros(
                        (self.num_embeddings, self.embedding_dim),
                        device=flat_input.device,
                        dtype=flat_input.dtype,
                    )
                    dw.index_add_(0, encoding_indices, flat_input)
                    self._ema_w.mul_(decay).add_(dw, alpha=1 - decay)

                    n = self._ema_cluster_size.sum()
                    denom = (
                        (self._ema_cluster_size + self.epsilon)
                        / (n + self.num_embeddings * self.epsilon)
                        * n
                    )
                    self.embedding.weight.data.copy_(self._ema_w / denom.unsqueeze(1))
                else:
                    cluster_size = torch.bincount(
                        encoding_indices, minlength=self.num_embeddings
                    ).float()

                self._maybe_refresh_codebook(
                    flat_input.detach(),
                    distances.detach(),
                    encoding_indices.detach(),
                    cluster_size.detach(),
                )

            quantized = self.embedding(encoding_indices).view_as(inputs)
            e_latent_loss = torch.mean((quantized.detach() - inputs) ** 2)
            q_latent_loss = torch.mean((quantized - inputs.detach()) ** 2)
            loss = e_latent_loss + self.commitment_cost * q_latent_loss

            quantized = inputs + (quantized - inputs).detach()

            probs = cluster_size / float(total_elems)
            perplexity = torch.exp(-torch.sum(probs * torch.log(probs + 1e-10)))

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

    class GumbelVectorQuantizer(nn.Module):
        """Gumbel-Softmax based differentiable vector quantizer.

        Unlike hard VQ-VAE which uses straight-through estimators and commitment loss,
        Gumbel-Softmax VQ is fully differentiable during training. This eliminates:
        - Commitment loss (no longer needed)
        - EMA updates (gradients flow directly)
        - Codebook refresh heuristics (natural gradient updates)

        During training, uses soft assignments via Gumbel-Softmax.
        During inference, uses hard assignments (argmax).

        The temperature τ controls the sharpness of assignments:
        - High τ → soft, uniform-like assignments (exploration)
        - Low τ → hard, one-hot-like assignments (exploitation)

        Temperature is annealed during training for best results.
        """

        def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            *,
            temperature_init: float = 1.0,
            temperature_min: float = 0.1,
            temperature_anneal_steps: int = 10000,
            straight_through: bool = True,
            entropy_weight: float = 0.01,
        ) -> None:
            """Initialize Gumbel-Softmax VQ.

            Args:
                num_embeddings: Number of codes in the codebook.
                embedding_dim: Dimension of each code vector.
                temperature_init: Initial Gumbel-Softmax temperature.
                temperature_min: Minimum temperature (annealing target).
                temperature_anneal_steps: Steps to anneal from init to min.
                straight_through: Use straight-through estimator for hard samples.
                entropy_weight: Weight for entropy regularization (encourages codebook usage).
            """
            super().__init__()

            if num_embeddings <= 0:
                raise ValueError("num_embeddings must be positive")
            if embedding_dim <= 0:
                raise ValueError("embedding_dim must be positive")
            if temperature_init <= 0:
                raise ValueError("temperature_init must be positive")
            if temperature_min <= 0:
                raise ValueError("temperature_min must be positive")
            if temperature_min > temperature_init:
                raise ValueError("temperature_min must be <= temperature_init")
            if temperature_anneal_steps <= 0:
                raise ValueError("temperature_anneal_steps must be positive")

            self.num_embeddings = num_embeddings
            self.embedding_dim = embedding_dim
            self.temperature_init = temperature_init
            self.temperature_min = temperature_min
            self.temperature_anneal_steps = temperature_anneal_steps
            self.straight_through = straight_through
            self.entropy_weight = entropy_weight

            # Codebook embeddings (learnable via gradients, no EMA needed)
            self.embedding = nn.Embedding(num_embeddings, embedding_dim)
            self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

            # Track training steps for temperature annealing
            self.register_buffer("_step", torch.tensor(0, dtype=torch.long))

        @property
        def temperature(self) -> float:
            """Get current temperature based on annealing schedule."""
            step = self._step.item()
            if step >= self.temperature_anneal_steps:
                return self.temperature_min

            # Linear annealing
            progress = step / self.temperature_anneal_steps
            return self.temperature_init - progress * (self.temperature_init - self.temperature_min)

        def forward(self, inputs: "torch.Tensor") -> VectorQuantizerOutput:
            """Quantize inputs using Gumbel-Softmax.

            Args:
                inputs: Tensor of shape (..., embedding_dim).

            Returns:
                VectorQuantizerOutput with quantized embeddings, loss, indices, perplexity.
            """
            if inputs.dim() < 2:
                raise ValueError("inputs must have shape (..., embedding_dim)")
            if inputs.size(-1) != self.embedding_dim:
                raise ValueError("last dimension must equal embedding_dim")

            original_shape = inputs.shape
            flat_input = inputs.reshape(-1, self.embedding_dim)
            batch_size = flat_input.shape[0]

            # Compute distances to codebook (negative distance = logit)
            # distances[i,j] = ||z_i - e_j||^2
            distances = (
                torch.sum(flat_input**2, dim=1, keepdim=True)
                - 2 * torch.matmul(flat_input, self.embedding.weight.t())
                + torch.sum(self.embedding.weight**2, dim=1)
            )

            # Convert distances to logits (smaller distance = higher logit)
            logits = -distances

            # Update step counter during training
            if self.training:
                self._step.add_(1)

            tau = self.temperature

            if self.training:
                # Gumbel-Softmax: differentiable sampling
                if self.straight_through:
                    # Hard samples with straight-through gradient
                    soft_samples = F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)
                else:
                    # Soft samples (fully differentiable)
                    soft_samples = F.gumbel_softmax(logits, tau=tau, hard=False, dim=-1)

                # Get hard indices for metrics (no gradient)
                with torch.no_grad():
                    encoding_indices = torch.argmax(logits, dim=1)
            else:
                # Inference: use hard assignments
                encoding_indices = torch.argmax(logits, dim=1)
                soft_samples = F.one_hot(encoding_indices, self.num_embeddings).float()

            # Quantized output: weighted sum of codebook vectors
            # With straight_through=True during training, this is effectively z_q = e[argmax]
            # but gradients flow through the soft weights
            quantized = torch.matmul(soft_samples, self.embedding.weight)
            quantized = quantized.view(*original_shape)

            # Compute entropy regularization loss (encourages uniform codebook usage)
            # Higher entropy = more uniform usage = better codebook utilization
            with torch.no_grad():
                cluster_size = torch.bincount(
                    encoding_indices, minlength=self.num_embeddings
                ).float()

            probs = cluster_size / max(1, batch_size)
            log_probs = torch.log(probs + 1e-10)
            entropy = -torch.sum(probs * log_probs)
            max_entropy = torch.log(torch.tensor(self.num_embeddings, dtype=torch.float32, device=inputs.device))

            # Loss: negative entropy (we want to maximize entropy)
            # Scaled to [0, 1] range where 0 = max entropy (uniform), 1 = min entropy (collapsed)
            entropy_loss = 1.0 - (entropy / max_entropy)
            loss = self.entropy_weight * entropy_loss

            # Perplexity (exponential of entropy)
            perplexity = torch.exp(entropy)

            indices = encoding_indices.view(*original_shape[:-1])
            return VectorQuantizerOutput(
                quantized=quantized,
                loss=loss,
                indices=indices,
                perplexity=perplexity,
            )

    def create_vector_quantizer(
        mode: VQMode,
        num_embeddings: int,
        embedding_dim: int,
        **kwargs,
    ) -> "VectorQuantizer | GumbelVectorQuantizer":
        """Factory function to create the appropriate VQ module.

        Args:
            mode: "hard" for standard VQ-VAE, "gumbel" for Gumbel-Softmax VQ.
            num_embeddings: Number of codes in the codebook.
            embedding_dim: Dimension of each code vector.
            **kwargs: Additional arguments passed to the VQ constructor.

        Returns:
            VectorQuantizer or GumbelVectorQuantizer instance.
        """
        if mode == "hard":
            # Filter kwargs for VectorQuantizer
            hard_kwargs = {
                k: v for k, v in kwargs.items()
                if k in {
                    "commitment_cost", "ema_decay", "epsilon",
                    "refresh_unused_codes", "refresh_interval", "refresh_usage_threshold"
                }
            }
            return VectorQuantizer(num_embeddings, embedding_dim, **hard_kwargs)
        elif mode == "gumbel":
            # Filter kwargs for GumbelVectorQuantizer
            gumbel_kwargs = {
                k: v for k, v in kwargs.items()
                if k in {
                    "temperature_init", "temperature_min", "temperature_anneal_steps",
                    "straight_through", "entropy_weight"
                }
            }
            return GumbelVectorQuantizer(num_embeddings, embedding_dim, **gumbel_kwargs)
        else:
            raise ValueError(f"Unknown VQ mode: {mode}. Must be 'hard' or 'gumbel'.")

else:  # pragma: no cover - executed only when torch is missing at import time

    class VectorQuantizer:  # type: ignore[override]
        def __init__(self, *args, **kwargs) -> None:  # noqa: D401
            raise VectorQuantizerUnavailable(
                "PyTorch is required to use training.modules.vq.VectorQuantizer"
            )

    class GumbelVectorQuantizer:  # type: ignore[override]
        def __init__(self, *args, **kwargs) -> None:  # noqa: D401
            raise VectorQuantizerUnavailable(
                "PyTorch is required to use training.modules.vq.GumbelVectorQuantizer"
            )

    def create_vector_quantizer(*args, **kwargs):  # noqa: D401
        raise VectorQuantizerUnavailable(
            "PyTorch is required to use training.modules.vq.create_vector_quantizer"
        )
