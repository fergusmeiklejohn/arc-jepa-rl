"""Embedding diagnostics and quality metrics for JEPA training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, MutableSequence

try:  # pragma: no cover - torch required during training
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


@dataclass(frozen=True)
class EmbeddingDiagnosticsConfig:
    """Configuration for projection embedding diagnostics."""

    enabled: bool = True
    log_interval: int = 200
    max_samples: int = 4096

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None) -> "EmbeddingDiagnosticsConfig":
        if data is None:
            return cls()
        interval = int(data.get("interval", data.get("log_interval", cls.log_interval)))
        max_samples = int(data.get("max_samples", cls.max_samples))
        if interval <= 0:
            raise ValueError("embedding diagnostics interval must be positive")
        if max_samples <= 0:
            raise ValueError("embedding diagnostics max_samples must be positive")
        return cls(
            enabled=bool(data.get("enabled", cls.enabled)),
            log_interval=interval,
            max_samples=max_samples,
        )


class EmbeddingDiagnosticsTracker:
    """Accumulates embeddings and emits statistics at a configured interval."""

    def __init__(
        self,
        config: EmbeddingDiagnosticsConfig,
        *,
        codebook_size: int | None,
    ) -> None:
        if torch is None:  # pragma: no cover
            raise RuntimeError("PyTorch is required for embedding diagnostics")

        self.config = config
        self.codebook_size = codebook_size
        self._step = 0
        self._buffers: dict[str, MutableSequence["torch.Tensor"]] = {
            "context": [],
            "target": [],
        }
        self._events: list[dict[str, object]] = []
        self._usage: "torch.Tensor | None" = (
            torch.zeros(codebook_size, dtype=torch.bool) if codebook_size else None
        )

    def observe(
        self,
        *,
        context_proj: "torch.Tensor",
        target_proj: "torch.Tensor",
        context_encoding=None,
        target_encoding=None,
    ) -> None:
        if not self.config.enabled:
            return

        self._step += 1
        self._append("context", context_proj.detach().float().cpu())
        self._append("target", target_proj.detach().float().cpu())
        self._update_usage(context_encoding)
        self._update_usage(target_encoding)

        if self._step % self.config.log_interval == 0:
            event = self._build_event()
            if event is not None:
                self._events.append(event)
            self._reset_buffers()

    def consume(self, *, flush: bool = False) -> list[dict[str, object]]:
        if flush and self.config.enabled:
            if any(self._buffers[key] for key in self._buffers):
                event = self._build_event()
                if event is not None:
                    self._events.append(event)
                self._reset_buffers()
        events = self._events
        self._events = []
        return events

    # ------------------------------------------------------------------ helpers
    def _append(self, key: str, tensor: "torch.Tensor") -> None:
        buffer = self._buffers[key]
        buffer.append(tensor)
        self._trim_buffer(buffer)

    def _trim_buffer(self, buffer: MutableSequence["torch.Tensor"]) -> None:
        total = sum(tensor.shape[0] for tensor in buffer)
        overflow = max(0, total - self.config.max_samples)
        while overflow > 0 and buffer:
            head = buffer[0]
            head_count = head.shape[0]
            if head_count <= overflow:
                buffer.pop(0)
                overflow -= head_count
            else:
                buffer[0] = head[overflow:]
                overflow = 0

    def _update_usage(self, encoding) -> None:
        if self._usage is None or encoding is None:
            return
        indices = getattr(encoding, "vq_indices", None)
        mask = getattr(encoding, "mask", None)
        if indices is None or mask is None:
            return
        indices = indices.detach().cpu().long()
        mask_tensor = mask.detach().cpu() > 0.5
        valid = indices[mask_tensor]
        if valid.numel() == 0:
            return
        unique = torch.unique(valid)
        self._usage[unique] = True

    def _build_event(self) -> dict[str, object] | None:
        context = self._stack("context")
        target = self._stack("target")
        if context is None and target is None:
            return None
        event: dict[str, object] = {"step": self._step}
        if context is not None:
            event["context"] = self._compute_stats(context)
            event["context_samples"] = int(context.shape[0])
        if target is not None:
            event["target"] = self._compute_stats(target)
            event["target_samples"] = int(target.shape[0])
        if self._usage is not None and self.codebook_size:
            active = int(self._usage.sum().item())
            event["vq_active_codes"] = active
            event["vq_usage_ratio"] = float(active / float(self.codebook_size))
        return event

    def _stack(self, key: str) -> "torch.Tensor | None":
        buffer = self._buffers[key]
        if not buffer:
            return None
        return torch.cat(list(buffer), dim=0)

    def _reset_buffers(self) -> None:
        for tensors in self._buffers.values():
            tensors.clear()
        if self._usage is not None:
            self._usage.zero_()

    def _compute_stats(self, embeddings: "torch.Tensor") -> dict[str, float]:
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be 2D for diagnostics")
        variance = torch.var(embeddings, dim=0, unbiased=False).mean()
        cov = self._covariance(embeddings)
        # Ensure numerical stability: add a tiny jitter if covariance is ill-conditioned
        jitter = torch.eye(cov.size(0), device=cov.device, dtype=cov.dtype) * 1e-6
        try:
            eigenvalues = torch.linalg.eigvalsh(cov + jitter)
        except torch.linalg.LinAlgError:
            return {
                "variance": float(variance.item()),
                "isotropy": 0.0,
                "effective_rank": 0.0,
                "gaussian_score": 0.0,
            }
        eigenvalues = torch.clamp(eigenvalues.real, min=0.0)
        max_eig = torch.max(eigenvalues).clamp(min=1e-8)
        min_eig = torch.min(eigenvalues).clamp(min=0.0)
        isotropy = (min_eig + 1e-8) / (max_eig + 1e-8)
        total = torch.sum(eigenvalues).clamp(min=1e-8)
        probs = eigenvalues / total
        entropy = -torch.sum(probs * torch.log(probs + 1e-12))
        effective_rank = torch.exp(entropy)
        gaussian_score = self._gaussian_score(embeddings)

        return {
            "variance": float(variance.item()),
            "isotropy": float(isotropy.item()),
            "effective_rank": float(effective_rank.item()),
            "gaussian_score": float(gaussian_score),
        }

    def _covariance(self, embeddings: "torch.Tensor") -> "torch.Tensor":
        centered = embeddings - embeddings.mean(dim=0, keepdim=True)
        denom = max(embeddings.shape[0] - 1, 1)
        cov = centered.t() @ centered
        return cov / float(denom)

    def _gaussian_score(self, embeddings: "torch.Tensor") -> float:
        centered = embeddings - embeddings.mean(dim=0, keepdim=True)
        std = centered.std(dim=0, unbiased=False).clamp(min=1e-6)
        normalized = centered / std
        skewness = torch.mean(normalized ** 3, dim=0).abs().mean()
        kurtosis = torch.mean(normalized ** 4, dim=0)
        excess = (kurtosis - 3.0).abs().mean()
        score = 1.0 / (1.0 + skewness.item() + excess.item())
        return score
