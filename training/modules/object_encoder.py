"""Placeholder encoder that maps object tokens into latent embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

try:  # pragma: no cover - torch optional
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = object  # type: ignore

from .vq import (
    VectorQuantizer,
    GumbelVectorQuantizer,
    VectorQuantizerUnavailable,
    create_vector_quantizer,
    VQMode,
)
from .relational import RelationalAggregator, RelationalModuleUnavailable


class ObjectTokenEncoderUnavailable(RuntimeError):
    """Raised when attempting to instantiate the encoder without PyTorch."""


if torch is not None:  # pragma: no branch

    class ObjectTokenEncoder(nn.Module):
        """Maps per-object feature tokens into latent vectors with optional VQ."""

        def __init__(
            self,
            feature_dim: int,
            *,
            hidden_dim: int = 128,
            num_embeddings: int | None = 128,
            commitment_cost: float = 0.25,
            ema_decay: float | None = 0.99,
            vq_refresh_enabled: bool = False,
            vq_refresh_interval: int = 100,
            vq_refresh_usage_threshold: float = 1e-3,
            vq_mode: VQMode = "hard",
            vq_temperature_init: float = 1.0,
            vq_temperature_min: float = 0.1,
            vq_temperature_anneal_steps: int = 10000,
            vq_straight_through: bool = True,
            vq_entropy_weight: float = 0.01,
            activation: str = "gelu",
            relational: bool = True,
            relational_layers: int = 2,
            relational_heads: int = 4,
            relational_dropout: float = 0.0,
        ) -> None:
            super().__init__()

            if feature_dim <= 0:
                raise ValueError("feature_dim must be positive")
            if hidden_dim <= 0:
                raise ValueError("hidden_dim must be positive")

            if activation == "relu":
                act = nn.ReLU()
            elif activation == "gelu":
                act = nn.GELU()
            else:
                raise ValueError("activation must be 'relu' or 'gelu'")

            self.proj = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                act,
                nn.Linear(hidden_dim, hidden_dim),
            )

            self.relational = (
                RelationalAggregator(
                    hidden_dim,
                    num_layers=relational_layers,
                    num_heads=relational_heads,
                    dropout=relational_dropout,
                )
                if relational
                else None
            )

            self.vq_mode = vq_mode
            if num_embeddings is not None:
                self.vq = create_vector_quantizer(
                    mode=vq_mode,
                    num_embeddings=num_embeddings,
                    embedding_dim=hidden_dim,
                    # Hard VQ params
                    commitment_cost=commitment_cost,
                    ema_decay=ema_decay,
                    refresh_unused_codes=vq_refresh_enabled,
                    refresh_interval=vq_refresh_interval,
                    refresh_usage_threshold=vq_refresh_usage_threshold,
                    # Gumbel VQ params
                    temperature_init=vq_temperature_init,
                    temperature_min=vq_temperature_min,
                    temperature_anneal_steps=vq_temperature_anneal_steps,
                    straight_through=vq_straight_through,
                    entropy_weight=vq_entropy_weight,
                )
            else:
                self.vq = None

        def forward(
            self,
            features: "torch.Tensor",
            mask: "torch.Tensor | None" = None,
            adjacency: "torch.Tensor | None" = None,
        ) -> dict:
            if features.dim() != 3:
                raise ValueError("features must have shape (batch, objects, feature_dim)")
            if mask is None:
                mask = torch.ones(features.shape[:2], device=features.device, dtype=features.dtype)
            if mask.shape != features.shape[:2]:
                raise ValueError("mask must have shape (batch, objects)")

            embeddings = self.proj(features)

            if self.relational is not None:
                if adjacency is None:
                    raise ValueError("adjacency matrix must be provided when relational aggregation is enabled")
                embeddings = self.relational(embeddings, adjacency, mask)

            vq_loss = None
            vq_indices = None

            if self.vq is not None:
                vq_out = self.vq(embeddings)
                embeddings = vq_out.quantized
                vq_loss = vq_out.loss
                vq_indices = vq_out.indices

            return {
                "embeddings": embeddings,
                "mask": mask,
                "vq_loss": vq_loss,
                "vq_indices": vq_indices,
            }

else:  # pragma: no cover - executed only when torch missing

    class ObjectTokenEncoder:  # type: ignore[override]
        def __init__(self, *args, **kwargs) -> None:  # noqa: D401
            raise ObjectTokenEncoderUnavailable(
                "PyTorch is required to use training.modules.object_encoder.ObjectTokenEncoder"
            )
