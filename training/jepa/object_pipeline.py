"""Object-centric helpers for JEPA training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from arcgen import Grid

from training.modules.object_tokenizer import (
    BASE_FEATURE_KEYS,
    TokenizedObjects,
    tokenize_grid_objects,
)
from training.modules.object_encoder import ObjectTokenEncoder

try:  # pragma: no cover - torch optional at import time
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


@dataclass(frozen=True)
class ObjectTokenizerConfig:
    """Configuration options for object tokenization."""

    max_objects: int = 16
    max_color_features: int = 10
    background: int | Iterable[int] = 0
    connectivity: int = 4
    normalize: bool = True
    respect_colors: bool = True

    @property
    def feature_dim(self) -> int:
        return len(BASE_FEATURE_KEYS) + self.max_color_features

    def as_kwargs(self) -> dict:
        return {
            "max_objects": self.max_objects,
            "max_color_features": self.max_color_features,
            "background": self.background,
            "connectivity": self.connectivity,
            "normalize": self.normalize,
            "respect_colors": self.respect_colors,
        }


@dataclass(frozen=True)
class ObjectTokenBatch:
    """Tensor batch produced from tokenized objects."""

    features: "torch.Tensor"
    mask: "torch.Tensor"
    adjacency: "torch.Tensor"

    def to(self, device: str | torch.device) -> "ObjectTokenBatch":  # pragma: no cover - thin convenience
        return ObjectTokenBatch(
            features=self.features.to(device),
            mask=self.mask.to(device),
            adjacency=self.adjacency.to(device),
        )


@dataclass(frozen=True)
class ObjectCentricEncoding:
    """Encoded JEPA tokens produced by the object-centric encoder."""

    embeddings: "torch.Tensor"
    mask: "torch.Tensor"
    adjacency: "torch.Tensor"
    vq_loss: "torch.Tensor | None"
    vq_indices: "torch.Tensor | None"


def _ensure_torch_available() -> None:
    if torch is None:  # pragma: no cover - safety net
        raise RuntimeError("PyTorch is required for object-centric JEPA encoding")


def build_object_token_batch(
    grids: Sequence[Grid],
    config: ObjectTokenizerConfig,
    *,
    device: "torch.device | None" = None,
) -> ObjectTokenBatch:
    """Tokenize a batch of grids and stack into tensors."""

    _ensure_torch_available()

    tokenized: list[TokenizedObjects] = [
        tokenize_grid_objects(grid, **config.as_kwargs()) for grid in grids
    ]

    feature_tensor = torch.tensor(
        [tokens.features for tokens in tokenized], dtype=torch.float32, device=device
    )
    mask_tensor = torch.tensor(
        [tokens.mask for tokens in tokenized], dtype=torch.float32, device=device
    )
    adjacency_tensor = torch.tensor(
        [tokens.adjacency for tokens in tokenized], dtype=torch.float32, device=device
    )

    return ObjectTokenBatch(features=feature_tensor, mask=mask_tensor, adjacency=adjacency_tensor)


class ObjectCentricJEPAEncoder:
    """High-level helper that encodes grids via object tokens."""

    def __init__(
        self,
        encoder: ObjectTokenEncoder,
        tokenizer_config: ObjectTokenizerConfig,
    ) -> None:
        _ensure_torch_available()

        if encoder is None:
            raise ValueError("encoder must be provided")

        if tokenizer_config.max_objects <= 0:
            raise ValueError("tokenizer_config.max_objects must be positive")

        self.encoder = encoder
        self.tokenizer_config = tokenizer_config

    @property
    def feature_dim(self) -> int:
        return self.tokenizer_config.feature_dim

    def encode(
        self,
        grids: Sequence[Grid],
        *,
        device: "torch.device | None" = None,
    ) -> ObjectCentricEncoding:
        if not grids:
            raise ValueError("grids must contain at least one entry")

        batch = build_object_token_batch(grids, self.tokenizer_config, device=device)
        encoder_out = self.encoder(batch.features, mask=batch.mask)

        return ObjectCentricEncoding(
            embeddings=encoder_out["embeddings"],
            mask=encoder_out["mask"],
            adjacency=batch.adjacency,
            vq_loss=encoder_out["vq_loss"],
            vq_indices=encoder_out["vq_indices"],
        )
