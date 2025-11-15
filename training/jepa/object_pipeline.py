"""Object-centric helpers for JEPA training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

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
class ObjectEncoderConfig:
    """Configuration options for the object token encoder."""

    hidden_dim: int = 256
    num_embeddings: int | None = 512
    commitment_cost: float = 0.25
    ema_decay: float | None = 0.99
    vq_refresh_enabled: bool = False
    vq_refresh_interval: int = 100
    vq_refresh_usage_threshold: float = 1e-3
    activation: str = "gelu"
    relational: bool = True
    relational_layers: int = 2
    relational_heads: int = 4
    relational_dropout: float = 0.0

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None) -> "ObjectEncoderConfig":
        if data is None:
            return cls()
        return cls(
            hidden_dim=int(data.get("hidden_dim", cls.hidden_dim)),
            num_embeddings=data.get("num_embeddings", cls.num_embeddings),
            commitment_cost=float(data.get("commitment_cost", cls.commitment_cost)),
            ema_decay=data.get("ema_decay", cls.ema_decay),
            vq_refresh_enabled=bool(data.get("vq_refresh_enabled", cls.vq_refresh_enabled)),
            vq_refresh_interval=int(data.get("vq_refresh_interval", cls.vq_refresh_interval)),
            vq_refresh_usage_threshold=float(
                data.get("vq_refresh_usage_threshold", cls.vq_refresh_usage_threshold)
            ),
            activation=str(data.get("activation", cls.activation)),
            relational=bool(data.get("relational", cls.relational)),
            relational_layers=int(data.get("relational_layers", cls.relational_layers)),
            relational_heads=int(data.get("relational_heads", cls.relational_heads)),
            relational_dropout=float(data.get("relational_dropout", cls.relational_dropout)),
        )


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
        encoder_out = self.encoder(batch.features, mask=batch.mask, adjacency=batch.adjacency)

        return ObjectCentricEncoding(
            embeddings=encoder_out["embeddings"],
            mask=encoder_out["mask"],
            adjacency=batch.adjacency,
            vq_loss=encoder_out["vq_loss"],
            vq_indices=encoder_out["vq_indices"],
        )


def build_object_tokenizer_config(data: Mapping[str, object] | None) -> ObjectTokenizerConfig:
    if data is None:
        return ObjectTokenizerConfig()
    return ObjectTokenizerConfig(
        max_objects=int(data.get("max_objects", ObjectTokenizerConfig.max_objects)),
        max_color_features=int(data.get("max_color_features", ObjectTokenizerConfig.max_color_features)),
        background=data.get("background", ObjectTokenizerConfig.background),
        connectivity=int(data.get("connectivity", ObjectTokenizerConfig.connectivity)),
        normalize=bool(data.get("normalize", ObjectTokenizerConfig.normalize)),
        respect_colors=bool(data.get("respect_colors", ObjectTokenizerConfig.respect_colors)),
    )


def build_object_encoder(
    tokenizer_cfg: ObjectTokenizerConfig,
    encoder_cfg: ObjectEncoderConfig,
) -> ObjectTokenEncoder:
    return ObjectTokenEncoder(
        feature_dim=tokenizer_cfg.feature_dim,
        hidden_dim=encoder_cfg.hidden_dim,
        num_embeddings=encoder_cfg.num_embeddings,
        commitment_cost=encoder_cfg.commitment_cost,
        ema_decay=encoder_cfg.ema_decay,
        vq_refresh_enabled=encoder_cfg.vq_refresh_enabled,
        vq_refresh_interval=encoder_cfg.vq_refresh_interval,
        vq_refresh_usage_threshold=encoder_cfg.vq_refresh_usage_threshold,
        activation=encoder_cfg.activation,
        relational=encoder_cfg.relational,
        relational_layers=encoder_cfg.relational_layers,
        relational_heads=encoder_cfg.relational_heads,
        relational_dropout=encoder_cfg.relational_dropout,
    )


def build_object_centric_encoder_from_config(
    tokenizer_cfg_data: Mapping[str, object] | None,
    encoder_cfg_data: Mapping[str, object] | None,
) -> ObjectCentricJEPAEncoder:
    tokenizer_cfg = build_object_tokenizer_config(tokenizer_cfg_data)
    encoder_cfg = ObjectEncoderConfig.from_mapping(encoder_cfg_data)
    encoder = build_object_encoder(tokenizer_cfg, encoder_cfg)
    return ObjectCentricJEPAEncoder(encoder, tokenizer_cfg)


@dataclass(frozen=True)
class EncodedPairBatch:
    context: ObjectCentricEncoding
    target: ObjectCentricEncoding


def encode_context_target(
    encoder: ObjectCentricJEPAEncoder,
    context_grids: Sequence[Grid],
    target_grids: Sequence[Grid],
    *,
    device: "torch.device | None" = None,
) -> EncodedPairBatch:
    if len(context_grids) != len(target_grids):
        raise ValueError("context_grids and target_grids must have the same length")

    context_enc = encoder.encode(context_grids, device=device)
    target_enc = encoder.encode(target_grids, device=device)
    return EncodedPairBatch(context=context_enc, target=target_enc)
