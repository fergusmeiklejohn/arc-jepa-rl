"""Neural modules for program-conditioned JEPA counterfactual prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .object_pipeline import ObjectCentricEncoding

try:  # pragma: no cover - torch optional for type-checking
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = object  # type: ignore[misc]
    F = object  # type: ignore


def _ensure_torch() -> None:
    if torch is None:  # pragma: no cover - defensive fallback
        raise RuntimeError("PyTorch is required for program-conditioned JEPA modules")


def aggregate_object_encoding(encoding: ObjectCentricEncoding) -> "torch.Tensor":
    """Average object embeddings while respecting the mask."""

    _ensure_torch()
    mask = encoding.mask.to(dtype=encoding.embeddings.dtype).unsqueeze(-1)
    summed = (encoding.embeddings * mask).sum(dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1.0)
    return summed / counts


@dataclass(frozen=True)
class ProgramConditionedModelConfig:
    """Configuration block for :class:`ProgramConditionedJEPA`."""

    program_embedding_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    transition_hidden_dim: int = 512

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None) -> "ProgramConditionedModelConfig":
        if data is None:
            return cls()
        return cls(
            program_embedding_dim=int(data.get("program_embedding_dim", cls.program_embedding_dim)),
            hidden_dim=int(data.get("hidden_dim", cls.hidden_dim)),
            num_layers=int(data.get("num_layers", cls.num_layers)),
            dropout=float(data.get("dropout", cls.dropout)),
            transition_hidden_dim=int(data.get("transition_hidden_dim", cls.transition_hidden_dim)),
        )


class ProgramSequenceEncoder(nn.Module if nn is not object else object):  # type: ignore[misc]
    """Encode primitive sequences (with parameters) into dense vectors."""

    def __init__(
        self,
        vocab_size: int,
        parameter_dim: int,
        *,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        padding_idx: int = 0,
    ) -> None:
        _ensure_torch()
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if parameter_dim < 0:
            raise ValueError("parameter_dim cannot be negative")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        self.primitive_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.parameter_dim = parameter_dim
        self.param_proj = nn.Linear(parameter_dim, embedding_dim) if parameter_dim > 0 else None
        input_dim = embedding_dim if self.param_proj is None else embedding_dim * 2
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output = nn.Linear(hidden_dim, embedding_dim)

    def forward(
        self,
        program_ids: "torch.Tensor",
        program_params: "torch.Tensor",
        program_mask: "torch.Tensor",
    ) -> "torch.Tensor":
        embeddings = self.primitive_embedding(program_ids)
        if self.param_proj is not None:
            param_features = self.param_proj(program_params)
            step_features = torch.cat([embeddings, param_features], dim=-1)
        else:
            step_features = embeddings
        step_features = self.dropout(step_features)

        lengths = program_mask.sum(dim=1).clamp(min=1).to(dtype=torch.long)
        packed = nn.utils.rnn.pack_padded_sequence(
            step_features,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        encoded = hidden[-1]
        return self.output(encoded)


class ProgramConditionedJEPA(nn.Module if nn is not object else object):  # type: ignore[misc]
    """Predict the latent effect of applying a program to the current state."""

    def __init__(
        self,
        latent_dim: int,
        vocab_size: int,
        parameter_dim: int,
        config: ProgramConditionedModelConfig | Mapping[str, object] | None = None,
    ) -> None:
        _ensure_torch()
        super().__init__()
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        cfg = (
            config
            if isinstance(config, ProgramConditionedModelConfig)
            else ProgramConditionedModelConfig.from_mapping(config)
        )

        self.program_encoder = ProgramSequenceEncoder(
            vocab_size=vocab_size,
            parameter_dim=parameter_dim,
            embedding_dim=cfg.program_embedding_dim,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
        )

        transition_input = latent_dim + cfg.program_embedding_dim
        self.transition = nn.Sequential(
            nn.Linear(transition_input, cfg.transition_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.transition_hidden_dim, latent_dim),
        )

    def forward(
        self,
        current_latents: "torch.Tensor",
        program_ids: "torch.Tensor",
        program_params: "torch.Tensor",
        program_mask: "torch.Tensor",
    ) -> "torch.Tensor":
        program_embedding = self.program_encoder(program_ids, program_params, program_mask)
        combined = torch.cat([current_latents, program_embedding], dim=-1)
        return self.transition(combined)

    def predict_counterfactual(
        self,
        latents: "torch.Tensor",
        *,
        program_ids: "torch.Tensor",
        program_params: "torch.Tensor",
        program_mask: "torch.Tensor",
    ) -> "torch.Tensor":
        """Alias for :meth:`forward` with keyword inputs for readability."""

        return self.forward(latents, program_ids, program_params, program_mask)

