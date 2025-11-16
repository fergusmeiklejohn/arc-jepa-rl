"""Relational consistency loss helpers for JEPA training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from training.jepa.object_pipeline import ObjectCentricEncoding


@dataclass(frozen=True)
class RelationalConsistencyConfig:
    """Configuration describing pairwise relational penalties."""

    weight: float = 0.0
    context_self_weight: float = 0.0
    eps: float = 1e-6

    @classmethod
    def from_mapping(cls, data: dict | None) -> "RelationalConsistencyConfig":
        if data is None:
            return cls()
        return cls(
            weight=float(data.get("weight", cls.weight)),
            context_self_weight=float(data.get("context_self_weight", cls.context_self_weight)),
            eps=float(data.get("eps", cls.eps)),
        )

    @property
    def enabled(self) -> bool:
        return self.weight > 0.0 or self.context_self_weight > 0.0


def _pairwise_relations(encoding: ObjectCentricEncoding) -> torch.Tensor:
    """Compute adjacency-masked pairwise relation magnitudes."""

    embeddings = encoding.embeddings
    mask = (encoding.mask > 0).float()
    adjacency = (encoding.adjacency > 0).float()

    diff = embeddings.unsqueeze(2) - embeddings.unsqueeze(1)
    relation = diff.pow(2).sum(dim=-1)
    pair_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
    return relation * adjacency * pair_mask


def relational_consistency_loss(
    context_encoding: ObjectCentricEncoding,
    target_encoding: ObjectCentricEncoding,
    *,
    batch_size: int,
    context_length: int,
    config: RelationalConsistencyConfig,
) -> torch.Tensor | None:
    if not config.enabled:
        return None

    if context_length <= 0:
        raise ValueError("context_length must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    total_entries = batch_size * context_length
    if context_encoding.embeddings.size(0) != total_entries:
        raise ValueError("context encoding size mismatch for relational loss")

    context_rel = _pairwise_relations(context_encoding)
    target_rel = _pairwise_relations(target_encoding)

    num_objects = target_rel.size(-1)
    context_rel = context_rel.view(batch_size, context_length, num_objects, num_objects)
    context_mean = context_rel.mean(dim=1)

    losses: list[torch.Tensor] = []

    if config.weight > 0.0:
        losses.append(config.weight * F.mse_loss(context_mean, target_rel))

    if config.context_self_weight > 0.0 and context_length > 1:
        deviations = context_rel - context_mean.unsqueeze(1)
        variance = deviations.pow(2).mean()
        losses.append(config.context_self_weight * variance)

    if not losses:
        return None
    return torch.stack(losses).sum()
