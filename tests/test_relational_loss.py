"""Tests for the relational consistency loss helpers."""

from __future__ import annotations

import torch

from training.jepa.object_pipeline import ObjectCentricEncoding
from training.jepa.relational_loss import (
    RelationalConsistencyConfig,
    relational_consistency_loss,
)


def _encoding_from_tensors(embeddings: torch.Tensor, mask: torch.Tensor, adjacency: torch.Tensor) -> ObjectCentricEncoding:
    return ObjectCentricEncoding(
        embeddings=embeddings,
        mask=mask,
        adjacency=adjacency,
        vq_loss=None,
        vq_indices=None,
    )


def test_relational_loss_zero_when_context_matches_target() -> None:
    embeddings = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ]
    )
    mask = torch.ones((2, 2))
    adjacency = torch.ones((2, 2, 2))
    context_enc = _encoding_from_tensors(embeddings, mask, adjacency)
    target_enc = _encoding_from_tensors(embeddings[0:1], mask[0:1], adjacency[0:1])

    cfg = RelationalConsistencyConfig(weight=1.0, context_self_weight=1.0)
    loss = relational_consistency_loss(
        context_enc,
        target_enc,
        batch_size=1,
        context_length=2,
        config=cfg,
    )
    assert loss is not None
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_relational_loss_positive_when_target_differs() -> None:
    context_embeddings = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ]
    )
    target_embeddings = torch.tensor([[[0.5, 0.0], [0.0, 1.5]]])
    mask = torch.ones((2, 2))
    adjacency = torch.ones((2, 2, 2))
    context_enc = _encoding_from_tensors(context_embeddings, mask, adjacency)
    target_enc = _encoding_from_tensors(target_embeddings, mask[:1], adjacency[:1])

    cfg = RelationalConsistencyConfig(weight=0.5, context_self_weight=0.0)
    loss = relational_consistency_loss(
        context_enc,
        target_enc,
        batch_size=1,
        context_length=2,
        config=cfg,
    )
    assert loss is not None
    assert loss.item() > 0.0
