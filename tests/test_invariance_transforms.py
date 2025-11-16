"""Unit tests for JEPA invariance token transforms."""

from __future__ import annotations

import torch

from training.jepa.dataset import TokenizedPairBatch
from training.jepa.invariance import (
    InvarianceLossConfig,
    color_permuted_batch,
    symmetry_flipped_batch,
    translated_batch,
)
from training.modules.object_tokenizer import BASE_FEATURE_KEYS


def _make_dummy_batch(feature_dim: int) -> TokenizedPairBatch:
    context = torch.zeros((1, 1, 1, feature_dim), dtype=torch.float32)
    target = torch.zeros((1, 1, feature_dim), dtype=torch.float32)
    context_mask = torch.ones((1, 1, 1), dtype=torch.float32)
    target_mask = torch.ones((1, 1), dtype=torch.float32)
    context_adj = torch.zeros((1, 1, 1, 1), dtype=torch.float32)
    target_adj = torch.zeros((1, 1, 1), dtype=torch.float32)
    return TokenizedPairBatch(
        context_features=context,
        context_mask=context_mask,
        context_adjacency=context_adj,
        target_features=target,
        target_mask=target_mask,
        target_adjacency=target_adj,
        metadata=(None,),
    )


def test_color_permutation_shuffle_preserves_shapes() -> None:
    torch.manual_seed(0)
    feature_dim = len(BASE_FEATURE_KEYS) + 2
    batch = _make_dummy_batch(feature_dim)
    # Populate dominant color and histogram slots
    dom_idx = BASE_FEATURE_KEYS.index("dominant_color")
    batch.context_features[..., dom_idx] = 1.0
    batch.target_features[..., dom_idx] = 2.0
    batch.context_features[..., -2:] = torch.tensor([0.5, 0.25])
    batch.target_features[..., -2:] = torch.tensor([0.75, 0.125])

    cfg = InvarianceLossConfig(color_weight=1.0, palette_size=5)
    mutated = color_permuted_batch(batch, cfg)

    assert mutated.context_features.shape == batch.context_features.shape
    assert mutated.target_features.shape == batch.target_features.shape
    assert not torch.equal(mutated.context_features, batch.context_features)
    assert not torch.equal(mutated.target_features, batch.target_features)
    # Masks/adjs unchanged
    assert torch.equal(mutated.context_mask, batch.context_mask)
    assert torch.equal(mutated.context_adjacency, batch.context_adjacency)


def test_translation_shift_respects_bounds() -> None:
    torch.manual_seed(1)
    feature_dim = len(BASE_FEATURE_KEYS) + 1
    batch = _make_dummy_batch(feature_dim)
    for key in ("centroid_y", "centroid_x", "bbox_min_y", "bbox_max_y", "bbox_min_x", "bbox_max_x"):
        idx = BASE_FEATURE_KEYS.index(key)
        batch.context_features[..., idx] = 0.5
        batch.target_features[..., idx] = 0.25

    cfg = InvarianceLossConfig(translation_weight=1.0, translation_max_delta=0.2)
    mutated = translated_batch(batch, cfg)

    cy_idx = BASE_FEATURE_KEYS.index("centroid_y")
    new_value = mutated.context_features[..., cy_idx].item()
    assert 0.0 <= new_value <= 1.0
    assert new_value != 0.5


def test_horizontal_symmetry_swaps_bounds() -> None:
    torch.manual_seed(2)
    feature_dim = len(BASE_FEATURE_KEYS) + 1
    batch = _make_dummy_batch(feature_dim)
    idx_cx = BASE_FEATURE_KEYS.index("centroid_x")
    idx_min = BASE_FEATURE_KEYS.index("bbox_min_x")
    idx_max = BASE_FEATURE_KEYS.index("bbox_max_x")
    batch.context_features[..., idx_cx] = 0.2
    batch.context_features[..., idx_min] = 0.1
    batch.context_features[..., idx_max] = 0.4

    cfg = InvarianceLossConfig(symmetry_weight=1.0, symmetry_modes=("horizontal",))
    mutated = symmetry_flipped_batch(batch, cfg)

    assert torch.isclose(mutated.context_features[..., idx_cx], torch.tensor(0.8))
    assert torch.isclose(mutated.context_features[..., idx_min], torch.tensor(0.6))
    assert torch.isclose(mutated.context_features[..., idx_max], torch.tensor(0.9))
