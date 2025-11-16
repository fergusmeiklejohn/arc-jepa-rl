"""Invariance-aware token transformations for JEPA training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from training.jepa.dataset import TokenizedPairBatch
from training.modules.object_tokenizer import BASE_FEATURE_KEYS


FEATURE_INDEX = {name: idx for idx, name in enumerate(BASE_FEATURE_KEYS)}
COLOR_HIST_START = len(BASE_FEATURE_KEYS)


@dataclass(frozen=True)
class InvarianceLossConfig:
    """Configuration describing invariance penalties."""

    color_weight: float = 0.0
    translation_weight: float = 0.0
    translation_max_delta: float = 0.05
    symmetry_weight: float = 0.0
    palette_size: int = 10
    symmetry_modes: Sequence[str] = ("horizontal", "vertical")
    coordinate_min: float = 0.0
    coordinate_max: float = 1.0

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None) -> "InvarianceLossConfig":
        if data is None:
            return cls()
        modes = data.get("symmetry_modes", cls.symmetry_modes)
        if isinstance(modes, Sequence):
            modes_value: Sequence[str] = tuple(str(mode) for mode in modes if str(mode))
        else:
            modes_value = cls.symmetry_modes
        return cls(
            color_weight=float(data.get("color_weight", cls.color_weight)),
            translation_weight=float(data.get("translation_weight", cls.translation_weight)),
            translation_max_delta=float(data.get("translation_max_delta", cls.translation_max_delta)),
            symmetry_weight=float(data.get("symmetry_weight", cls.symmetry_weight)),
            palette_size=int(data.get("palette_size", cls.palette_size)),
            symmetry_modes=modes_value if modes_value else cls.symmetry_modes,
            coordinate_min=float(data.get("coordinate_min", cls.coordinate_min)),
            coordinate_max=float(data.get("coordinate_max", cls.coordinate_max)),
        )

    @property
    def enabled(self) -> bool:
        return any(weight > 0.0 for weight in (self.color_weight, self.translation_weight, self.symmetry_weight))


def clone_token_batch(batch: TokenizedPairBatch) -> TokenizedPairBatch:
    return TokenizedPairBatch(
        context_features=batch.context_features.clone(),
        context_mask=batch.context_mask,
        context_adjacency=batch.context_adjacency,
        target_features=batch.target_features.clone(),
        target_mask=batch.target_mask,
        target_adjacency=batch.target_adjacency,
        metadata=batch.metadata,
    )


def color_permuted_batch(batch: TokenizedPairBatch, cfg: InvarianceLossConfig) -> TokenizedPairBatch:
    mutated = clone_token_batch(batch)
    _permute_color_tokens(mutated.context_features, mutated.context_mask, cfg)
    _permute_color_tokens(mutated.target_features, mutated.target_mask, cfg)
    return mutated


def translated_batch(batch: TokenizedPairBatch, cfg: InvarianceLossConfig) -> TokenizedPairBatch:
    mutated = clone_token_batch(batch)
    _apply_translation(mutated.context_features, mutated.context_mask, cfg)
    _apply_translation(mutated.target_features, mutated.target_mask, cfg)
    return mutated


def symmetry_flipped_batch(batch: TokenizedPairBatch, cfg: InvarianceLossConfig) -> TokenizedPairBatch:
    mutated = clone_token_batch(batch)
    _apply_symmetry(mutated.context_features, mutated.context_mask, cfg)
    _apply_symmetry(mutated.target_features, mutated.target_mask, cfg)
    return mutated


def _active_mask(mask: torch.Tensor | None) -> torch.Tensor | None:
    if mask is None:
        return None
    return (mask > 0.0).unsqueeze(-1)


def _permute_color_tokens(features: torch.Tensor, mask: torch.Tensor | None, cfg: InvarianceLossConfig) -> None:
    if cfg.palette_size <= 1 or features.size(-1) <= COLOR_HIST_START:
        return
    device = features.device
    mask_factor = _active_mask(mask)

    dom_idx = FEATURE_INDEX["dominant_color"]
    permutation = torch.randperm(cfg.palette_size, device=device).float()
    dominant = features[..., dom_idx]
    dominant_indices = dominant.round().clamp(0, cfg.palette_size - 1).long()
    remapped = permutation[dominant_indices]
    if mask_factor is not None:
        features[..., dom_idx] = torch.where(mask_factor.squeeze(-1), remapped, dominant)
    else:
        features[..., dom_idx] = remapped

    hist = features[..., COLOR_HIST_START:]
    if hist.numel() == 0:
        return
    hist_perm = torch.randperm(hist.size(-1), device=device)
    shuffled = hist.index_select(-1, hist_perm)
    if mask_factor is not None:
        mask_expand = mask_factor.expand_as(shuffled)
        features[..., COLOR_HIST_START:] = torch.where(mask_expand, shuffled, hist)
    else:
        features[..., COLOR_HIST_START:] = shuffled


def _apply_translation(features: torch.Tensor, mask: torch.Tensor | None, cfg: InvarianceLossConfig) -> None:
    if cfg.translation_weight <= 0.0 or cfg.translation_max_delta <= 0.0:
        return
    device = features.device
    coord_min = cfg.coordinate_min
    coord_max = cfg.coordinate_max
    delta_shape = features.shape[:-1]
    delta_y = (torch.rand(delta_shape, device=device) * 2 - 1) * cfg.translation_max_delta
    delta_x = (torch.rand(delta_shape, device=device) * 2 - 1) * cfg.translation_max_delta
    if mask is not None:
        delta_y = delta_y * (mask > 0.0)
        delta_x = delta_x * (mask > 0.0)

    cy_idx = FEATURE_INDEX["centroid_y"]
    cx_idx = FEATURE_INDEX["centroid_x"]
    min_y_idx = FEATURE_INDEX["bbox_min_y"]
    max_y_idx = FEATURE_INDEX["bbox_max_y"]
    min_x_idx = FEATURE_INDEX["bbox_min_x"]
    max_x_idx = FEATURE_INDEX["bbox_max_x"]

    features[..., cy_idx] = torch.clamp(features[..., cy_idx] + delta_y, coord_min, coord_max)
    features[..., min_y_idx] = torch.clamp(features[..., min_y_idx] + delta_y, coord_min, coord_max)
    features[..., max_y_idx] = torch.clamp(features[..., max_y_idx] + delta_y, coord_min, coord_max)
    features[..., cx_idx] = torch.clamp(features[..., cx_idx] + delta_x, coord_min, coord_max)
    features[..., min_x_idx] = torch.clamp(features[..., min_x_idx] + delta_x, coord_min, coord_max)
    features[..., max_x_idx] = torch.clamp(features[..., max_x_idx] + delta_x, coord_min, coord_max)


def _apply_symmetry(features: torch.Tensor, mask: torch.Tensor | None, cfg: InvarianceLossConfig) -> None:
    if cfg.symmetry_weight <= 0.0 or not cfg.symmetry_modes:
        return

    device = features.device
    mode_idx = torch.randint(0, len(cfg.symmetry_modes), (1,), device=device).item()
    mode = cfg.symmetry_modes[mode_idx]
    coord_min = cfg.coordinate_min
    coord_max = cfg.coordinate_max
    reflect = coord_min + coord_max

    mask_factor = _active_mask(mask)
    if mode == "horizontal":
        _flip_axis(
            features,
            FEATURE_INDEX["centroid_x"],
            FEATURE_INDEX["bbox_min_x"],
            FEATURE_INDEX["bbox_max_x"],
            reflect,
            coord_min,
            coord_max,
            mask_factor,
        )
    else:
        _flip_axis(
            features,
            FEATURE_INDEX["centroid_y"],
            FEATURE_INDEX["bbox_min_y"],
            FEATURE_INDEX["bbox_max_y"],
            reflect,
            coord_min,
            coord_max,
            mask_factor,
        )


def _flip_axis(
    features: torch.Tensor,
    centroid_idx: int,
    min_idx: int,
    max_idx: int,
    reflect_sum: float,
    coord_min: float,
    coord_max: float,
    mask_factor: torch.Tensor | None,
) -> None:
    centroid = features[..., centroid_idx]
    min_val = features[..., min_idx]
    max_val = features[..., max_idx]

    new_centroid = torch.clamp(reflect_sum - centroid, coord_min, coord_max)
    new_min = torch.clamp(reflect_sum - max_val, coord_min, coord_max)
    new_max = torch.clamp(reflect_sum - min_val, coord_min, coord_max)

    if mask_factor is not None:
        mask_scalar = mask_factor.squeeze(-1)
        features[..., centroid_idx] = torch.where(mask_scalar, new_centroid, centroid)
        features[..., min_idx] = torch.where(mask_scalar, new_min, min_val)
        features[..., max_idx] = torch.where(mask_scalar, new_max, max_val)
    else:
        features[..., centroid_idx] = new_centroid
        features[..., min_idx] = new_min
        features[..., max_idx] = new_max
