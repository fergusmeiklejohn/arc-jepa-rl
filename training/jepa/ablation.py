"""Utilities for running LeJEPA ablation sweeps."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping

from training.jepa import load_jepa_config


def _deep_update(base: dict, updates: Mapping[str, object]) -> dict:
    """Recursively merge ``updates`` into ``base`` without mutating inputs."""

    merged = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, Mapping):
            merged[key] = _deep_update(merged.get(key, {}) if isinstance(merged.get(key), Mapping) else {}, value)
        else:
            merged[key] = value
    return merged


@dataclass(frozen=True)
class AblationVariant:
    name: str
    description: str
    overrides: Mapping[str, object]


def build_ablation_variants() -> List[AblationVariant]:
    """Define the LeJEPA ablation matrix."""

    variants = [
        AblationVariant(
            name="baseline_infonce",
            description="InfoNCE-only encoder (no VQ, no relational, invariance off, SIGReg off).",
            overrides={
                "object_encoder": {
                    "num_embeddings": None,
                    "relational": False,
                },
                "invariance": {"enabled": False, "color_weight": 0.0, "translation_weight": 0.0, "symmetry_weight": 0.0},
                "sigreg": {"weight": 0.0},
                "diagnostics": {"embedding_metrics": {"enabled": True, "interval": 1, "max_samples": 64}},
            },
        ),
        AblationVariant(
            name="vq_only",
            description="+VQ bottleneck, otherwise baseline.",
            overrides={
                "object_encoder": {
                    "num_embeddings": 128,
                    "relational": False,
                },
                "invariance": {"enabled": False},
                "sigreg": {"weight": 0.0},
                "diagnostics": {"embedding_metrics": {"enabled": True, "interval": 1, "max_samples": 64}},
            },
        ),
        AblationVariant(
            name="vq_relational",
            description="+VQ + relational attention stack.",
            overrides={
                "object_encoder": {
                    "num_embeddings": 128,
                    "relational": True,
                    "relational_layers": 2,
                    "relational_heads": 4,
                },
                "invariance": {"enabled": False},
                "sigreg": {"weight": 0.0},
                "diagnostics": {"embedding_metrics": {"enabled": True, "interval": 1, "max_samples": 64}},
            },
        ),
        AblationVariant(
            name="vq_relational_invariance",
            description="+VQ + relational + invariance penalties (color/translation/symmetry).",
            overrides={
                "object_encoder": {
                    "num_embeddings": 128,
                    "relational": True,
                    "relational_layers": 2,
                    "relational_heads": 4,
                },
                "invariance": {
                    "enabled": True,
                    "color_weight": 0.1,
                    "translation_weight": 0.1,
                    "translation_max_delta": 1,
                    "symmetry_weight": 0.05,
                    "symmetry_modes": ["h", "v"],
                },
                "sigreg": {"weight": 0.0},
                "diagnostics": {"embedding_metrics": {"enabled": True, "interval": 1, "max_samples": 64}},
            },
        ),
        AblationVariant(
            name="vq_relational_invariance_sigreg",
            description="+VQ + relational + invariance + SIGReg regularisation.",
            overrides={
                "object_encoder": {
                    "num_embeddings": 128,
                    "relational": True,
                    "relational_layers": 2,
                    "relational_heads": 4,
                },
                "invariance": {
                    "enabled": True,
                    "color_weight": 0.1,
                    "translation_weight": 0.1,
                    "translation_max_delta": 1,
                    "symmetry_weight": 0.05,
                    "symmetry_modes": ["h", "v"],
                },
                "sigreg": {"weight": 0.05, "num_slices": 32, "num_points": 17},
                "diagnostics": {"embedding_metrics": {"enabled": True, "interval": 1, "max_samples": 64}},
            },
        ),
    ]
    return variants


def load_base_config(path: str | Path) -> Dict[str, object]:
    """Load a base JEPA YAML config."""

    cfg_path = Path(path)
    return dict(load_jepa_config(cfg_path))


def apply_variant(base_config: Mapping[str, object], variant: AblationVariant) -> Dict[str, object]:
    """Apply variant overrides to a base config."""

    return _deep_update(dict(base_config), variant.overrides)


__all__ = ["AblationVariant", "apply_variant", "build_ablation_variants", "load_base_config"]
