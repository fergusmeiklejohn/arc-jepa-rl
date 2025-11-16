"""Shared data loader helpers for JEPA scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

try:  # pragma: no cover - torch optional in some environments
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

from training.jepa.dataset import (
    ManifestTokenizedPairDataset,
    TokenizedPairDataset,
    collate_tokenized_samples,
)

__all__ = [
    "build_manifest_loader",
    "build_tokenized_loader",
    "build_jepa_dataloader",
    "loader_settings",
]


def loader_settings(
    config: Mapping[str, object],
    *,
    shuffle_override: Optional[bool] = None,
    drop_last_override: Optional[bool] = None,
    seed_override: Optional[int] = None,
) -> dict:
    data_cfg = config.get("data", {})
    if not isinstance(data_cfg, Mapping):
        raise ValueError("config['data'] must be a mapping")

    training_cfg = config.get("training", {})
    if not isinstance(training_cfg, Mapping):
        raise ValueError("config['training'] must be a mapping")

    batch_size = int(training_cfg.get("batch_size", 32))
    shuffle = bool(shuffle_override if shuffle_override is not None else data_cfg.get("shuffle", True))
    drop_last_cfg = bool(data_cfg.get("drop_last", False))
    training_drop_last = bool(training_cfg.get("drop_last", False))
    drop_last_value = drop_last_override if drop_last_override is not None else (drop_last_cfg or training_drop_last)
    num_workers = int(training_cfg.get("num_workers", 0))
    pin_memory = bool(training_cfg.get("pin_memory", False))
    seed_value = seed_override if seed_override is not None else data_cfg.get("seed", config.get("seed"))

    generator = None
    if torch is not None and seed_value is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed_value))

    return {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "drop_last": bool(drop_last_value),
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "generator": generator,
        "seed": seed_value,
    }


def _create_tokenized_loader(dataset, loader_cfg: Mapping[str, object]):
    if torch is None:
        raise RuntimeError("PyTorch is required for JEPA training")

    num_workers = int(loader_cfg.get("num_workers", 0))
    loader_kwargs = {
        "batch_size": int(loader_cfg.get("batch_size", 32)),
        "shuffle": bool(loader_cfg.get("shuffle", True)),
        "drop_last": bool(loader_cfg.get("drop_last", False)),
        "collate_fn": collate_tokenized_samples,
        "num_workers": num_workers,
        "pin_memory": bool(loader_cfg.get("pin_memory", False)),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    generator = loader_cfg.get("generator")
    if generator is not None:
        loader_kwargs["generator"] = generator
    return torch.utils.data.DataLoader(dataset, **loader_kwargs)


def build_manifest_loader(
    config: Mapping[str, object],
    manifest_path: Path,
    tokenizer_config,
):
    data_cfg = config.get("data", {})
    if not isinstance(data_cfg, Mapping):
        raise ValueError("config['data'] must be a mapping")

    context_window = int(data_cfg.get("context_window", data_cfg.get("context_length", 3)))
    if context_window <= 0:
        raise ValueError("data.context_window must be positive")

    loader_cfg = loader_settings(config)

    dataset = ManifestTokenizedPairDataset(
        manifest_path,
        context_window=context_window,
        target_offset=int(data_cfg.get("target_offset", 1)),
        augmentations=config.get("augmentations"),
        tokenizer_config=tokenizer_config,
        seed=loader_cfg.get("seed"),
    )
    return _create_tokenized_loader(dataset, loader_cfg)


def build_tokenized_loader(config: Mapping[str, object], dataset_cfg: Mapping[str, object]):
    if torch is None:
        raise RuntimeError("PyTorch is required for tokenized dataset loading")
    if not isinstance(dataset_cfg, Mapping):
        raise ValueError("config['pre_tokenized'] must be a mapping")

    dataset_path = dataset_cfg.get("path")
    if not dataset_path:
        raise ValueError("pre_tokenized.path must be set when using pre-tokenized data")

    shuffle_override = dataset_cfg.get("shuffle")
    if shuffle_override is not None:
        shuffle_override = bool(shuffle_override)
    drop_last_override = dataset_cfg.get("drop_last")
    if drop_last_override is not None:
        drop_last_override = bool(drop_last_override)
    seed_override = dataset_cfg.get("seed")
    seed_override = int(seed_override) if seed_override is not None else None

    loader_cfg = loader_settings(
        config,
        shuffle_override=shuffle_override,
        drop_last_override=drop_last_override,
        seed_override=seed_override,
    )

    dataset = TokenizedPairDataset(dataset_path)
    return _create_tokenized_loader(dataset, loader_cfg)


def build_jepa_dataloader(
    config: Mapping[str, object],
    tokenizer_config,
) -> tuple["torch.utils.data.DataLoader", Optional[Path], Optional[Path]]:
    """Build the JEPA training dataloader plus bookkeeping about its source."""

    tokenized_cfg = config.get("pre_tokenized")
    manifest_path: Path | None = None
    tokenized_path: Path | None = None

    if isinstance(tokenized_cfg, Mapping) and tokenized_cfg.get("path"):
        data_loader = build_tokenized_loader(config, tokenized_cfg)
        tokenized_path = Path(tokenized_cfg["path"])
    else:
        manifest_value = config.get("dataset_manifest")
        if not manifest_value:
            raise ValueError("JEPA config must define 'dataset_manifest' when pre_tokenized.path is not provided")
        manifest_path = Path(manifest_value)
        if not manifest_path.exists():
            raise FileNotFoundError(f"dataset manifest not found: {manifest_path}")
        data_loader = build_manifest_loader(config, manifest_path, tokenizer_config)

    return data_loader, manifest_path, tokenized_path
