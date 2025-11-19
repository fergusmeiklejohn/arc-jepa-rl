"""Shared data loader helpers for JEPA scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

try:  # pragma: no cover - torch optional in some environments
    import torch
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    Dataset = object  # type: ignore[misc]

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


def _split_train_validation(
    dataset: "Dataset",
    validation_cfg: Mapping[str, object] | None,
    loader_cfg: Mapping[str, object],
) -> tuple["Dataset", "Dataset | None"]:
    if validation_cfg is None:
        return dataset, None
    if torch is None:
        raise RuntimeError("PyTorch is required for JEPA validation splits")
    if not isinstance(validation_cfg, Mapping):
        raise ValueError("data.validation must be a mapping when provided")
    split_value = validation_cfg.get("split")
    if split_value is None:
        split_value = validation_cfg.get("split_ratio")
    if split_value is None:
        raise ValueError("data.validation must define split or split_ratio")
    split_ratio = float(split_value)
    if not 0.0 < split_ratio < 1.0:
        raise ValueError("validation split must be between 0 and 1")
    dataset_len = len(dataset)
    if dataset_len < 2:
        raise ValueError("dataset too small to create a validation split")
    val_size = max(1, int(round(dataset_len * split_ratio)))
    if val_size >= dataset_len:
        raise ValueError("validation split leaves no training samples")

    seed_override = validation_cfg.get("seed")
    seed_value = int(seed_override) if seed_override is not None else loader_cfg.get("seed")
    generator = torch.Generator()
    if seed_value is not None:
        generator.manual_seed(int(seed_value))
    train_size = dataset_len - val_size
    train_subset, val_subset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)
    return train_subset, val_subset


def _validation_loader_settings(
    loader_cfg: Mapping[str, object],
    validation_cfg: Mapping[str, object] | None,
) -> Mapping[str, object]:
    if validation_cfg is None:
        return loader_cfg
    cfg = dict(loader_cfg)
    cfg["shuffle"] = False
    cfg["drop_last"] = False
    batch_override = validation_cfg.get("batch_size") if isinstance(validation_cfg, Mapping) else None
    if batch_override is not None:
        cfg["batch_size"] = int(batch_override)
    return cfg


def build_manifest_loader(
    config: Mapping[str, object],
    manifest_path: Path,
    tokenizer_config,
) -> tuple["torch.utils.data.DataLoader", "torch.utils.data.DataLoader | None"]:
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
    train_dataset, val_dataset = _split_train_validation(dataset, data_cfg.get("validation"), loader_cfg)
    train_loader = _create_tokenized_loader(train_dataset, loader_cfg)
    val_loader = None
    if val_dataset is not None:
        val_loader_cfg = _validation_loader_settings(loader_cfg, data_cfg.get("validation"))
        val_loader = _create_tokenized_loader(val_dataset, val_loader_cfg)
    return train_loader, val_loader


def build_tokenized_loader(
    config: Mapping[str, object],
    dataset_cfg: Mapping[str, object],
) -> tuple["torch.utils.data.DataLoader", "torch.utils.data.DataLoader | None"]:
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
    data_cfg = config.get("data", {})
    validation_cfg = None
    if isinstance(data_cfg, Mapping):
        validation_cfg = data_cfg.get("validation")
    train_dataset, val_dataset = _split_train_validation(dataset, validation_cfg, loader_cfg)
    train_loader = _create_tokenized_loader(train_dataset, loader_cfg)
    val_loader = None
    if val_dataset is not None:
        val_loader_cfg = _validation_loader_settings(loader_cfg, validation_cfg)
        val_loader = _create_tokenized_loader(val_dataset, val_loader_cfg)
    return train_loader, val_loader


def build_jepa_dataloader(
    config: Mapping[str, object],
    tokenizer_config,
) -> tuple["torch.utils.data.DataLoader", Optional["torch.utils.data.DataLoader"], Optional[Path], Optional[Path]]:
    """Build the JEPA training dataloader plus bookkeeping about its source."""

    tokenized_cfg = config.get("pre_tokenized")
    manifest_path: Path | None = None
    tokenized_path: Path | None = None

    if isinstance(tokenized_cfg, Mapping) and tokenized_cfg.get("path"):
        data_loader, val_loader = build_tokenized_loader(config, tokenized_cfg)
        tokenized_path = Path(tokenized_cfg["path"])
    else:
        manifest_value = config.get("dataset_manifest")
        if not manifest_value:
            raise ValueError("JEPA config must define 'dataset_manifest' when pre_tokenized.path is not provided")
        manifest_path = Path(manifest_value)
        if not manifest_path.exists():
            raise FileNotFoundError(f"dataset manifest not found: {manifest_path}")
        data_loader, val_loader = build_manifest_loader(config, manifest_path, tokenizer_config)

    return data_loader, val_loader, manifest_path, tokenized_path
