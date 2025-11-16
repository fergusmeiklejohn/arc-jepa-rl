"""Train the object-centric JEPA encoder on manifest datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from arcgen import Grid

try:  # pragma: no cover - handled by ObjectCentricJEPAExperiment
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

from training.jepa import ObjectCentricJEPAExperiment, load_jepa_config
from training.jepa.dataset import (
    ManifestTokenizedPairDataset,
    TokenizedPairDataset,
    collate_tokenized_samples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the object-centric JEPA encoder")
    parser.add_argument("--config", type=Path, required=True, help="YAML configuration file")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run a single dummy training step instead of full training",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device to run training on (overrides training.device in config)",
    )
    return parser.parse_args()


def _loader_settings(
    config: dict,
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


def _build_manifest_loader(
    config: dict,
    manifest_path: Path,
    tokenizer_config,
):
    data_cfg = config.get("data", {})
    if not isinstance(data_cfg, Mapping):
        raise ValueError("config['data'] must be a mapping")

    context_window = int(data_cfg.get("context_window", data_cfg.get("context_length", 3)))
    if context_window <= 0:
        raise ValueError("data.context_window must be positive")

    loader_cfg = _loader_settings(config)

    dataset = ManifestTokenizedPairDataset(
        manifest_path,
        context_window=context_window,
        target_offset=int(data_cfg.get("target_offset", 1)),
        augmentations=config.get("augmentations"),
        tokenizer_config=tokenizer_config,
        seed=loader_cfg.get("seed"),
    )
    return _create_tokenized_loader(dataset, loader_cfg)


def _build_tokenized_loader(config: dict, dataset_cfg: Mapping[str, object]):
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

    loader_cfg = _loader_settings(
        config,
        shuffle_override=shuffle_override,
        drop_last_override=drop_last_override,
        seed_override=seed_override,
    )

    dataset = TokenizedPairDataset(dataset_path)
    return _create_tokenized_loader(dataset, loader_cfg)


def main() -> None:
    args = parse_args()
    config = dict(load_jepa_config(args.config))

    training_cfg = config.get("training", {})
    if not isinstance(training_cfg, dict):
        raise ValueError("config['training'] must be a mapping")

    device = args.device or training_cfg.get("device") or "cpu"
    experiment = ObjectCentricJEPAExperiment(config, device=device)

    if args.dry_run:
        grid = Grid([[0, 1], [0, 1]])
        context_sequence = tuple(grid for _ in range(experiment.context_length))
        result = experiment.train_step([context_sequence], [grid])
        print(f"Dry-run loss: {result.loss:.6f}")
        return

    if torch is None:
        raise RuntimeError("PyTorch is required for full JEPA training")

    tokenized_cfg = config.get("pre_tokenized")
    data_loader = None
    manifest_path: Path | None = None
    tokenized_path: Path | None = None

    if isinstance(tokenized_cfg, Mapping) and tokenized_cfg.get("path"):
        data_loader = _build_tokenized_loader(config, tokenized_cfg)
        tokenized_path = Path(tokenized_cfg["path"])
    else:
        manifest_value = config.get("dataset_manifest")
        if not manifest_value:
            raise ValueError("JEPA config must define 'dataset_manifest' when pre_tokenized.path is not provided")
        manifest_path = Path(manifest_value)
        if not manifest_path.exists():
            raise FileNotFoundError(f"dataset manifest not found: {manifest_path}")
        data_loader = _build_manifest_loader(config, manifest_path, experiment.trainer.tokenizer_config)

    epochs = int(training_cfg.get("epochs", 1))

    checkpoint_dir = Path(training_cfg.get("checkpoint_dir", "artifacts/jepa/pretrain"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logging_cfg = config.get("logging", {})
    if logging_cfg and not isinstance(logging_cfg, dict):
        raise ValueError("config['logging'] must be a mapping if provided")

    logger = None
    if isinstance(logging_cfg, dict) and logging_cfg.get("enabled", True):
        from training.utils.logging import create_experiment_logger

        log_dir = logging_cfg.get("log_dir", checkpoint_dir / "tensorboard")
        run_name = logging_cfg.get("run_name")
        flush_secs = int(logging_cfg.get("flush_secs", 15))
        logger = create_experiment_logger(log_dir, run_name=run_name, flush_secs=flush_secs)
        if logging_cfg.get("run_metadata"):
            logger.log_scalars(
                "config/meta",
                {
                    "batch_size": float(training_cfg.get("batch_size", 32)),
                    "epochs": float(epochs),
                },
                step=0,
            )

    losses: list[float] = []
    assert data_loader is not None

    for epoch in range(1, epochs + 1):
        epoch_loss = experiment.train_epoch(data_loader)
        losses.append(epoch_loss)
        print(f"Epoch {epoch}/{epochs}: loss={epoch_loss:.6f}")

        if logger is not None:
            logger.log_scalar("train/loss", epoch_loss, step=epoch)

        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "config": config,
                "model_state": experiment.trainer.encoder.state_dict(),
                "projection_state": experiment.projection_head.state_dict(),
                "optimizer_state": experiment.optimizer.state_dict(),
                "queue_state": experiment.queue.state_dict(),
                "device": device,
            },
            checkpoint_path,
        )

    if logger is not None:
        logger.close()

    metrics = {
        "config": str(args.config),
        "manifest": str(manifest_path) if manifest_path is not None else None,
        "tokenized_dataset": str(tokenized_path) if tokenized_path is not None else None,
        "epochs": epochs,
        "batch_size": int(training_cfg.get("batch_size", 32)),
        "device": device,
        "losses": losses,
    }
    metrics_path = checkpoint_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
