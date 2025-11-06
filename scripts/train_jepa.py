"""Train the object-centric JEPA encoder on manifest datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from arcgen import Grid

try:  # pragma: no cover - handled by ObjectCentricJEPAExperiment
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

from training.jepa import ObjectCentricJEPAExperiment, load_jepa_config
from training.jepa.dataset import ManifestGridPairDataset


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


def _build_dataset(config: dict, manifest_path: Path) -> ManifestGridPairDataset:
    data_cfg = config.get("data", {})
    if not isinstance(data_cfg, dict):
        raise ValueError("config['data'] must be a mapping")

    training_cfg = config.get("training", {})
    if not isinstance(training_cfg, dict):
        raise ValueError("config['training'] must be a mapping")

    batch_size = int(training_cfg.get("batch_size", 32))
    shuffle = bool(data_cfg.get("shuffle", True))
    drop_last = bool(data_cfg.get("drop_last", False) or training_cfg.get("drop_last", False))
    dataset_seed = data_cfg.get("seed", config.get("seed"))

    return ManifestGridPairDataset(
        manifest_path,
        batch_size=batch_size,
        context_window=int(data_cfg.get("context_window", 1)),
        target_offset=int(data_cfg.get("target_offset", 1)),
        shuffle=shuffle,
        drop_last=drop_last,
        augmentations=config.get("augmentations"),
        seed=dataset_seed,
    )


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
        result = experiment.train_step([grid], [grid])
        print(f"Dry-run loss: {result.loss:.6f}")
        return

    if torch is None:
        raise RuntimeError("PyTorch is required for full JEPA training")

    manifest_value = config.get("dataset_manifest")
    if not manifest_value:
        raise ValueError("JEPA config must define 'dataset_manifest'")
    manifest_path = Path(manifest_value)
    if not manifest_path.exists():
        raise FileNotFoundError(f"dataset manifest not found: {manifest_path}")

    dataset = _build_dataset(config, manifest_path)
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
    for epoch in range(1, epochs + 1):
        epoch_loss = experiment.train_epoch(dataset)
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
        "manifest": str(manifest_path),
        "epochs": epochs,
        "batch_size": int(training_cfg.get("batch_size", 32)),
        "device": device,
        "losses": losses,
    }
    metrics_path = checkpoint_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
