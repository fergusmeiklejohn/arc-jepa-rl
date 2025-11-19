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
from scripts._jepa_loader import build_jepa_dataloader


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

    data_loader, manifest_path, tokenized_path = build_jepa_dataloader(
        config,
        experiment.trainer.tokenizer_config,
    )

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

    embedding_metrics_handle = None
    embedding_metrics_path = checkpoint_dir / "embedding_metrics.jsonl"

    def _handle_embedding_events(events: list[dict[str, object]]) -> None:
        nonlocal embedding_metrics_handle
        if not events:
            return
        if logger is not None:
            for event in events:
                context_metrics = event.get("context")
                target_metrics = event.get("target")
                step = int(event.get("step", 0))
                if context_metrics:
                    logger.log_scalars("diagnostics/context", context_metrics, step=step)
                if target_metrics:
                    logger.log_scalars("diagnostics/target", target_metrics, step=step)
                if "vq_usage_ratio" in event:
                    logger.log_scalars(
                        "diagnostics/vq",
                        {
                            "usage_ratio": float(event["vq_usage_ratio"]),
                            "active_codes": float(event.get("vq_active_codes", 0)),
                        },
                        step=step,
                    )
        if embedding_metrics_handle is None:
            embedding_metrics_handle = embedding_metrics_path.open("a", encoding="utf-8")
        for event in events:
            embedding_metrics_handle.write(json.dumps(event) + "\n")
        embedding_metrics_handle.flush()

    try:
        for epoch in range(1, epochs + 1):
            epoch_loss = experiment.train_epoch(data_loader)
            losses.append(epoch_loss)
            print(f"Epoch {epoch}/{epochs}: loss={epoch_loss:.6f}")

            if logger is not None:
                logger.log_scalar("train/loss", epoch_loss, step=epoch)

            events = experiment.consume_embedding_metrics()
            _handle_embedding_events(events)

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

        final_events = experiment.consume_embedding_metrics(flush=True)
        _handle_embedding_events(final_events)
    finally:
        if embedding_metrics_handle is not None:
            embedding_metrics_handle.close()
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
