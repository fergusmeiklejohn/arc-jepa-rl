"""Train the object-centric JEPA encoder on manifest datasets."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Mapping

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from arcgen import Grid

try:  # pragma: no cover - handled by ObjectCentricJEPAExperiment
    import torch
    import torch.distributed as dist
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    dist = None  # type: ignore

from training.jepa import ObjectCentricJEPAExperiment, load_jepa_config
from scripts._jepa_loader import build_jepa_dataloader
from training.utils import EarlyStopping, EarlyStoppingConfig


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
    parser.add_argument(
        "--mixed-precision",
        type=str,
        choices=("none", "fp16", "bf16"),
        default=None,
        help="Override mixed precision mode (default: use config)",
    )
    parser.add_argument(
        "--ddp",
        action="store_true",
        help="Enable torch.distributed DistributedDataParallel (single-node).",
    )
    parser.add_argument(
        "--ddp-backend",
        type=str,
        default="nccl",
        help="DDP backend (nccl/gloo).",
    )
    parser.add_argument(
        "--ddp-world-size",
        type=int,
        default=None,
        help="World size override; defaults to env WORLD_SIZE when using torchrun.",
    )
    parser.add_argument(
        "--ddp-rank",
        type=int,
        default=None,
        help="Global rank override; defaults to env RANK when using torchrun.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = dict(load_jepa_config(args.config))

    training_cfg = config.get("training", {})
    if not isinstance(training_cfg, dict):
        raise ValueError("config['training'] must be a mapping")

    if args.mixed_precision is not None:
        training_cfg["mixed_precision"] = args.mixed_precision
    device = args.device or training_cfg.get("device") or "cpu"
    ddp_enabled = bool(args.ddp)

    if ddp_enabled:
        if torch is None or dist is None:
            raise RuntimeError("PyTorch is required for DDP training")
        if not dist.is_available():
            raise RuntimeError("torch.distributed is not available")
        backend = args.ddp_backend
        dist.init_process_group(
            backend=backend,
            world_size=args.ddp_world_size or None,
            rank=args.ddp_rank or None,
        )
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    experiment = ObjectCentricJEPAExperiment(config, device=device)

    if args.dry_run:
        grid = Grid([[0, 1], [0, 1]])
        context_sequence = tuple(grid for _ in range(experiment.context_length))
        result = experiment.train_step([context_sequence], [grid])
        print(f"Dry-run loss: {result.loss:.6f}")
        if ddp_enabled:
            dist.destroy_process_group()
        return

    if torch is None:
        raise RuntimeError("PyTorch is required for full JEPA training")

    train_loader, val_loader, manifest_path, tokenized_path = build_jepa_dataloader(
        config,
        experiment.trainer.tokenizer_config,
    )

    early_stopping_cfg = EarlyStoppingConfig.from_mapping(training_cfg.get("early_stopping"))
    if early_stopping_cfg.enabled and val_loader is None:
        raise ValueError("training.early_stopping.enabled requires data.validation.split to be configured")
    early_stopper = EarlyStopping(early_stopping_cfg)

    epochs = int(training_cfg.get("epochs", 1))
    experiment.set_planned_epochs(epochs)

    total_steps = None
    try:
        steps_per_epoch = len(train_loader)
        if steps_per_epoch > 0:
            optimizer_steps = math.ceil(steps_per_epoch / experiment.grad_accum_steps)
            total_steps = optimizer_steps * max(1, epochs)
    except TypeError:
        total_steps = None
    if total_steps is not None:
        experiment.configure_scheduler(total_steps)

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
    val_losses: list[float] = []
    assert train_loader is not None

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
        completed_epochs = 0
        for epoch in range(1, epochs + 1):
            epoch_loss = experiment.train_epoch(train_loader)
            losses.append(epoch_loss)
            if not ddp_enabled or dist.get_rank() == 0:
                print(f"Epoch {epoch}/{epochs}: loss={epoch_loss:.6f}")

            if logger is not None:
                logger.log_scalar("train/loss", epoch_loss, step=epoch)

            val_loss = None
            stop_training = False
            if val_loader is not None:
                val_loss = experiment.evaluate_epoch(val_loader)
                val_losses.append(val_loss)
                if not ddp_enabled or dist.get_rank() == 0:
                    print(f"  Validation loss: {val_loss:.6f}")
                if logger is not None:
                    logger.log_scalar("val/loss", val_loss, step=epoch)
                if early_stopper.update(val_loss, step=epoch):
                    stop_training = True
                    if not ddp_enabled or dist.get_rank() == 0:
                        print(
                            "  Early stopping triggered: "
                            f"best_val={early_stopper.best_metric:.6f} at epoch {early_stopper.best_step}",
                        )

            loss_events = experiment.consume_loss_metrics()
            if logger is not None:
                for event in loss_events:
                    scalars = {
                        "info_nce": float(event.get("info_nce", 0.0)),
                        "total": float(event.get("total", 0.0)),
                        "sigreg": float(event.get("sigreg", 0.0)),
                    }
                    if "invariance" in event:
                        scalars["invariance"] = float(event["invariance"])
                    if "relational" in event:
                        scalars["relational"] = float(event["relational"])
                    logger.log_scalars("loss", scalars, step=int(event.get("step", epoch)))

            events = experiment.consume_embedding_metrics()
            _handle_embedding_events(events)

            if not ddp_enabled or dist.get_rank() == 0:
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
            completed_epochs = epoch
            if stop_training:
                break

        final_events = experiment.consume_embedding_metrics(flush=True)
        _handle_embedding_events(final_events)
    finally:
        if ddp_enabled and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        if embedding_metrics_handle is not None:
            embedding_metrics_handle.close()
        if logger is not None:
            logger.close()

    metrics = {
        "config": str(args.config),
        "manifest": str(manifest_path) if manifest_path is not None else None,
        "tokenized_dataset": str(tokenized_path) if tokenized_path is not None else None,
        "epochs": epochs,
        "completed_epochs": completed_epochs,
        "batch_size": int(training_cfg.get("batch_size", 32)),
        "device": device,
        "losses": losses,
        "val_losses": val_losses if val_losses else None,
        "best_val_loss": early_stopper.best_metric if val_losses else None,
        "early_stopping_epoch": early_stopper.stopped_step,
    }
    metrics_path = checkpoint_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
