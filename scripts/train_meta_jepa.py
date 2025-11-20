"""Train the Meta-JEPA rule-family encoder from JSONL tasks."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.eval import load_synthetic_tasks_jsonl
from training.meta_jepa import MetaJEPATrainer, TrainingConfig, build_rule_family_examples
from training.utils import EarlyStoppingConfig, GradientClippingConfig, LRSchedulerConfig, MixedPrecisionConfig

try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


def _ensure_torch() -> None:
    if torch is None:  # pragma: no cover - defensive
        raise RuntimeError("PyTorch is required for Meta-JEPA training")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Meta-JEPA on rule family datasets")
    parser.add_argument("--tasks", type=Path, required=True, help="Path to JSONL file of synthetic tasks")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--temperature", type=float, default=0.1, help="Contrastive temperature")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension for graph encoder")
    parser.add_argument("--embedding-dim", type=int, default=64, help="Output embedding dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout for encoder blocks")
    parser.add_argument("--attn-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--attn-layers", type=int, default=2, help="Number of attention layers")
    parser.add_argument(
        "--enable-relational-task",
        action="store_true",
        help="Enable relational prediction auxiliary head",
    )
    parser.add_argument(
        "--relational-weight",
        type=float,
        default=0.0,
        help="Loss weight for the relational auxiliary task",
    )
    parser.add_argument(
        "--temperature-init",
        type=float,
        default=0.1,
        help="Initial temperature when --learnable-temperature is enabled",
    )
    parser.add_argument(
        "--temperature-bounds",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(0.03, 0.3),
        help="Clamping range for learnable temperature",
    )
    parser.add_argument(
        "--learnable-temperature",
        action="store_true",
        help="Enable learnable temperature instead of fixed value",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Torch device (default: cpu)")
    parser.add_argument("--min-family-size", type=int, default=2, help="Minimum examples per rule family")
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.0,
        help="Fraction of data reserved for validation (0 disables)",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Optional seed for deterministic train/val splitting",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Stop after N epochs without validation improvement (0 disables)",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Minimum improvement in validation loss to reset patience",
    )
    parser.add_argument("--grad-clip-norm", type=float, default=0.0, help="Max gradient norm (0 disables)")
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default="none",
        choices=["none", "linear", "cosine"],
        help="Enable LR scheduler",
    )
    parser.add_argument(
        "--lr-warmup-steps",
        type=int,
        default=0,
        help="Warmup steps for LR scheduler",
    )
    parser.add_argument(
        "--lr-min-scale",
        type=float,
        default=0.1,
        help="Minimum LR scale for scheduler decay (0-1)",
    )
    parser.add_argument(
        "--mixed-precision",
        type=str,
        choices=["none", "fp16", "bf16"],
        default="none",
        help="Mixed-precision mode (default: none)",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional path to write trained weights (.pt)")
    return parser.parse_args()


def main() -> None:
    _ensure_torch()
    args = parse_args()
    tasks = load_synthetic_tasks_jsonl(args.tasks)
    if not tasks:
        raise RuntimeError("No tasks loaded from dataset")

    trainer = MetaJEPATrainer.from_tasks(
        tasks,
        min_family_size=args.min_family_size,
        model_kwargs={
            "hidden_dim": args.hidden_dim,
            "embedding_dim": args.embedding_dim,
            "dropout": args.dropout,
            "attn_heads": args.attn_heads,
            "attn_layers": args.attn_layers,
            "relational_decoder": args.enable_relational_task or args.relational_weight > 0,
        },
    )
    if args.early_stopping_patience > 0 and args.val_split <= 0:
        raise ValueError("--early-stopping-patience requires --val-split > 0")
    early_stopping_cfg = EarlyStoppingConfig(
        enabled=args.early_stopping_patience > 0,
        patience=max(1, args.early_stopping_patience or 1),
        min_delta=max(0.0, args.early_stopping_min_delta),
        mode="min",
    )
    config = TrainingConfig(
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        temperature=args.temperature,
        temperature_init=args.temperature_init,
        temperature_bounds=tuple(args.temperature_bounds),
        learnable_temperature=args.learnable_temperature,
        relational_weight=args.relational_weight,
        device=args.device,
        validation_split=max(0.0, args.val_split),
        split_seed=args.split_seed,
        early_stopping=early_stopping_cfg,
        grad_clip=GradientClippingConfig(max_norm=max(0.0, args.grad_clip_norm)),
        lr_scheduler=LRSchedulerConfig(
            name=args.lr_scheduler,
            warmup_steps=max(0, args.lr_warmup_steps),
            min_lr_scale=max(0.0, min(1.0, args.lr_min_scale)),
        ),
        mixed_precision=MixedPrecisionConfig(mode=args.mixed_precision),
    )
    result = trainer.fit(config)

    print("Finished training Meta-JEPA")
    for epoch, loss in enumerate(result.history, start=1):
        print(f"Epoch {epoch}: loss={loss:.4f}")
    if result.val_history:
        for epoch, loss in enumerate(result.val_history, start=1):
            print(f"  Val {epoch}: loss={loss:.4f}")
        print(f"Best validation loss: {min(result.val_history):.4f}")
    print(f"Final temperature: {result.temperature:.4f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": trainer.model.state_dict(), "history": result.history}, args.output)
        print(f"Saved weights to {args.output}")


if __name__ == "__main__":
    main()
