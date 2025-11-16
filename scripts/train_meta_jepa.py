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
    )
    config = TrainingConfig(
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        temperature=args.temperature,
        temperature_init=args.temperature_init,
        temperature_bounds=tuple(args.temperature_bounds),
        learnable_temperature=args.learnable_temperature,
        device=args.device,
    )
    result = trainer.fit(config)

    print("Finished training Meta-JEPA")
    for epoch, loss in enumerate(result.history, start=1):
        print(f"Epoch {epoch}: loss={loss:.4f}")
    print(f"Final temperature: {result.temperature:.4f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": trainer.model.state_dict(), "history": result.history}, args.output)
        print(f"Saved weights to {args.output}")


if __name__ == "__main__":
    main()
