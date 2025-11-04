"""CLI stub for running the object-centric JEPA encoder training."""

from __future__ import annotations

import argparse
from pathlib import Path

from arcgen import Grid

from training.jepa import ObjectCentricJEPAExperiment, load_jepa_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the object-centric JEPA encoder")
    parser.add_argument("--config", type=Path, required=True, help="YAML configuration file")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run a single dummy training step instead of full training",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_jepa_config(args.config)
    experiment = ObjectCentricJEPAExperiment(config)

    if args.dry_run:
        grid = Grid([[0, 1], [0, 1]])
        result = experiment.train_step([grid], [grid])
        print(f"Dry-run loss: {result.loss:.6f}")
        return

    raise NotImplementedError("Dataset loading and full JEPA training loop are pending implementation")


if __name__ == "__main__":
    main()
