"""Entry point for generating synthetic ARC datasets.

This script will load a YAML configuration and hand the parameters to the
`arcgen` data generator once implemented.
"""

from __future__ import annotations

import argparse
import pathlib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic ARC data")
    parser.add_argument("--config", type=pathlib.Path, required=True, help="Path to YAML config")
    parser.add_argument("--output", type=pathlib.Path, default=None, help="Optional override for output root")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Placeholder until generator exists. This ensures the script can be wired into CI early.
    raise NotImplementedError(
        f"Dataset generation not yet implemented. Config received: {args.config}"
    )


if __name__ == "__main__":
    main()
