"""Pretrain option policies via behavioral cloning on recorded traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Tuple

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.rllib_utils.bc_data import BehavioralCloningDataset, load_option_traces, split_records
from training.rllib_utils.bc_trainer import train_behavioral_cloning


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to BC YAML config")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/bc"), help="Where to store checkpoints/metrics")
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("BC config must be a mapping")
    return data


def prepare_datasets(config: Dict[str, Any]) -> Tuple[BehavioralCloningDataset, BehavioralCloningDataset]:
    dataset_cfg = config.get("dataset") or {}
    traces_path = dataset_cfg.get("path")
    if not traces_path:
        raise ValueError("dataset.path is required")
    trace_records = load_option_traces(Path(traces_path), max_steps=dataset_cfg.get("max_steps"))

    val_path = dataset_cfg.get("val_path")
    if val_path:
        val_records = load_option_traces(Path(val_path), max_steps=dataset_cfg.get("max_steps"))
        train_records = trace_records
    else:
        val_ratio = float(dataset_cfg.get("val_ratio", 0.2))
        seed = int(dataset_cfg.get("seed", 0))
        train_records, val_records = split_records(trace_records, val_ratio=val_ratio, seed=seed)

    return BehavioralCloningDataset(train_records), BehavioralCloningDataset(val_records)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    train_dataset, val_dataset = prepare_datasets(config)

    result = train_behavioral_cloning(
        train_dataset,
        val_dataset,
        model_cfg=config.get("model", {}),
        optim_cfg=config.get("training", {}),
        output_dir=args.output_dir,
    )
    summary_path = args.output_dir / "run_summary.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(f"Saved BC artifacts to {args.output_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()
