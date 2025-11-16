"""Surface and optionally promote composite options from RL traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.dsl.primitives import build_default_primitive_registry
from training.options import (
    discover_option_sequences,
    promote_discovered_option,
)
from training.options.traces import load_option_episodes_from_traces
from training.rllib_utils import build_options_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-config", type=Path, required=True, help="YAML env config describing available options")
    parser.add_argument("--traces", type=Path, required=True, help="JSON/JSONL file with RL/rollout traces")
    parser.add_argument("--output", type=Path, default=Path("artifacts/options/discovered_options.json"), help="Where to store the discovery summary")
    parser.add_argument("--min-support", type=int, default=2, help="Minimum support count for discovered sequences")
    parser.add_argument("--max-length", type=int, default=3, help="Maximum sequence length to mine")
    parser.add_argument("--min-success", type=float, default=0.6, help="Minimum empirical success rate")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional cap on steps per episode when loading traces")
    parser.add_argument("--allow-singleton", action="store_true", help="Allow single-option discoveries (defaults to False)")
    parser.add_argument("--promote", action="store_true", help="Register discovered options in a fresh DSL registry for validation")
    return parser.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at top of {path}")
    return data


def summarise(discovered) -> List[Dict[str, Any]]:
    summary = []
    for item in discovered:
        summary.append(
            {
                "name": item.name,
                "sequence": list(item.sequence_names),
                "support": item.support,
                "success_rate": item.success_rate,
                "avg_reward": item.avg_reward,
            }
        )
    return summary


def main() -> None:
    args = parse_args()
    env_cfg = load_yaml(args.env_config)
    options = build_options_from_config(env_cfg.get("options"))
    if not options:
        raise ValueError("env options config produced an empty option list")

    episodes = load_option_episodes_from_traces(args.traces, options, max_steps=args.max_steps)

    discovered = discover_option_sequences(
        episodes,
        min_support=args.min_support,
        max_sequence_length=args.max_length,
        min_success_rate=args.min_success,
        allow_singleton=args.allow_singleton,
    )

    promoted: List[str] = []
    if args.promote and discovered:
        registry = build_default_primitive_registry()
        for option in discovered:
            promoted.append(promote_discovered_option(option, registry))

    total_steps = sum(len(ep.steps) for ep in episodes)
    output = {
        "env_config": str(args.env_config),
        "trace_file": str(args.traces),
        "episodes": len(episodes),
        "steps": total_steps,
        "parameters": {
            "min_support": args.min_support,
            "max_length": args.max_length,
            "min_success_rate": args.min_success,
            "allow_singleton": args.allow_singleton,
            "max_steps": args.max_steps,
        },
        "discovered_options": summarise(discovered),
        "promoted": promoted,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)

    print(f"Processed {len(episodes)} episodes / {total_steps} steps")
    print(f"Discovered {len(discovered)} option(s); summary saved to {args.output}")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()

