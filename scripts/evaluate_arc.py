"""Run ARC evaluation and ablation suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs import TorchUnavailableError
from training.eval import (
    EvaluationSuite,
    EvaluationVariant,
    build_summary,
    load_arc_dev_tasks,
    load_synthetic_tasks_jsonl,
    LatentDistanceTracker,
)
from training.rllib_utils.builders import build_latent_scorer_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ARC evaluation and ablations")
    parser.add_argument(
        "--tasks",
        type=Path,
        default=None,
        help="Path to synthetic JSONL manifest (use --arc-dev-root for ARC dev)",
    )
    parser.add_argument(
        "--arc-dev-root",
        type=Path,
        default=None,
        help="Path to ARC dev (training) directory or JSON file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write JSON summary",
    )
    parser.add_argument(
        "--surprise-tasks",
        type=Path,
        default=None,
        help="Optional JSONL manifest for human-crafted surprise tasks",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top primitives to keep for the meta-guided ablation",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=3,
        help="Maximum program size for the default DSL-only ablation",
    )
    parser.add_argument(
        "--latent-config",
        type=Path,
        default=None,
        help="Optional JEPA config to enable latent distance tracking",
    )
    parser.add_argument(
        "--latent-device",
        type=str,
        default="cpu",
        help="Torch device to use for latent distance tracking (default: cpu)",
    )
    parser.add_argument(
        "--latent-metric",
        type=str,
        default="cosine",
        choices=("cosine", "l2"),
        help="Latent distance metric (matches LatentScorer.metric)",
    )
    args = parser.parse_args()
    if bool(args.tasks) == bool(args.arc_dev_root):
        parser.error("Specify exactly one of --tasks or --arc-dev-root")
    return args


def _build_latent_tracker(
    config_path: Path | str,
    *,
    device: str,
    metric: str,
) -> LatentDistanceTracker:
    scorer = build_latent_scorer_from_config(config_path, device=device)

    def embed(grid):
        return scorer.embed(grid)

    def distance(a, b):
        value = scorer.distance(a, b, metric=metric)
        return float(value.item() if hasattr(value, "item") else value)

    return LatentDistanceTracker(embed, distance)


def build_default_variants(top_k: int, max_nodes: int) -> List[EvaluationVariant]:
    return [
        EvaluationVariant(
            name="dsl_only",
            description="Typed DSL enumeration without meta priors",
            max_nodes=max_nodes,
        ),
        EvaluationVariant(
            name="meta_guided",
            description="Filter primitives using top-k families from Meta-JEPA statistics",
            max_nodes=max_nodes,
            top_k_primitives=top_k,
        ),
    ]


def main() -> None:
    args = parse_args()
    if args.arc_dev_root:
        tasks = load_arc_dev_tasks(args.arc_dev_root)
        dataset = "arc_dev"
        task_source: Path | None = args.arc_dev_root
    else:
        tasks = load_synthetic_tasks_jsonl(args.tasks)
        dataset = "synthetic_jsonl"
        task_source = args.tasks
    if not tasks:
        raise RuntimeError("No evaluation tasks provided")

    latent_tracker = None
    if args.latent_config:
        try:
            latent_tracker = _build_latent_tracker(
                args.latent_config,
                device=args.latent_device,
                metric=args.latent_metric,
            )
        except TorchUnavailableError as exc:  # pragma: no cover - depends on torch install
            raise RuntimeError("Latent distance tracking requires PyTorch") from exc

    suite = EvaluationSuite(tasks, latent_tracker=latent_tracker)
    variants = build_default_variants(args.top_k, args.max_nodes)
    results = suite.run(variants)

    surprise_summary = None
    if args.surprise_tasks:
        surprise_tasks = load_synthetic_tasks_jsonl(args.surprise_tasks)
        if not surprise_tasks:
            raise RuntimeError("No surprise tasks provided")
        surprise_metrics = EvaluationSuite(surprise_tasks).run(variants)
        surprise_summary = {
            "label": "surprise_tasks",
            "task_count": len(surprise_tasks),
            "metrics": surprise_metrics,
            "source": args.surprise_tasks,
        }

    summary = build_summary(
        dataset,
        metrics=results,
        task_count=len(tasks),
        task_source=task_source,
        surprise=surprise_summary,
    )

    print(json.dumps(summary, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
