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

from training.eval import (
    EvaluationSuite,
    EvaluationVariant,
    build_summary,
    load_arc_dev_tasks,
    load_synthetic_tasks_jsonl,
)


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
    args = parser.parse_args()
    if bool(args.tasks) == bool(args.arc_dev_root):
        parser.error("Specify exactly one of --tasks or --arc-dev-root")
    return args


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

    suite = EvaluationSuite(tasks)
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
