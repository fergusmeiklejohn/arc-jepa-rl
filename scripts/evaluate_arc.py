"""Run ARC evaluation and ablation suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from training.eval import EvaluationSuite, EvaluationVariant, load_synthetic_tasks_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ARC evaluation and ablations")
    parser.add_argument("--tasks", type=Path, required=True, help="Path to JSONL file of evaluation tasks")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write JSON summary",
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
    return parser.parse_args()


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
    tasks = load_synthetic_tasks_jsonl(args.tasks)
    if not tasks:
        raise RuntimeError("No evaluation tasks provided")

    suite = EvaluationSuite(tasks)
    variants = build_default_variants(args.top_k, args.max_nodes)
    results = suite.run(variants)

    summary = {"results": [metrics.to_dict() for metrics in results]}

    print(json.dumps(summary, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
