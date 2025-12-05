"""Convert real ARC tasks from arckit to JEPA manifest format.

This script loads the official ARC-AGI training set via arckit and converts
each task's input/output examples to our manifest.jsonl format for JEPA training.

Usage:
    pip install -U arckit
    python scripts/convert_arc_to_manifest.py --output data/arc_real/manifest.jsonl

The output manifest contains one entry per input→output example, with:
- id: task_id + example index
- input: the input grid
- output: the output grid
- metadata: task_id, example_index, dataset source
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    import arckit
except ImportError:
    print("Error: arckit not installed. Run: pip install -U arckit")
    sys.exit(1)

import numpy as np


def grid_to_list(grid: np.ndarray) -> list[list[int]]:
    """Convert numpy grid to nested list for JSON serialization."""
    return grid.tolist()


def convert_task_to_examples(task: Any, dataset_name: str = "arcagi") -> list[dict]:
    """Convert a single ARC task to manifest entries.

    Each task has multiple train examples (input→output pairs).
    We create one manifest entry per example.
    """
    examples = []
    task_id = task.id

    # Process training examples
    for idx, (input_grid, output_grid) in enumerate(task.train):
        entry = {
            "id": f"{dataset_name}_{task_id}_train_{idx:02d}",
            "input": grid_to_list(input_grid),
            "output": grid_to_list(output_grid),
            "metadata": {
                "task_id": task_id,
                "example_type": "train",
                "example_index": idx,
                "dataset": dataset_name,
                "input_height": input_grid.shape[0],
                "input_width": input_grid.shape[1],
                "output_height": output_grid.shape[0],
                "output_width": output_grid.shape[1],
            }
        }
        examples.append(entry)

    # Optionally process test examples (input only, no output for evaluation)
    # For JEPA training, we skip test examples since we need input→output pairs

    return examples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert ARC tasks from arckit to manifest format"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/arc_real/manifest.jsonl"),
        help="Output manifest file path",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["arcagi", "kaggle", "arcagi1"],
        default="arcagi",
        help="Which ARC dataset to load (default: arcagi = ARC-AGI-2)",
    )
    parser.add_argument(
        "--include-eval",
        action="store_true",
        help="Include evaluation set tasks (default: train only)",
    )
    args = parser.parse_args()

    # Load ARC data
    print(f"Loading ARC dataset: {args.dataset}")
    if args.dataset == "arcagi1":
        train_set, eval_set = arckit.load_data("arcagi")
    elif args.dataset == "kaggle":
        train_set, eval_set = arckit.load_data("kaggle")
    else:
        train_set, eval_set = arckit.load_data()

    print(f"Loaded {len(train_set)} training tasks")
    if args.include_eval:
        print(f"Loaded {len(eval_set)} evaluation tasks")

    # Convert all tasks
    all_examples = []

    for task in train_set:
        examples = convert_task_to_examples(task, args.dataset)
        all_examples.extend(examples)

    if args.include_eval:
        for task in eval_set:
            examples = convert_task_to_examples(task, f"{args.dataset}_eval")
            all_examples.extend(examples)

    print(f"Total examples: {len(all_examples)}")

    # Write manifest
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for entry in all_examples:
            f.write(json.dumps(entry) + "\n")

    print(f"Written to: {output_path}")

    # Print summary statistics
    input_heights = [e["metadata"]["input_height"] for e in all_examples]
    input_widths = [e["metadata"]["input_width"] for e in all_examples]
    output_heights = [e["metadata"]["output_height"] for e in all_examples]
    output_widths = [e["metadata"]["output_width"] for e in all_examples]

    print("\nStatistics:")
    print(f"  Input grid sizes: {min(input_heights)}x{min(input_widths)} to {max(input_heights)}x{max(input_widths)}")
    print(f"  Output grid sizes: {min(output_heights)}x{min(output_widths)} to {max(output_heights)}x{max(output_widths)}")

    # Count unique tasks
    unique_tasks = set(e["metadata"]["task_id"] for e in all_examples)
    print(f"  Unique tasks: {len(unique_tasks)}")
    print(f"  Examples per task: {len(all_examples) / len(unique_tasks):.1f} avg")


if __name__ == "__main__":
    main()
