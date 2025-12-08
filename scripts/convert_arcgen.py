#!/usr/bin/env python3
"""Convert ARC-GEN dataset to manifest.jsonl format for JEPA training.

Usage:
    python scripts/convert_arcgen.py data/arc_gen_100k/raw data/arc_gen_100k/manifest.jsonl

The ARC-GEN-100K dataset from Kaggle contains JSON files with structure:
    {
        "task_id": "007bbfb7",
        "input": [[...]],
        "output": [[...]]
    }

This script converts to our manifest format:
    {"id": "arcgen_007bbfb7_00001", "input": [[...]], "output": [[...]]}
"""

import argparse
import json
import sys
from pathlib import Path


def convert_arcgen_to_manifest(input_dir: Path, output_path: Path) -> int:
    """Convert ARC-GEN data to manifest.jsonl format.

    Handles multiple possible input formats:
    1. Single JSON file with list of examples
    2. Directory of JSON files (one per example or one per task)
    3. JSONL file with one example per line

    Returns:
        Number of examples written
    """
    examples = []

    if input_dir.is_file():
        # Single file - could be JSON or JSONL
        if input_dir.suffix == '.jsonl':
            with open(input_dir) as f:
                for line in f:
                    if line.strip():
                        examples.append(json.loads(line))
        else:
            with open(input_dir) as f:
                data = json.load(f)
                if isinstance(data, list):
                    examples = data
                else:
                    examples = [data]
    else:
        # Directory of files
        for json_file in sorted(input_dir.glob('**/*.json')):
            with open(json_file) as f:
                data = json.load(f)
                if isinstance(data, list):
                    examples.extend(data)
                else:
                    examples.append(data)

        # Also check for JSONL files
        for jsonl_file in sorted(input_dir.glob('**/*.jsonl')):
            with open(jsonl_file) as f:
                for line in f:
                    if line.strip():
                        examples.append(json.loads(line))

    if not examples:
        print(f"Error: No examples found in {input_dir}", file=sys.stderr)
        return 0

    # Convert to manifest format
    output_path.parent.mkdir(parents=True, exist_ok=True)

    task_counts: dict[str, int] = {}
    written = 0

    with open(output_path, 'w') as f:
        for example in examples:
            # Extract task_id if present, otherwise use index
            task_id = example.get('task_id', example.get('id', 'unknown'))

            # Handle case where task_id might be numeric
            if isinstance(task_id, int):
                task_id = f"task_{task_id:03d}"

            # Count examples per task for unique IDs
            task_counts[task_id] = task_counts.get(task_id, 0) + 1
            example_num = task_counts[task_id]

            # Get input/output grids
            input_grid = example.get('input')
            output_grid = example.get('output')

            if input_grid is None or output_grid is None:
                print(f"Warning: Skipping example missing input/output: {example.get('id', 'unknown')}", file=sys.stderr)
                continue

            # Create manifest entry
            manifest_entry = {
                'id': f"arcgen_{task_id}_{example_num:05d}",
                'task_id': task_id,  # Preserve task ID for analysis
                'input': input_grid,
                'output': output_grid,
            }

            f.write(json.dumps(manifest_entry) + '\n')
            written += 1

    return written


def main():
    parser = argparse.ArgumentParser(description='Convert ARC-GEN to manifest.jsonl')
    parser.add_argument('input', type=Path, help='Input directory or file')
    parser.add_argument('output', type=Path, help='Output manifest.jsonl path')
    args = parser.parse_args()

    count = convert_arcgen_to_manifest(args.input, args.output)
    print(f"Converted {count} examples to {args.output}")

    # Print task distribution summary
    with open(args.output) as f:
        task_ids = set()
        for line in f:
            entry = json.loads(line)
            task_ids.add(entry.get('task_id', 'unknown'))

    print(f"Tasks represented: {len(task_ids)}")
    print(f"Average examples per task: {count / len(task_ids):.1f}")


if __name__ == '__main__':
    main()
