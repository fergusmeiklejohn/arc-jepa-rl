"""Convert ARC generator manifests into (input, program, output) triples."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract (input, program, output) triples from a manifest")
    parser.add_argument("--input", type=Path, required=True, help="Path to synthetic manifest JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSONL file for triples")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of records to convert (for smoke tests)",
    )
    parser.add_argument(
        "--skip-missing-programs",
        action="store_true",
        help="Skip manifest entries without rule traces instead of raising an error",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> Iterable[Mapping[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:  # pragma: no cover - depends on user input
                raise ValueError(f"failed to parse JSON on line {line_number}") from exc
            if not isinstance(record, Mapping):
                raise ValueError(f"manifest line {line_number} must be a JSON object")
            yield record


def _extract_program(record: Mapping[str, object]) -> Sequence[Mapping[str, object]]:
    program = record.get("program")
    if program is None:
        program = record.get("program_steps") or record.get("rule_trace")
    if program is None:
        raise ValueError("record missing program information")
    if isinstance(program, Mapping) and "steps" in program:
        program = program["steps"]
    if not isinstance(program, Sequence):
        raise ValueError("program must be a sequence of steps")
    return program


def convert_record(record: Mapping[str, object]) -> dict:
    if "input" not in record or "output" not in record:
        raise ValueError("record must contain input/output grids")
    program_steps = []
    for step in _extract_program(record):
        if not isinstance(step, Mapping):
            raise ValueError("program steps must be mappings with 'primitive'")
        primitive = step.get("primitive")
        if not isinstance(primitive, str):
            raise ValueError("program step missing 'primitive'")
        params = step.get("params") or {}
        if not isinstance(params, Mapping):
            raise ValueError("program step params must be a mapping")
        program_steps.append(
            {
                "primitive": primitive,
                "params": dict(params),
            }
        )
    triple = {
        "id": record.get("id"),
        "input": record["input"],
        "output": record["output"],
        "program": program_steps,
    }
    metadata = record.get("metadata")
    if isinstance(metadata, Mapping):
        triple["metadata"] = dict(metadata)
    return triple


def main() -> None:
    args = parse_args()
    manifest_path = args.input
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    primitive_counts: Counter[str] = Counter()
    converted = 0

    with args.output.open("w", encoding="utf-8") as sink:
        for record in load_manifest(manifest_path):
            if args.limit is not None and converted >= args.limit:
                break
            try:
                triple = convert_record(record)
            except ValueError:
                if args.skip_missing_programs:
                    continue
                raise
            sink.write(json.dumps(triple) + "\n")
            converted += 1
            for step in triple["program"]:
                primitive_counts[step["primitive"]] += 1

    print(f"Converted {converted} triples into {args.output}")
    if primitive_counts:
        most_common = primitive_counts.most_common(5)
        stats = ", ".join(f"{name}={count}" for name, count in most_common)
        print(f"Most frequent primitives: {stats}")


if __name__ == "__main__":
    main()

