"""Utility helpers for evaluation."""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import IO, List, Sequence

from arcgen import Grid, ProgramStep, SyntheticTask


@dataclass(frozen=True)
class ArcExample:
    """Single ARC example consisting of an input grid and optional output."""

    input_grid: Grid
    output_grid: Grid | None = None


@dataclass(frozen=True)
class ArcTask:
    """ARC dev task containing training and test examples."""

    task_id: str
    train_examples: Sequence[ArcExample]
    test_examples: Sequence[ArcExample]


def load_synthetic_tasks_jsonl(path: Path | str) -> List[SyntheticTask]:
    """Load synthetic ARC tasks from a JSONL manifest."""

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"dataset not found: {file_path}")

    tasks: List[SyntheticTask] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("each JSONL record must be an object")

            rule_trace = [
                ProgramStep(step["primitive"], step.get("params", {}))
                for step in record.get("rule_trace", [])
            ]
            metadata = record.get("metadata", {})
            task = SyntheticTask(
                task_id=str(record.get("id")),
                phase=str(metadata.get("phase_name", metadata.get("phase", ""))),
                input_grid=Grid(record["input"]),
                output_grid=Grid(record["output"]),
                rule_trace=rule_trace,
                metadata=metadata,
            )
            tasks.append(task)
    return tasks


def load_arc_dev_tasks(root: Path | str) -> List[ArcTask]:
    """Load ARC dev (training) tasks from a directory of JSON or JSONL files."""

    base_path = Path(root)
    if not base_path.exists():
        raise FileNotFoundError(f"ARC dataset not found: {base_path}")

    if base_path.is_file():
        files = [base_path]
    else:
        files = sorted(
            [
                path
                for path in base_path.iterdir()
                if path.is_file() and path.suffix in {".json", ".jsonl", ".gz"}
            ]
        )

    if not files:
        raise FileNotFoundError(f"No ARC task files found under {base_path}")

    tasks: List[ArcTask] = []
    for file_path in files:
        record = _read_json_record(file_path)
        train_pairs = _parse_examples(record.get("train"), require_output=True, source=file_path)
        test_pairs = _parse_examples(record.get("test"), require_output=False, source=file_path)
        task_id = str(record.get("id") or _derive_task_id(file_path))
        tasks.append(
            ArcTask(
                task_id=task_id,
                train_examples=tuple(train_pairs),
                test_examples=tuple(test_pairs),
            )
        )
    return tasks


def _read_json_record(path: Path) -> dict:
    if path.suffix == ".gz":
        handle: IO[str]
        handle = gzip.open(path, "rt", encoding="utf-8")
    else:
        handle = path.open("r", encoding="utf-8")
    with handle:
        return json.load(handle)


def _parse_examples(
    entries: object,
    *,
    require_output: bool,
    source: Path,
) -> List[ArcExample]:
    if entries is None:
        return []
    if not isinstance(entries, list):
        raise ValueError(f"Expected list of examples in {source}")

    parsed: List[ArcExample] = []
    for idx, example in enumerate(entries):
        if not isinstance(example, dict):
            raise ValueError(f"Example {idx} in {source} must be an object")
        if "input" not in example:
            raise ValueError(f"Example {idx} in {source} missing 'input'")
        input_grid = Grid(example["input"])

        output_data = example.get("output")
        if require_output and output_data is None:
            raise ValueError(f"Example {idx} in {source} missing 'output'")
        output_grid = Grid(output_data) if output_data is not None else None
        parsed.append(ArcExample(input_grid=input_grid, output_grid=output_grid))
    return parsed


def _derive_task_id(path: Path) -> str:
    stem = path.stem
    if stem.endswith(".json"):
        return Path(stem).stem
    return stem
