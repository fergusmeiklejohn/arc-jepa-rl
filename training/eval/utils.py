"""Utility helpers for evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Sequence

from arcgen import Grid, ProgramStep, SyntheticTask


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

