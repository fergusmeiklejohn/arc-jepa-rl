"""Reporting utilities for evaluation suites."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence

from .suite import VariantMetrics


def _serialise_metrics(metrics: Iterable[VariantMetrics]) -> list[dict]:
    """Convert variant metrics into JSON-ready dictionaries."""

    return [entry.to_dict() for entry in metrics]


def build_summary(
    dataset_label: str,
    *,
    metrics: Sequence[VariantMetrics],
    task_count: int,
    task_source: Path | str | None = None,
    surprise: Mapping[str, object] | None = None,
) -> MutableMapping[str, object]:
    """Build a combined JSON summary for primary and optional surprise datasets."""

    summary: MutableMapping[str, object] = {
        "dataset": dataset_label,
        "task_count": task_count,
        "results": _serialise_metrics(metrics),
    }
    if task_source is not None:
        summary["task_source"] = str(task_source)

    if surprise is not None:
        label = str(surprise.get("label", "surprise"))
        surprise_entry: MutableMapping[str, object] = {
            "dataset": label,
            "task_count": int(surprise.get("task_count", 0)),
            "results": _serialise_metrics(surprise.get("metrics", ())),
        }
        source = surprise.get("source")
        if source is not None:
            surprise_entry["task_source"] = str(source)
        summary["surprise_results"] = surprise_entry

    return summary
