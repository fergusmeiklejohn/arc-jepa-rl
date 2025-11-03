"""Utilities for experiment logging (TensorBoard or JSON fallback)."""

from __future__ import annotations

import importlib
import json
import time
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Optional


def _default_run_name() -> str:
    return datetime.utcnow().strftime("%Y%m%d-%H%M%S")


@lru_cache(maxsize=1)
def _load_summary_writer_cls():
    try:
        spec = importlib.util.find_spec("torch.utils.tensorboard")
    except ModuleNotFoundError:
        return None
    if spec is None:
        return None

    try:
        module = importlib.import_module("torch.utils.tensorboard")
    except Exception:
        return None

    return getattr(module, "SummaryWriter", None)


@dataclass
class _JsonSummaryWriter:
    """Very small JSONL-based fallback when TensorBoard is unavailable."""

    log_path: Path

    def __post_init__(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.log_path.open("a", encoding="utf-8")

    def add_scalar(self, tag: str, scalar_value: float, global_step: Optional[int] = None) -> None:
        record = {
            "type": "scalar",
            "tag": tag,
            "value": float(scalar_value),
            "step": global_step,
            "timestamp": time.time(),
        }
        self._handle.write(json.dumps(record) + "\n")
        self._handle.flush()

    def add_scalars(self, main_tag: str, tag_scalar_dict: Mapping[str, float], global_step: Optional[int] = None) -> None:
        for sub_tag, value in tag_scalar_dict.items():
            self.add_scalar(f"{main_tag}/{sub_tag}", value, global_step)

    def add_histogram(self, tag: str, values: Any, global_step: Optional[int] = None, bins: str = "tensorflow") -> None:
        record = {
            "type": "histogram",
            "tag": tag,
            "values": list(values) if isinstance(values, (list, tuple)) else values,
            "step": global_step,
            "bins": bins,
            "timestamp": time.time(),
        }
        self._handle.write(json.dumps(record) + "\n")
        self._handle.flush()

    def flush(self) -> None:
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    # Compatibility helpers -------------------------------------------------
    def __enter__(self) -> "_JsonSummaryWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class ExperimentLogger:
    """Wrapper around TensorBoard (with JSON fallback) for consistent logging."""

    def __init__(self, log_dir: Path | str, run_name: Optional[str] = None, *, flush_secs: int = 10) -> None:
        self.base_dir = Path(log_dir)
        self.run_name = run_name or _default_run_name()
        self.run_dir = self.base_dir / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        summary_writer_cls = _load_summary_writer_cls()
        if summary_writer_cls is not None:
            self._writer = summary_writer_cls(log_dir=str(self.run_dir), flush_secs=flush_secs)
            self.backend = "tensorboard"
        else:
            self._writer = _JsonSummaryWriter(self.run_dir / "events.jsonl")
            self.backend = "jsonl"

    # ------------------------------------------------------------------ API
    def log_scalar(self, tag: str, value: float, step: Optional[int] = None) -> None:
        self._writer.add_scalar(tag, value, global_step=step)

    def log_scalars(self, main_tag: str, values: Mapping[str, float], step: Optional[int] = None) -> None:
        add_scalars = getattr(self._writer, "add_scalars", None)
        if callable(add_scalars):
            add_scalars(main_tag, values, global_step=step)
        else:
            for key, val in values.items():
                self.log_scalar(f"{main_tag}/{key}", val, step)

    def log_histogram(self, tag: str, values: Any, step: Optional[int] = None) -> None:
        add_hist = getattr(self._writer, "add_histogram", None)
        if callable(add_hist):
            add_hist(tag, values, global_step=step)

    def flush(self) -> None:
        flush_fn = getattr(self._writer, "flush", None)
        if callable(flush_fn):
            flush_fn()

    def close(self) -> None:
        close_fn = getattr(self._writer, "close", None)
        if callable(close_fn):
            close_fn()

    # Context manager support ----------------------------------------------
    def __enter__(self) -> "ExperimentLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def create_experiment_logger(log_dir: Path | str, *, run_name: Optional[str] = None, flush_secs: int = 10) -> ExperimentLogger:
    """Factory helper mirroring the old API used in Hydra/Lightning scripts."""

    return ExperimentLogger(log_dir=log_dir, run_name=run_name, flush_secs=flush_secs)
