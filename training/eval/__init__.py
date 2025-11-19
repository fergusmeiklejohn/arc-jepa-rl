"""Evaluation and ablation tooling for ARC pipelines."""

from .latents import LatentDistanceRecord, LatentDistanceTracker
from .suite import EvaluationSuite, EvaluationVariant, TaskEvaluation, VariantMetrics
from .utils import ArcExample, ArcTask, load_arc_dev_tasks, load_synthetic_tasks_jsonl
from .reporting import build_summary

__all__ = [
    "EvaluationSuite",
    "EvaluationVariant",
    "VariantMetrics",
    "TaskEvaluation",
    "ArcExample",
    "ArcTask",
    "load_synthetic_tasks_jsonl",
    "load_arc_dev_tasks",
    "build_summary",
    "LatentDistanceRecord",
    "LatentDistanceTracker",
]
