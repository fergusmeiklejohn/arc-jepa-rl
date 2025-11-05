"""Evaluation and ablation suite for ARC pipelines."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from arcgen import SyntheticTask

from training.dsl.primitives import PrimitiveRegistry, build_default_primitive_registry
from training.meta_jepa.data import RuleFamilyExample, build_rule_family_examples
from training.solver import FewShotSolver


@dataclass(frozen=True)
class EvaluationVariant:
    """Configuration for a single ablation/evaluation run."""

    name: str
    description: str
    max_nodes: int = 4
    allowed_primitives: Optional[Sequence[str]] = None
    top_k_primitives: Optional[int] = None


@dataclass(frozen=True)
class TaskEvaluation:
    """Per-task evaluation metrics."""

    task_id: str
    success: bool
    programs_tested: int


@dataclass(frozen=True)
class VariantMetrics:
    """Aggregated metrics for a variant."""

    variant: EvaluationVariant
    total_tasks: int
    successes: int
    success_rate: float
    avg_programs_tested: float
    details: Sequence[TaskEvaluation]

    def to_dict(self) -> dict:
        return {
            "variant": self.variant.name,
            "description": self.variant.description,
            "total_tasks": self.total_tasks,
            "successes": self.successes,
            "success_rate": self.success_rate,
            "avg_programs_tested": self.avg_programs_tested,
            "details": [
                {
                    "task_id": detail.task_id,
                    "success": detail.success,
                    "programs_tested": detail.programs_tested,
                }
                for detail in self.details
            ],
        }


class EvaluationSuite:
    """Runs ablation variants over a collection of ARC tasks."""

    def __init__(self, tasks: Sequence[SyntheticTask]) -> None:
        if not tasks:
            raise ValueError("EvaluationSuite requires at least one task")
        self.tasks = list(tasks)
        self._family_examples = build_rule_family_examples(tasks, min_family_size=1)

    def run(self, variants: Sequence[EvaluationVariant]) -> List[VariantMetrics]:
        if not variants:
            raise ValueError("variants must contain at least one entry")
        results: List[VariantMetrics] = []
        for variant in variants:
            metrics = self._run_variant(variant)
            results.append(metrics)
        return results

    def _run_variant(self, variant: EvaluationVariant) -> VariantMetrics:
        registry = self._build_registry(variant)
        solver = FewShotSolver(registry)

        details: List[TaskEvaluation] = []
        successes = 0
        total_programs = 0

        for task in self.tasks:
            result = solver.solve([(task.input_grid, task.output_grid)], max_nodes=variant.max_nodes)
            success = result.solved()
            if success:
                successes += 1
            programs_tested = result.evaluated
            total_programs += programs_tested
            details.append(
                TaskEvaluation(
                    task_id=task.task_id,
                    success=success,
                    programs_tested=programs_tested,
                )
            )

        total = len(self.tasks)
        success_rate = successes / total if total else 0.0
        avg_programs = total_programs / total if total else 0.0

        return VariantMetrics(
            variant=variant,
            total_tasks=total,
            successes=successes,
            success_rate=success_rate,
            avg_programs_tested=avg_programs,
            details=tuple(details),
        )

    def _build_registry(self, variant: EvaluationVariant) -> PrimitiveRegistry:
        base_registry = build_default_primitive_registry()
        allowed: Optional[Sequence[str]] = variant.allowed_primitives

        if variant.top_k_primitives is not None:
            allowed = self._top_primitives(variant.top_k_primitives)

        if allowed is None:
            return base_registry

        allowed_set = set(allowed)
        filtered = PrimitiveRegistry()
        for primitive in base_registry.list():
            if primitive.name in allowed_set:
                filtered.register(primitive)

        if not filtered.list():
            raise ValueError(f"no primitives left after filtering for variant '{variant.name}'")
        return filtered

    def _top_primitives(self, count: int) -> List[str]:
        counter: Counter[str] = Counter()
        for example in self._family_examples:
            counter.update(example.primitive_counts)
        return [name for name, _ in counter.most_common(count)]
