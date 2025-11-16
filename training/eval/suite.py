"""Evaluation and ablation suite for ARC pipelines."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from arcgen import Grid, SyntheticTask

from training.dsl.enumerator import Program
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
        traced_tasks = [task for task in self.tasks if getattr(task, "rule_trace", None)]
        if traced_tasks:
            self._family_examples = build_rule_family_examples(traced_tasks, min_family_size=1)
        else:
            self._family_examples = []

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
            examples = self._training_examples(task)
            tests = self._test_examples(task)

            result = solver.solve(examples, max_nodes=variant.max_nodes)
            success = result.solved()
            if success and tests and result.program is not None:
                success = self._evaluate_test_examples(solver, result.program, tests)

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

        if not allowed:
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

    def _training_examples(self, task: object) -> Sequence[Tuple[Grid, Grid]]:
        raw_examples = getattr(task, "train_examples", None)
        if raw_examples is None:
            return ((task.input_grid, task.output_grid),)

        normalised: List[Tuple[Grid, Grid]] = []
        for example in raw_examples:
            input_grid, output_grid = self._coerce_example(example, require_output=True)
            if output_grid is None:
                raise ValueError(f"Training example for task '{task.task_id}' is missing an output grid")
            normalised.append((input_grid, output_grid))
        if not normalised:
            raise ValueError(f"Task '{task.task_id}' does not contain training examples")
        return tuple(normalised)

    def _test_examples(self, task: object) -> Sequence[Tuple[Grid, Grid | None]]:
        raw_examples = getattr(task, "test_examples", None)
        if not raw_examples:
            return ()
        return tuple(self._coerce_example(example, require_output=False) for example in raw_examples)

    def _coerce_example(self, example: object, *, require_output: bool) -> Tuple[Grid, Grid | None]:
        if isinstance(example, tuple):
            if len(example) != 2:
                raise ValueError("Example tuples must contain exactly two elements")
            input_grid, output_grid = example
        else:
            input_grid = getattr(example, "input_grid", None)
            output_grid = getattr(example, "output_grid", None)

        if not isinstance(input_grid, Grid):
            raise TypeError("Example inputs must be Grid instances")

        if output_grid is None and require_output:
            raise ValueError("Example outputs must be provided for training data")
        if output_grid is not None and not isinstance(output_grid, Grid):
            raise TypeError("Example outputs must be Grid instances")
        return input_grid, output_grid

    def _evaluate_test_examples(
        self,
        solver: FewShotSolver,
        program: Program,
        tests: Sequence[Tuple[Grid, Grid | None]],
    ) -> bool:
        interpreter = solver.interpreter
        for input_grid, expected in tests:
            try:
                prediction = interpreter.evaluate(program, {"grid": input_grid})
            except Exception:
                return False
            if not isinstance(prediction, Grid):
                return False
            if expected is not None and prediction.cells != expected.cells:
                return False
        return True
