"""Synthetic ARC task generator and JSONL exporter."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, MutableMapping, Sequence

from . import Grid, PRIMITIVE_REGISTRY, SeededRNG
from .primitives import PrimitiveSpec


PHASE_CODES: Mapping[str, str] = {
    "atomic": "I",
    "sequential": "II",
}


@dataclass(frozen=True)
class GeneratorConfig:
    """Configuration controlling grid sampling and program synthesis."""

    min_grid_size: int = 5
    max_grid_size: int = 12
    min_colors: int = 3
    max_colors: int = 6
    background_color: int = 0
    fill_probability: float = 0.75
    max_parameter_retries: int = 5
    max_task_retries: int = 12

    def __post_init__(self) -> None:
        if self.min_grid_size <= 0 or self.max_grid_size <= 0:
            raise ValueError("grid sizes must be positive")
        if self.min_grid_size > self.max_grid_size:
            raise ValueError("min_grid_size cannot exceed max_grid_size")
        if self.min_colors <= 0 or self.max_colors <= 0:
            raise ValueError("color counts must be positive")
        if self.min_colors > self.max_colors:
            raise ValueError("min_colors cannot exceed max_colors")
        if not 0.0 < self.fill_probability <= 1.0:
            raise ValueError("fill_probability must be in (0, 1]")
        if self.max_parameter_retries <= 0:
            raise ValueError("max_parameter_retries must be positive")
        if self.max_task_retries <= 0:
            raise ValueError("max_task_retries must be positive")


@dataclass(frozen=True)
class ProgramStep:
    """One primitive invocation inside a generated program."""

    primitive: str
    params: Dict[str, int]


@dataclass(frozen=True)
class SyntheticTask:
    """Synthetic ARC task containing grids, program trace, and metadata."""

    task_id: str
    phase: str
    input_grid: Grid
    output_grid: Grid
    rule_trace: Sequence[ProgramStep]
    metadata: Mapping[str, object]

    def to_json_record(self) -> Dict[str, object]:
        return {
            "id": self.task_id,
            "input": self.input_grid.to_lists(),
            "output": self.output_grid.to_lists(),
            "rule_trace": [
                {"primitive": step.primitive, "params": dict(step.params)}
                for step in self.rule_trace
            ],
            "metadata": dict(self.metadata),
        }


class SyntheticARCGenerator:
    """Generates ARC-style transformation tasks for JEPA pretraining."""

    def __init__(
        self,
        config: GeneratorConfig | None = None,
        *,
        seed: int | None = None,
        allowed_primitives: Sequence[str] | None = None,
    ) -> None:
        self.config = config or GeneratorConfig()
        self._rng = SeededRNG(seed)
        self._counter = 0

        all_specs = PRIMITIVE_REGISTRY.list()
        if allowed_primitives is not None:
            allowed = set(allowed_primitives)
            specs = [spec for spec in all_specs if spec.name in allowed]
            if not specs:
                raise ValueError("allowed_primitives filtered out all primitives")
            self._primitive_pool = specs
        else:
            self._primitive_pool = all_specs

    # ---------------------------------------------------------------- sampling
    def sample_task(self, phase: str) -> SyntheticTask:
        """Sample a single synthetic task for the given curriculum phase."""

        phase_key = phase.lower()
        if phase_key not in PHASE_CODES:
            raise ValueError(f"unsupported phase '{phase}'")

        for _ in range(self.config.max_task_retries):
            palette = self._sample_palette()
            input_grid = self._sample_input_grid(palette)
            try:
                output_grid, trace = self._execute_program(
                    input_grid,
                    palette,
                    phase_key,
                )
            except RuntimeError:
                continue

            if input_grid.cells == output_grid.cells:
                continue

            metadata = self._build_metadata(
                phase_key,
                input_grid,
                output_grid,
                palette,
                trace,
            )
            task_id = self._next_task_id(phase_key)
            return SyntheticTask(
                task_id=task_id,
                phase=PHASE_CODES[phase_key],
                input_grid=input_grid,
                output_grid=output_grid,
                rule_trace=trace,
                metadata=metadata,
            )

        raise RuntimeError(f"failed to generate task for phase '{phase}'")

    def sample_many(self, count: int, phase: str) -> List[SyntheticTask]:
        if count <= 0:
            raise ValueError("count must be positive")
        return [self.sample_task(phase) for _ in range(count)]

    # -------------------------------------------------------------- execution
    def _execute_program(
        self,
        input_grid: Grid,
        palette: Sequence[int],
        phase: str,
    ) -> tuple[Grid, List[ProgramStep]]:
        specs = self._sample_program_specs(phase)

        current = input_grid
        trace: List[ProgramStep] = []

        for spec in specs:
            result = self._apply_primitive_with_retries(spec, current, palette)
            if result is None:
                raise RuntimeError(f"failed to apply primitive '{spec.name}'")
            current, step = result
            trace.append(step)
        return current, trace

    def _apply_primitive_with_retries(
        self,
        spec: PrimitiveSpec,
        grid: Grid,
        palette: Sequence[int],
    ) -> tuple[Grid, ProgramStep] | None:
        last_error: Exception | None = None

        for _ in range(self.config.max_parameter_retries):
            params = self._sample_parameters(spec, grid, palette)
            try:
                new_grid = spec.apply(grid, **params)
            except Exception as exc:  # pragma: no cover - depends on primitive internals
                last_error = exc
                continue

            if new_grid.cells == grid.cells:
                last_error = None
                continue

            return new_grid, ProgramStep(spec.name, params)

        if last_error is not None:
            return None
        return None

    def _sample_program_specs(self, phase: str) -> List[PrimitiveSpec]:
        rng = self._rng
        if phase == "atomic":
            length = 1
        elif phase == "sequential":
            length = rng.randint(2, 3)
        else:  # pragma: no cover - guarded by caller
            raise ValueError(f"unsupported phase '{phase}'")

        return [rng.choice(self._primitive_pool) for _ in range(length)]

    # --------------------------------------------------------------- sampling
    def _sample_palette(self) -> List[int]:
        rng = self._rng
        color_count = rng.randint(self.config.min_colors, self.config.max_colors)
        non_background = list(range(self.config.background_color + 1, self.config.background_color + color_count))
        rng.shuffle(non_background)
        return [self.config.background_color] + non_background

    def _sample_input_grid(self, palette: Sequence[int]) -> Grid:
        rng = self._rng
        height = rng.randint(self.config.min_grid_size, self.config.max_grid_size)
        width = rng.randint(self.config.min_grid_size, self.config.max_grid_size)
        return Grid.random(
            height,
            width,
            palette,
            fill_prob=self.config.fill_probability,
            background=self.config.background_color,
            rng=rng,
        )

    def _sample_parameters(
        self,
        spec: PrimitiveSpec,
        grid: Grid,
        palette: Sequence[int],
    ) -> Dict[str, int]:
        rng = self._rng
        params: Dict[str, int] = {}
        height, width = grid.shape

        palette_list = list(palette)
        palette_set = set(palette_list)
        non_background = [value for value in palette_list if value != self.config.background_color]
        min_value = min(palette_list)
        max_value = max(palette_list)

        for parameter in spec.parameters:
            name = parameter.name

            if parameter.choices:
                params[name] = rng.choice(tuple(parameter.choices))
                continue

            if name == "k":
                params[name] = rng.randint(1, 3)
            elif name == "dx":
                max_shift = max(1, width // 2)
                params[name] = rng.randint(-max_shift, max_shift)
            elif name == "dy":
                max_shift = max(1, height // 2)
                params[name] = rng.randint(-max_shift, max_shift)
            elif name == "x":
                params[name] = rng.randint(0, max(width - 1, 0))
            elif name == "y":
                params[name] = rng.randint(0, max(height - 1, 0))
            elif name in {"source", "value_a", "value_b"}:
                if non_background:
                    params[name] = rng.choice(non_background)
                else:
                    params[name] = self.config.background_color
            elif name in {"target", "fill", "high", "low", "background"}:
                pool = palette_list or [self.config.background_color]
                params[name] = rng.choice(pool)
            elif name == "threshold":
                params[name] = rng.randint(min_value, max_value)
            else:
                params[name] = rng.randint(-3, 3)

        if "source" in params and "target" in params and params["source"] == params["target"]:
            alternatives = [value for value in palette_set if value != params["source"]]
            if alternatives:
                params["target"] = rng.choice(alternatives)

        if "value_a" in params and "value_b" in params and params["value_a"] == params["value_b"]:
            alternatives = [value for value in palette_set if value != params["value_a"]]
            if alternatives:
                params["value_b"] = rng.choice(alternatives)

        return params

    # -------------------------------------------------------------- metadata
    def _build_metadata(
        self,
        phase: str,
        input_grid: Grid,
        output_grid: Grid,
        palette: Sequence[int],
        trace: Sequence[ProgramStep],
    ) -> Dict[str, object]:
        metadata: MutableMapping[str, object] = {
            "phase": PHASE_CODES[phase],
            "phase_name": phase,
            "grid_height": input_grid.height,
            "grid_width": input_grid.width,
            "color_count": len(set(palette)),
            "program_length": len(trace),
            "complexity": float(len(trace)),
        }
        metadata["palette"] = list(palette)
        metadata["changed_cells"] = self._changed_cell_count(input_grid, output_grid)
        return dict(metadata)

    def _changed_cell_count(self, before: Grid, after: Grid) -> int:
        before_rows = before.to_lists()
        after_rows = after.to_lists()
        height = max(len(before_rows), len(after_rows))
        width_before = len(before_rows[0]) if before_rows else 0
        width_after = len(after_rows[0]) if after_rows else 0
        width = max(width_before, width_after)

        background = self.config.background_color
        diffs = 0
        for y in range(height):
            for x in range(width):
                if y < len(before_rows) and x < len(before_rows[y]):
                    val_before = before_rows[y][x]
                else:
                    val_before = background

                if y < len(after_rows) and x < len(after_rows[y]):
                    val_after = after_rows[y][x]
                else:
                    val_after = background

                if val_before != val_after:
                    diffs += 1
        return diffs

    def _next_task_id(self, phase: str) -> str:
        self._counter += 1
        return f"{phase}_{self._counter:06d}"

    # ----------------------------------------------------------------- export
    @staticmethod
    def export_jsonl(path: str | Path, tasks: Sequence[SyntheticTask]) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        with destination.open("w", encoding="utf-8") as handle:
            for task in tasks:
                json.dump(task.to_json_record(), handle)
                handle.write("\n")

    # --------------------------------------------------------------- utility
    def iter_curriculum(
        self,
        schedule: Mapping[str, int],
    ) -> Iterator[SyntheticTask]:
        for phase, count in schedule.items():
            for _ in range(count):
                yield self.sample_task(phase)
