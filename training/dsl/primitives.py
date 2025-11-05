"""Primitive definitions and registry for the ARC DSL."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence

from arcgen import PRIMITIVE_REGISTRY as ARC_PRIMITIVES, Grid as ArcGrid, extract_objects

from .types import DSLType, Grid, Color, GridValue


PrimitiveImpl = Callable[..., object]


@dataclass(frozen=True)
class Primitive:
    name: str
    input_types: Sequence[DSLType]
    output_type: DSLType
    implementation: PrimitiveImpl
    description: str = ""
    complexity: float = 1.0

    def __call__(self, *args) -> object:  # pragma: no cover - convenience
        return self.implementation(*args)


class PrimitiveRegistry:
    def __init__(self) -> None:
        self._primitives: Dict[str, Primitive] = {}

    def register(self, primitive: Primitive) -> None:
        if primitive.name in self._primitives:
            raise ValueError(f"primitive '{primitive.name}' already registered")
        self._primitives[primitive.name] = primitive

    def register_many(self, primitives: Iterable[Primitive]) -> None:
        for primitive in primitives:
            self.register(primitive)

    def get(self, name: str) -> Primitive:
        try:
            return self._primitives[name]
        except KeyError as exc:  # pragma: no cover - caller error
            raise KeyError(f"primitive '{name}' not found") from exc

    def list(self) -> List[Primitive]:
        return list(self._primitives.values())


def _ensure_grid(value: object) -> ArcGrid:
    if not isinstance(value, ArcGrid):
        raise TypeError(f"expected ArcGrid, got {type(value)!r}")
    return value


def _mirror(axis: str, grid: ArcGrid) -> ArcGrid:
    op_name = f"mirror_{axis}"
    primitive = ARC_PRIMITIVES.get(op_name)
    return primitive.apply(grid)


def _rotate(grid: ArcGrid, quarter_turns: int) -> ArcGrid:
    primitive = ARC_PRIMITIVES.get("rotate90")
    return primitive.apply(grid, k=quarter_turns)


def _color_constant(value: int) -> int:
    return value


def _recolor(grid: ArcGrid, source: int, target: int) -> ArcGrid:
    return grid.replace(source, target)


def _dominant_color(grid: ArcGrid) -> int:
    counter = Counter(value for row in grid.cells for value in row)
    if not counter:
        return 0
    return counter.most_common(1)[0][0]


def _object_count(grid: ArcGrid) -> int:
    return len(extract_objects(grid))


def _identity(grid: ArcGrid) -> ArcGrid:
    return grid


def build_default_primitive_registry(
    *,
    color_constants: Iterable[int] = (0, 1, 2, 3),
) -> PrimitiveRegistry:
    registry = PrimitiveRegistry()

    def make_grid_unary(
        name: str,
        impl: Callable[[ArcGrid], ArcGrid],
        description: str,
        *,
        complexity: float = 1.0,
    ) -> Primitive:
        return Primitive(
            name=name,
            input_types=(Grid,),
            output_type=Grid,
            implementation=lambda grid: impl(_ensure_grid(grid)),
            description=description,
            complexity=complexity,
        )

    registry.register_many(
        [
            make_grid_unary("identity", _identity, "Return the grid unchanged", complexity=0.5),
            make_grid_unary("mirror_x", lambda g: _mirror("x", g), "Reflect grid across x-axis"),
            make_grid_unary("mirror_y", lambda g: _mirror("y", g), "Reflect grid across y-axis"),
            make_grid_unary("rotate_cw", lambda g: _rotate(g, 1), "Rotate grid clockwise", complexity=1.2),
            make_grid_unary("rotate_ccw", lambda g: _rotate(g, 3), "Rotate grid counter-clockwise", complexity=1.2),
            make_grid_unary("rotate_180", lambda g: _rotate(g, 2), "Rotate grid by 180 degrees", complexity=1.3),
        ]
    )

    registry.register(
        Primitive(
            name="recolor",
            input_types=(Grid, Color, Color),
            output_type=Grid,
            implementation=lambda grid, src, dst: _recolor(_ensure_grid(grid), int(src), int(dst)),
            description="Replace all occurrences of source color with target color",
            complexity=1.5,
        )
    )

    registry.register(
        Primitive(
            name="dominant_color",
            input_types=(Grid,),
            output_type=Color,
            implementation=lambda grid: _dominant_color(_ensure_grid(grid)),
            description="Return the most frequent color in the grid",
            complexity=1.2,
        )
    )

    registry.register(
        Primitive(
            name="count_objects",
            input_types=(Grid,),
            output_type=GridValue,
            implementation=lambda grid: _object_count(_ensure_grid(grid)),
            description="Count connected components in the grid",
            complexity=1.3,
        )
    )

    for value in color_constants:
        registry.register(
            Primitive(
                name=f"const_color_{value}",
                input_types=(),
                output_type=Color,
                implementation=lambda v=value: _color_constant(v),
                description=f"Constant color {value}",
                complexity=0.4,
            )
        )

    return registry
