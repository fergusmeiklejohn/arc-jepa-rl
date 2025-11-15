"""Primitive definitions and registry for the ARC DSL."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from arcgen import (
    PRIMITIVE_REGISTRY as ARC_PRIMITIVES,
    Grid as ArcGrid,
    GridObject,
    extract_objects,
)

from .types import (
    DSLType,
    Bool,
    Color,
    Grid,
    GridList,
    GridValue,
    Position,
    Shape,
    ShapeList,
)


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


def _ensure_int(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got bool")
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value)!r}")
    return value


def _ensure_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    raise TypeError(f"expected bool, got {type(value)!r}")


def _ensure_position(value: object) -> Tuple[int, int]:
    if (
        isinstance(value, tuple)
        and len(value) == 2
        and all(isinstance(coord, int) for coord in value)
    ):
        return value  # type: ignore[return-value]
    raise TypeError(f"expected (y, x) tuple, got {type(value)!r}")


def _ensure_shape(value: object) -> GridObject:
    if not isinstance(value, GridObject):
        raise TypeError(f"expected GridObject, got {type(value)!r}")
    return value


def _ensure_shape_list(value: object) -> Tuple[GridObject, ...]:
    if isinstance(value, tuple):
        shapes = value
    elif isinstance(value, list):
        shapes = tuple(value)
    else:
        raise TypeError(f"expected sequence of GridObject, got {type(value)!r}")
    for shape in shapes:
        _ensure_shape(shape)
    return tuple(shapes)


def _ensure_grid_list(value: object) -> Tuple[ArcGrid, ...]:
    if isinstance(value, tuple):
        grids = value
    elif isinstance(value, list):
        grids = tuple(value)
    else:
        raise TypeError(f"expected sequence of Grid, got {type(value)!r}")
    for grid in grids:
        _ensure_grid(grid)
    return tuple(grids)


def _flood_fill(grid: ArcGrid, y: int, x: int, fill: int) -> ArcGrid:
    primitive = ARC_PRIMITIVES.get("flood_fill")
    return primitive.apply(grid, y=y, x=x, fill=fill)


def _connected_components(grid: ArcGrid) -> Tuple[GridObject, ...]:
    return tuple(extract_objects(grid))


def _shapes_to_subgrids(shapes: Sequence[GridObject], background: int) -> Tuple[ArcGrid, ...]:
    return tuple(shape.to_subgrid(background=background) for shape in shapes)


def _draw_shapes(
    base: ArcGrid,
    shapes: Sequence[GridObject],
    *,
    fill: Optional[int] = None,
) -> ArcGrid:
    cells = [list(row) for row in base.cells]
    height = len(cells)
    width = len(cells[0]) if cells else 0

    for shape in shapes:
        for y, x, value in shape.pixel_values:
            if 0 <= y < height and 0 <= x < width:
                cells[y][x] = fill if fill is not None else value

    return ArcGrid(cells)


def _shape_list_get(shapes: object, index: object) -> GridObject:
    shape_tuple = _ensure_shape_list(shapes)
    if not shape_tuple:
        raise ValueError("shape list is empty")
    idx = _ensure_int(index, name="index") % len(shape_tuple)
    return shape_tuple[idx]


def _grid_list_get(grids: object, index: object) -> ArcGrid:
    grid_tuple = _ensure_grid_list(grids)
    if not grid_tuple:
        raise ValueError("grid list is empty")
    idx = _ensure_int(index, name="index") % len(grid_tuple)
    return grid_tuple[idx]


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
    value_constants: Iterable[int] = (0, 1, 2, 3),
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

    registry.register(
        Primitive(
            name="flood_fill",
            input_types=(Grid, GridValue, GridValue, Color),
            output_type=Grid,
            implementation=lambda grid, y, x, fill: _flood_fill(
                _ensure_grid(grid),
                _ensure_int(y, name="y"),
                _ensure_int(x, name="x"),
                int(fill),
            ),
            description="Flood fill region starting at (y, x) with the provided color",
            complexity=2.0,
        )
    )

    registry.register(
        Primitive(
            name="connected_components",
            input_types=(Grid,),
            output_type=ShapeList,
            implementation=lambda grid: _connected_components(_ensure_grid(grid)),
            description="Extract connected components as shape objects",
            complexity=2.0,
        )
    )

    registry.register_many(
        [
            Primitive(
                name="shape_list_len",
                input_types=(ShapeList,),
                output_type=GridValue,
                implementation=lambda shapes: len(_ensure_shape_list(shapes)),
                description="Return number of shapes in the list",
                complexity=0.6,
            ),
            Primitive(
                name="shape_list_get",
                input_types=(ShapeList, GridValue),
                output_type=Shape,
                implementation=lambda shapes, idx: _shape_list_get(shapes, idx),
                description="Return shape at index modulo list length",
                complexity=1.1,
            ),
        ]
    )

    registry.register_many(
        [
            Primitive(
                name="shape_area",
                input_types=(Shape,),
                output_type=GridValue,
                implementation=lambda shape: _ensure_shape(shape).area,
                description="Return the area of a shape",
                complexity=0.8,
            ),
            Primitive(
                name="shape_bbox",
                input_types=(Shape, Color),
                output_type=Grid,
                implementation=lambda shape, background: _ensure_shape(shape).to_subgrid(background=int(background)),
                description="Crop shape to its bounding box",
                complexity=1.4,
            ),
            Primitive(
                name="shape_centroid",
                input_types=(Shape,),
                output_type=Position,
                implementation=lambda shape: tuple(
                    int(round(coord)) for coord in _ensure_shape(shape).centroid
                ),
                description="Centroid (y, x) of a shape",
                complexity=1.1,
            ),
            Primitive(
                name="position_y",
                input_types=(Position,),
                output_type=GridValue,
                implementation=lambda position: _ensure_position(position)[0],
                description="Extract row coordinate",
                complexity=0.4,
            ),
            Primitive(
                name="position_x",
                input_types=(Position,),
                output_type=GridValue,
                implementation=lambda position: _ensure_position(position)[1],
                description="Extract column coordinate",
                complexity=0.4,
            ),
        ]
    )

    registry.register_many(
        [
            Primitive(
                name="components_filter_by_color",
                input_types=(ShapeList, Color),
                output_type=ShapeList,
                implementation=lambda shapes, color: tuple(
                    shape for shape in _ensure_shape_list(shapes) if shape.dominant_color == int(color)
                ),
                description="Filter shapes by dominant color (filter combinator)",
                complexity=1.2,
            ),
            Primitive(
                name="components_filter_by_area",
                input_types=(ShapeList, GridValue),
                output_type=ShapeList,
                implementation=lambda shapes, min_area: tuple(
                    shape
                    for shape in _ensure_shape_list(shapes)
                    if shape.area >= _ensure_int(min_area, name="min_area")
                ),
                description="Filter shapes with area >= min_area",
                complexity=1.2,
            ),
            Primitive(
                name="components_map_to_subgrids",
                input_types=(ShapeList, Color),
                output_type=GridList,
                implementation=lambda shapes, background: _shapes_to_subgrids(
                    _ensure_shape_list(shapes), int(background)
                ),
                description="Map each shape to a cropped grid (map combinator)",
                complexity=1.6,
            ),
            Primitive(
                name="components_fold_overlay",
                input_types=(ShapeList, Grid),
                output_type=Grid,
                implementation=lambda shapes, base: _draw_shapes(
                    _ensure_grid(base), _ensure_shape_list(shapes)
                ),
                description="Overlay shapes on top of a base grid (fold combinator)",
                complexity=1.8,
            ),
        ]
    )

    registry.register_many(
        [
            Primitive(
                name="grid_list_len",
                input_types=(GridList,),
                output_type=GridValue,
                implementation=lambda grids: len(_ensure_grid_list(grids)),
                description="Number of grids in the list",
                complexity=0.6,
            ),
            Primitive(
                name="grid_list_get",
                input_types=(GridList, GridValue),
                output_type=Grid,
                implementation=lambda grids, idx: _grid_list_get(grids, idx),
                description="Select grid by index modulo list length",
                complexity=1.1,
            ),
        ]
    )

    registry.register_many(
        [
            Primitive(
                name="shape_list_nonempty",
                input_types=(ShapeList,),
                output_type=Bool,
                implementation=lambda shapes: bool(_ensure_shape_list(shapes)),
                description="Return True if list contains at least one shape",
                complexity=0.6,
            ),
            Primitive(
                name="value_greater_than",
                input_types=(GridValue, GridValue),
                output_type=Bool,
                implementation=lambda lhs, rhs: _ensure_int(lhs, name="lhs") > _ensure_int(rhs, name="rhs"),
                description="Return True when lhs > rhs",
                complexity=0.5,
            ),
            Primitive(
                name="value_equals",
                input_types=(GridValue, GridValue),
                output_type=Bool,
                implementation=lambda lhs, rhs: _ensure_int(lhs, name="lhs") == _ensure_int(rhs, name="rhs"),
                description="Return True when lhs == rhs",
                complexity=0.5,
            ),
            Primitive(
                name="if_then_else",
                input_types=(Bool, Grid, Grid),
                output_type=Grid,
                implementation=lambda condition, true_grid, false_grid: _ensure_grid(true_grid)
                if _ensure_bool(condition)
                else _ensure_grid(false_grid),
                description="Return true_grid when condition else false_grid",
                complexity=1.3,
            ),
        ]
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

    for value in value_constants:
        registry.register(
            Primitive(
                name=f"const_value_{value}",
                input_types=(),
                output_type=GridValue,
                implementation=lambda v=value: _ensure_int(v, name="value"),
                description=f"Constant integer {value}",
                complexity=0.4,
            )
        )

    return registry
