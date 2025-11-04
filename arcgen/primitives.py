"""Primitive registry for ARC transformations."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


PrimitiveFn = Callable[..., Any]


@dataclass(frozen=True)
class ParameterSpec:
    """Metadata describing a primitive parameter."""

    name: str
    type: str
    description: str = ""
    default: Any = field(default_factory=lambda: _NoDefault)
    choices: Optional[Tuple[Any, ...]] = None

    def has_default(self) -> bool:
        return self.default is not _NoDefault


@dataclass(frozen=True)
class PrimitiveSpec:
    """Registered primitive entry."""

    name: str
    func: PrimitiveFn
    category: str
    description: str = ""
    parameters: Tuple[ParameterSpec, ...] = ()

    def apply(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)


class PrimitiveRegistry:
    """Registry managing ARC transformation primitives."""

    def __init__(self) -> None:
        self._entries: Dict[str, PrimitiveSpec] = {}

    # -------------------------------------------------------------- registration
    def register(
        self,
        name: str,
        *,
        category: str,
        description: str = "",
        parameters: Iterable[ParameterSpec] | None = None,
    ) -> Callable[[PrimitiveFn], PrimitiveFn]:
        """Decorator to register a primitive function."""

        if name in self._entries:
            raise ValueError(f"primitive '{name}' is already registered")

        params_tuple = tuple(parameters or ())

        def decorator(func: PrimitiveFn) -> PrimitiveFn:
            spec = PrimitiveSpec(
                name=name,
                func=func,
                category=category,
                description=description,
                parameters=params_tuple,
            )
            self._entries[name] = spec
            return func

        return decorator

    # ------------------------------------------------------------------- queries
    def get(self, name: str) -> PrimitiveSpec:
        try:
            return self._entries[name]
        except KeyError as exc:
            raise KeyError(f"unknown primitive '{name}'") from exc

    def list(self, *, category: str | None = None) -> List[PrimitiveSpec]:
        if category is None:
            return list(self._entries.values())
        return [spec for spec in self._entries.values() if spec.category == category]

    def categories(self) -> List[str]:
        return sorted({spec.category for spec in self._entries.values()})


class _NoDefaultType:
    pass


_NoDefault = _NoDefaultType()


# Global registry --------------------------------------------------------------
REGISTRY = PrimitiveRegistry()


def register_primitive(
    name: str,
    *,
    category: str,
    description: str = "",
    parameters: Iterable[ParameterSpec] | None = None,
) -> Callable[[PrimitiveFn], PrimitiveFn]:
    """Public helper to register primitives into the global registry."""

    return REGISTRY.register(
        name,
        category=category,
        description=description,
        parameters=parameters,
    )


__all__ = [
    "ParameterSpec",
    "PrimitiveSpec",
    "PrimitiveRegistry",
    "register_primitive",
    "REGISTRY",
]


# Geometry primitives ---------------------------------------------------------


@register_primitive(
    "mirror_x",
    category="geometry",
    description="Reflect grid across the horizontal axis (flip vertically).",
)
def mirror_x(grid: "Grid") -> "Grid":
    from .grid import Grid

    return Grid(list(reversed(grid.cells)))


@register_primitive(
    "mirror_y",
    category="geometry",
    description="Reflect grid across the vertical axis (flip horizontally).",
)
def mirror_y(grid: "Grid") -> "Grid":
    from .grid import Grid

    return Grid([list(reversed(row)) for row in grid.cells])


@register_primitive(
    "rotate90",
    category="geometry",
    description="Rotate grid clockwise by 90 degrees multiplied by k.",
    parameters=[
        ParameterSpec(
            name="k",
            type="int",
            description="Number of quarter turns (positive rotates clockwise).",
            default=1,
        )
    ],
)
def rotate90(grid: "Grid", k: int = 1) -> "Grid":
    from .grid import Grid

    if not isinstance(k, int):
        raise TypeError("k must be an integer")

    rotations = k % 4
    cells = grid.cells
    for _ in range(rotations):
        cells = tuple(zip(*cells[::-1]))  # type: ignore[assignment]
    return Grid([list(row) for row in cells])


@register_primitive(
    "translate",
    category="geometry",
    description="Translate grid by (dx, dy) within bounds, filling exposed cells.",
    parameters=[
        ParameterSpec(
            name="dx",
            type="int",
            description="Horizontal shift (positive = right).",
            default=0,
        ),
        ParameterSpec(
            name="dy",
            type="int",
            description="Vertical shift (positive = down).",
            default=0,
        ),
        ParameterSpec(
            name="fill",
            type="int",
            description="Value used to fill uncovered cells.",
            default=0,
        ),
    ],
)
def translate(grid: "Grid", dx: int = 0, dy: int = 0, *, fill: int = 0) -> "Grid":
    from .grid import Grid

    if not all(isinstance(val, int) for val in (dx, dy, fill)):
        raise TypeError("dx, dy, and fill must be integers")

    height, width = grid.shape
    new_cells = [[fill for _ in range(width)] for _ in range(height)]

    for y, row in enumerate(grid.cells):
        ny = y + dy
        if 0 <= ny < height:
            for x, value in enumerate(row):
                nx = x + dx
                if 0 <= nx < width:
                    new_cells[ny][nx] = value

    return Grid(new_cells)


# Color/value primitives -------------------------------------------------------


@register_primitive(
    "recolor",
    category="color",
    description="Map one value to another across the entire grid.",
    parameters=[
        ParameterSpec("source", "int", "Value to replace."),
        ParameterSpec("target", "int", "Replacement value."),
    ],
)
def recolor(grid: "Grid", source: int, target: int) -> "Grid":
    from .grid import Grid

    if not all(isinstance(val, int) for val in (source, target)):
        raise TypeError("source and target must be integers")
    return grid.replace(source, target)


@register_primitive(
    "invert_palette",
    category="color",
    description="Invert the palette order so lowest value becomes highest and vice versa.",
)
def invert_palette(grid: "Grid") -> "Grid":
    from .grid import Grid

    palette = grid.palette()
    if not palette:
        return grid.copy()
    mapping = {value: palette[-idx - 1] for idx, value in enumerate(palette)}
    return Grid([[mapping[value] for value in row] for row in grid.cells])


@register_primitive(
    "threshold_gt",
    category="color",
    description="Return binary grid where cells > threshold become high value.",
    parameters=[
        ParameterSpec("threshold", "int", "Threshold to compare against."),
        ParameterSpec(
            "low",
            "int",
            "Value for cells <= threshold.",
            default=0,
        ),
        ParameterSpec(
            "high",
            "int",
            "Value for cells > threshold.",
            default=1,
        ),
    ],
)
def threshold_gt(grid: "Grid", threshold: int, *, low: int = 0, high: int = 1) -> "Grid":
    from .grid import Grid

    if not all(isinstance(val, int) for val in (threshold, low, high)):
        raise TypeError("threshold, low, and high must be integers")

    return Grid([[high if value > threshold else low for value in row] for row in grid.cells])


# Topology/Object primitives ---------------------------------------------------


def _flood_fill_data(cells, start_y: int, start_x: int):
    height = len(cells)
    width = len(cells[0])
    target = cells[start_y][start_x]
    visited = set()
    queue = [(start_y, start_x)]
    component = []

    while queue:
        y, x = queue.pop()
        if (y, x) in visited:
            continue
        visited.add((y, x))
        component.append((y, x))

        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < height and 0 <= nx < width and cells[ny][nx] == target:
                queue.append((ny, nx))

    return component, target


def _collect_non_background_component(cells, start_y: int, start_x: int, background_value: int):
    height = len(cells)
    width = len(cells[0])
    visited = set()
    queue = [(start_y, start_x)]
    component = []

    while queue:
        y, x = queue.pop()
        if (y, x) in visited:
            continue
        visited.add((y, x))

        if cells[y][x] == background_value:
            continue

        component.append((y, x))

        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < height and 0 <= nx < width:
                queue.append((ny, nx))

    return component


@register_primitive(
    "flood_fill",
    category="topology",
    description="Flood fill connected region starting at (y, x).",
    parameters=[
        ParameterSpec("y", "int", "Row index of seed cell."),
        ParameterSpec("x", "int", "Column index of seed cell."),
        ParameterSpec("fill", "int", "Value to fill the region with."),
    ],
)
def flood_fill(grid: "Grid", y: int, x: int, fill: int) -> "Grid":
    from .grid import Grid

    if not all(isinstance(val, int) for val in (y, x, fill)):
        raise TypeError("y, x, and fill must be integers")

    height, width = grid.shape
    if not (0 <= y < height and 0 <= x < width):
        raise ValueError("seed coordinates out of bounds")

    component, target = _flood_fill_data(grid.cells, y, x)
    if target == fill:
        return grid.copy()

    new_cells = [list(row) for row in grid.cells]
    for cy, cx in component:
        new_cells[cy][cx] = fill

    return Grid(new_cells)


@register_primitive(
    "extract_shape",
    category="topology",
    description="Extract a connected component seeded at (y, x) into a minimal bounding box.",
    parameters=[
        ParameterSpec("y", "int", "Row index of seed cell."),
        ParameterSpec("x", "int", "Column index of seed cell."),
        ParameterSpec(
            "background",
            "int",
            "Background fill for the bounding box outside component.",
            default=0,
        ),
    ],
)
def extract_shape(grid: "Grid", y: int, x: int, *, background: int = 0) -> "Grid":
    from .grid import Grid

    if not all(isinstance(val, int) for val in (y, x, background)):
        raise TypeError("y, x, and background must be integers")

    height, width = grid.shape
    if not (0 <= y < height and 0 <= x < width):
        raise ValueError("seed coordinates out of bounds")

    palette_counts = Counter(value for row in grid.cells for value in row)
    background_value, _ = palette_counts.most_common(1)[0]
    component = _collect_non_background_component(grid.cells, y, x, background_value)

    if not component:
        return Grid([[background]])
    rows = [cy for cy, _ in component]
    cols = [cx for _, cx in component]
    min_y, max_y = min(rows), max(rows)
    min_x, max_x = min(cols), max(cols)

    box_height = max_y - min_y + 1
    box_width = max_x - min_x + 1

    new_cells = [[background for _ in range(box_width)] for _ in range(box_height)]
    for cy, cx in component:
        new_cells[cy - min_y][cx - min_x] = grid.cells[cy][cx]

    return Grid(new_cells)


@register_primitive(
    "merge_touching",
    category="topology",
    description="Merge two values when adjacent (4-neighbour) into target value.",
    parameters=[
        ParameterSpec("value_a", "int", "First value to merge."),
        ParameterSpec("value_b", "int", "Second value to merge."),
        ParameterSpec("target", "int", "Merged value."),
    ],
)
def merge_touching(grid: "Grid", value_a: int, value_b: int, target: int) -> "Grid":
    from .grid import Grid

    if not all(isinstance(val, int) for val in (value_a, value_b, target)):
        raise TypeError("value_a, value_b, and target must be integers")

    height, width = grid.shape
    new_cells = [list(row) for row in grid.cells]

    directions = (
        (0, 1),
        (1, 0),
        (0, -1),
        (-1, 0),
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
    )

    for y in range(height):
        for x in range(width):
            for dy, dx in directions:
                ny, nx = y + dy, x + dx
                if not (0 <= ny < height and 0 <= nx < width):
                    continue
                v1 = grid.cells[y][x]
                v2 = grid.cells[ny][nx]
                if {v1, v2} == {value_a, value_b}:
                    new_cells[y][x] = target
                    new_cells[ny][nx] = target

    return Grid(new_cells)
