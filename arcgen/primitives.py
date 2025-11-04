"""Primitive registry for ARC transformations."""

from __future__ import annotations

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
