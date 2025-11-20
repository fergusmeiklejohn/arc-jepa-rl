"""Type system definitions for the ARC DSL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple, Union, overload

try:  # pragma: no cover - optional dependency in some environments
    from arcgen import Grid as ArcGrid, GridObject
except Exception:  # pragma: no cover
    ArcGrid = None  # type: ignore
    GridObject = None  # type: ignore


@dataclass(frozen=True)
class DSLType:
    name: str

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.name

    def __hash__(self) -> int:  # pragma: no cover - data class default not stable across subclasses
        return hash((self.name, self.__class__))

    def is_supertype_of(self, other: "DSLType") -> bool:
        return self == other


@dataclass(frozen=True)
class ProductType(DSLType):
    """Product type (e.g., tuple components)."""

    components: Tuple[DSLType, ...]

    def __hash__(self) -> int:  # pragma: no cover - structural hashing
        return hash((self.name, self.components))

    def is_supertype_of(self, other: "DSLType") -> bool:
        return isinstance(other, ProductType) and self.components == other.components


@dataclass(frozen=True)
class SumType(DSLType):
    """Sum/union type."""

    options: Tuple[DSLType, ...]

    def __hash__(self) -> int:  # pragma: no cover - structural hashing
        return hash((self.name, tuple(self.options)))

    def is_supertype_of(self, other: "DSLType") -> bool:
        return any(option == other or option.is_supertype_of(other) for option in self.options)


Color = DSLType("Color")
Position = DSLType("Position")
Shape = DSLType("Shape")
ShapeList = DSLType("ShapeList")
Grid = DSLType("Grid")
GridList = DSLType("GridList")
GridValue = DSLType("GridValue")
Bool = DSLType("Bool")


# Structured aliases
Position = ProductType("Position", components=(GridValue, GridValue))


def _is_shape(value: object) -> bool:
    return GridObject is not None and isinstance(value, GridObject)


def _is_grid(value: object) -> bool:
    return ArcGrid is not None and isinstance(value, ArcGrid)


def is_value_of_type(value: object, expected: DSLType) -> bool:
    """Return True when runtime value matches the expected DSL type."""

    if isinstance(expected, ProductType):
        if not isinstance(value, (tuple, list)) or len(value) != len(expected.components):
            return False
        return all(is_value_of_type(val, component) for val, component in zip(value, expected.components))

    if isinstance(expected, SumType):
        return any(is_value_of_type(value, option) for option in expected.options)

    if expected == Grid:
        return _is_grid(value)
    if expected == GridList:
        return isinstance(value, (tuple, list)) and all(_is_grid(v) for v in value)
    if expected == Shape:
        return _is_shape(value)
    if expected == ShapeList:
        return isinstance(value, (tuple, list)) and all(_is_shape(v) for v in value)
    if expected == Bool:
        return isinstance(value, bool)
    if expected == GridValue:
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == Color:
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == Position:
        return isinstance(value, (tuple, list)) and len(value) == 2 and all(
            isinstance(coord, int) and not isinstance(coord, bool) for coord in value
        )
    return True  # Fallback: treat unrecognized types as permissive


def infer_type(value: object) -> DSLType | None:
    """Best-effort inference from runtime value to DSL type."""

    if _is_grid(value):
        return Grid
    if isinstance(value, (tuple, list)) and value and all(_is_grid(v) for v in value):
        return GridList
    if _is_shape(value):
        return Shape
    if isinstance(value, (tuple, list)) and value and all(_is_shape(v) for v in value):
        return ShapeList
    if isinstance(value, bool):
        return Bool
    if isinstance(value, int) and not isinstance(value, bool):
        return GridValue
    if isinstance(value, (tuple, list)) and len(value) == 2 and all(
        isinstance(coord, int) and not isinstance(coord, bool) for coord in value
    ):
        return Position
    return None
