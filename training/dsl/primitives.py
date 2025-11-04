"""Primitive definitions and registry for the ARC DSL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence

from .types import DSLType, Grid, Color, Position, Shape, GridValue


@dataclass(frozen=True)
class Primitive:
    name: str
    input_types: Sequence[DSLType]
    output_type: DSLType
    description: str = ""


class PrimitiveRegistry:
    def __init__(self) -> None:
        self._primitives: Dict[str, Primitive] = {}

    def register(self, primitive: Primitive) -> None:
        if primitive.name in self._primitives:
            raise ValueError(f"primitive '{primitive.name}' already registered")
        self._primitives[primitive.name] = primitive

    def get(self, name: str) -> Primitive:
        return self._primitives[name]

    def list(self) -> List[Primitive]:
        return list(self._primitives.values())


def build_default_primitive_registry() -> PrimitiveRegistry:
    registry = PrimitiveRegistry()
    registry.register(Primitive("mirror_x", input_types=(Grid,), output_type=Grid, description="Reflect grid across x-axis"))
    registry.register(Primitive("mirror_y", input_types=(Grid,), output_type=Grid, description="Reflect grid across y-axis"))
    registry.register(Primitive("rotate90", input_types=(Grid,), output_type=Grid, description="Rotate grid by 90 degrees"))
    registry.register(Primitive("color_filter", input_types=(Grid, Color), output_type=Grid, description="Keep only cells of a given color"))
    registry.register(Primitive("count_objects", input_types=(Grid,), output_type=GridValue, description="Count connected components"))
    registry.register(Primitive("bounding_box", input_types=(Grid,), output_type=Shape, description="Return bounding box shape"))
    return registry
