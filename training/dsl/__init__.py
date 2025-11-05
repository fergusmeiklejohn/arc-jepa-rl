"""Typed DSL for ARC program synthesis scaffolding."""

from .types import DSLType, Color, Position, Shape, GridValue, Grid
from .primitives import Primitive, PrimitiveRegistry, build_default_primitive_registry
from .enumerator import ProgramEnumerator, Program, InputVar, Expression, ProgramInterpreter
from .scoring import NeuralGuidedScorer, ProgramScore

__all__ = [
    "DSLType",
    "Color",
    "Position",
    "Shape",
    "GridValue",
    "Grid",
    "Primitive",
    "PrimitiveRegistry",
    "build_default_primitive_registry",
    "ProgramEnumerator",
    "Program",
    "InputVar",
    "Expression",
    "ProgramInterpreter",
    "NeuralGuidedScorer",
    "ProgramScore",
]
