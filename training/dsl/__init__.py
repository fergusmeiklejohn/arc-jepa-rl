"""Typed DSL for ARC program synthesis scaffolding."""

from .types import DSLType, Color, Position, Shape, GridValue
from .primitives import Primitive, PrimitiveRegistry, build_default_primitive_registry
from .enumerator import ProgramEnumerator, Program, ProgramNode
from .scoring import NeuralGuidedScorer, ProgramScore

__all__ = [
    "DSLType",
    "Color",
    "Position",
    "Shape",
    "GridValue",
    "Primitive",
    "PrimitiveRegistry",
    "build_default_primitive_registry",
    "ProgramEnumerator",
    "Program",
    "ProgramNode",
    "NeuralGuidedScorer",
    "ProgramScore",
]
