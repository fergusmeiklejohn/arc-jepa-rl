"""Typed DSL for ARC program synthesis scaffolding."""

from .types import (
    DSLType,
    Bool,
    Color,
    Position,
    Shape,
    ShapeList,
    GridList,
    GridValue,
    Grid,
    ProductType,
    SumType,
    is_value_of_type,
    infer_type,
)
from .primitives import Primitive, PrimitiveRegistry, build_default_primitive_registry
from .enumerator import ProgramEnumerator, Program, InputVar, Expression, ProgramInterpreter
from .guide import GuidanceScorer, ProgramEncoder, GuidedBeamSearch, encode_program_features
from .data import GuidanceDataset, GuidanceExample, build_guidance_examples
from .scoring import NeuralGuidedScorer, ProgramScore

__all__ = [
    "DSLType",
    "Bool",
    "Color",
    "Position",
    "Shape",
    "ShapeList",
    "GridList",
    "GridValue",
    "Grid",
    "ProductType",
    "SumType",
    "is_value_of_type",
    "infer_type",
    "Primitive",
    "PrimitiveRegistry",
    "build_default_primitive_registry",
    "ProgramEnumerator",
    "Program",
    "InputVar",
    "Expression",
    "ProgramInterpreter",
    "ProgramEncoder",
    "GuidanceScorer",
    "GuidedBeamSearch",
    "encode_program_features",
    "GuidanceDataset",
    "GuidanceExample",
    "build_guidance_examples",
    "NeuralGuidedScorer",
    "ProgramScore",
]
