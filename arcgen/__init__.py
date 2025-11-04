"""Core modules for synthetic ARC generation."""

from .grid import Grid, SeededRNG
from .primitives import (
    REGISTRY as PRIMITIVE_REGISTRY,
    ParameterSpec,
    PrimitiveRegistry,
    PrimitiveSpec,
    register_primitive,
)

__all__ = [
    "Grid",
    "SeededRNG",
    "PrimitiveRegistry",
    "PrimitiveSpec",
    "ParameterSpec",
    "register_primitive",
    "PRIMITIVE_REGISTRY",
]
