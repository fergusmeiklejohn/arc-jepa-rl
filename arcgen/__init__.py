"""Core modules for synthetic ARC generation."""

from .grid import Grid, SeededRNG
from .objects import GridObject, compute_adjacency, extract_objects
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
    "GridObject",
    "extract_objects",
    "compute_adjacency",
    "PrimitiveRegistry",
    "PrimitiveSpec",
    "ParameterSpec",
    "register_primitive",
    "PRIMITIVE_REGISTRY",
]
