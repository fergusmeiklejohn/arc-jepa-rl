"""Utilities for promoting discovered options into the DSL primitive registry."""

from __future__ import annotations

from typing import Optional

from arcgen import PRIMITIVE_REGISTRY as ARC_PRIMITIVE_REGISTRY, Grid
from arcgen.primitives import PrimitiveRegistry as ArcPrimitiveRegistry

from training.dsl.primitives import Primitive, PrimitiveRegistry as DSLPrimitiveRegistry
from training.dsl.types import Grid as GridType

from .discovery import DiscoveredOption


def promote_discovered_option(
    discovered: DiscoveredOption,
    dsl_registry: DSLPrimitiveRegistry,
    *,
    arc_registry: ArcPrimitiveRegistry = ARC_PRIMITIVE_REGISTRY,
    category: str = "discovered",
    description: Optional[str] = None,
) -> str:
    """Register a discovered option as both an ARC primitive and DSL primitive."""

    if not discovered.sequence:
        raise ValueError("discovered option must contain at least one underlying option")

    primitive_name = discovered.name
    description = description or _format_description(discovered)

    if not _arc_has_primitive(arc_registry, primitive_name):
        arc_registry.register(
            primitive_name,
            category=category,
            description=description,
        )(_build_arc_wrapper(discovered))

    if not _dsl_has_primitive(dsl_registry, primitive_name):
        dsl_registry.register(
            Primitive(
                name=primitive_name,
                input_types=(GridType,),
                output_type=GridType,
                implementation=_build_dsl_wrapper(arc_registry, primitive_name),
                description=description,
            )
        )

    return primitive_name


def _build_arc_wrapper(discovered: DiscoveredOption):
    def composite(grid: Grid) -> Grid:
        return discovered.apply(grid)

    composite.__name__ = discovered.name  # type: ignore[attr-defined]
    composite.__doc__ = _format_description(discovered)
    return composite


def _build_dsl_wrapper(arc_registry: ArcPrimitiveRegistry, primitive_name: str):
    def impl(grid: Grid) -> Grid:
        primitive = arc_registry.get(primitive_name)
        return primitive.apply(grid)

    impl.__name__ = f"dsl_{primitive_name}"  # type: ignore[attr-defined]
    return impl


def _arc_has_primitive(arc_registry: ArcPrimitiveRegistry, name: str) -> bool:
    try:
        arc_registry.get(name)
    except KeyError:
        return False
    return True


def _dsl_has_primitive(dsl_registry: DSLPrimitiveRegistry, name: str) -> bool:
    return any(primitive.name == name for primitive in dsl_registry.list())


def _format_description(discovered: DiscoveredOption) -> str:
    seq = ", ".join(discovered.sequence_names)
    return (
        f"Auto-discovered composite option: [{seq}] "
        f"(support={discovered.support}, success_rate={discovered.success_rate:.2f})"
    )

