"""Tests for the primitive registry."""

from __future__ import annotations

import pytest

from arcgen import (
    Grid,
    ParameterSpec,
    PrimitiveRegistry,
    register_primitive,
)


def test_primitive_register_and_retrieve() -> None:
    registry = PrimitiveRegistry()

    @registry.register(
        "identity",
        category="utility",
        description="Return grid as-is",
    )
    def identity(grid: Grid) -> Grid:
        return grid

    spec = registry.get("identity")
    assert spec.name == "identity"
    assert spec.category == "utility"
    assert spec.description == "Return grid as-is"

    sample = Grid([[1, 2], [3, 4]])
    assert spec.apply(sample) is sample


def test_duplicate_registration_raises() -> None:
    registry = PrimitiveRegistry()

    @registry.register("noop", category="utility")
    def noop(grid: Grid) -> Grid:
        return grid

    with pytest.raises(ValueError):
        registry.register("noop", category="utility")(noop)


def test_parameter_metadata_and_global_helper() -> None:
    spec = ParameterSpec(
        name="color",
        type="int",
        description="Fill color",
        default=3,
        choices=(1, 2, 3),
    )

    registry = PrimitiveRegistry()

    @registry.register(
        "color_fill",
        category="color",
        parameters=[spec],
    )
    def color_fill(grid: Grid, color: int = 0) -> Grid:
        return Grid.full(grid.height, grid.width, value=color)

    entry = registry.get("color_fill")
    assert entry.parameters[0].name == "color"
    assert entry.parameters[0].choices == (1, 2, 3)
    assert entry.parameters[0].has_default()

    grid = Grid([[0, 1], [1, 0]])
    result = entry.apply(grid, color=2)
    assert result.flatten() == [2, 2, 2, 2]


def test_global_registry_helper_registers() -> None:
    # Register a temporary primitive using the module-level helper.
    @register_primitive("test_double", category="utility")
    def double(grid: Grid) -> Grid:
        data = [[value * 2 for value in row] for row in grid.cells]
        return Grid(data)

    from arcgen import PRIMITIVE_REGISTRY

    spec = PRIMITIVE_REGISTRY.get("test_double")
    sample = Grid([[1]])
    assert spec.apply(sample).flatten() == [2]
