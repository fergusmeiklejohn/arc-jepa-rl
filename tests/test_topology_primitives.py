"""Tests for topology/object primitives."""

from __future__ import annotations

import pytest

from arcgen import Grid, PRIMITIVE_REGISTRY


def get(name: str):
    return PRIMITIVE_REGISTRY.get(name)


def test_flood_fill_changes_connected_component() -> None:
    grid = Grid([
        [1, 1, 0],
        [1, 0, 0],
        [2, 2, 0],
    ])
    filled = get("flood_fill").apply(grid, y=0, x=0, fill=9)
    assert filled.cells == ((9, 9, 0), (9, 0, 0), (2, 2, 0))


def test_flood_fill_no_change_when_same_value() -> None:
    grid = Grid([[1, 1], [1, 1]])
    filled = get("flood_fill").apply(grid, y=0, x=0, fill=1)
    assert filled.cells == grid.cells


def test_flood_fill_validates_arguments() -> None:
    with pytest.raises(TypeError):
        get("flood_fill").apply(Grid([[1]]), y="0", x=0, fill=2)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        get("flood_fill").apply(Grid([[1]]), y=1, x=0, fill=2)


def test_extract_shape_returns_bounding_box() -> None:
    grid = Grid([
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 2, 2],
        [0, 0, 0, 0],
    ])
    extracted = get("extract_shape").apply(grid, y=1, x=1, background=9)
    assert extracted.cells == ((1, 1, 9), (1, 2, 2))


def test_extract_shape_validates_arguments() -> None:
    with pytest.raises(TypeError):
        get("extract_shape").apply(Grid([[1]]), y=0, x=0, background="0")  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        get("extract_shape").apply(Grid([[1]]), y=2, x=0)


def test_merge_touching_merges_pairs() -> None:
    grid = Grid([
        [1, 2, 0],
        [2, 0, 1],
        [0, 0, 0],
    ])
    merged = get("merge_touching").apply(grid, value_a=1, value_b=2, target=7)
    assert merged.cells == ((7, 7, 0), (7, 0, 7), (0, 0, 0))


def test_merge_touching_validates_arguments() -> None:
    with pytest.raises(TypeError):
        get("merge_touching").apply(Grid([[1]]), value_a=1, value_b="2", target=3)  # type: ignore[arg-type]
