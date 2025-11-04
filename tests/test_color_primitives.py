"""Tests for color/value primitives."""

from __future__ import annotations

import pytest

from arcgen import Grid, PRIMITIVE_REGISTRY


def get(name: str):
    return PRIMITIVE_REGISTRY.get(name)


def test_recolor_swaps_values() -> None:
    grid = Grid([[1, 2], [2, 3]])
    result = get("recolor").apply(grid, source=2, target=5)
    assert result.cells == ((1, 5), (5, 3))


def test_recolor_validates_arguments() -> None:
    with pytest.raises(TypeError):
        get("recolor").apply(Grid([[1]]), source="a", target=1)  # type: ignore[arg-type]


def test_invert_palette() -> None:
    grid = Grid([[0, 1, 2], [2, 1, 0]])
    inverted = get("invert_palette").apply(grid)
    assert inverted.cells == ((2, 1, 0), (0, 1, 2))


def test_threshold_gt_respects_params() -> None:
    grid = Grid([[0, 3, 5], [6, 1, 4]])
    result = get("threshold_gt").apply(grid, threshold=3, low=9, high=7)
    assert result.cells == ((9, 9, 7), (7, 9, 7))


def test_threshold_gt_validates_types() -> None:
    with pytest.raises(TypeError):
        get("threshold_gt").apply(Grid([[1]]), threshold=1, low="0")  # type: ignore[arg-type]
