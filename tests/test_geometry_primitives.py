"""Tests for geometry primitives."""

from __future__ import annotations

import pytest

from arcgen import Grid, PRIMITIVE_REGISTRY


def get(name: str):
    return PRIMITIVE_REGISTRY.get(name)


def test_mirror_x_flips_vertically() -> None:
    grid = Grid([[1, 2], [3, 4], [5, 6]])
    mirrored = get("mirror_x").apply(grid)
    assert mirrored.cells == ((5, 6), (3, 4), (1, 2))


def test_mirror_y_flips_horizontally() -> None:
    grid = Grid([[1, 2], [3, 4]])
    mirrored = get("mirror_y").apply(grid)
    assert mirrored.cells == ((2, 1), (4, 3))


def test_rotate90_clockwise_multiple_times() -> None:
    grid = Grid([[1, 2], [3, 4]])

    rot90 = get("rotate90").apply(grid)
    assert rot90.cells == ((3, 1), (4, 2))

    rot180 = get("rotate90").apply(grid, k=2)
    assert rot180.cells == ((4, 3), (2, 1))

    rot270 = get("rotate90").apply(grid, k=3)
    assert rot270.cells == ((2, 4), (1, 3))

    identity = get("rotate90").apply(grid, k=4)
    assert identity.cells == grid.cells


def test_rotate90_requires_integer() -> None:
    with pytest.raises(TypeError):
        get("rotate90").apply(Grid([[1]]), k=1.5)


def test_translate_shifts_with_fill() -> None:
    grid = Grid([[1, 2], [3, 4]])
    translated = get("translate").apply(grid, dx=1, dy=1, fill=0)
    assert translated.cells == ((0, 0), (0, 1))

    translated_left = get("translate").apply(grid, dx=-1, dy=0, fill=9)
    assert translated_left.cells == ((2, 9), (4, 9))


def test_translate_validates_types() -> None:
    with pytest.raises(TypeError):
        get("translate").apply(Grid([[1]]), dx=1.2)
