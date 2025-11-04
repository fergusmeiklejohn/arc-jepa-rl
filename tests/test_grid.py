"""Tests for the ARC grid engine."""

from __future__ import annotations

import pytest

from arcgen import Grid, SeededRNG


def test_grid_validates_rectangular_input() -> None:
    with pytest.raises(ValueError):
        Grid([])  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        Grid([[1, 2], [3]])  # ragged rows

    with pytest.raises(ValueError):
        Grid([[1, -1]])


def test_grid_basic_properties() -> None:
    grid = Grid([[1, 2, 3], [4, 5, 6]])
    assert grid.height == 2
    assert grid.width == 3
    assert grid.shape == (2, 3)
    assert grid.flatten() == [1, 2, 3, 4, 5, 6]
    assert grid.palette() == [1, 2, 3, 4, 5, 6]


def test_grid_from_flat_and_copy() -> None:
    grid = Grid.from_flat([1, 2, 3, 4], width=2)
    assert grid.shape == (2, 2)

    clone = grid.copy()
    assert clone.cells == grid.cells
    assert clone is not grid


def test_grid_full_and_replace() -> None:
    grid = Grid.full(3, 3, value=2)
    assert grid.flatten() == [2] * 9

    replaced = grid.replace(2, 5)
    assert replaced.flatten() == [5] * 9


def test_grid_transpose() -> None:
    grid = Grid([[1, 2], [3, 4], [5, 6]])
    transposed = grid.transpose()
    assert transposed.shape == (2, 3)
    assert transposed.cells == ((1, 3, 5), (2, 4, 6))


def test_grid_random_reproducible() -> None:
    rng1 = SeededRNG(42)
    rng2 = SeededRNG(42)

    grid_a = Grid.random(4, 4, palette=[1, 2, 3], rng=rng1, fill_prob=0.7)
    grid_b = Grid.random(4, 4, palette=[1, 2, 3], rng=rng2, fill_prob=0.7)

    assert grid_a.cells == grid_b.cells


def test_seeded_rng_spawn_produces_different_streams() -> None:
    parent = SeededRNG(1)
    child_a = parent.spawn(0)
    child_b = parent.spawn(0)

    assert child_a.randint(0, 1000) != child_b.randint(0, 1000)
