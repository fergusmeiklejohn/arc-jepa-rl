import math

import pytest

from arcgen import Grid, compute_adjacency, extract_objects


def test_extract_objects_basic_properties():
    grid = Grid(
        [
            [0, 1, 1],
            [0, 0, 2],
            [0, 0, 2],
        ]
    )

    objects = extract_objects(grid, background=0)

    assert len(objects) == 2

    first, second = objects

    assert first.area == 2
    assert first.bbox == (0, 1, 0, 2)
    assert first.dominant_color == 1
    assert math.isclose(first.centroid[0], 0.0)
    assert math.isclose(first.centroid[1], 1.5)
    assert first.perimeter == 6

    assert second.area == 2
    assert second.bbox == (1, 2, 2, 2)
    assert second.unique_colors == (2,)


def test_extract_objects_without_respecting_colors_merges_components():
    grid = Grid(
        [
            [0, 1, 1],
            [0, 0, 2],
            [0, 0, 2],
        ]
    )

    objects = extract_objects(grid, background=0, respect_colors=False)
    assert len(objects) == 1
    assert objects[0].unique_colors == (1, 2)


def test_extract_objects_with_connectivity_and_min_size():
    grid = Grid(
        [
            [0, 3, 0],
            [0, 0, 3],
            [0, 0, 0],
        ]
    )

    objects_4 = extract_objects(grid, background=0, connectivity=4)
    objects_8 = extract_objects(grid, background=0, connectivity=8)

    assert len(objects_4) == 2
    assert len(objects_8) == 1  # diagonals join under 8-connectivity

    small_filtered = extract_objects(grid, background=0, connectivity=8, min_size=3)
    assert small_filtered == []


def test_feature_dict_contains_color_frequencies():
    grid = Grid(
        [
            [0, 7, 5],
            [0, 7, 5],
        ]
    )

    (obj,) = extract_objects(grid, background=0, connectivity=4, respect_colors=False)
    features = obj.as_feature_dict(normalize=False)

    assert math.isclose(features["area"], 4.0)
    assert features["color_count"] == 2.0
    assert math.isclose(features["color_freq_7"], 2.0)
    assert math.isclose(features["color_freq_5"], 2.0)


def test_compute_adjacency_touching_mode():
    grid = Grid(
        [
            [1, 0, 0],
            [1, 0, 2],
            [0, 2, 2],
        ]
    )

    objects = extract_objects(grid, background=0)
    adjacency = compute_adjacency(objects, mode="touching")

    assert adjacency[objects[0].id] == (objects[1].id,)
    assert adjacency[objects[1].id] == (objects[0].id,)
