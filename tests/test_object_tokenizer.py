import math

import pytest

from arcgen import Grid
from training.modules import tokenize_grid_objects


def test_tokenize_grid_objects_basic_behavior():
    grid = Grid(
        [
            [0, 1, 1],
            [0, 0, 2],
            [0, 0, 2],
        ]
    )

    tokens = tokenize_grid_objects(grid, max_objects=4, max_color_features=2)

    assert len(tokens.features) == 4
    assert tokens.mask == [1, 1, 0, 0]

    first = tokens.features[0]
    second = tokens.features[1]

    assert math.isclose(first[0], 2 / 9)
    assert math.isclose(first[1], 6 / 9)
    assert first[5] == 1.0  # single color
    assert math.isclose(first[-2], 2 / 9)  # color_freq_1
    assert math.isclose(first[-1], 0.0)

    assert math.isclose(second[0], 2 / 9)
    assert math.isclose(sum(second[-2:]), 2 / 9)

    adjacency = tokens.adjacency
    assert adjacency[0][1] == 1
    assert adjacency[1][0] == 1
    assert adjacency[2][0] == 0


def test_tokenize_grid_objects_without_color_partitioning():
    grid = Grid(
        [
            [0, 1, 1],
            [0, 0, 2],
            [0, 0, 2],
        ]
    )

    tokens = tokenize_grid_objects(grid, respect_colors=False, max_objects=2, max_color_features=2)
    assert tokens.mask == [1, 0]
    freq_features = tokens.features[0][-2:]
    assert all(math.isclose(value, 2 / 9) for value in freq_features)
    assert math.isclose(sum(freq_features), 4 / 9)


def test_tokenized_objects_to_tensors_optional():
    grid = Grid([[0, 1], [0, 1]])
    tokens = tokenize_grid_objects(grid, max_objects=1, max_color_features=1)

    pytest.importorskip("torch", reason="token conversion requires torch")
    features, mask, adjacency = tokens.to_tensors()
    assert features.shape[0] == 1
    assert mask.shape[0] == 1
    assert adjacency.shape == (1, 1)
