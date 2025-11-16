import math

import pytest

from arcgen import Grid
from training.modules import tokenize_grid_objects
from training.modules.object_tokenizer_legacy import tokenize_grid_objects_legacy


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


@pytest.mark.parametrize("respect_colors", [True, False])
@pytest.mark.parametrize("normalize", [True, False])
def test_vectorized_matches_legacy(respect_colors, normalize):
    grids = [
        Grid([[0, 1, 2], [2, 2, 0], [0, 3, 3]]),
        Grid([[1, 1, 1, 0], [0, 2, 2, 2], [3, 0, 3, 0], [4, 4, 0, 4]]),
    ]
    kwargs = {
        "max_objects": 4,
        "max_color_features": 3,
        "connectivity": 8,
        "respect_colors": respect_colors,
        "normalize": normalize,
    }

    for grid in grids:
        new_tokens = tokenize_grid_objects(grid, **kwargs)
        old_tokens = tokenize_grid_objects_legacy(grid, **kwargs)

        assert new_tokens.mask == old_tokens.mask
        assert new_tokens.adjacency == old_tokens.adjacency

        for new, old in zip(new_tokens.features, old_tokens.features):
            assert len(new) == len(old)
            for a, b in zip(new, old):
                assert math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)
