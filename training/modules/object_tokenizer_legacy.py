"""Legacy reference implementation for object tokenization.

This module preserves the pre-vectorized tokenizer so tests/benchmarks can
compare outputs and speedups against the new optimized path.
"""

from __future__ import annotations

from typing import Iterable, List

from arcgen import Grid, extract_objects

from .object_tokenizer import BASE_FEATURE_KEYS, TokenizedObjects


def tokenize_grid_objects_legacy(
    grid: Grid,
    *,
    max_objects: int = 16,
    max_color_features: int = 10,
    background: int | Iterable[int] = 0,
    connectivity: int = 4,
    normalize: bool = True,
    respect_colors: bool = True,
) -> TokenizedObjects:
    if max_objects <= 0:
        raise ValueError("max_objects must be positive")
    if max_color_features < 0:
        raise ValueError("max_color_features must be non-negative")

    objects = extract_objects(
        grid,
        background=background,
        connectivity=connectivity,
        respect_colors=respect_colors,
    )

    objects = sorted(objects, key=lambda obj: (-obj.area, obj.id))[:max_objects]
    feature_vectors: List[List[float]] = []
    mask: List[int] = []
    adjacency_matrix: List[List[int]] = [[0 for _ in range(max_objects)] for _ in range(max_objects)]

    for idx, obj in enumerate(objects):
        feature_dict = obj.as_feature_dict(normalize=normalize)
        vector: List[float] = [float(feature_dict[key]) for key in BASE_FEATURE_KEYS]

        color_entries = sorted(
            (
                (int(key.removeprefix("color_freq_")), value)
                for key, value in feature_dict.items()
                if key.startswith("color_freq_")
            ),
            key=lambda item: item[0],
        )
        color_features = [0.0 for _ in range(max_color_features)]
        for slot, (_, value) in enumerate(color_entries[:max_color_features]):
            color_features[slot] = float(value)

        vector.extend(color_features)
        feature_vectors.append(vector)
        mask.append(1)

    for i, left in enumerate(objects):
        for j, right in enumerate(objects):
            if i == j:
                continue
            if _objects_touch(left, right):
                adjacency_matrix[i][j] = 1

    feature_length = len(feature_vectors[0]) if feature_vectors else len(BASE_FEATURE_KEYS) + max_color_features
    while len(feature_vectors) < max_objects:
        feature_vectors.append([0.0 for _ in range(feature_length)])
        mask.append(0)

    return TokenizedObjects(features=feature_vectors, mask=mask, adjacency=adjacency_matrix)


def _objects_touch(a, b) -> bool:
    return not (
        a.max_y + 1 < b.min_y
        or b.max_y + 1 < a.min_y
        or a.max_x + 1 < b.min_x
        or b.max_x + 1 < a.min_x
    )
