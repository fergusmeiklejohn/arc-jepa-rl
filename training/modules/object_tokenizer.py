"""Utilities to convert ARC grids into object-centric feature tokens."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from arcgen import Grid, extract_objects

try:  # pragma: no cover - torch is optional at runtime
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


BASE_FEATURE_KEYS = (
    "area",
    "perimeter",
    "height",
    "width",
    "aspect_ratio",
    "color_count",
    "dominant_color",
    "centroid_y",
    "centroid_x",
    "bbox_min_y",
    "bbox_min_x",
    "bbox_max_y",
    "bbox_max_x",
)


@dataclass(frozen=True)
class TokenizedObjects:
    """Structured representation of object tokens.

    All arrays are Python lists to avoid enforcing a heavyweight dependency.
    Call :meth:`to_tensors` to obtain PyTorch tensors when torch is available.
    """

    features: List[List[float]]
    mask: List[int]
    adjacency: List[List[int]]

    def to_tensors(self):  # pragma: no cover - exercised in integration tests
        if torch is None:
            raise RuntimeError("PyTorch is required for TokenizedObjects.to_tensors()")

        features_tensor = torch.tensor(self.features, dtype=torch.float32)
        mask_tensor = torch.tensor(self.mask, dtype=torch.float32)
        adjacency_tensor = torch.tensor(self.adjacency, dtype=torch.float32)
        return features_tensor, mask_tensor, adjacency_tensor


def tokenize_grid_objects(
    grid: Grid,
    *,
    max_objects: int = 16,
    max_color_features: int = 10,
    background: int | Iterable[int] = 0,
    connectivity: int = 4,
    normalize: bool = True,
    respect_colors: bool = True,
) -> TokenizedObjects:
    """Convert ``grid`` into a padded set of object feature tokens.

    ``max_objects`` controls the number of slots; excess objects are truncated
    after sorting by area (descending). ``max_color_features`` limits how many
    per-color frequency slots are included in each feature vector.
    """

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
            ((int(key.removeprefix("color_freq_")), value) for key, value in feature_dict.items() if key.startswith("color_freq_")),
            key=lambda item: item[0],
        )
        color_features = [0.0 for _ in range(max_color_features)]
        for slot, (_, value) in enumerate(color_entries[:max_color_features]):
            color_features[slot] = float(value)

        vector.extend(color_features)
        feature_vectors.append(vector)
        mask.append(1)

    # Compute adjacency among selected objects.
    for i, left in enumerate(objects):
        for j, right in enumerate(objects):
            if i == j:
                continue
            if _objects_touch(left, right):
                adjacency_matrix[i][j] = 1

    # Pad remaining slots.
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
