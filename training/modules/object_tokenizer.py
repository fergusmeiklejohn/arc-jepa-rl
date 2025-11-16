"""Utilities to convert ARC grids into object-centric feature tokens."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from arcgen import Grid

try:  # pragma: no cover - torch is optional at runtime
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:  # pragma: no cover - numpy optional at runtime
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:  # pragma: no cover - scipy optional at runtime
    from scipy import ndimage
except Exception:  # pragma: no cover
    ndimage = None  # type: ignore


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


@dataclass(frozen=True)
class _ComponentRecord:
    order: int
    area: int
    perimeter: int
    min_y: int
    min_x: int
    max_y: int
    max_x: int
    centroid_y: float
    centroid_x: float
    color_ids: "np.ndarray"
    color_counts: "np.ndarray"
    color_first_indices: "np.ndarray"

    @property
    def height(self) -> int:
        return self.max_y - self.min_y + 1

    @property
    def width(self) -> int:
        return self.max_x - self.min_x + 1

    @property
    def color_count(self) -> int:
        return int(self.color_ids.size)

    @property
    def dominant_color(self) -> int:
        if self.color_count == 0:
            return 0
        max_count = self.color_counts.max()
        candidates = np.flatnonzero(self.color_counts == max_count)
        if candidates.size == 1:
            idx = int(candidates[0])
        else:
            idx = int(candidates[np.argmin(self.color_first_indices[candidates])])
        return int(self.color_ids[idx])

    @property
    def aspect_ratio(self) -> float:
        if self.height == 0:
            return 0.0
        return float(self.width) / float(self.height)


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
    """Convert ``grid`` into a padded set of object feature tokens."""

    if max_objects <= 0:
        raise ValueError("max_objects must be positive")
    if max_color_features < 0:
        raise ValueError("max_color_features must be non-negative")
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")

    if np is None:  # pragma: no cover - fallback when numpy missing
        from .object_tokenizer_legacy import tokenize_grid_objects_legacy

        return tokenize_grid_objects_legacy(
            grid,
            max_objects=max_objects,
            max_color_features=max_color_features,
            background=background,
            connectivity=connectivity,
            normalize=normalize,
            respect_colors=respect_colors,
        )

    grid_array = _grid_to_numpy(grid)
    components = _extract_components(
        grid_array,
        background=background,
        connectivity=connectivity,
        respect_colors=respect_colors,
    )
    components = sorted(components, key=lambda obj: (-obj.area, obj.order))
    selected = components[:max_objects]

    feature_vectors, mask = _build_feature_matrix(
        selected,
        grid_shape=grid_array.shape,
        normalize=normalize,
        max_color_features=max_color_features,
        max_objects=max_objects,
    )
    adjacency = _build_adjacency_matrix(selected, max_objects=max_objects)
    return TokenizedObjects(features=feature_vectors, mask=mask, adjacency=adjacency)


def _grid_to_numpy(grid: Grid) -> "np.ndarray":
    return np.array(grid.cells, dtype=np.int64)


def _normalize_background(background: int | Iterable[int]) -> "np.ndarray":
    if isinstance(background, int):
        return np.array([int(background)], dtype=np.int64)
    values = [int(value) for value in background]
    if not values:
        raise ValueError("background must contain at least one value")
    return np.array(values, dtype=np.int64)


def _extract_components(
    grid_array: "np.ndarray",
    *,
    background: int | Iterable[int],
    connectivity: int,
    respect_colors: bool,
) -> List[_ComponentRecord]:
    allowed = np.isin(grid_array, _normalize_background(background), invert=True)
    if not allowed.any():
        return []

    labels, num_components = _label_components(
        grid_array,
        allowed=allowed,
        connectivity=connectivity,
        respect_colors=respect_colors,
    )
    return _records_from_labels(grid_array, labels, num_components)


def _label_components(
    grid_array: "np.ndarray",
    *,
    allowed: "np.ndarray",
    connectivity: int,
    respect_colors: bool,
) -> Tuple["np.ndarray", int]:
    if ndimage is not None:
        return _label_components_scipy(
            grid_array,
            allowed=allowed,
            connectivity=connectivity,
            respect_colors=respect_colors,
        )
    return _label_components_union(
        grid_array,
        allowed=allowed,
        connectivity=connectivity,
        respect_colors=respect_colors,
    )


def _label_components_scipy(
    grid_array: "np.ndarray",
    *,
    allowed: "np.ndarray",
    connectivity: int,
    respect_colors: bool,
) -> Tuple["np.ndarray", int]:
    structure = _build_structure(connectivity)
    labels = np.zeros_like(grid_array, dtype=np.int32)
    next_label = 1

    if respect_colors:
        color_order: list[int] = []
        seen: set[int] = set()
        for y in range(grid_array.shape[0]):
            for x in range(grid_array.shape[1]):
                color = int(grid_array[y, x])
                if allowed[y, x] and color not in seen:
                    seen.add(color)
                    color_order.append(color)
        for color in color_order:
            mask = allowed & (grid_array == color)
            if not mask.any():
                continue
            labeled, count = ndimage.label(mask, structure=structure)
            if count == 0:
                continue
            labeled[labeled > 0] += next_label - 1
            labels += labeled.astype(np.int32)
            next_label += count
    else:
        labeled, count = ndimage.label(allowed, structure=structure)
        labels = labeled.astype(np.int32)
        next_label = count + 1
    relabeled = np.zeros_like(labels)
    label_map: dict[int, int] = {}
    next_id = 1
    for y in range(labels.shape[0]):
        for x in range(labels.shape[1]):
            label = labels[y, x]
            if label == 0:
                continue
            mapped = label_map.setdefault(label, next_id)
            if mapped == next_id:
                next_id += 1
            relabeled[y, x] = mapped

    return relabeled, next_id - 1


def _build_structure(connectivity: int) -> "np.ndarray":
    if connectivity == 4:
        return np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.int8)
    return np.ones((3, 3), dtype=np.int8)


def _label_components_union(
    grid_array: "np.ndarray",
    *,
    allowed: "np.ndarray",
    connectivity: int,
    respect_colors: bool,
) -> Tuple["np.ndarray", int]:
    height, width = grid_array.shape
    labels = np.zeros((height, width), dtype=np.int32)
    parent: List[int] = [0]
    next_label = 1

    if connectivity == 4:
        neighbor_offsets = [(-1, 0), (0, -1)]
    else:
        neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1)]

    for y in range(height):
        for x in range(width):
            if not allowed[y, x]:
                continue
            neighbors: List[int] = []
            for dy, dx in neighbor_offsets:
                ny, nx = y + dy, x + dx
                if ny < 0 or nx < 0 or ny >= height or nx >= width:
                    continue
                if not allowed[ny, nx]:
                    continue
                if respect_colors and grid_array[ny, nx] != grid_array[y, x]:
                    continue
                neighbor_label = labels[ny, nx]
                if neighbor_label > 0:
                    neighbors.append(neighbor_label)

            if not neighbors:
                label = next_label
                labels[y, x] = label
                parent.append(label)
                next_label += 1
            else:
                label = min(neighbors)
                labels[y, x] = label
                for other in neighbors:
                    if other != label:
                        _union(parent, label, other)

    label_map: dict[int, int] = {}
    next_comp = 1
    for y in range(height):
        for x in range(width):
            label = labels[y, x]
            if label == 0:
                continue
            root = _find(parent, label)
            mapped = label_map.setdefault(root, next_comp)
            if mapped == next_comp:
                next_comp += 1
            labels[y, x] = mapped

    return labels, next_comp - 1


def _find(parent: List[int], label: int) -> int:
    while parent[label] != label:
        parent[label] = parent[parent[label]]
        label = parent[label]
    return label


def _union(parent: List[int], a: int, b: int) -> None:
    root_a = _find(parent, a)
    root_b = _find(parent, b)
    if root_a == root_b:
        return
    if root_a < root_b:
        parent[root_b] = root_a
    else:
        parent[root_a] = root_b


def _records_from_labels(
    grid_array: "np.ndarray",
    labels: "np.ndarray",
    num_components: int,
) -> List[_ComponentRecord]:
    if num_components == 0:
        return []

    height, width = grid_array.shape
    flat_labels = labels.reshape(-1)
    mask = flat_labels > 0
    if not mask.any():
        return []

    ys = np.repeat(np.arange(height), width)
    xs = np.tile(np.arange(width), height)
    labels_nonzero = flat_labels[mask]
    ys_nonzero = ys[mask]
    xs_nonzero = xs[mask]
    colors = grid_array.reshape(-1)[mask]
    flat_indices = ys_nonzero * width + xs_nonzero

    areas = np.bincount(labels_nonzero, minlength=num_components + 1).astype(np.int64)
    sum_y = np.bincount(labels_nonzero, weights=ys_nonzero, minlength=num_components + 1)
    sum_x = np.bincount(labels_nonzero, weights=xs_nonzero, minlength=num_components + 1)

    min_y = np.full(num_components + 1, height, dtype=np.int32)
    min_x = np.full(num_components + 1, width, dtype=np.int32)
    max_y = np.full(num_components + 1, -1, dtype=np.int32)
    max_x = np.full(num_components + 1, -1, dtype=np.int32)

    np.minimum.at(min_y, labels_nonzero, ys_nonzero)
    np.minimum.at(min_x, labels_nonzero, xs_nonzero)
    np.maximum.at(max_y, labels_nonzero, ys_nonzero)
    np.maximum.at(max_x, labels_nonzero, xs_nonzero)

    packed = (labels_nonzero.astype(np.int64) << 32) | colors.astype(np.int64)
    unique_pairs, first_idx, counts = np.unique(packed, return_counts=True, return_index=True)
    pair_labels = (unique_pairs >> 32).astype(np.int32)
    pair_colors = (unique_pairs & ((1 << 32) - 1)).astype(np.int64)

    color_map: dict[int, Tuple["np.ndarray", "np.ndarray", "np.ndarray"]] = {}
    for label in range(1, num_components + 1):
        mask_label = pair_labels == label
        color_ids = pair_colors[mask_label]
        color_counts = counts[mask_label].astype(np.int64)
        first_positions = flat_indices[first_idx[mask_label]].astype(np.int64)
        color_map[label] = (color_ids, color_counts, first_positions)

    label_int = labels.astype(np.int32)
    neighbor_count = np.zeros_like(label_int, dtype=np.int32)
    vertical = (label_int[:-1, :] == label_int[1:, :]) & (label_int[:-1, :] > 0)
    neighbor_count[:-1, :] += vertical.astype(np.int32)
    neighbor_count[1:, :] += vertical.astype(np.int32)
    horizontal = (label_int[:, :-1] == label_int[:, 1:]) & (label_int[:, :-1] > 0)
    neighbor_count[:, :-1] += horizontal.astype(np.int32)
    neighbor_count[:, 1:] += horizontal.astype(np.int32)
    mask_nonzero = label_int > 0
    perimeter_counts = np.zeros(num_components + 1, dtype=np.int32)
    np.add.at(perimeter_counts, label_int[mask_nonzero], (4 - neighbor_count[mask_nonzero]))

    records: List[_ComponentRecord] = []
    for label in range(1, num_components + 1):
        area = int(areas[label])
        if area == 0:
            continue
        centroid_y = float(sum_y[label] / area)
        centroid_x = float(sum_x[label] / area)
        color_ids, color_counts, first_positions = color_map.get(
            label,
            (
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
            ),
        )
        records.append(
            _ComponentRecord(
                order=label - 1,
                area=area,
                perimeter=int(perimeter_counts[label]),
                min_y=int(min_y[label]),
                min_x=int(min_x[label]),
                max_y=int(max_y[label]),
                max_x=int(max_x[label]),
                centroid_y=centroid_y,
                centroid_x=centroid_x,
                color_ids=color_ids,
                color_counts=color_counts,
                color_first_indices=first_positions,
            )
        )

    return records


def _build_feature_matrix(
    objects: Sequence[_ComponentRecord],
    *,
    grid_shape: Tuple[int, int],
    normalize: bool,
    max_color_features: int,
    max_objects: int,
) -> Tuple[List[List[float]], List[int]]:
    feature_vectors: List[List[float]] = []
    mask: List[int] = []
    feature_length = len(BASE_FEATURE_KEYS) + max_color_features

    for obj in objects:
        vector = _build_feature_vector(
            obj,
            grid_shape=grid_shape,
            normalize=normalize,
            max_color_features=max_color_features,
        )
        feature_vectors.append(vector)
        mask.append(1)

    while len(feature_vectors) < max_objects:
        feature_vectors.append([0.0 for _ in range(feature_length)])
        mask.append(0)

    return feature_vectors, mask


def _build_feature_vector(
    obj: _ComponentRecord,
    *,
    grid_shape: Tuple[int, int],
    normalize: bool,
    max_color_features: int,
) -> List[float]:
    grid_height, grid_width = grid_shape
    norm = float(grid_height * grid_width) if normalize else 1.0
    height_scale = float(grid_height) if normalize else 1.0
    width_scale = float(grid_width) if normalize else 1.0

    area = obj.area / norm
    perimeter = obj.perimeter / norm
    height_value = obj.height / height_scale if normalize else float(obj.height)
    width_value = obj.width / width_scale if normalize else float(obj.width)
    centroid_y = obj.centroid_y / height_scale if normalize else obj.centroid_y
    centroid_x = obj.centroid_x / width_scale if normalize else obj.centroid_x
    bbox_min_y = obj.min_y / height_scale if normalize else float(obj.min_y)
    bbox_max_y = obj.max_y / height_scale if normalize else float(obj.max_y)
    bbox_min_x = obj.min_x / width_scale if normalize else float(obj.min_x)
    bbox_max_x = obj.max_x / width_scale if normalize else float(obj.max_x)

    vector: List[float] = [
        float(area),
        float(perimeter),
        float(height_value),
        float(width_value),
        float(obj.aspect_ratio),
        float(obj.color_count),
        float(obj.dominant_color),
        float(centroid_y),
        float(centroid_x),
        float(bbox_min_y),
        float(bbox_min_x),
        float(bbox_max_y),
        float(bbox_max_x),
    ]

    color_features = _build_color_features(obj, norm, max_color_features, normalize=normalize)
    vector.extend(color_features)
    return vector


def _build_color_features(
    obj: _ComponentRecord,
    norm: float,
    max_color_features: int,
    *,
    normalize: bool,
) -> List[float]:
    if max_color_features == 0:
        return []

    color_features = [0.0 for _ in range(max_color_features)]
    if obj.color_count == 0:
        return color_features

    sorted_idx = np.argsort(obj.color_ids)
    for slot, idx in enumerate(sorted_idx[:max_color_features]):
        value = obj.color_counts[idx]
        color_features[slot] = float(value / norm if normalize else value)
    return color_features


def _build_adjacency_matrix(
    objects: Sequence[_ComponentRecord],
    *,
    max_objects: int,
) -> List[List[int]]:
    if not objects:
        return [[0 for _ in range(max_objects)] for _ in range(max_objects)]

    count = min(len(objects), max_objects)
    boxes = np.array(
        [(obj.min_y, obj.min_x, obj.max_y, obj.max_x) for obj in objects[:count]],
        dtype=np.int32,
    )
    min_y = boxes[:, 0]
    min_x = boxes[:, 1]
    max_y = boxes[:, 2]
    max_x = boxes[:, 3]

    cond_y1 = (max_y[:, None] + 1) < min_y[None, :]
    cond_y2 = (max_y[None, :] + 1) < min_y[:, None]
    cond_x1 = (max_x[:, None] + 1) < min_x[None, :]
    cond_x2 = (max_x[None, :] + 1) < min_x[:, None]
    touches = ~(cond_y1 | cond_y2 | cond_x1 | cond_x2)
    np.fill_diagonal(touches, False)

    adjacency = np.zeros((max_objects, max_objects), dtype=np.int32)
    adjacency[:count, :count] = touches.astype(np.int32)
    return adjacency.tolist()
