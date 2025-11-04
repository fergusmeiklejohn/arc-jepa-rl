"""Object-centric utilities for ARC-style grids.

The extraction logic intentionally avoids third-party dependencies so that it
can run inside lightweight tooling and unit tests. The resulting
``GridObject`` instances capture rich metadata that downstream components (for
example JEPA encoders, relational GNNs, or symbolic search) can consume.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

from .grid import Grid


Coords = Tuple[int, int]


def _normalize_background(background: int | Iterable[int]) -> Tuple[int, ...]:
    if isinstance(background, int):
        return (background,)
    values = tuple(int(value) for value in background)
    if not values:
        raise ValueError("background must contain at least one value")
    return values


def _iter_neighbors(y: int, x: int, height: int, width: int, connectivity: int) -> Iterator[Coords]:
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        directions.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])

    for dy, dx in directions:
        ny, nx = y + dy, x + dx
        if 0 <= ny < height and 0 <= nx < width:
            yield ny, nx


@dataclass(frozen=True)
class GridObject:
    """Connected component extracted from a :class:`~arcgen.grid.Grid`.

    Attributes include geometric and color statistics commonly required by ARC
    solvers. Instances are hashable, allowing use as dictionary keys.
    """

    id: int
    pixels: Tuple[Coords, ...]
    colors: Counter
    pixel_values: Tuple[Tuple[int, int, int], ...]
    bbox: Tuple[int, int, int, int]  # (min_y, min_x, max_y, max_x)
    area: int
    perimeter: int
    centroid: Tuple[float, float]
    grid_shape: Tuple[int, int]

    @property
    def min_y(self) -> int:
        return self.bbox[0]

    @property
    def min_x(self) -> int:
        return self.bbox[1]

    @property
    def max_y(self) -> int:
        return self.bbox[2]

    @property
    def max_x(self) -> int:
        return self.bbox[3]

    @property
    def height(self) -> int:
        return self.max_y - self.min_y + 1

    @property
    def width(self) -> int:
        return self.max_x - self.min_x + 1

    @property
    def dominant_color(self) -> int:
        return self.colors.most_common(1)[0][0]

    @property
    def unique_colors(self) -> Tuple[int, ...]:
        return tuple(sorted(self.colors.keys()))

    def to_subgrid(self, background: int = 0) -> Grid:
        """Return a :class:`Grid` cropped to this object's bounding box."""

        subgrid = [[background for _ in range(self.width)] for _ in range(self.height)]
        for y, x, value in self.pixel_values:
            subgrid[y - self.min_y][x - self.min_x] = value
        return Grid(subgrid)

    def as_feature_dict(self, normalize: bool = True) -> Dict[str, float]:
        """Return a feature dictionary suitable for model consumption."""

        height, width = self.grid_shape
        norm = float(height * width) if normalize else 1.0
        bbox_norm = float(height + width) if normalize else 1.0

        features: Dict[str, float] = {
            "area": self.area / norm,
            "perimeter": self.perimeter / norm,
            "height": self.height / height if normalize else float(self.height),
            "width": self.width / width if normalize else float(self.width),
            "aspect_ratio": self.width / self.height,
            "color_count": float(len(self.colors)),
            "dominant_color": float(self.dominant_color),
            "centroid_y": self.centroid[0] / height if normalize else self.centroid[0],
            "centroid_x": self.centroid[1] / width if normalize else self.centroid[1],
            "bbox_min_y": self.min_y / height if normalize else float(self.min_y),
            "bbox_min_x": self.min_x / width if normalize else float(self.min_x),
            "bbox_max_y": self.max_y / height if normalize else float(self.max_y),
            "bbox_max_x": self.max_x / width if normalize else float(self.max_x),
        }

        for color, count in self.colors.items():
            features[f"color_freq_{color}"] = count / norm

        return features

    def _color_at(self, coord: Coords) -> int:
        lookup = {(y, x): value for y, x, value in self.pixel_values}
        return lookup[coord]


def extract_objects(
    grid: Grid,
    *,
    background: int | Iterable[int] = 0,
    connectivity: int = 4,
    min_size: int = 1,
    respect_colors: bool = True,
) -> List[GridObject]:
    """Extract connected components from ``grid``.

    Returns objects sorted by discovery order. Components smaller than
    ``min_size`` are ignored. ``background`` accepts either a single value or
    an iterable of values to exclude. By default objects are defined as
    contiguous regions with the same color; set ``respect_colors=False`` to
    group all non-background colors into a single component when connected.
    """

    if min_size <= 0:
        raise ValueError("min_size must be positive")

    background_values = set(_normalize_background(background))
    height, width = grid.shape
    visited = [[False for _ in range(width)] for _ in range(height)]
    objects: List[GridObject] = []

    for y in range(height):
        for x in range(width):
            if visited[y][x]:
                continue
            color = grid.cells[y][x]
            if color in background_values:
                visited[y][x] = True
                continue

            seed_color = grid.cells[y][x]
            pixels: List[Coords] = []
            pixel_values: List[Tuple[int, int, int]] = []
            colors: Counter = Counter()
            queue: deque[Coords] = deque([(y, x)])

            while queue:
                cy, cx = queue.popleft()
                if visited[cy][cx]:
                    continue

                visited[cy][cx] = True
                cval = grid.cells[cy][cx]
                if cval in background_values:
                    continue

                pixels.append((cy, cx))
                pixel_values.append((cy, cx, cval))
                colors[cval] += 1

                for ny, nx in _iter_neighbors(cy, cx, height, width, connectivity):
                    if visited[ny][nx]:
                        continue
                    neighbor_value = grid.cells[ny][nx]
                    if neighbor_value in background_values:
                        continue
                    if respect_colors and neighbor_value != seed_color:
                        continue
                    queue.append((ny, nx))

            if len(pixels) < min_size:
                continue

            min_y = min(py for py, _ in pixels)
            max_y = max(py for py, _ in pixels)
            min_x = min(px for _, px in pixels)
            max_x = max(px for _, px in pixels)

            area = len(pixels)
            centroid_y = sum(py for py, _ in pixels) / area
            centroid_x = sum(px for _, px in pixels) / area

            pixel_set = set(pixels)
            perimeter = 0
            for py, px in pixels:
                for ny, nx in _iter_neighbors(py, px, height, width, 4):
                    if (ny, nx) not in pixel_set:
                        perimeter += 1
                # Account for borders outside grid bounds.
                if py == 0:
                    perimeter += 1
                if py == height - 1:
                    perimeter += 1
                if px == 0:
                    perimeter += 1
                if px == width - 1:
                    perimeter += 1

            obj = GridObject(
                id=len(objects),
                pixels=tuple(sorted(pixels)),
                colors=colors,
                pixel_values=tuple(sorted(pixel_values)),
                bbox=(min_y, min_x, max_y, max_x),
                area=area,
                perimeter=perimeter,
                centroid=(centroid_y, centroid_x),
                grid_shape=grid.shape,
            )
            objects.append(obj)

    return objects


def compute_adjacency(
    objects: Sequence[GridObject],
    *,
    mode: str = "touching",
) -> Dict[int, Tuple[int, ...]]:
    """Compute adjacency relationships between objects.

    ``mode="touching"`` marks objects as adjacent when their bounding boxes are
    at most one cell apart in either axis. ``mode="overlap"`` marks adjacency
    when bounding boxes intersect.
    """

    if mode not in {"touching", "overlap"}:
        raise ValueError("mode must be 'touching' or 'overlap'")

    adjacency: Dict[int, set[int]] = {obj.id: set() for obj in objects}

    for i, left in enumerate(objects):
        for right in objects[i + 1 :]:
            if mode == "overlap" and _bbox_overlap(left.bbox, right.bbox):
                adjacency[left.id].add(right.id)
                adjacency[right.id].add(left.id)
            elif mode == "touching" and _bbox_touching(left.bbox, right.bbox):
                adjacency[left.id].add(right.id)
                adjacency[right.id].add(left.id)

    return {key: tuple(sorted(value)) for key, value in adjacency.items()}


def _bbox_overlap(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _bbox_touching(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    return not (a[2] + 1 < b[0] or b[2] + 1 < a[0] or a[3] + 1 < b[1] or b[3] + 1 < a[1])
