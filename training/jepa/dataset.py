"""Dataset scaffolding for object-centric JEPA training."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Mapping, MutableMapping, Sequence, Tuple

from arcgen import Grid


@dataclass(frozen=True)
class GridPairBatch:
    context: Sequence[Grid]
    target: Sequence[Grid]

    def __post_init__(self) -> None:
        if len(self.context) != len(self.target):
            raise ValueError("context and target sequences must have the same length")


@dataclass(frozen=True)
class AugmentationConfig:
    """Configuration for grid-level data augmentations."""

    mask_ratio: float = 0.0
    random_crop_radius: int = 0
    palette_permutation: bool = False
    gaussian_noise_std: float = 0.0
    min_grid_size_for_crop: int = 0
    background_color: int = 0

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None) -> "AugmentationConfig":
        if data is None:
            return cls()

        mask_ratio = float(data.get("mask_ratio", cls.mask_ratio))
        if not 0.0 <= mask_ratio <= 1.0:
            raise ValueError("mask_ratio must be in [0, 1]")

        random_crop_radius = int(data.get("random_crop_radius", cls.random_crop_radius))
        if random_crop_radius < 0:
            raise ValueError("random_crop_radius must be non-negative")

        gaussian_noise_std = float(data.get("gaussian_noise_std", cls.gaussian_noise_std))
        if gaussian_noise_std < 0:
            raise ValueError("gaussian_noise_std must be non-negative")

        min_grid_size_for_crop = int(data.get("min_grid_size_for_crop", cls.min_grid_size_for_crop))
        if min_grid_size_for_crop < 0:
            raise ValueError("min_grid_size_for_crop must be non-negative")

        background_color = int(data.get("background_color", cls.background_color))
        if background_color < 0:
            raise ValueError("background_color must be non-negative")

        return cls(
            mask_ratio=mask_ratio,
            random_crop_radius=random_crop_radius,
            palette_permutation=bool(data.get("palette_permutation", cls.palette_permutation)),
            gaussian_noise_std=gaussian_noise_std,
            min_grid_size_for_crop=min_grid_size_for_crop,
            background_color=background_color,
        )

    def is_identity(self) -> bool:
        return (
            self.mask_ratio == 0.0
            and self.random_crop_radius == 0
            and not self.palette_permutation
            and self.gaussian_noise_std == 0.0
        )


@dataclass(frozen=True)
class ManifestExample:
    """Single context/target pair extracted from a manifest record."""

    context: Grid
    target: Grid
    metadata: Mapping[str, object] | None = None
    context_sequence: Sequence[Grid] | None = None


def _is_grid_like(value: object) -> bool:
    if isinstance(value, Grid):
        return True
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return False
    if len(value) == 0:
        return False
    first_row = value[0]
    if not isinstance(first_row, Sequence) or isinstance(first_row, (str, bytes)):
        return False
    try:
        for row in value:
            if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
                return False
            for cell in row:
                if not isinstance(cell, int):
                    return False
    except TypeError:
        return False
    return True


def _grid_from_data(value: object) -> Grid:
    if isinstance(value, Grid):
        return value
    if _is_grid_like(value):
        return Grid(value)  # type: ignore[arg-type]
    raise ValueError("value cannot be interpreted as an ARC grid")


def _normalize_context(value: object) -> Tuple[Grid, ...]:
    if _is_grid_like(value):
        return (_grid_from_data(value),)
    if isinstance(value, Sequence) and value and not isinstance(value, (str, bytes)):
        grids = tuple(_grid_from_data(item) for item in value)
        if not grids:
            raise ValueError("context sequence must contain at least one grid")
        return grids
    raise ValueError("context must be a grid or sequence of grids")


def _normalize_target(value: object) -> Grid:
    if _is_grid_like(value):
        return _grid_from_data(value)
    if isinstance(value, Sequence) and value and not isinstance(value, (str, bytes)):
        first = value[0]
        if _is_grid_like(first):
            return _grid_from_data(first)
    raise ValueError("target must be a grid or sequence containing a grid")


def _merge_metadata(base: Mapping[str, object] | None, extra: Mapping[str, object]) -> Mapping[str, object]:
    if base is None:
        return dict(extra)
    merged: MutableMapping[str, object] = dict(base)
    merged.update(extra)
    return merged


def _examples_from_record(
    data: Mapping[str, object],
    *,
    context_window: int,
    target_offset: int,
) -> Iterator[ManifestExample]:
    metadata = data.get("metadata")
    base_meta = metadata if isinstance(metadata, Mapping) else None

    if "context" in data and "target" in data:
        context_sequence = _normalize_context(data["context"])
        target_grid = _normalize_target(data["target"])
        yield ManifestExample(
            context=context_sequence[-1],
            target=target_grid,
            metadata=base_meta,
            context_sequence=context_sequence,
        )
        return

    if "input" in data and "output" in data:
        context_grid = _grid_from_data(data["input"])
        target_grid = _grid_from_data(data["output"])
        yield ManifestExample(
            context=context_grid,
            target=target_grid,
            metadata=base_meta,
            context_sequence=(context_grid,),
        )
        return

    if "frames" in data:
        frames_data = data["frames"]
        if not isinstance(frames_data, Sequence):
            raise ValueError("frames must be a sequence")
        frames = tuple(_grid_from_data(frame) for frame in frames_data)
        window = context_window + target_offset
        if window <= 0:
            raise ValueError("context_window + target_offset must be positive")
        if len(frames) < window:
            return

        for start in range(len(frames) - window + 1):
            context_slice = frames[start : start + context_window]
            target_index = start + context_window + target_offset - 1
            target_grid = frames[target_index]
            extra_meta: Mapping[str, object] = {
                "record_id": data.get("id"),
                "context_indices": list(range(start, start + context_window)),
                "target_index": target_index,
            }
            yield ManifestExample(
                context=context_slice[-1],
                target=target_grid,
                metadata=_merge_metadata(base_meta, extra_meta),
                context_sequence=context_slice,
            )
        return

    raise ValueError("manifest record must contain either context/target or frames")


def _random_translate(
    cells: List[List[int]],
    *,
    radius: int,
    rng: random.Random,
    background: int,
) -> List[List[int]]:
    if radius <= 0:
        return cells
    height = len(cells)
    width = len(cells[0]) if height > 0 else 0
    if height == 0 or width == 0:
        return cells

    dy = rng.randint(-radius, radius)
    dx = rng.randint(-radius, radius)

    translated = [[background for _ in range(width)] for _ in range(height)]
    for y in range(height):
        ny = y + dy
        if 0 <= ny < height:
            for x in range(width):
                nx = x + dx
                if 0 <= nx < width:
                    translated[ny][nx] = cells[y][x]
    return translated


def _augment_grid(grid: Grid, config: AugmentationConfig, rng: random.Random) -> Grid:
    if config.is_identity():
        return grid

    cells = [list(row) for row in grid.cells]
    background = config.background_color

    if config.palette_permutation:
        palette = sorted({value for row in cells for value in row if value != background})
        if len(palette) > 1:
            permuted = palette[:]
            rng.shuffle(permuted)
            mapping = {src: dst for src, dst in zip(palette, permuted)}
            cells = [
                [mapping.get(value, value) for value in row]
                for row in cells
            ]

    if config.mask_ratio > 0.0:
        for y, row in enumerate(cells):
            for x, value in enumerate(row):
                if value == background:
                    continue
                if rng.random() < config.mask_ratio:
                    cells[y][x] = background

    height = len(cells)
    width = len(cells[0]) if height > 0 else 0
    min_dim = min(height, width) if height and width else 0
    if (
        config.random_crop_radius > 0
        and min_dim >= max(config.min_grid_size_for_crop, 1)
    ):
        cells = _random_translate(
            cells,
            radius=config.random_crop_radius,
            rng=rng,
            background=background,
        )

    if config.gaussian_noise_std > 0.0:
        max_value = max((value for row in grid.cells for value in row), default=background)
        for y, row in enumerate(cells):
            for x, value in enumerate(row):
                if value == background:
                    continue
                noisy = value + rng.gauss(0.0, config.gaussian_noise_std)
                clamped = int(round(noisy))
                if clamped < background:
                    clamped = background
                if clamped > max_value:
                    clamped = max_value
                cells[y][x] = clamped

    return Grid(cells)


class InMemoryGridPairDataset:
    """Minimal iterable dataset backed by in-memory grid pairs."""

    def __init__(self, pairs: Iterable[Tuple[Sequence[Grid], Sequence[Grid]]]) -> None:
        self._pairs: List[Tuple[Sequence[Grid], Sequence[Grid]]] = [(
            tuple(context),
            tuple(target),
        ) for context, target in pairs]

    def __len__(self) -> int:
        return len(self._pairs)

    def __iter__(self) -> Iterator[GridPairBatch]:
        for context, target in self._pairs:
            yield GridPairBatch(context=context, target=target)


class ManifestGridPairDataset:
    """Dataset that materialises batches from a JSONL manifest."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        batch_size: int,
        context_window: int = 1,
        target_offset: int = 1,
        shuffle: bool = True,
        drop_last: bool = False,
        augmentations: AugmentationConfig | Mapping[str, object] | None = None,
        seed: int | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if context_window <= 0:
            raise ValueError("context_window must be positive")
        if target_offset <= 0:
            raise ValueError("target_offset must be positive")

        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"manifest path does not exist: {self.manifest_path}")

        self.batch_size = batch_size
        self.context_window = context_window
        self.target_offset = target_offset
        self.shuffle = shuffle
        self.drop_last = drop_last
        if augmentations is None:
            self.augmentations = AugmentationConfig()
        elif isinstance(augmentations, AugmentationConfig):
            self.augmentations = augmentations
        else:
            self.augmentations = AugmentationConfig.from_mapping(augmentations)

        self._seed = seed
        self._epoch = 0
        self._examples = self._load_examples()
        if not self._examples:
            raise ValueError(f"manifest {self.manifest_path} produced no samples")

    def _load_examples(self) -> List[ManifestExample]:
        examples: List[ManifestExample] = []
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            for line_no, raw_line in enumerate(handle, 1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"failed to parse JSON on line {line_no} of {self.manifest_path}"
                    ) from exc
                if not isinstance(parsed, Mapping):
                    raise ValueError(
                        f"manifest line {line_no} in {self.manifest_path} must decode to a mapping",
                    )
                try:
                    for example in _examples_from_record(
                        parsed,
                        context_window=self.context_window,
                        target_offset=self.target_offset,
                    ):
                        examples.append(example)
                except Exception as exc:
                    raise ValueError(
                        f"invalid manifest entry on line {line_no} of {self.manifest_path}: {exc}"
                    ) from exc
        return examples

    def __len__(self) -> int:
        if self.drop_last:
            return len(self._examples) // self.batch_size
        return (len(self._examples) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[GridPairBatch]:
        if self._seed is None:
            rng = random.Random()
        else:
            rng = random.Random(self._seed + self._epoch)
        self._epoch += 1

        indices = list(range(len(self._examples)))
        if self.shuffle:
            rng.shuffle(indices)

        batch_context: List[Grid] = []
        batch_target: List[Grid] = []

        for idx in indices:
            example = self._examples[idx]
            batch_context.append(self._apply_augmentations(example.context, rng))
            batch_target.append(self._apply_augmentations(example.target, rng))

            if len(batch_context) == self.batch_size:
                yield GridPairBatch(context=list(batch_context), target=list(batch_target))
                batch_context.clear()
                batch_target.clear()

        if batch_context and not self.drop_last:
            yield GridPairBatch(context=list(batch_context), target=list(batch_target))

    def _apply_augmentations(self, grid: Grid, rng: random.Random) -> Grid:
        return _augment_grid(grid, self.augmentations, rng)


def build_dummy_dataset(num_batches: int = 4) -> InMemoryGridPairDataset:
    grid = Grid([[0, 1], [0, 1]])
    pairs = [([grid, grid], [grid, grid]) for _ in range(num_batches)]
    return InMemoryGridPairDataset(pairs)
