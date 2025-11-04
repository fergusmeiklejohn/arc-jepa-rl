"""Grid utilities for ARC-style tasks.

The implementation sticks to pure Python/standard library data structures to
avoid platform-specific dependencies. NumPy interoperability is provided as an
optional convenience when the package is installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import random


class SeededRNG:
    """Wrapper around ``random.Random`` with a minimal convenience API."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def randint(self, a: int, b: int) -> int:
        return self._rng.randint(a, b)

    def choice(self, seq: Sequence[int]) -> int:
        return self._rng.choice(seq)

    def random(self) -> float:
        return self._rng.random()

    def shuffle(self, seq: List[int]) -> None:
        self._rng.shuffle(seq)

    def spawn(self, offset: int) -> "SeededRNG":
        """Derive a new RNG deterministically from this one."""

        return SeededRNG(self.randint(-(1 << 31), 1 << 31) + offset)


def _validate_rectangular(cells: Sequence[Sequence[int]]) -> Tuple[int, int]:
    if not cells:
        raise ValueError("grid must contain at least one row")

    width = len(cells[0])
    if width == 0:
        raise ValueError("grid must contain at least one column")

    for row in cells:
        if len(row) != width:
            raise ValueError("grid rows must all be the same length")
        for value in row:
            if not isinstance(value, int):
                raise TypeError("grid values must be integers")
            if value < 0:
                raise ValueError("grid values must be non-negative")

    return len(cells), width


@dataclass(frozen=True)
class Grid:
    """Immutable grid representation backed by nested Python lists."""

    cells: Tuple[Tuple[int, ...], ...]

    def __init__(self, cells: Sequence[Sequence[int]]) -> None:
        height, width = _validate_rectangular(cells)
        object.__setattr__(self, "cells", tuple(tuple(row) for row in cells))
        object.__setattr__(self, "_size", (height, width))

    # ------------------------------------------------------------------ basic
    @property
    def height(self) -> int:
        return self._size[0]

    @property
    def width(self) -> int:
        return self._size[1]

    @property
    def shape(self) -> Tuple[int, int]:
        return self._size

    def flatten(self) -> List[int]:
        return [value for row in self.cells for value in row]

    def palette(self) -> List[int]:
        return sorted(set(self.flatten()))

    def copy(self) -> "Grid":
        return Grid([list(row) for row in self.cells])

    def to_lists(self) -> List[List[int]]:
        return [list(row) for row in self.cells]

    def to_numpy(self):  # pragma: no cover - optional dependency
        """Return the grid as a NumPy array if NumPy is available."""

        try:
            import numpy as np
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("NumPy is required for to_numpy()") from exc

        return np.array(self.cells, dtype=np.int64)

    # ----------------------------------------------------------- constructions
    @classmethod
    def from_flat(cls, values: Sequence[int], width: int) -> "Grid":
        if width <= 0:
            raise ValueError("width must be positive")

        if len(values) % width != 0:
            raise ValueError("values length must be divisible by width")

        rows = [list(values[i : i + width]) for i in range(0, len(values), width)]
        return cls(rows)

    @classmethod
    def full(cls, height: int, width: int, value: int = 0) -> "Grid":
        if height <= 0 or width <= 0:
            raise ValueError("height and width must be positive")
        return cls([[value for _ in range(width)] for _ in range(height)])

    @classmethod
    def random(
        cls,
        height: int,
        width: int,
        palette: Iterable[int],
        *,
        fill_prob: float = 0.8,
        background: int = 0,
        rng: SeededRNG | None = None,
    ) -> "Grid":
        if height <= 0 or width <= 0:
            raise ValueError("height and width must be positive")
        if not 0 <= fill_prob <= 1:
            raise ValueError("fill_prob must be in [0, 1]")

        palette_list = list(palette)
        if background not in palette_list:
            palette_list.append(background)

        if not palette_list:
            raise ValueError("palette must contain at least one color")

        rng = rng or SeededRNG()

        cells: List[List[int]] = []
        for _ in range(height):
            row = []
            for _ in range(width):
                if rng.random() <= fill_prob:
                    row.append(rng.choice(palette_list))
                else:
                    row.append(background)
            cells.append(row)

        return cls(cells)

    # ------------------------------------------------------------ manipulation
    def replace(self, source: int, target: int) -> "Grid":
        return Grid([[target if value == source else value for value in row] for row in self.cells])

    def transpose(self) -> "Grid":
        return Grid(list(zip(*self.cells)))
