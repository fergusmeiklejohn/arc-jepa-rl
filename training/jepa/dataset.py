"""Dataset scaffolding for object-centric JEPA training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, List, Sequence, Tuple

from arcgen import Grid


@dataclass(frozen=True)
class GridPairBatch:
    context: Sequence[Grid]
    target: Sequence[Grid]

    def __post_init__(self) -> None:
        if len(self.context) != len(self.target):
            raise ValueError("context and target sequences must have the same length")


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


def build_dummy_dataset(num_batches: int = 4) -> InMemoryGridPairDataset:
    grid = Grid([[0, 1], [0, 1]])
    pairs = [([grid, grid], [grid, grid]) for _ in range(num_batches)]
    return InMemoryGridPairDataset(pairs)
