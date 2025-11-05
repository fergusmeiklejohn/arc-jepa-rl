"""Reusable evaluation cache for DSL program execution."""

from __future__ import annotations

from typing import Dict, Tuple

from arcgen import Grid
from training.dsl.enumerator import Program

EVALUATION_ERROR = object()


class EvaluationCache:
    """Memoises program executions on specific grids."""

    ERROR = EVALUATION_ERROR

    def __init__(self) -> None:
        self._store: Dict[Tuple, object] = {}

    @staticmethod
    def _key(program: Program, grid: Grid) -> Tuple:
        return (program.root.signature, grid.cells)

    def get(self, program: Program, grid: Grid) -> object | None:
        return self._store.get(self._key(program, grid))

    def store_success(self, program: Program, grid: Grid, result: Grid) -> None:
        self._store[self._key(program, grid)] = result

    def store_failure(self, program: Program, grid: Grid) -> None:
        self._store[self._key(program, grid)] = EVALUATION_ERROR


__all__ = ["EvaluationCache", "EVALUATION_ERROR"]
