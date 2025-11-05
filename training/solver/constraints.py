"""Heuristic constraints to prune infeasible DSL programs early."""

from __future__ import annotations

from typing import Iterable, Set

from arcgen import Grid

from training.dsl.enumerator import Program
from training.utils import count_changed_cells


class ConstraintChecker:
    """Applies lightweight heuristics before/after program execution."""

    COLOR_CHANGE_CUES: Set[str] = {
        "recolor",
        "flood_fill",
        "paint",
    }

    def __init__(self, *, allow_palette_mismatch: bool = False, change_slack: int = 4) -> None:
        self.allow_palette_mismatch = allow_palette_mismatch
        self.change_slack = change_slack

    def pre_check(self, program: Program, context: Grid, target: Grid) -> bool:
        """Return True if program can be pruned before execution."""

        if self.allow_palette_mismatch:
            return False

        context_palette = set(value for row in context.cells for value in row)
        target_palette = set(value for row in target.cells for value in row)
        introduces_new_colors = bool(target_palette - context_palette)

        if not introduces_new_colors:
            return False

        if self._uses_color_change(program):
            return False

        return True

    def post_check(self, program: Program, context: Grid, candidate: Grid, target: Grid) -> bool:
        """Return True if program should be discarded after execution."""

        candidate_palette = set(value for row in candidate.cells for value in row)
        target_palette = set(value for row in target.cells for value in row)

        if candidate_palette - target_palette:
            # Candidate introduces colors absent from target; reject unless program
            # explicitly manipulates colors.
            if not self._uses_color_change(program):
                return True

        target_change = count_changed_cells(context, target)
        candidate_change = count_changed_cells(context, candidate)
        if candidate_change > target_change + self.change_slack:
            return True

        return False

    def _uses_color_change(self, program: Program) -> bool:
        return any(
            expr.primitive is not None and expr.primitive.name in self.COLOR_CHANGE_CUES
            for expr in program.traverse()
        )


__all__ = ["ConstraintChecker"]
