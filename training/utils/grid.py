"""Grid utility helpers shared across training modules."""

from __future__ import annotations

from arcgen import Grid


def count_changed_cells(before: Grid, after: Grid, background: int = 0) -> int:
    """Return the number of cells that differ between two grids.

    Grids may differ in shape; missing cells are treated as the provided
    ``background`` value so rectangular comparisons remain well-defined.
    """

    before_rows = before.to_lists()
    after_rows = after.to_lists()
    height = max(len(before_rows), len(after_rows))

    width_before = len(before_rows[0]) if before_rows else 0
    width_after = len(after_rows[0]) if after_rows else 0
    width = max(width_before, width_after)

    diffs = 0
    for y in range(height):
        for x in range(width):
            if y < len(before_rows) and x < len(before_rows[y]):
                prev = before_rows[y][x]
            else:
                prev = background

            if y < len(after_rows) and x < len(after_rows[y]):
                curr = after_rows[y][x]
            else:
                curr = background

            if prev != curr:
                diffs += 1
    return diffs


__all__ = ["count_changed_cells"]
