"""Type system definitions for the ARC DSL."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DSLType:
    name: str

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.name


Color = DSLType("Color")
Position = DSLType("Position")
Shape = DSLType("Shape")
Grid = DSLType("Grid")
GridValue = DSLType("GridValue")
