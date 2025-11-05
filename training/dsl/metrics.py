"""Lightweight metrics for reasoning about DSL programs."""

from __future__ import annotations

from training.dsl.enumerator import Program


def description_length(program: Program) -> float:
    """Approximate MDL cost using primitive complexities."""

    cost = 0.0
    for expr in program.traverse():
        primitive = expr.primitive
        if primitive is None:
            continue
        cost += getattr(primitive, "complexity", 1.0)
    return cost


__all__ = ["description_length"]
