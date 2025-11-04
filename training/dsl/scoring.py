"""Scoring hooks connecting DSL programs to JEPA embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .enumerator import Program


@dataclass(frozen=True)
class ProgramScore:
    program: Program
    score: float


class NeuralGuidedScorer:
    def __init__(self, scoring_fn: Callable[[Program], float]) -> None:
        self.scoring_fn = scoring_fn

    def score(self, program: Program) -> ProgramScore:
        return ProgramScore(program=program, score=float(self.scoring_fn(program)))
