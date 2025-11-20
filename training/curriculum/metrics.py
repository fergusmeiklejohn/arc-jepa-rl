"""Metric helpers for curriculum progression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class PhaseMetrics:
    """Aggregated metrics for a curriculum phase."""

    episodes: int
    successes: int
    solve_rate: float
    codebook_usage: float | None = None
    option_diversity: float | None = None

    @classmethod
    def from_counts(
        cls,
        *,
        successes: int,
        episodes: int,
        codebook_usage: float | None = None,
        active_codes: int | None = None,
        codebook_size: int | None = None,
        option_counts: Mapping[str, int] | None = None,
    ) -> "PhaseMetrics":
        if episodes <= 0:
            raise ValueError("episodes must be positive")
        if successes < 0:
            raise ValueError("successes must be non-negative")
        solve_rate = successes / float(episodes)
        usage = codebook_usage
        if usage is None and active_codes is not None and codebook_size:
            if codebook_size <= 0:
                raise ValueError("codebook_size must be positive when provided")
            usage = active_codes / float(codebook_size)

        diversity = None
        if option_counts is not None:
            distinct = sum(1 for _, count in option_counts.items() if count > 0)
            total = sum(option_counts.values())
            total = total if total > 0 else distinct
            if total > 0:
                diversity = distinct / float(total)

        return cls(
            episodes=episodes,
            successes=successes,
            solve_rate=solve_rate,
            codebook_usage=usage,
            option_diversity=diversity,
        )


def compute_phase_metrics(
    *,
    episodes: int,
    successes: int,
    codebook_usage: float | None = None,
    active_codes: int | None = None,
    codebook_size: int | None = None,
    option_counts: Mapping[str, int] | None = None,
) -> PhaseMetrics:
    """Convenience wrapper mirroring :meth:`PhaseMetrics.from_counts`."""

    return PhaseMetrics.from_counts(
        successes=successes,
        episodes=episodes,
        codebook_usage=codebook_usage,
        active_codes=active_codes,
        codebook_size=codebook_size,
        option_counts=option_counts,
    )
