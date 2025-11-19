"""Helpers for tracking JEPA latent distance-to-goal metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from arcgen import Grid


@dataclass(frozen=True)
class LatentDistanceRecord:
    """Single measurement of latent distance progress for an example."""

    example_kind: str
    index: int
    start_distance: float
    final_distance: float | None


class LatentDistanceTracker:
    """Compute latent distance-to-goal stats from arbitrary embedders."""

    def __init__(
        self,
        embed_fn: Callable[[Grid], object],
        distance_fn: Callable[[object, object], float],
    ) -> None:
        if not callable(embed_fn):
            raise TypeError("embed_fn must be callable")
        if not callable(distance_fn):
            raise TypeError("distance_fn must be callable")
        self._embed = embed_fn
        self._distance = distance_fn

    def record(
        self,
        example_kind: str,
        index: int,
        *,
        start: Grid,
        target: Grid | None,
        prediction: Grid | None,
    ) -> LatentDistanceRecord | None:
        """Return latent distance measurements for a single inference step."""

        if target is None:
            return None

        target_embedding = self._embed(target)
        start_embedding = self._embed(start)
        start_distance = float(self._distance(start_embedding, target_embedding))
        final_distance: Optional[float] = None
        if prediction is not None:
            prediction_embedding = self._embed(prediction)
            final_distance = float(self._distance(prediction_embedding, target_embedding))

        return LatentDistanceRecord(
            example_kind=str(example_kind),
            index=int(index),
            start_distance=start_distance,
            final_distance=final_distance,
        )


__all__ = ["LatentDistanceRecord", "LatentDistanceTracker"]
