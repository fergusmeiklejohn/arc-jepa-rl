"""Heuristics to mine option sequences and prepare them for DSL promotion."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from arcgen import Grid
from envs import Option


InvariantFn = Callable[[Grid, Grid], bool]


def _default_invariants() -> Tuple[InvariantFn, ...]:
    return (
        lambda before, after: before.shape == after.shape,
    )


def _apply_sequence(sequence: Sequence[Option], grid: Grid) -> Grid:
    """Apply a sequence of options to a grid, returning the final state."""
    state = grid
    for option in sequence:
        state = option.apply(state)
    return state


def _slugify(value: str) -> str:
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", value)
    slug = slug.strip("_")
    return slug or "option"


@dataclass(frozen=True)
class OptionApplication:
    """Single option application captured from rollouts."""

    option: Option
    before: Grid
    after: Grid
    reward: float = 0.0
    success: Optional[bool] = None


@dataclass(frozen=True)
class OptionEpisode:
    """Episode trace consisting of ordered option applications."""

    steps: Tuple[OptionApplication, ...]
    task_id: Optional[str] = None

    def __iter__(self) -> Iterator[OptionApplication]:  # pragma: no cover - convenience
        return iter(self.steps)


class _SequenceStats:
    """Mutable accumulator for discovered option statistics."""

    def __init__(self, sequence: Sequence[Option]) -> None:
        self.sequence: Tuple[Option, ...] = tuple(sequence)
        self.occurrences: List[Tuple[Grid, Grid]] = []
        self.successes = 0
        self.success_observations = 0
        self.total = 0
        self.changed = False
        self.rewards: List[float] = []

    def record(self, before: Grid, after: Grid, reward: float, success: Optional[bool]) -> None:
        self.occurrences.append((before, after))
        self.total += 1
        if before.cells != after.cells:
            self.changed = True
        if success is not None:
            self.success_observations += 1
            if success:
                self.successes += 1
        self.rewards.append(reward)

    def support(self) -> int:
        return len(self.occurrences)

    def success_rate(self) -> float:
        if self.success_observations == 0:
            return 1.0
        return self.successes / max(1, self.success_observations)

    def avg_reward(self) -> float:
        if not self.rewards:
            return 0.0
        return sum(self.rewards) / len(self.rewards)


@dataclass(frozen=True)
class DiscoveredOption:
    """Candidate composite option ready for DSL promotion."""

    name: str
    sequence: Tuple[Option, ...]
    sequence_names: Tuple[str, ...]
    support: int
    success_rate: float
    avg_reward: float
    occurrences: Tuple[Tuple[Grid, Grid], ...]

    def apply(self, grid: Grid) -> Grid:
        return _apply_sequence(self.sequence, grid)


def discover_option_sequences(
    episodes: Iterable[OptionEpisode],
    *,
    min_support: int = 2,
    max_sequence_length: int = 3,
    min_success_rate: float = 0.6,
    invariants: Optional[Sequence[InvariantFn]] = None,
    allow_singleton: bool = False,
    name_prefix: str = "auto",
) -> List[DiscoveredOption]:
    """Discover frequently successful option sequences that behave like new primitives.

    The miner walks each episode, checking every contiguous subsequence up to
    ``max_sequence_length``. A sequence is kept when:
      - It meets ``min_support`` (occurs across episodes at least this many times).
      - It changes the grid (filters out pure no-ops).
      - Its observed success rate >= ``min_success_rate`` (defaults to 0.6).
      - All provided invariants hold (defaults to shape preservation).

    Examples:
        >>> from envs import make_primitive_option
        >>> opt_a = make_primitive_option("mirror_x")
        >>> opt_b = make_primitive_option("translate", dx=1, dy=0, fill=0)
        >>> # Build OptionEpisode steps elsewhere...
        >>> discovered = discover_option_sequences([episode], min_support=1, allow_singleton=False)
        >>> discovered[0].name.startswith("auto_")
        True

    Edge cases:
      - ``allow_singleton=False`` prunes single-option sequences so we only promote
        genuine composites.
      - Setting ``invariants=()`` disables shape checks (e.g., for cropping options).
      - ``max_sequence_length=1`` + ``allow_singleton=True`` can be used to mine
        frequently successful primitive options without recomposition.
    """

    if min_support <= 0:
        raise ValueError("min_support must be positive")
    if max_sequence_length <= 0:
        raise ValueError("max_sequence_length must be positive")
    if not 0.0 <= min_success_rate <= 1.0:
        raise ValueError("min_success_rate must lie in [0, 1]")

    invariant_fns = tuple(invariants or _default_invariants())

    stats: Dict[Tuple[str, ...], _SequenceStats] = {}

    for episode in episodes:
        steps = list(episode.steps)
        for start in range(len(steps)):
            base_grid = steps[start].before
            sequence: List[Option] = []
            names: List[str] = []

            cumulative_reward = 0.0
            for local_idx in range(start, min(len(steps), start + max_sequence_length)):
                step = steps[local_idx]
                sequence.append(step.option)
                names.append(step.option.name)
                cumulative_reward += step.reward

                try:
                    predicted = _apply_sequence(sequence, base_grid)
                except Exception:
                    break

                if predicted.cells != step.after.cells:
                    break

                if any(not invariant(base_grid, step.after) for invariant in invariant_fns):
                    break

                key = tuple(names)
                if len(key) == 1 and not allow_singleton:
                    continue

                seq_stats = stats.get(key)
                if seq_stats is None:
                    seq_stats = _SequenceStats(sequence)
                    stats[key] = seq_stats

                # Verify the canonical sequence reproduces the transformation.
                try:
                    canonical_after = _apply_sequence(seq_stats.sequence, base_grid)
                except Exception:
                    break
                if canonical_after.cells != step.after.cells:
                    break

                seq_stats.record(base_grid, step.after, cumulative_reward, step.success)

    discovered: List[DiscoveredOption] = []
    for names, seq_stats in stats.items():
        if seq_stats.support() < min_support:
            continue
        if not seq_stats.changed:
            continue
        success = seq_stats.success_rate()
        if success < min_success_rate:
            continue

        slug = "__".join(_slugify(name) for name in names)
        digest = hashlib.sha1("::".join(names).encode("utf-8")).hexdigest()[:8]
        primitive_name = f"{name_prefix}_{slug}_{digest}"

        discovered.append(
            DiscoveredOption(
                name=primitive_name,
                sequence=seq_stats.sequence,
                sequence_names=names,
                support=seq_stats.support(),
                success_rate=success,
                avg_reward=seq_stats.avg_reward(),
                occurrences=tuple(seq_stats.occurrences),
            )
        )

    discovered.sort(key=lambda item: (item.support, item.success_rate, item.avg_reward), reverse=True)
    return discovered
