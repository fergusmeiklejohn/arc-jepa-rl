"""Curriculum progression logic for multi-phase training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping

from .metrics import PhaseMetrics


@dataclass(frozen=True)
class PhaseThresholds:
    solve_rate: float = 0.6
    codebook_usage: float = 0.1
    option_diversity: float = 0.1

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None) -> "PhaseThresholds":
        if data is None:
            return cls()
        return cls(
            solve_rate=float(data.get("solve_rate", cls.solve_rate)),
            codebook_usage=float(data.get("codebook_usage", cls.codebook_usage)),
            option_diversity=float(data.get("option_diversity", cls.option_diversity)),
        )


@dataclass(frozen=True)
class PhaseConfig:
    name: str
    task_schedule: Mapping[str, int]
    thresholds: PhaseThresholds = PhaseThresholds()
    min_episodes: int = 10

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "PhaseConfig":
        name = str(data.get("name", ""))
        if not name:
            raise ValueError("phase config requires a name")
        schedule = data.get("task_schedule") or {}
        if not isinstance(schedule, Mapping) or not schedule:
            raise ValueError("phase task_schedule must be a non-empty mapping")
        min_episodes = int(data.get("min_episodes", cls.min_episodes))
        if min_episodes <= 0:
            raise ValueError("min_episodes must be positive")
        thresholds = PhaseThresholds.from_mapping(data.get("thresholds"))
        return cls(
            name=name,
            task_schedule=schedule,
            thresholds=thresholds,
            min_episodes=min_episodes,
        )


@dataclass(frozen=True)
class CurriculumConfig:
    phases: List[PhaseConfig]

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "CurriculumConfig":
        phases_raw = data.get("phases")
        if not isinstance(phases_raw, Iterable):
            raise ValueError("curriculum config must include a 'phases' sequence")
        phases = [PhaseConfig.from_mapping(entry) for entry in phases_raw]
        if not phases:
            raise ValueError("curriculum must include at least one phase")
        return cls(phases=phases)


class CurriculumManager:
    """Track curriculum phases and decide when to advance."""

    def __init__(self, config: CurriculumConfig):
        self.config = config
        self._phase_index = 0
        self.metrics_history: list[tuple[str, PhaseMetrics]] = []

    @property
    def current_phase(self) -> PhaseConfig:
        return self.config.phases[self._phase_index]

    @property
    def is_last_phase(self) -> bool:
        return self._phase_index >= len(self.config.phases) - 1

    def record_metrics(self, metrics: PhaseMetrics) -> None:
        self.metrics_history.append((self.current_phase.name, metrics))

    def should_advance(self, metrics: PhaseMetrics) -> bool:
        cfg = self.current_phase
        if metrics.episodes < cfg.min_episodes:
            return False
        thresholds = cfg.thresholds
        if metrics.solve_rate < thresholds.solve_rate:
            return False
        if thresholds.codebook_usage > 0 and metrics.codebook_usage is not None:
            if metrics.codebook_usage < thresholds.codebook_usage:
                return False
        if thresholds.option_diversity > 0 and metrics.option_diversity is not None:
            if metrics.option_diversity < thresholds.option_diversity:
                return False
        return True

    def advance_phase(self) -> bool:
        if self.is_last_phase:
            return False
        self._phase_index += 1
        return True

    def update(self, metrics: PhaseMetrics) -> bool:
        """Record metrics and advance when thresholds are satisfied.

        Returns True when the phase advanced.
        """

        self.record_metrics(metrics)
        if self.should_advance(metrics):
            return self.advance_phase()
        return False
