"""Lightweight curriculum runner that delegates per-phase training."""

from __future__ import annotations

from typing import Callable, Iterable, List

from .manager import CurriculumManager, PhaseConfig
from .metrics import PhaseMetrics


def run_curriculum(
    manager: CurriculumManager,
    run_phase: Callable[[PhaseConfig], PhaseMetrics],
) -> List[PhaseMetrics]:
    """Iterate curriculum phases, invoking ``run_phase`` per phase.

    ``run_phase`` should perform training/eval for the given phase and return
    aggregated metrics. The manager will record metrics and advance when
    thresholds are satisfied.
    """

    history: list[PhaseMetrics] = []
    while True:
        phase = manager.current_phase
        metrics = run_phase(phase)
        history.append(metrics)
        advanced = manager.update(metrics)
        if not advanced and manager.is_last_phase:
            break
        if advanced and manager.is_last_phase:
            # Train/evaluate final phase once before exit.
            metrics = run_phase(manager.current_phase)
            history.append(metrics)
            manager.record_metrics(metrics)
            break
    return history
