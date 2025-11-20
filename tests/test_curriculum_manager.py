import pytest

from training.curriculum import (
    CurriculumConfig,
    CurriculumManager,
    PhaseConfig,
    PhaseMetrics,
    PhaseThresholds,
    compute_phase_metrics,
    run_curriculum,
)


def _phase(name: str, solve: float = 0.5) -> PhaseConfig:
    return PhaseConfig(
        name=name,
        task_schedule={"atomic": 1},
        thresholds=PhaseThresholds(solve_rate=solve, codebook_usage=0.1, option_diversity=0.1),
        min_episodes=5,
    )


def test_should_advance_when_thresholds_met():
    manager = CurriculumManager(CurriculumConfig(phases=[_phase("I"), _phase("II")]))
    metrics = compute_phase_metrics(
        episodes=5,
        successes=4,
        codebook_usage=0.5,
        option_counts={"a": 3, "b": 2},
    )
    assert manager.should_advance(metrics)
    manager.update(metrics)
    assert manager.current_phase.name == "II"


def test_does_not_advance_when_thresholds_unmet():
    manager = CurriculumManager(CurriculumConfig(phases=[_phase("I", solve=0.8), _phase("II")]))
    metrics = compute_phase_metrics(
        episodes=5,
        successes=3,
        codebook_usage=0.5,
        option_counts={"a": 1},
    )
    assert not manager.should_advance(metrics)
    advanced = manager.update(metrics)
    assert not advanced
    assert manager.current_phase.name == "I"


def test_run_curriculum_advances_all_phases():
    manager = CurriculumManager(CurriculumConfig(phases=[_phase("I"), _phase("II")]))

    def _run_phase(phase: PhaseConfig) -> PhaseMetrics:
        return compute_phase_metrics(
            episodes=phase.min_episodes,
            successes=phase.min_episodes,
            codebook_usage=1.0,
            option_counts={"opt": phase.min_episodes},
        )

    history = run_curriculum(manager, _run_phase)
    assert manager.is_last_phase
    assert len(history) >= 2
    assert manager.current_phase.name == "II"
