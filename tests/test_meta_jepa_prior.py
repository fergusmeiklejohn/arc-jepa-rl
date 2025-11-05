import pytest

torch = pytest.importorskip("torch")

from arcgen import Grid, PRIMITIVE_REGISTRY, ProgramStep, SyntheticTask
from training.dsl.enumerator import InputVar, ProgramEnumerator, ProgramInterpreter
from training.dsl.primitives import build_default_primitive_registry
from training.dsl.types import Grid as GridType
from training.meta_jepa import MetaJEPAPrior, MetaJEPATrainer, TrainingConfig
from training.solver import FewShotSolver
from training.utils import count_changed_cells


def make_task(task_id: str, primitive: str) -> SyntheticTask:
    grid_in = Grid([[0, 1], [0, 0]])
    spec = PRIMITIVE_REGISTRY.get(primitive)
    grid_out = spec.apply(grid_in)
    metadata = {"changed_cells": count_changed_cells(grid_in, grid_out), "phase": "I"}
    return SyntheticTask(
        task_id=task_id,
        phase="I",
        input_grid=grid_in,
        output_grid=grid_out,
        rule_trace=[ProgramStep(primitive, {})],
        metadata=metadata,
    )


def build_trained_prior() -> MetaJEPAPrior:
    tasks = [
        make_task("mirror_y_1", "mirror_y"),
        make_task("mirror_y_2", "mirror_y"),
        make_task("mirror_x_1", "mirror_x"),
        make_task("mirror_x_2", "mirror_x"),
    ]
    trainer = MetaJEPATrainer.from_tasks(tasks, min_family_size=1, model_kwargs={"embedding_dim": 16})
    config = TrainingConfig(epochs=3, batch_size=2, lr=1e-3, temperature=0.5)
    trainer.fit(config)
    return MetaJEPAPrior(trainer)


def test_meta_prior_scores_matching_program_higher():
    prior = build_trained_prior()
    registry = build_default_primitive_registry()
    context = Grid([[0, 1], [0, 0]])
    target = PRIMITIVE_REGISTRY.get("mirror_y").apply(context)

    enumerator = ProgramEnumerator(
        registry,
        inputs=[InputVar("grid", GridType)],
        target_type=GridType,
        max_nodes=1,
    )

    good_score = None
    bad_score = None

    interpreter = ProgramInterpreter()
    for program in enumerator.enumerate():
        try:
            output = interpreter.evaluate(program, {"grid": context})
        except Exception:
            continue
        if not isinstance(output, Grid):
            continue
        score = prior.score_program(program, context, output)
        if output.cells == target.cells:
            good_score = score
        else:
            if bad_score is None:
                bad_score = score
        if good_score is not None and bad_score is not None:
            break

    assert good_score is not None
    assert bad_score is not None
    assert good_score > bad_score


def test_few_shot_solver_uses_meta_prior_for_ordering():
    prior = build_trained_prior()
    registry = build_default_primitive_registry()
    solver = FewShotSolver(registry)

    context = Grid([[0, 1], [0, 0]])
    target = PRIMITIVE_REGISTRY.get("mirror_y").apply(context)

    baseline = solver.solve([(context, target)], max_nodes=1)
    assert baseline.evaluated > 1

    guided = solver.solve([(context, target)], max_nodes=1, meta_prior=prior)
    assert guided.solved()
    assert guided.evaluated <= baseline.evaluated
