import pytest

torch = pytest.importorskip("torch")

from arcgen import Grid, PRIMITIVE_REGISTRY, ProgramStep, SyntheticTask
from training.dsl.enumerator import InputVar, ProgramEnumerator, ProgramInterpreter
from training.dsl.primitives import build_default_primitive_registry
from training.dsl.types import Grid as GridType
from training.meta_jepa import MetaJEPAPrior, MetaJEPATrainer, TrainingConfig
from training.solver import FewShotSolver
from training.utils import count_changed_cells


def make_task(task_id: str, primitive) -> SyntheticTask:
    grid_in = Grid([[0, 1], [0, 0]])
    primitives = (primitive,) if isinstance(primitive, str) else tuple(primitive)
    grid_out = grid_in
    for name in primitives:
        spec = PRIMITIVE_REGISTRY.get(name)
        grid_out = spec.apply(grid_out)
    metadata = {"changed_cells": count_changed_cells(grid_in, grid_out), "phase": "I"}
    return SyntheticTask(
        task_id=task_id,
        phase="I",
        input_grid=grid_in,
        output_grid=grid_out,
        rule_trace=[ProgramStep(name, {}) for name in primitives],
        metadata=metadata,
    )


def build_trained_prior(extra_tasks=None, *, cluster_levels=None) -> MetaJEPAPrior:
    tasks = [
        make_task("mirror_y_1", "mirror_y"),
        make_task("mirror_y_2", "mirror_y"),
        make_task("mirror_x_1", "mirror_x"),
        make_task("mirror_x_2", "mirror_x"),
    ]
    if extra_tasks:
        tasks.extend(extra_tasks)
    trainer = MetaJEPATrainer.from_tasks(tasks, min_family_size=1, model_kwargs={"embedding_dim": 16})
    config = TrainingConfig(epochs=3, batch_size=2, lr=1e-3, temperature=0.5)
    trainer.fit(config)
    return MetaJEPAPrior(trainer, cluster_levels=cluster_levels)


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


def test_meta_prior_cluster_levels_group_similar_families():
    extra_tasks = [
        make_task("mirror_y_stack_1", ("mirror_y", "mirror_y")),
        make_task("mirror_y_stack_2", ("mirror_y", "mirror_y")),
    ]
    prior = build_trained_prior(extra_tasks=extra_tasks, cluster_levels=(3, 2))

    level = 2
    hierarchy = prior.hierarchy
    examples = prior.trainer.dataset.examples
    mirror_y_id = next(example.family_id for example in examples if example.family_id == "mirror_y")
    mirror_x_id = next(example.family_id for example in examples if example.family_id == "mirror_x")
    mirror_y_stack = next(example.family_id for example in examples if "mirror_y::mirror_y" == example.family_id)

    assert hierarchy.cluster_id(mirror_y_id, level) == hierarchy.cluster_id(mirror_y_stack, level)
    assert hierarchy.cluster_id(mirror_y_id, level) != hierarchy.cluster_id(mirror_x_id, level)

    registry = build_default_primitive_registry()
    enumerator = ProgramEnumerator(
        registry,
        inputs=[InputVar("grid", GridType)],
        target_type=GridType,
        max_nodes=1,
    )

    context = Grid([[0, 1], [0, 0]])
    target = PRIMITIVE_REGISTRY.get("mirror_y").apply(context)
    interpreter = ProgramInterpreter()
    good_program = None
    bad_program = None
    for program in enumerator.enumerate():
        try:
            output = interpreter.evaluate(program, {"grid": context})
        except Exception:
            continue
        if not isinstance(output, Grid):
            continue
        if output.cells == target.cells:
            good_program = program
        else:
            bad_program = program
        if good_program and bad_program:
            break

    assert good_program is not None and bad_program is not None

    predicted_good = prior.predict_cluster(good_program, context, target, level=level)
    predicted_bad = prior.predict_cluster(bad_program, context, target, level=level)

    assert predicted_good == hierarchy.cluster_id(mirror_y_id, level)
    assert predicted_bad != predicted_good
