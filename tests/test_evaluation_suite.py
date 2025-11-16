from arcgen import Grid, ProgramStep, SyntheticTask
from training.eval import ArcExample, ArcTask, EvaluationSuite, EvaluationVariant


def make_task(task_id: str, primitive: str):
    rule_trace = [ProgramStep(primitive, {})]
    grid_in = Grid([[0, 1], [0, 0]])
    if primitive == "mirror_y":
        grid_out = Grid([[1, 0], [0, 0]])
    else:
        grid_out = Grid([[0, 1], [0, 0]])
    return SyntheticTask(
        task_id=task_id,
        phase="I",
        input_grid=grid_in,
        output_grid=grid_out,
        rule_trace=rule_trace,
        metadata={"changed_cells": 1},
    )


def test_evaluation_suite_runs_variants():
    tasks = [
        make_task("mirror", "mirror_y"),
        make_task("identity", "identity"),
    ]

    suite = EvaluationSuite(tasks)
    variants = [
        EvaluationVariant(name="full", description="all primitives", max_nodes=1),
        EvaluationVariant(name="filtered", description="limited primitives", max_nodes=1, allowed_primitives=("mirror_y",)),
    ]

    results = suite.run(variants)
    assert len(results) == 2

    full = next(result for result in results if result.variant.name == "full")
    filtered = next(result for result in results if result.variant.name == "filtered")

    assert full.successes >= filtered.successes
    assert filtered.total_tasks == len(tasks)
    assert all(detail.programs_tested >= 0 for detail in full.details)


def test_evaluation_suite_handles_arc_dev_tasks():
    train_examples = (
        ArcExample(Grid([[0]]), Grid([[0]])),
        ArcExample(Grid([[0, 1], [1, 0]]), Grid([[0, 1], [1, 0]])),
    )
    test_examples = (ArcExample(Grid([[2, 2]]), Grid([[2, 2]])),)
    task = ArcTask(task_id="dev_task", train_examples=train_examples, test_examples=test_examples)

    suite = EvaluationSuite([task])
    variant = EvaluationVariant(name="arc_dev", description="dev harness", max_nodes=2)
    result = suite.run([variant])[0]

    assert result.successes == 1
    assert result.details[0].success


def test_evaluation_suite_marks_failed_test_predictions():
    train_examples = (
        ArcExample(Grid([[0]]), Grid([[0]])),
        ArcExample(Grid([[0, 1]]), Grid([[0, 1]])),
    )
    failing_test = (ArcExample(Grid([[3]]), Grid([[1]])),)
    task = ArcTask(task_id="dev_task_bad", train_examples=train_examples, test_examples=failing_test)

    suite = EvaluationSuite([task])
    variant = EvaluationVariant(name="arc_dev", description="dev harness", max_nodes=2)
    result = suite.run([variant])[0]

    assert result.successes == 0
    assert not result.details[0].success
