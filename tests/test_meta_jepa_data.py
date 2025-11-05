import pytest

torch = pytest.importorskip("torch")

from arcgen import Grid, ProgramStep, SyntheticTask
from training.meta_jepa.data import (
    PrimitiveVocabulary,
    RuleFamilyDataset,
    build_rule_family_dataset,
    build_rule_family_examples,
)


def make_task(task_id: str, primitives, changed: int = 4) -> SyntheticTask:
    rule_trace = [ProgramStep(name, params or {}) for name, params in primitives]
    metadata = {"changed_cells": changed, "phase": "I"}
    grid = Grid([[0, 1], [1, 0]])
    return SyntheticTask(
        task_id=task_id,
        phase="I",
        input_grid=grid,
        output_grid=grid,
        rule_trace=rule_trace,
        metadata=metadata,
    )


def test_build_rule_family_examples_groups_by_signature():
    tasks = [
        make_task("a1", [("mirror_x", None), ("translate", {"dx": 1, "dy": 0})]),
        make_task("a2", [("mirror_x", None), ("translate", {"dx": 1, "dy": 0})]),
        make_task("b1", [("mirror_y", None)], changed=2),
    ]

    examples = build_rule_family_examples(tasks, min_family_size=1)

    assert len(examples) == 2
    first = next(example for example in examples if example.family_size == 2)
    assert first.primitive_counts["mirror_x"] == 2
    assert "translate" in first.primitive_counts

    second = next(example for example in examples if example.family_size == 1)
    assert second.mean_changed_cells == pytest.approx(2.0)


def test_build_rule_family_dataset_creates_feature_tensor():
    tasks = [
        make_task("a1", [("mirror_x", None), ("translate", None)]),
        make_task("a2", [("mirror_x", None), ("translate", None)]),
        make_task("b1", [("mirror_y", None)]),
        make_task("b2", [("mirror_y", None)]),
    ]

    dataset, vocab, examples = build_rule_family_dataset(tasks, min_family_size=2)

    assert isinstance(dataset, RuleFamilyDataset)
    assert isinstance(vocab, PrimitiveVocabulary)
    assert len(dataset) == 2
    feature_dim = len(vocab) + 3  # primitive histogram + stats
    assert dataset.features.shape == (2, feature_dim)
    assert dataset.labels.tolist() == [0, 1]
