import pytest

torch = pytest.importorskip("torch")

from arcgen import Grid, ProgramStep, SyntheticTask
from training.meta_jepa import MetaJEPATrainer, TrainingConfig


def make_task(task_id: str, primitives, changed: int = 4) -> SyntheticTask:
    rule_trace = [ProgramStep(name, params or {}) for name, params in primitives]
    metadata = {"changed_cells": changed, "phase": "I"}
    grid_in = Grid([[0, 1], [1, 0]])
    grid_out = Grid([[1, 0], [0, 1]])
    return SyntheticTask(
        task_id=task_id,
        phase="I",
        input_grid=grid_in,
        output_grid=grid_out,
        rule_trace=rule_trace,
        metadata=metadata,
    )


def build_tasks():
    return [
        make_task("a1", [("mirror_x", None), ("translate", None)]),
        make_task("a2", [("mirror_x", None), ("translate", None)]),
        make_task("b1", [("mirror_y", None)]),
        make_task("b2", [("mirror_y", None)]),
    ]


def test_meta_jepa_trainer_fit_and_encode():
    trainer = MetaJEPATrainer.from_tasks(build_tasks(), min_family_size=2, model_kwargs={"embedding_dim": 16})

    config = TrainingConfig(epochs=2, batch_size=2, lr=1e-3, temperature=0.5)
    result = trainer.fit(config)

    assert len(result.history) == 2
    assert all(loss >= 0 for loss in result.history)
    assert pytest.approx(result.temperature, rel=1e-5) == 0.5

    vocab_size = len(trainer.vocabulary)
    assert trainer.dataset.adjacency.shape == (len(trainer.dataset), vocab_size, vocab_size)

    features = trainer.dataset.features
    embeddings = trainer.encode(features)
    assert embeddings.shape == (len(trainer.dataset), 16)
    norms = embeddings.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    embeddings_with_adj = trainer.encode(features, adjacency=trainer.dataset.adjacency)
    assert embeddings_with_adj.shape == (len(trainer.dataset), 16)


def test_meta_jepa_trainer_learnable_temperature_clamps_and_trains():
    trainer = MetaJEPATrainer.from_tasks(build_tasks(), min_family_size=2, model_kwargs={"embedding_dim": 16})

    config = TrainingConfig(
        epochs=2,
        batch_size=2,
        lr=1e-3,
        learnable_temperature=True,
        temperature_init=0.2,
        temperature_bounds=(0.1, 0.3),
    )
    result = trainer.fit(config)

    assert len(result.history) == 2
    assert config.temperature_bounds[0] <= result.temperature <= config.temperature_bounds[1]


def test_meta_jepa_trainer_relational_auxiliary_task():
    trainer = MetaJEPATrainer.from_tasks(
        build_tasks(),
        min_family_size=1,
        model_kwargs={"embedding_dim": 16, "relational_decoder": True},
    )

    config = TrainingConfig(
        epochs=1,
        batch_size=2,
        lr=1e-3,
        relational_weight=0.5,
    )
    trainer.fit(config)

    assert trainer.model.relational_decoder is not None
    features = trainer.dataset.features
    adjacency = trainer.dataset.adjacency
    with torch.no_grad():
        _, logits = trainer.model(features, adjacency=adjacency, return_relations=True)
    assert logits is not None
    assert logits.shape == adjacency.shape
