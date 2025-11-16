import math

import pytest

torch = pytest.importorskip("torch")

from arcgen import Grid, PRIMITIVE_REGISTRY, ProgramStep, SyntheticTask
from training.dsl import GuidanceScorer, GuidedBeamSearch, ProgramEncoder, build_default_primitive_registry
from training.dsl.enumerator import InputVar, ProgramEnumerator, ProgramInterpreter
from training.dsl.primitives import PrimitiveRegistry
from training.dsl.types import Grid as GridType
from training.jepa import InMemoryGridPairDataset, ObjectCentricJEPAExperiment
from training.meta_jepa import MetaJEPAPrior, MetaJEPATrainer, TrainingConfig
from training.solver import FewShotSolver
from training.utils import count_changed_cells


ACTIVE_PRIMITIVES = ("identity", "mirror_x", "mirror_y")


def _make_task(task_id: str, primitive: str, cells: list[list[int]]) -> SyntheticTask:
    grid_in = Grid(cells)
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


def _build_demo_tasks() -> list[SyntheticTask]:
    examples = [
        ("mirror_y", [[0, 1, 0], [0, 0, 0], [0, 0, 0]]),
        ("mirror_y", [[0, 0, 2], [0, 0, 0], [0, 0, 0]]),
        ("mirror_x", [[0, 0, 0], [1, 0, 0], [0, 0, 0]]),
        ("mirror_x", [[0, 0, 0], [0, 0, 0], [0, 3, 0]]),
    ]
    tasks = []
    for idx, (primitive, cells) in enumerate(examples):
        tasks.append(_make_task(f"{primitive}_{idx}", primitive, cells))
    return tasks


def _tiny_jepa_config() -> dict:
    return {
        "tokenizer": {
            "max_objects": 4,
            "max_color_features": 3,
        },
        "object_encoder": {
            "hidden_dim": 16,
            "num_embeddings": 32,
            "relational_layers": 1,
            "relational_heads": 2,
        },
        "optimizer": {
            "lr": 1e-3,
            "weight_decay": 0.0,
        },
        "loss": {
            "queue_size": 16,
            "projection_dim": 16,
            "temperature_bounds": (0.05, 0.5),
        },
        "data": {
            "context_length": 1,
        },
        "training": {
            "grad_accum_steps": 1,
        },
    }


def test_e2e_pipeline_runs_jepa_meta_solver_round_trip():
    torch.manual_seed(0)
    tasks = _build_demo_tasks()

    experiment = ObjectCentricJEPAExperiment(_tiny_jepa_config())
    context_sequences = [(task.input_grid,) for task in tasks]
    target_grids = [task.output_grid for task in tasks]
    dataset = InMemoryGridPairDataset([(context_sequences, target_grids)])
    loss = experiment.train_epoch(dataset)
    assert math.isfinite(loss)

    def embed_grid(grid: Grid) -> torch.Tensor:
        encoding = experiment.trainer.object_encoder.encode([grid], device=experiment.device)
        pooled = experiment._masked_mean(encoding.embeddings, encoding.mask.to(experiment.device))
        return pooled.squeeze(0).detach()

    trainer = MetaJEPATrainer.from_tasks(
        tasks,
        min_family_size=1,
        model_kwargs={"embedding_dim": 16, "hidden_dim": 32},
    )
    trainer.fit(TrainingConfig(epochs=5, batch_size=2, lr=1e-3, temperature=0.5))
    meta_prior = MetaJEPAPrior(trainer)

    base_registry = build_default_primitive_registry(color_constants=(0, 1))
    registry = PrimitiveRegistry()
    for name in ACTIVE_PRIMITIVES:
        registry.register(base_registry.get(name))

    program_encoder = ProgramEncoder(registry, embedding_dim=8)
    latent_dim = embed_grid(tasks[0].input_grid).shape[0]
    enumerator = ProgramEnumerator(
        registry,
        inputs=[InputVar("grid", GridType)],
        target_type=GridType,
        max_nodes=1,
    )
    sample_program = next(enumerator.enumerate())
    program_dim = program_encoder(sample_program).shape[0]
    feature_dim = latent_dim * 3 + program_dim + 1
    scorer = GuidanceScorer(feature_dim, hidden_dim=16)
    with torch.no_grad():
        for param in scorer.parameters():
            param.zero_()

    beam = GuidedBeamSearch(
        registry,
        scorer,
        program_encoder,
        ProgramInterpreter(),
        embed_grid,
        beam_width=16,
        length_penalty=0.01,
        meta_prior=meta_prior,
        meta_weight=0.3,
    )
    solver = FewShotSolver(registry, mdl_weight=0.01)

    results = []
    for task in tasks:
        latent_context = embed_grid(task.input_grid)
        latent_target = embed_grid(task.output_grid)
        outcome = solver.solve(
            [(task.input_grid, task.output_grid)],
            max_nodes=1,
            beam_search=beam,
            latent_context=latent_context,
            latent_target=latent_target,
        )
        results.append(outcome)

    successes = sum(result.solved() for result in results)
    avg_programs = sum(result.evaluated for result in results) / len(results)

    assert successes == len(tasks)
    assert avg_programs <= len(ACTIVE_PRIMITIVES) + 1
