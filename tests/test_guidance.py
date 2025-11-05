import pytest

torch = pytest.importorskip("torch")

from arcgen import Grid
from training.dsl import (
    build_default_primitive_registry,
    ProgramEncoder,
    GuidanceScorer,
    GuidedBeamSearch,
    GuidanceDataset,
    build_guidance_examples,
    InputVar,
    ProgramEnumerator,
    ProgramInterpreter,
    Grid as GridType,
)


def dummy_latent_embedder(grid: Grid) -> torch.Tensor:
    flat = torch.tensor(grid.flatten(), dtype=torch.float32)
    if flat.numel() == 0:
        return torch.zeros(1)
    if torch.all(flat == 0):
        return flat + 1e-6
    return torch.nn.functional.normalize(flat, dim=0)


def test_guidance_dataset_builds_examples():
    registry = build_default_primitive_registry(color_constants=(0, 1))
    program_encoder = ProgramEncoder(registry, embedding_dim=8)
    tasks = [(Grid([[0, 1], [0, 0]]), Grid([[1, 0], [0, 0]]))]

    examples = build_guidance_examples(
        tasks,
        registry,
        dummy_latent_embedder,
        program_encoder,
        max_nodes=3,
        max_programs_per_task=8,
    )

    assert examples
    dataset = GuidanceDataset(examples)
    vector, label, success = dataset[0]
    assert isinstance(vector, torch.Tensor)
    assert isinstance(label.item(), float)
    assert isinstance(success, bool)


def test_guided_beam_search_scores_programs():
    registry = build_default_primitive_registry(color_constants=(0, 1))
    program_encoder = ProgramEncoder(registry, embedding_dim=8)
    examples = build_guidance_examples(
        [(Grid([[0, 1], [0, 0]]), Grid([[1, 0], [0, 0]]))],
        registry,
        dummy_latent_embedder,
        program_encoder,
        max_nodes=3,
        max_programs_per_task=8,
    )
    dataset = GuidanceDataset(examples)
    feature_dim = dataset[0][0].shape[0]
    scorer = GuidanceScorer(feature_dim, hidden_dim=16)
    beam = GuidedBeamSearch(
        registry,
        scorer,
        program_encoder,
        ProgramInterpreter(),
        dummy_latent_embedder,
        beam_width=2,
        length_penalty=0.01,
    )

    enumerator = ProgramEnumerator(
        registry,
        inputs=[InputVar("grid", GridType)],
        target_type=GridType,
        max_nodes=3,
    )

    context = Grid([[0, 1], [0, 0]])
    target = Grid([[1, 0], [0, 0]])
    results = beam.search(
        dummy_latent_embedder(context),
        dummy_latent_embedder(target),
        enumerator,
        context,
    )

    assert results
    assert len(results) <= 2
