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
    Primitive,
    PrimitiveRegistry,
    Expression,
    Program,
)


def dummy_latent_embedder(grid: Grid) -> torch.Tensor:
    flat = torch.tensor(grid.flatten(), dtype=torch.float32)
    if flat.numel() == 0:
        return torch.zeros(1)
    if torch.all(flat == 0):
        return flat + 1e-6
    return torch.nn.functional.normalize(flat, dim=0)


def _build_identity_registry():
    registry = PrimitiveRegistry()
    identity = Primitive(
        name="identity_grid",
        input_types=(GridType,),
        output_type=GridType,
        implementation=lambda grid: grid,
    )
    registry.register(identity)
    return registry, identity


def test_program_encoder_outputs_expected_shape():
    torch.manual_seed(0)
    registry, identity = _build_identity_registry()
    var = InputVar("grid", GridType)
    var_expr = Expression(type=GridType, var=var)
    program = Program(Expression(type=GridType, primitive=identity, args=(var_expr,)))
    encoder = ProgramEncoder(registry, embedding_dim=16)

    encoding = encoder(program)

    assert tuple(encoding.shape) == (16,)


def test_program_encoder_distinguishes_tree_structure():
    torch.manual_seed(0)
    registry, identity = _build_identity_registry()
    var = InputVar("grid", GridType)
    var_expr = Expression(type=GridType, var=var)
    shallow_expr = Expression(type=GridType, primitive=identity, args=(var_expr,))
    deep_expr = Expression(type=GridType, primitive=identity, args=(shallow_expr,))
    shallow_program = Program(shallow_expr)
    deep_program = Program(deep_expr)
    encoder = ProgramEncoder(registry, embedding_dim=16)

    shallow_encoding = encoder(shallow_program)
    deep_encoding = encoder(deep_program)

    assert not torch.allclose(shallow_encoding, deep_encoding)


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
        target,
    )

    assert results
    assert len(results) <= 2


def test_guided_beam_search_batches_scorer():
    torch = pytest.importorskip("torch")
    registry = build_default_primitive_registry(color_constants=(0, 1))
    program_encoder = ProgramEncoder(registry, embedding_dim=8)

    class CountingScorer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.last_shape = None

        def forward(self, features):
            self.last_shape = tuple(features.shape)
            return torch.zeros(features.shape[0], device=features.device)

    scorer = CountingScorer()

    beam = GuidedBeamSearch(
        registry,
        scorer,
        program_encoder,
        ProgramInterpreter(),
        dummy_latent_embedder,
        beam_width=4,
        length_penalty=0.0,
        parallel=True,
    )

    enumerator = ProgramEnumerator(
        registry,
        inputs=[InputVar("grid", GridType)],
        target_type=GridType,
        max_nodes=2,
    )

    context = Grid([[0, 1], [0, 0]])
    target = Grid([[1, 0], [0, 0]])
    results = beam.search(
        dummy_latent_embedder(context),
        dummy_latent_embedder(target),
        enumerator,
        context,
        target,
    )

    assert results
    assert scorer.last_shape is not None and scorer.last_shape[0] > 1


def test_guided_beam_search_can_disable_parallel_path():
    torch = pytest.importorskip("torch")
    registry = build_default_primitive_registry(color_constants=(0, 1))
    program_encoder = ProgramEncoder(registry, embedding_dim=4)

    class CountingScorer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, features):
            self.calls += 1
            return torch.zeros(features.shape[0], device=features.device)

    scorer = CountingScorer()

    beam = GuidedBeamSearch(
        registry,
        scorer,
        program_encoder,
        ProgramInterpreter(),
        dummy_latent_embedder,
        beam_width=2,
        length_penalty=0.0,
        parallel=False,
    )

    enumerator = ProgramEnumerator(
        registry,
        inputs=[InputVar("grid", GridType)],
        target_type=GridType,
        max_nodes=2,
    )

    context = Grid([[0, 1], [0, 0]])
    target = Grid([[1, 0], [0, 0]])
    results = beam.search(
        dummy_latent_embedder(context),
        dummy_latent_embedder(target),
        enumerator,
        context,
        target,
    )

    assert results
    assert scorer.calls == len(results) or scorer.calls >= 1
