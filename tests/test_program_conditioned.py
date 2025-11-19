import pytest

torch = pytest.importorskip("torch")

from training.jepa import (
    ObjectCentricEncoding,
    ProgramConditionedJEPA,
    ProgramConditionedModelConfig,
    aggregate_object_encoding,
)


def test_program_conditioned_forward_shapes():
    config = ProgramConditionedModelConfig(program_embedding_dim=32, hidden_dim=64, transition_hidden_dim=48)
    model = ProgramConditionedJEPA(latent_dim=16, vocab_size=10, parameter_dim=3, config=config)

    batch_size = 5
    latents = torch.randn(batch_size, 16)
    program_ids = torch.randint(0, 10, (batch_size, 4))
    program_params = torch.randn(batch_size, 4, 3)
    program_mask = torch.ones(batch_size, 4)

    predicted = model(latents, program_ids, program_params, program_mask)
    assert predicted.shape == (batch_size, 16)


def test_aggregate_object_encoding_matches_mean():
    embeddings = torch.tensor([[[1.0, 2.0], [2.0, 0.0]], [[0.0, 2.0], [4.0, 4.0]]])
    mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    adjacency = torch.zeros_like(embeddings)
    encoding = ObjectCentricEncoding(
        embeddings=embeddings,
        mask=mask,
        adjacency=adjacency,
        vq_loss=None,
        vq_indices=None,
    )

    aggregated = aggregate_object_encoding(encoding)
    expected = torch.tensor([[1.5, 1.0], [0.0, 2.0]])
    assert torch.allclose(aggregated, expected, atol=1e-6)
