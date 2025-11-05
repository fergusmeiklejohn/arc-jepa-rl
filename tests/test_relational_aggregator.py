import pytest

torch = pytest.importorskip("torch")

from training.modules.relational import RelationalAggregator


def test_relational_aggregator_respects_mask_and_shape():
    aggregator = RelationalAggregator(hidden_dim=8, num_layers=2, num_heads=2, dropout=0.0)

    embeddings = torch.randn(1, 3, 8)
    adjacency = torch.tensor(
        [[[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]]],
        dtype=torch.float32,
    )
    mask = torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32)

    output = aggregator(embeddings, adjacency, mask)

    assert output.shape == (1, 3, 8)
    assert torch.allclose(output[:, 2, :], torch.zeros_like(output[:, 2, :]), atol=1e-6)


def test_relational_aggregator_handles_isolated_nodes():
    aggregator = RelationalAggregator(hidden_dim=6, num_layers=1, num_heads=3, dropout=0.0)

    embeddings = torch.randn(2, 2, 6)
    adjacency = torch.zeros(2, 2, 2)
    mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]], dtype=torch.float32)

    output = aggregator(embeddings, adjacency, mask)

    assert output.shape == (2, 2, 6)
    assert torch.allclose(output[1, 1], torch.zeros(6), atol=1e-6)

