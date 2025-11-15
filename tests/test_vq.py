import pytest

torch = pytest.importorskip("torch")

from training.modules import VectorQuantizer


def test_vector_quantizer_forward_backward():
    torch.manual_seed(0)
    vq = VectorQuantizer(num_embeddings=8, embedding_dim=4, commitment_cost=0.5)
    vq.train()

    inputs = torch.randn(2, 3, 4, dtype=torch.float32, requires_grad=True)
    output = vq(inputs)

    assert output.quantized.shape == inputs.shape
    assert output.indices.shape == inputs.shape[:-1]
    assert output.loss.ndim == 0
    assert torch.isfinite(output.perplexity)

    output.loss.backward()
    assert inputs.grad is not None


def test_vector_quantizer_ema_updates_embedding():
    torch.manual_seed(0)
    vq = VectorQuantizer(num_embeddings=4, embedding_dim=2, ema_decay=0.9)
    vq.train()

    initial_weight = vq.embedding.weight.detach().clone()

    inputs = torch.randn(16, 2, dtype=torch.float32)
    _ = vq(inputs)

    updated_weight = vq.embedding.weight.detach().clone()
    assert not torch.allclose(initial_weight, updated_weight)


def test_vector_quantizer_refresh_revives_unused_codes():
    torch.manual_seed(0)
    vq = VectorQuantizer(
        num_embeddings=3,
        embedding_dim=2,
        ema_decay=0.9,
        refresh_unused_codes=True,
        refresh_interval=1,
        refresh_usage_threshold=0.2,
    )
    vq.train()

    with torch.no_grad():
        weights = torch.tensor(
            [
                [0.0, 0.0],
                [10.0, 10.0],
                [20.0, 20.0],
            ],
            dtype=torch.float32,
        )
        vq.embedding.weight.copy_(weights)
        vq._ema_w.copy_(weights)

    cluster_a = torch.randn(8, 2, dtype=torch.float32) * 0.05
    cluster_b = torch.randn(8, 2, dtype=torch.float32) * 0.05 + torch.tensor([5.0, 5.0])
    inputs = torch.cat([cluster_a, cluster_b], dim=0)

    _ = vq(inputs)  # first pass triggers refresh
    result = vq(inputs)

    flat_indices = result.indices.reshape(-1)
    assert (flat_indices == 1).any() or (flat_indices == 2).any()
