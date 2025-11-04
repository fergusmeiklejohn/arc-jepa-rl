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
