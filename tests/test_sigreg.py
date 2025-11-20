import pytest

torch = pytest.importorskip("torch")

from training.jepa.sigreg import SIGRegLoss, SIGRegLossConfig


def test_sigreg_config_validation():
    with pytest.raises(ValueError):
        SIGRegLossConfig.from_mapping({"num_slices": 0})
    with pytest.raises(ValueError):
        SIGRegLossConfig.from_mapping({"num_points": 4})


def test_sigreg_penalty_is_finite_and_backward():
    torch.manual_seed(0)
    loss_fn = SIGRegLoss(num_slices=8, num_points=5)
    embeddings = torch.randn(4, 6, requires_grad=True)

    penalty = loss_fn(embeddings)
    assert penalty.dim() == 0
    assert torch.isfinite(penalty)
    assert penalty >= 0

    penalty.backward()
    assert embeddings.grad is not None


def test_sigreg_rejects_non_matrix_inputs():
    loss_fn = SIGRegLoss(num_slices=4, num_points=5)
    with pytest.raises(ValueError):
        loss_fn(torch.randn(6))
