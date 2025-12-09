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


def test_sigreg_detects_collapse():
    """SIGReg should penalize collapsed embeddings more than Gaussian ones.

    This test verifies the L-JEPA paper insight: SIGReg encourages isotropic
    Gaussian distribution, so collapsed/non-Gaussian embeddings get higher penalty.
    """
    torch.manual_seed(42)
    loss_fn = SIGRegLoss(num_slices=128, num_points=17)

    # Perfect isotropic Gaussian - should have LOW penalty
    gaussian = torch.randn(256, 64)
    gaussian_penalty = loss_fn(gaussian)

    # Complete collapse (all identical) - should have HIGH penalty
    collapsed = torch.ones(256, 64) * 5.0
    collapsed_penalty = loss_fn(collapsed)

    # Bimodal (two clusters) - should have HIGH penalty (not Gaussian)
    bimodal = torch.cat([torch.randn(128, 64) + 5, torch.randn(128, 64) - 5])
    bimodal_penalty = loss_fn(bimodal)

    # Verify the ordering
    assert gaussian_penalty < collapsed_penalty, (
        f"Gaussian ({gaussian_penalty:.4f}) should have lower penalty than "
        f"collapsed ({collapsed_penalty:.4f})"
    )
    assert gaussian_penalty < bimodal_penalty, (
        f"Gaussian ({gaussian_penalty:.4f}) should have lower penalty than "
        f"bimodal ({bimodal_penalty:.4f})"
    )
    # Gaussian penalty should be relatively small
    # Note: The Epps-Pulley statistic scales with N (batch size), so the raw value
    # depends on sample count. What matters is it's much smaller than non-Gaussian cases.
    assert gaussian_penalty < 2.0, (
        f"Gaussian penalty ({gaussian_penalty:.4f}) should be < 2.0"
    )
