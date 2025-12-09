"""VICReg (Variance-Invariance-Covariance Regularization) for JEPA.

VICReg prevents representation collapse by enforcing three properties:
1. Variance: Each dimension should have variance > epsilon (prevents constant outputs)
2. Invariance: Matched pairs should have similar representations (the prediction objective)
3. Covariance: Dimensions should be decorrelated (prevents redundant dimensions)

Reference: Bardes et al. "VICReg: Variance-Invariance-Covariance Regularization
           for Self-Supervised Learning" (ICLR 2022)

For I-JEPA, we apply variance and covariance to the target embeddings (which are
the EMA encoder outputs). This prevents the target from collapsing, which in turn
prevents the predictor from finding trivial solutions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None  # type: ignore
    nn = object  # type: ignore
    F = object  # type: ignore


@dataclass(frozen=True)
class VICRegConfig:
    """Configuration for VICReg regularization."""

    enabled: bool = False

    # Variance loss: hinge loss on std dev, std should be >= variance_target
    variance_weight: float = 1.0
    variance_target: float = 1.0  # Target std dev (usually 1.0)
    variance_epsilon: float = 1e-4  # Small constant for numerical stability

    # Covariance loss: push off-diagonal covariance entries toward 0
    covariance_weight: float = 0.04  # Lower weight since many off-diag terms

    # Which embeddings to regularize
    regularize_target: bool = True  # Apply to target (EMA) embeddings
    regularize_predictor: bool = False  # Apply to predictor outputs

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None) -> "VICRegConfig":
        if data is None:
            return cls()

        weight = data.get("weight")
        if weight is not None:
            # Shorthand: vicreg.weight sets both variance and covariance
            var_weight = float(weight)
            cov_weight = float(weight) * 0.04  # Standard ratio from paper
        else:
            var_weight = float(data.get("variance_weight", cls.variance_weight))
            cov_weight = float(data.get("covariance_weight", cls.covariance_weight))

        return cls(
            enabled=bool(data.get("enabled", weight is not None and float(weight) > 0)),
            variance_weight=var_weight,
            variance_target=float(data.get("variance_target", cls.variance_target)),
            variance_epsilon=float(data.get("variance_epsilon", cls.variance_epsilon)),
            covariance_weight=cov_weight,
            regularize_target=bool(data.get("regularize_target", cls.regularize_target)),
            regularize_predictor=bool(data.get("regularize_predictor", cls.regularize_predictor)),
        )


if torch is not None:

    class VICRegLoss(nn.Module):
        """VICReg loss for preventing representation collapse.

        Computes:
        - Variance loss: ReLU(target_std - std(x, dim=0)).mean()
        - Covariance loss: (off_diagonal(cov(x)) ** 2).sum() / dim
        """

        def __init__(
            self,
            variance_weight: float = 1.0,
            covariance_weight: float = 0.04,
            variance_target: float = 1.0,
            variance_epsilon: float = 1e-4,
        ) -> None:
            super().__init__()
            self.variance_weight = variance_weight
            self.covariance_weight = covariance_weight
            self.variance_target = variance_target
            self.variance_epsilon = variance_epsilon

        def variance_loss(self, x: torch.Tensor) -> torch.Tensor:
            """Hinge loss on standard deviation.

            Encourages std >= variance_target along each dimension.
            """
            # x shape: (batch, dim)
            std = torch.sqrt(x.var(dim=0) + self.variance_epsilon)
            # Hinge: penalize if std < target
            return F.relu(self.variance_target - std).mean()

        def covariance_loss(self, x: torch.Tensor) -> torch.Tensor:
            """Push off-diagonal covariance entries toward zero.

            Decorrelates the dimensions to prevent redundancy.
            """
            # x shape: (batch, dim)
            batch_size, dim = x.shape

            # Center the representations
            x_centered = x - x.mean(dim=0, keepdim=True)

            # Compute covariance matrix
            cov = (x_centered.T @ x_centered) / (batch_size - 1)

            # Off-diagonal elements (exclude diagonal)
            off_diag = cov.pow(2).sum() - cov.diagonal().pow(2).sum()

            # Normalize by dimension
            return off_diag / dim

        def forward(
            self,
            x: torch.Tensor,
            return_components: bool = False,
        ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
            """Compute VICReg loss.

            Args:
                x: Embeddings of shape (batch, dim)
                return_components: If True, also return individual loss components

            Returns:
                Total VICReg loss (and optionally component dict)
            """
            var_loss = self.variance_loss(x)
            cov_loss = self.covariance_loss(x)

            total = self.variance_weight * var_loss + self.covariance_weight * cov_loss

            if return_components:
                return total, {
                    "variance": var_loss,
                    "covariance": cov_loss,
                }
            return total

else:

    class VICRegLoss:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("PyTorch is required for VICRegLoss")
