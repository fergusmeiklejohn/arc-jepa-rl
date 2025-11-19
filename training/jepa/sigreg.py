"""Sketched isotropic Gaussian regularization utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F

from lejepa.multivariate import SlicingUnivariateTest
from lejepa.univariate import EppsPulley


@dataclass(frozen=True)
class SIGRegLossConfig:
    """User-configurable knobs for SIGReg regularisation."""

    weight: float = 0.0
    num_slices: int = 128
    num_points: int = 17

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None) -> "SIGRegLossConfig":
        if data is None:
            return cls()

        weight = float(data.get("weight", cls.weight))
        num_slices = int(data.get("num_slices", cls.num_slices))
        num_points = int(data.get("num_points", cls.num_points))

        if num_slices <= 0:
            raise ValueError("sigreg.num_slices must be positive")
        if num_points <= 0 or num_points % 2 == 0:
            raise ValueError("sigreg.num_points must be a positive odd integer")

        return cls(weight=weight, num_slices=num_slices, num_points=num_points)

    @property
    def enabled(self) -> bool:
        return self.weight > 0.0


class SIGRegLoss(torch.nn.Module):
    """Differentiable SIGReg penalty built from the LeJEPA test suite."""

    def __init__(self, num_slices: int, num_points: int) -> None:
        super().__init__()
        self._univariate_test = EppsPulley(n_points=num_points)
        self._sliced_test = SlicingUnivariateTest(
            univariate_test=self._univariate_test,
            num_slices=num_slices,
            reduction="mean",
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute the SIGReg penalty for a batch of embeddings.

        Args:
            embeddings: Tensor with shape (batch, dim).

        Returns:
            Scalar torch.Tensor containing the penalty value.
        """
        if embeddings.dim() < 2:
            raise ValueError("expected embeddings with shape (batch, dim)")

        flat = embeddings.reshape(-1, embeddings.size(-1))
        normalized = F.normalize(flat, dim=-1)
        normalized = normalized.to(torch.float32)
        penalty = self._sliced_test(normalized)
        if penalty.dim() != 0:
            penalty = penalty.mean()
        return penalty

