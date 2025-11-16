"""Meta-JEPA neural modules."""

from __future__ import annotations

try:  # pragma: no cover - optional dependency
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = object  # type: ignore
    F = None  # type: ignore


def _ensure_torch() -> None:
    if torch is None:  # pragma: no cover - defensive
        raise RuntimeError("PyTorch is required for Meta-JEPA models")


if torch is not None:  # pragma: no branch

    class MetaJEPAModel(nn.Module):
        """Simple feed-forward encoder for rule family features."""

        def __init__(
            self,
            input_dim: int,
            *,
            hidden_dim: int = 128,
            embedding_dim: int = 64,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            if input_dim <= 0:
                raise ValueError("input_dim must be positive")
            if hidden_dim <= 0 or embedding_dim <= 0:
                raise ValueError("hidden_dim and embedding_dim must be positive")
            if not 0.0 <= dropout < 1.0:
                raise ValueError("dropout must lie in [0, 1)")

            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, embedding_dim),
            )

        def forward(self, features: "torch.Tensor") -> "torch.Tensor":
            embeddings = self.net(features)
            return F.normalize(embeddings, dim=-1)

else:  # pragma: no cover

    class MetaJEPAModel:  # type: ignore[misc]
        def __init__(self, *args, **kwargs) -> None:  # noqa: D401
            raise RuntimeError("PyTorch is required for MetaJEPAModel")


def contrastive_loss(
    embeddings: "torch.Tensor",
    labels: "torch.Tensor",
    *,
    temperature: float | "torch.Tensor" = 0.1,
) -> "torch.Tensor":
    """Compute an InfoNCE-style contrastive loss over rule families."""

    _ensure_torch()

    if embeddings.ndim != 2:
        raise ValueError("embeddings must be 2D (batch, dim)")
    if labels.ndim != 1 or labels.shape[0] != embeddings.shape[0]:
        raise ValueError("labels must be 1D and align with embeddings")
    if isinstance(temperature, torch.Tensor):
        temp_tensor = temperature.to(device=embeddings.device, dtype=embeddings.dtype)
    else:
        temp_tensor = torch.tensor(float(temperature), dtype=embeddings.dtype, device=embeddings.device)

    if temp_tensor.numel() != 1:
        raise ValueError("temperature must be a scalar")
    if torch.any(temp_tensor <= 0):
        raise ValueError("temperature must be positive")

    sim_matrix = torch.matmul(embeddings, embeddings.T) / temp_tensor
    logits_mask = torch.ones_like(sim_matrix, dtype=torch.bool)
    logits_mask.fill_diagonal_(False)

    positives = labels.unsqueeze(0) == labels.unsqueeze(1)
    positives = positives & logits_mask

    if positives.sum() == 0:
        return torch.tensor(0.0, dtype=embeddings.dtype, device=embeddings.device)

    logits = sim_matrix.masked_select(logits_mask).view(embeddings.shape[0], -1)
    targets = positives.masked_select(logits_mask).view(embeddings.shape[0], -1)
    target_weights = targets.float()
    row_sums = target_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    target_weights = target_weights / row_sums

    log_probs = F.log_softmax(logits, dim=-1)
    loss = -(target_weights * log_probs).sum(dim=-1)
    return loss.mean()
