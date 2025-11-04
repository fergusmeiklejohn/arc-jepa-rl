"""Relational aggregation layers for object-centric encoders."""

from __future__ import annotations

try:  # pragma: no cover - optional torch dependency
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = object  # type: ignore


class RelationalModuleUnavailable(RuntimeError):
    pass


if torch is not None:  # pragma: no branch

    class RelationalAggregator(nn.Module):
        """Lightweight graph-style aggregator using adjacency matrices."""

        def __init__(self, hidden_dim: int, *, add_self_loops: bool = True) -> None:
            super().__init__()
            if hidden_dim <= 0:
                raise ValueError("hidden_dim must be positive")

            self.add_self_loops = add_self_loops
            self.message_linear = nn.Linear(hidden_dim, hidden_dim)
            self.update_linear = nn.Linear(hidden_dim * 2, hidden_dim)
            self.activation = nn.GELU()

        def forward(
            self,
            embeddings: "torch.Tensor",
            adjacency: "torch.Tensor",
            mask: "torch.Tensor",
        ) -> "torch.Tensor":
            if embeddings.dim() != 3:
                raise ValueError("embeddings must have shape (batch, objects, hidden_dim)")
            if adjacency.shape != embeddings.shape[:2] + (embeddings.size(1),):
                raise ValueError("adjacency must have shape (batch, objects, objects)")
            if mask.shape != embeddings.shape[:2]:
                raise ValueError("mask must have shape (batch, objects)")

            B, N, _ = embeddings.shape

            adjacency = adjacency.clone()
            if self.add_self_loops:
                adjacency = adjacency + torch.eye(N, device=adjacency.device).unsqueeze(0)

            mask_expanded = mask.unsqueeze(-1)
            adjacency = adjacency * mask.unsqueeze(-1) * mask.unsqueeze(1)

            degrees = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
            weights = adjacency / degrees

            messages = torch.matmul(weights, self.message_linear(embeddings))

            combined = torch.cat([embeddings, messages], dim=-1)
            updated = self.update_linear(combined)
            updated = self.activation(updated)

            # Preserve masked slots by zeroing them out.
            updated = updated * mask_expanded
            return updated

else:  # pragma: no cover

    class RelationalAggregator:  # type: ignore[misc]
        def __init__(self, *args, **kwargs) -> None:  # noqa: D401
            raise RelationalModuleUnavailable("PyTorch is required for RelationalAggregator")
