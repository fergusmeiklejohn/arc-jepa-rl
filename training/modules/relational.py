"""Relational attention layers for object-centric encoders."""

from __future__ import annotations

import math

try:  # pragma: no cover - optional torch dependency
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = object  # type: ignore
    F = None  # type: ignore


class RelationalModuleUnavailable(RuntimeError):
    pass


if torch is not None:  # pragma: no branch

    class RelationalAttentionLayer(nn.Module):
        """Single relational self-attention block restricted by adjacency."""

        def __init__(
            self,
            hidden_dim: int,
            *,
            num_heads: int = 4,
            dropout: float = 0.0,
            add_self_loops: bool = True,
        ) -> None:
            super().__init__()
            if hidden_dim <= 0:
                raise ValueError("hidden_dim must be positive")
            if num_heads <= 0:
                raise ValueError("num_heads must be positive")
            if hidden_dim % num_heads != 0:
                raise ValueError("hidden_dim must be divisible by num_heads")
            if not 0.0 <= dropout <= 1.0:
                raise ValueError("dropout must lie in [0, 1]")

            self.add_self_loops = add_self_loops
            self.num_heads = num_heads
            self.head_dim = hidden_dim // num_heads

            self.q_proj = nn.Linear(hidden_dim, hidden_dim)
            self.k_proj = nn.Linear(hidden_dim, hidden_dim)
            self.v_proj = nn.Linear(hidden_dim, hidden_dim)
            self.out_proj = nn.Linear(hidden_dim, hidden_dim)

            self.dropout = nn.Dropout(dropout)
            self.ffn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )

            self.norm1 = nn.LayerNorm(hidden_dim)
            self.norm2 = nn.LayerNorm(hidden_dim)

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

            residual = embeddings
            x = self.norm1(embeddings)

            attn_out = self._attention(x, adjacency, mask)
            x = residual + self.dropout(attn_out)

            residual_ff = x
            x = self.norm2(x)
            x = residual_ff + self.dropout(self.ffn(x))

            # Zero-out masked slots.
            return x * mask.unsqueeze(-1)

        def _attention(
            self,
            embeddings: "torch.Tensor",
            adjacency: "torch.Tensor",
            mask: "torch.Tensor",
        ) -> "torch.Tensor":
            B, N, hidden_dim = embeddings.shape

            q = self.q_proj(embeddings)
            k = self.k_proj(embeddings)
            v = self.v_proj(embeddings)

            q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, D)
            k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

            adjacency = adjacency.clone()
            if self.add_self_loops:
                eye = torch.eye(N, device=adjacency.device, dtype=adjacency.dtype)
                adjacency = adjacency + eye.unsqueeze(0)

            mask_bool = mask > 0.5
            adjacency = adjacency * mask_bool.unsqueeze(-1) * mask_bool.unsqueeze(-2)

            if self.add_self_loops:
                eye = torch.eye(N, device=adjacency.device, dtype=adjacency.dtype).unsqueeze(0)
                adjacency = adjacency + eye * (~mask_bool).unsqueeze(-1)

            adjacency = adjacency.clamp(max=1.0)
            attn_mask = adjacency > 0
            attn_logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)

            attn_logits = attn_logits.masked_fill(~attn_mask.unsqueeze(1), float("-inf"))

            attn_weights = F.softmax(attn_logits, dim=-1)
            attn_weights = attn_weights.masked_fill(~attn_mask.unsqueeze(1), 0.0)
            attn_weights = self.dropout(attn_weights)

            attended = torch.matmul(attn_weights, v)  # (B, H, N, D)
            attended = attended.transpose(1, 2).contiguous().view(B, N, hidden_dim)
            out = self.out_proj(attended)
            return out * mask.unsqueeze(-1)


    class RelationalAggregator(nn.Module):
        """Stack of relational attention layers operating on adjacency graphs."""

        def __init__(
            self,
            hidden_dim: int,
            *,
            num_layers: int = 2,
            num_heads: int = 4,
            dropout: float = 0.0,
            add_self_loops: bool = True,
        ) -> None:
            super().__init__()
            if num_layers <= 0:
                raise ValueError("num_layers must be positive")

            self.layers = nn.ModuleList(
                [
                    RelationalAttentionLayer(
                        hidden_dim,
                        num_heads=num_heads,
                        dropout=dropout,
                        add_self_loops=add_self_loops,
                    )
                    for _ in range(num_layers)
                ]
            )

        def forward(
            self,
            embeddings: "torch.Tensor",
            adjacency: "torch.Tensor",
            mask: "torch.Tensor",
        ) -> "torch.Tensor":
            output = embeddings
            for layer in self.layers:
                output = layer(output, adjacency, mask)
            return output

else:  # pragma: no cover

    class RelationalAggregator:  # type: ignore[misc]
        def __init__(self, *args, **kwargs) -> None:  # noqa: D401
            raise RelationalModuleUnavailable("PyTorch is required for RelationalAggregator")
