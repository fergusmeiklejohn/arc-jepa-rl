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

    class GraphAttentionBlock(nn.Module):
        """Relational attention block with residual connections."""

        def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
            super().__init__()
            self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.norm2 = nn.LayerNorm(hidden_dim)
            self.ffn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Dropout(dropout),
            )

        def forward(self, x: "torch.Tensor", attn_mask: "torch.Tensor") -> "torch.Tensor":
            attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask)
            x = self.norm1(x + attn_out)
            feed_forward = self.ffn(x)
            return self.norm2(x + feed_forward)

    class MetaJEPAModel(nn.Module):
        """Graph/attention encoder over rule family structures."""

        def __init__(
            self,
            vocabulary_size: int,
            stats_dim: int,
            *,
            hidden_dim: int = 128,
            embedding_dim: int = 64,
            dropout: float = 0.1,
            num_layers: int = 2,
            num_heads: int = 4,
        ) -> None:
            super().__init__()
            if vocabulary_size <= 0:
                raise ValueError("vocabulary_size must be positive")
            if stats_dim <= 0:
                raise ValueError("stats_dim must be positive")
            if hidden_dim <= 0 or embedding_dim <= 0:
                raise ValueError("hidden_dim and embedding_dim must be positive")
            if num_layers <= 0 or num_heads <= 0:
                raise ValueError("num_layers and num_heads must be positive")
            if not 0.0 <= dropout < 1.0:
                raise ValueError("dropout must lie in [0, 1)")

            self.vocab_size = vocabulary_size
            self.stats_dim = stats_dim
            self.hidden_dim = hidden_dim
            self.embedding_dim = embedding_dim
            self.sequence_length = vocabulary_size + 2  # [CLS] + stats + primitives
            self.num_heads = num_heads

            self.cls_token = nn.Parameter(torch.randn(hidden_dim))
            self.primitive_embeddings = nn.Parameter(torch.randn(vocabulary_size, hidden_dim))
            self.stats_proj = nn.Sequential(
                nn.Linear(stats_dim, hidden_dim),
                nn.GELU(),
            )
            self.layers = nn.ModuleList(
                GraphAttentionBlock(hidden_dim, num_heads, dropout) for _ in range(num_layers)
            )
            self.out_norm = nn.LayerNorm(hidden_dim)
            self.out_proj = nn.Linear(hidden_dim, embedding_dim)
            self.dropout = nn.Dropout(dropout)

        def forward(
            self,
            features: "torch.Tensor",
            *,
            adjacency: "torch.Tensor" | None = None,
        ) -> "torch.Tensor":
            if features.ndim != 2:
                raise ValueError("features must be 2D (batch, dim)")
            if features.shape[1] < self.vocab_size + self.stats_dim:
                raise ValueError("features have insufficient dimension for vocabulary + stats")

            primitive_counts = features[:, : self.vocab_size]
            stats = features[:, self.vocab_size : self.vocab_size + self.stats_dim]
            primitive_mask = (primitive_counts > 0).float()
            batch_size = features.size(0)

            primitive_tokens = primitive_counts.unsqueeze(-1) * self.primitive_embeddings.unsqueeze(0)
            primitive_tokens = self.dropout(primitive_tokens)

            stats_token = self.stats_proj(stats)
            cls_token = self.cls_token.unsqueeze(0).expand(batch_size, -1)

            tokens = torch.cat(
                [cls_token.unsqueeze(1), stats_token.unsqueeze(1), primitive_tokens],
                dim=1,
            )

            primitive_adj = self._prepare_primitive_adjacency(primitive_mask, adjacency)
            attn_mask = self._build_attention_mask(primitive_adj)

            x = tokens
            for layer in self.layers:
                x = layer(x, attn_mask)

            cls_repr = self.out_norm(x[:, 0])
            embedding = self.out_proj(cls_repr)
            return F.normalize(embedding, dim=-1)

        def _prepare_primitive_adjacency(
            self,
            primitive_mask: "torch.Tensor",
            adjacency: "torch.Tensor" | None,
        ) -> "torch.Tensor":
            device = primitive_mask.device
            batch_size = primitive_mask.shape[0]
            if adjacency is None:
                primitive_adj = primitive_mask.unsqueeze(2) * primitive_mask.unsqueeze(1)
            else:
                if adjacency.ndim != 3 or adjacency.shape[1] != self.vocab_size:
                    raise ValueError("adjacency must match (batch, vocab, vocab)")
                primitive_adj = adjacency.to(device)
                primitive_adj = 0.5 * (primitive_adj + primitive_adj.transpose(-1, -2))

            diag_indices = torch.arange(self.vocab_size, device=device)
            diag_values = torch.where(
                primitive_mask > 0,
                torch.ones_like(primitive_mask),
                torch.zeros_like(primitive_mask),
            )
            primitive_adj[:, diag_indices, diag_indices] = diag_values

            inactive = (primitive_mask == 0).unsqueeze(2) | (primitive_mask == 0).unsqueeze(1)
            primitive_adj = primitive_adj.masked_fill(inactive, 0.0)

            full_adj = primitive_adj.new_zeros((batch_size, self.sequence_length, self.sequence_length))
            full_adj[:, 2:, 2:] = primitive_adj
            full_adj[:, :2, :] = 1.0
            full_adj[:, :, :2] = 1.0

            eye = torch.eye(self.sequence_length, device=device).unsqueeze(0)
            full_adj = torch.clamp(full_adj + eye, max=1.0)
            return full_adj

        def _build_attention_mask(self, adjacency: "torch.Tensor") -> "torch.Tensor":
            batch_size, seq_len, _ = adjacency.shape
            mask = adjacency <= 0
            attn_mask = adjacency.new_zeros((batch_size, seq_len, seq_len))
            attn_mask.masked_fill_(mask, float("-inf"))
            attn_mask = attn_mask.repeat_interleave(self.num_heads, dim=0)
            return attn_mask

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
