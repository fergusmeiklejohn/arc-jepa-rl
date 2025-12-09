"""Projection heads and InfoNCE utilities for JEPA training."""

from __future__ import annotations

try:  # pragma: no cover - torch optional
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = object  # type: ignore
    F = object  # type: ignore


class ProjectionModuleUnavailable(RuntimeError):
    pass


if torch is not None:  # pragma: no branch

    class ProjectionHead(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, layers: int = 2, activation: str = "relu") -> None:
            super().__init__()

            if layers < 1:
                raise ValueError("layers must be >= 1")

            acts: dict[str, nn.Module] = {
                "relu": nn.ReLU(),
                "gelu": nn.GELU(),
                "tanh": nn.Tanh(),
            }
            if activation not in acts:
                raise ValueError("activation must be one of 'relu', 'gelu', 'tanh'")

            modules = []
            hidden_dim = output_dim
            modules.append(nn.Linear(input_dim, hidden_dim))
            modules.append(acts[activation])
            for _ in range(layers - 1):
                modules.append(nn.Linear(hidden_dim, hidden_dim))
                modules.append(acts[activation])
            self.proj = nn.Sequential(*modules)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.proj(x)
            return F.normalize(out, dim=-1)


    class JEPAPredictor(nn.Module):
        """Predictor network for I-JEPA style asymmetric prediction.

        The predictor transforms context representations to predict target
        representations. This is the key component that prevents collapse
        in JEPA - the predictor must do actual work since context is masked
        and cannot simply copy to match the target.

        Architecture follows I-JEPA: narrow MLP with skip connection.
        """

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int | None = None,
            output_dim: int | None = None,
            depth: int = 4,
            activation: str = "gelu",
        ) -> None:
            super().__init__()

            if hidden_dim is None:
                hidden_dim = input_dim
            if output_dim is None:
                output_dim = input_dim

            self.input_dim = input_dim
            self.output_dim = output_dim

            acts: dict[str, nn.Module] = {
                "relu": nn.ReLU(),
                "gelu": nn.GELU(),
                "tanh": nn.Tanh(),
            }
            if activation not in acts:
                raise ValueError(f"activation must be one of {list(acts.keys())}")

            # Input projection
            self.input_proj = nn.Linear(input_dim, hidden_dim)

            # Transformer-like blocks (simplified)
            layers = []
            for _ in range(depth):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.LayerNorm(hidden_dim))
                layers.append(acts[activation])
            self.layers = nn.Sequential(*layers)

            # Output projection
            self.output_proj = nn.Linear(hidden_dim, output_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Predict target representation from context representation."""
            h = self.input_proj(x)
            h = self.layers(h)
            out = self.output_proj(h)
            # Don't normalize - we want to match the target embedding directly
            return out


    class InfoNCEQueue(nn.Module):
        def __init__(self, embedding_dim: int, queue_size: int) -> None:
            super().__init__()
            if embedding_dim <= 0:
                raise ValueError("embedding_dim must be positive")
            if queue_size <= 0:
                raise ValueError("queue_size must be positive")

            self.embedding_dim = embedding_dim
            self.queue_size = queue_size

            self.register_buffer("queue", torch.zeros(queue_size, embedding_dim))
            self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))
            self.register_buffer("filled", torch.zeros(1, dtype=torch.long))

        @torch.no_grad()
        def enqueue(self, embeddings: torch.Tensor) -> None:
            embeddings = F.normalize(embeddings, dim=-1)
            batch_size = embeddings.shape[0]
            ptr = int(self.ptr)

            if batch_size >= self.queue_size:
                self.queue.copy_(embeddings[-self.queue_size :])
                self.ptr[0] = 0
                self.filled[0] = self.queue_size
                return

            end = ptr + batch_size
            if end <= self.queue_size:
                self.queue[ptr:end] = embeddings
            else:
                first = self.queue_size - ptr
                self.queue[ptr:] = embeddings[:first]
                self.queue[: end - self.queue_size] = embeddings[first:]
            self.ptr[0] = (ptr + batch_size) % self.queue_size
            self.filled[0] = torch.clamp(self.filled + batch_size, max=self.queue_size)

        def get_negatives(self) -> torch.Tensor:
            count = int(self.filled.item())
            if count == 0:
                return self.queue.new_empty((0, self.embedding_dim))
            return self.queue[:count].clone().detach()

else:  # pragma: no cover

    class ProjectionHead:  # type: ignore[misc]
        def __init__(self, *args, **kwargs) -> None:  # noqa: D401
            raise ProjectionModuleUnavailable("PyTorch is required for ProjectionHead")

    class JEPAPredictor:  # type: ignore[misc]
        def __init__(self, *args, **kwargs) -> None:  # noqa: D401
            raise ProjectionModuleUnavailable("PyTorch is required for JEPAPredictor")

    class InfoNCEQueue:  # type: ignore[misc]
        def __init__(self, *args, **kwargs) -> None:  # noqa: D401
            raise ProjectionModuleUnavailable("PyTorch is required for InfoNCEQueue")
