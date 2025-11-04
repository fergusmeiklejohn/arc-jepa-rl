"""Simple training loop scaffold for the object-centric JEPA encoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from arcgen import Grid

from .dataset import GridPairBatch, InMemoryGridPairDataset
from .trainer import ObjectCentricJEPATrainer

try:  # pragma: no cover - optional
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


def _ensure_torch_available() -> None:
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required for the JEPA training loop")


@dataclass(frozen=True)
class OptimizerConfig:
    lr: float = 1e-4
    weight_decay: float = 0.0

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None) -> "OptimizerConfig":
        if data is None:
            return cls()
        return cls(
            lr=float(data.get("lr", cls.lr)),
            weight_decay=float(data.get("weight_decay", cls.weight_decay)),
        )


@dataclass(frozen=True)
class TrainStepResult:
    loss: float
    encoded_context: torch.Tensor
    encoded_target: torch.Tensor


class ObjectCentricJEPAExperiment:
    """Minimal experiment wrapper that performs optimisation steps."""

    def __init__(
        self,
        config: Mapping[str, object],
        *,
        device: str | torch.device | None = None,
    ) -> None:
        _ensure_torch_available()

        self.config = config
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.trainer = ObjectCentricJEPATrainer(config)
        self.trainer.encoder.to(self.device)

        opt_cfg = OptimizerConfig.from_mapping(config.get("optimizer"))
        self.optimizer = torch.optim.Adam(
            self.trainer.encoder.parameters(),
            lr=opt_cfg.lr,
            weight_decay=opt_cfg.weight_decay,
        )

    def encode_batch(
        self,
        context_grids: Sequence[Grid],
        target_grids: Sequence[Grid],
    ):
        return self.trainer.encode_batch(
            context_grids,
            target_grids,
            device=self.device,
        )

    def _alignment_loss(self, context_embeddings: torch.Tensor, target_embeddings: torch.Tensor) -> torch.Tensor:
        # Simple placeholder: align mean pooled embeddings.
        context_vec = context_embeddings.mean(dim=(1, 2))
        target_vec = target_embeddings.mean(dim=(1, 2))
        return torch.mean((context_vec - target_vec) ** 2)

    def train_step(
        self,
        context_grids: Sequence[Grid],
        target_grids: Sequence[Grid],
    ) -> TrainStepResult:
        batch = self.encode_batch(context_grids, target_grids)

        embeddings_context = batch.context.embeddings
        embeddings_target = batch.target.embeddings

        loss = self._alignment_loss(embeddings_context, embeddings_target)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        return TrainStepResult(
            loss=float(loss.detach().cpu().item()),
            encoded_context=embeddings_context.detach().cpu(),
            encoded_target=embeddings_target.detach().cpu(),
        )

    def train_epoch(
        self,
        dataset: Iterable[GridPairBatch],
    ) -> float:
        total_loss = 0.0
        batches = 0
        for batch in dataset:
            result = self.train_step(batch.context, batch.target)
            total_loss += result.loss
            batches += 1

        if batches == 0:
            raise ValueError("dataset yielded no batches")

        return total_loss / batches

    def train(
        self,
        dataset: Iterable[GridPairBatch],
        epochs: int = 1,
    ) -> Sequence[float]:
        if epochs <= 0:
            raise ValueError("epochs must be positive")

        losses = []
        for _ in range(epochs):
            epoch_loss = self.train_epoch(dataset)
            losses.append(epoch_loss)
        return losses
