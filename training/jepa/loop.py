"""Simple training loop scaffold for the object-centric JEPA encoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from arcgen import Grid

from .dataset import GridPairBatch, InMemoryGridPairDataset
from .trainer import ObjectCentricJEPATrainer

from training.modules.projection import InfoNCEQueue, ProjectionHead

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
class InfoNCELossConfig:
    temperature_init: float = 0.07
    temperature_bounds: Sequence[float] = (0.03, 0.3)
    learnable_temperature: bool = True
    queue_size: int = 4096
    projection_dim: int = 256
    projection_layers: int = 2
    projection_activation: str = "relu"

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None) -> "InfoNCELossConfig":
        if data is None:
            return cls()
        bounds = data.get("temperature_bounds", cls.temperature_bounds)
        if not isinstance(bounds, Sequence) or len(bounds) != 2:
            raise ValueError("temperature_bounds must be a sequence of length 2")
        return cls(
            temperature_init=float(data.get("temperature_init", cls.temperature_init)),
            temperature_bounds=bounds,
            learnable_temperature=bool(data.get("learnable_temperature", cls.learnable_temperature)),
            queue_size=int(data.get("queue_size", cls.queue_size)),
            projection_dim=int(data.get("projection_dim", cls.projection_dim)),
            projection_layers=int(data.get("projection_layers", cls.projection_layers)),
            projection_activation=str(data.get("projection_activation", cls.projection_activation)),
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

        self.loss_config = InfoNCELossConfig.from_mapping(config.get("loss"))

        embedding_dim = self.trainer.encoder_config.hidden_dim
        self.projection_head = ProjectionHead(
            input_dim=embedding_dim,
            output_dim=self.loss_config.projection_dim,
            layers=self.loss_config.projection_layers,
            activation=self.loss_config.projection_activation,
        ).to(self.device)

        self.queue = InfoNCEQueue(
            embedding_dim=self.loss_config.projection_dim,
            queue_size=self.loss_config.queue_size,
        ).to(self.device)

        temperature_value = torch.tensor(self.loss_config.temperature_init, dtype=torch.float32)
        if self.loss_config.learnable_temperature:
            self.log_temperature = torch.nn.Parameter(torch.log(temperature_value))
        else:
            self.register_buffer("log_temperature", torch.log(temperature_value))

        opt_cfg = OptimizerConfig.from_mapping(config.get("optimizer"))
        params = list(self.trainer.encoder.parameters()) + list(self.projection_head.parameters())
        if self.loss_config.learnable_temperature:
            params.append(self.log_temperature)

        self.optimizer = torch.optim.Adam(
            params,
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

    def _masked_mean(self, embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.unsqueeze(-1)
        summed = (embeddings * mask).sum(dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1.0)
        return summed / counts

    def _info_nce_loss(self, context_proj: torch.Tensor, target_proj: torch.Tensor) -> torch.Tensor:
        negatives_queue = self.queue.get_negatives()

        logits_pos = torch.sum(context_proj * target_proj, dim=-1, keepdim=True)

        logits_inbatch = context_proj @ target_proj.t()
        batch_size = context_proj.size(0)
        eye = torch.eye(batch_size, device=logits_inbatch.device, dtype=torch.bool)
        logits_inbatch = logits_inbatch[~eye].view(batch_size, -1) if batch_size > 1 else logits_inbatch.new_empty((batch_size, 0))

        logits_queue = (
            context_proj @ negatives_queue.t() if negatives_queue.size(0) > 0 else logits_pos.new_empty((batch_size, 0))
        )

        logits = torch.cat([logits_pos, logits_inbatch, logits_queue], dim=1)

        temperature = torch.exp(self.log_temperature)
        min_temp, max_temp = self.loss_config.temperature_bounds
        temperature = torch.clamp(temperature, min=min_temp, max=max_temp)

        logits = logits / temperature
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

        loss = torch.nn.functional.cross_entropy(logits, labels)
        return loss

    def train_step(
        self,
        context_grids: Sequence[Grid],
        target_grids: Sequence[Grid],
    ) -> TrainStepResult:
        batch = self.encode_batch(context_grids, target_grids)

        embeddings_context = batch.context.embeddings
        embeddings_target = batch.target.embeddings
        mask_context = batch.context.mask.to(self.device)
        mask_target = batch.target.mask.to(self.device)

        context_repr = self._masked_mean(embeddings_context, mask_context)
        target_repr = self._masked_mean(embeddings_target, mask_target)

        context_proj = self.projection_head(context_repr)
        target_proj = self.projection_head(target_repr)

        loss = self._info_nce_loss(context_proj, target_proj)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            self.queue.enqueue(target_proj.detach())

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
