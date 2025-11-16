"""Simple training loop scaffold for the object-centric JEPA encoder."""

from __future__ import annotations

import copy
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence
import warnings

from arcgen import Grid

from .dataset import GridPairBatch, InMemoryGridPairDataset, TokenizedPairBatch
from .invariance import (
    InvarianceLossConfig,
    color_permuted_batch,
    symmetry_flipped_batch,
    translated_batch,
)
from .relational_loss import RelationalConsistencyConfig, relational_consistency_loss
from .trainer import ObjectCentricJEPATrainer
from .object_pipeline import ObjectCentricEncoding, ObjectCentricJEPAEncoder, ObjectTokenBatch

from training.modules.projection import InfoNCEQueue, ProjectionHead

try:  # pragma: no cover - optional
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

if torch is not None:  # pragma: no branch
    try:  # pragma: no cover - CUDA optional on CI
        from torch.cuda.amp import GradScaler
        from torch.cuda.amp import autocast as cuda_autocast
    except Exception:  # pragma: no cover
        GradScaler = None  # type: ignore
        cuda_autocast = None  # type: ignore
else:  # pragma: no cover
    GradScaler = None  # type: ignore
    cuda_autocast = None  # type: ignore


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
    use_target_encoder: bool = False
    target_ema_decay: float = 0.99

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
            use_target_encoder=bool(data.get("use_target_encoder", cls.use_target_encoder)),
            target_ema_decay=float(data.get("target_ema_decay", cls.target_ema_decay)),
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
        self._use_non_blocking = self.device.type != "cpu"
        data_cfg = config.get("data", {})
        if isinstance(data_cfg, Mapping):
            context_length_value = data_cfg.get("context_length", data_cfg.get("context_window", 3))
        else:
            context_length_value = 3
        self.context_length = int(context_length_value)
        if self.context_length <= 0:
            raise ValueError("context length must be positive")

        training_cfg = config.get("training", {})
        if isinstance(training_cfg, Mapping):
            self.grad_accum_steps = max(1, int(training_cfg.get("grad_accum_steps", 1)))
            self._amp_requested = bool(training_cfg.get("amp", False))
        else:
            self.grad_accum_steps = 1
            self._amp_requested = False

        self._amp_enabled = False
        self._grad_scaler: GradScaler | None = None
        self._amp_dtype = torch.float16
        self._init_amp_state()

        self.trainer = ObjectCentricJEPATrainer(config)
        self.trainer.encoder.to(self.device)

        self.loss_config = InfoNCELossConfig.from_mapping(config.get("loss"))
        if not 0.0 < self.loss_config.target_ema_decay <= 1.0:
            raise ValueError("target_ema_decay must be in (0, 1]")
        self._use_target_encoder = self.loss_config.use_target_encoder
        self.invariance_config = InvarianceLossConfig.from_mapping(config.get("invariance"))
        self.relational_config = RelationalConsistencyConfig.from_mapping(config.get("relational_loss"))

        embedding_dim = self.trainer.encoder_config.hidden_dim
        self.projection_head = ProjectionHead(
            input_dim=embedding_dim,
            output_dim=self.loss_config.projection_dim,
            layers=self.loss_config.projection_layers,
            activation=self.loss_config.projection_activation,
        ).to(self.device)
        self._target_encoder = None
        self._target_object_encoder = None
        self._target_projection_head = None
        if self._use_target_encoder:
            self._build_target_network()

        self.queue = InfoNCEQueue(
            embedding_dim=self.loss_config.projection_dim,
            queue_size=self.loss_config.queue_size,
        ).to(self.device)

        self._temperature_bounds = (
            float(self.loss_config.temperature_bounds[0]),
            float(self.loss_config.temperature_bounds[1]),
        )

        temperature_value = torch.tensor(
            self.loss_config.temperature_init,
            dtype=torch.float32,
            device=self.device,
        )
        if self.loss_config.learnable_temperature:
            self.log_temperature = torch.nn.Parameter(torch.log(temperature_value))
        else:
            self._log_temperature = torch.log(temperature_value).detach()

        opt_cfg = OptimizerConfig.from_mapping(config.get("optimizer"))
        params = list(self.trainer.encoder.parameters()) + list(self.projection_head.parameters())
        if self.loss_config.learnable_temperature:
            params.append(self.log_temperature)

        self.optimizer = torch.optim.Adam(
            params,
            lr=opt_cfg.lr,
            weight_decay=opt_cfg.weight_decay,
        )

    def _masked_mean(self, embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.unsqueeze(-1)
        summed = (embeddings * mask).sum(dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1.0)
        return summed / counts

    def _context_length_from_sequences(self, context_sequences: Sequence[Sequence[Grid]]) -> int:
        if not context_sequences:
            raise ValueError("context_sequences must contain at least one entry")
        lengths = {len(sequence) for sequence in context_sequences}
        if len(lengths) != 1:
            raise ValueError("all context sequences must have the same length")
        context_length = lengths.pop()
        if context_length != self.context_length:
            raise ValueError(
                f"context length mismatch: expected {self.context_length}, received {context_length}",
            )
        if context_length <= 0:
            raise ValueError("context sequences must have positive length")
        return context_length

    def _encode_context_sequences(
        self,
        context_sequences: Sequence[Sequence[Grid]],
    ) -> torch.Tensor:
        context_length = self._context_length_from_sequences(context_sequences)
        batch_size = len(context_sequences)
        flat_context: list[Grid] = [
            grid for sequence in context_sequences for grid in sequence
        ]
        encoding = self.trainer.object_encoder.encode(flat_context, device=self.device)
        per_grid_repr = self._masked_mean(encoding.embeddings, encoding.mask.to(self.device))
        reshaped = per_grid_repr.view(batch_size, context_length, -1)
        return reshaped.mean(dim=1)

    def _info_nce_loss(self, context_proj: torch.Tensor, target_proj: torch.Tensor) -> torch.Tensor:
        if self.loss_config.learnable_temperature:
            log_temperature = self.log_temperature
        else:
            log_temperature = self._log_temperature

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

        temperature = torch.exp(log_temperature)
        min_temp, max_temp = self._temperature_bounds
        temperature = torch.clamp(temperature, min=min_temp, max=max_temp)

        logits = logits / temperature
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

        loss = torch.nn.functional.cross_entropy(logits.float(), labels)
        return loss

    def train_step(
        self,
        context_sequences: Sequence[Sequence[Grid]],
        target_grids: Sequence[Grid],
    ) -> TrainStepResult:
        if not target_grids:
            raise ValueError("target_grids must contain at least one entry")

        loss, context_repr, target_repr, target_proj = self._forward_from_grids(context_sequences, target_grids)

        self.optimizer.zero_grad(set_to_none=True)
        self._step_optimizer(loss)
        if self._use_target_encoder:
            self._update_target_network()

        with torch.no_grad():
            self.queue.enqueue(self._prepare_queue_projection(target_proj))

        return TrainStepResult(
            loss=float(loss.detach().float().cpu().item()),
            encoded_context=self._tensor_to_cpu(context_repr),
            encoded_target=self._tensor_to_cpu(target_repr),
        )

    def train_step_tokenized(self, batch: TokenizedPairBatch) -> TrainStepResult:
        token_batch = batch.to(self.device, non_blocking=self._use_non_blocking)
        loss, context_repr, target_repr, target_proj = self._forward_from_token_batch(token_batch)

        self.optimizer.zero_grad(set_to_none=True)
        self._step_optimizer(loss)
        if self._use_target_encoder:
            self._update_target_network()

        with torch.no_grad():
            self.queue.enqueue(self._prepare_queue_projection(target_proj))

        return TrainStepResult(
            loss=float(loss.detach().float().cpu().item()),
            encoded_context=self._tensor_to_cpu(context_repr),
            encoded_target=self._tensor_to_cpu(target_repr),
        )

    def _forward_from_grids(
        self,
        context_sequences: Sequence[Sequence[Grid]],
        target_grids: Sequence[Grid],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with self._autocast_context():
            context_repr = self._encode_context_sequences(context_sequences)
            context_proj = self.projection_head(context_repr)

            if self._use_target_encoder:
                with torch.no_grad():
                    target_repr = self._encode_grids(target_grids, object_encoder=self._target_object_encoder)
                target_proj = self._target_projection_head(target_repr)
            else:
                target_repr = self._encode_grids(target_grids)
                target_proj = self.projection_head(target_repr)
            loss = self._info_nce_loss(context_proj, target_proj)
        return loss, context_repr, target_repr, target_proj

    def _forward_from_token_batch(
        self,
        batch: TokenizedPairBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with self._autocast_context():
            context_repr, context_encoding = self._encode_tokenized_context(batch, return_encoding=True)  # type: ignore[assignment]
            context_proj = self.projection_head(context_repr)
            if self._use_target_encoder:
                with torch.no_grad():
                    target_repr, target_encoding = self._encode_tokenized_target(
                        batch,
                        object_encoder=self._target_object_encoder,
                        return_encoding=True,
                    )  # type: ignore[assignment]
                target_proj = self._target_projection_head(target_repr)
            else:
                target_repr, target_encoding = self._encode_tokenized_target(batch, return_encoding=True)  # type: ignore[assignment]
                target_proj = self.projection_head(target_repr)
            loss = self._info_nce_loss(context_proj, target_proj)
        invariance_loss = self._token_invariance_loss(batch, context_repr, target_repr)
        if invariance_loss is not None:
            loss = loss + invariance_loss
        relational_loss = self._relational_consistency_loss(
            context_encoding,
            target_encoding,
            batch_size=batch.context_features.size(0),
            context_length=batch.context_length,
        )
        if relational_loss is not None:
            loss = loss + relational_loss
        return loss, context_repr, target_repr, target_proj

    def _encode_tokenized_context(
        self,
        batch: TokenizedPairBatch,
        *,
        object_encoder: ObjectCentricJEPAEncoder | None = None,
        return_encoding: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ObjectCentricEncoding]:
        batch_size = batch.context_features.size(0)
        context_length = batch.context_length
        max_objects = batch.context_features.size(2)
        feature_dim = batch.context_features.size(3)

        flattened_features = batch.context_features.reshape(batch_size * context_length, max_objects, feature_dim)
        flattened_mask = batch.context_mask.reshape(batch_size * context_length, max_objects)
        flattened_adj = batch.context_adjacency.reshape(batch_size * context_length, max_objects, max_objects)

        token_batch = ObjectTokenBatch(
            features=flattened_features,
            mask=flattened_mask,
            adjacency=flattened_adj,
        )
        encoder = object_encoder or self.trainer.object_encoder
        encoding = encoder.encode_tokens(
            token_batch,
            device=self.device,
            non_blocking=self._use_non_blocking,
        )
        per_grid_repr = self._masked_mean(encoding.embeddings, encoding.mask.to(self.device))
        reshaped = per_grid_repr.view(batch_size, context_length, -1)
        aggregated = reshaped.mean(dim=1)
        if return_encoding:
            return aggregated, encoding
        return aggregated

    def _encode_tokenized_target(
        self,
        batch: TokenizedPairBatch,
        *,
        object_encoder: ObjectCentricJEPAEncoder | None = None,
        return_encoding: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ObjectCentricEncoding]:
        token_batch = ObjectTokenBatch(
            features=batch.target_features,
            mask=batch.target_mask,
            adjacency=batch.target_adjacency,
        )
        encoder = object_encoder or self.trainer.object_encoder
        encoding = encoder.encode_tokens(
            token_batch,
            device=self.device,
            non_blocking=self._use_non_blocking,
        )
        aggregated = self._masked_mean(encoding.embeddings, encoding.mask.to(self.device))
        if return_encoding:
            return aggregated, encoding
        return aggregated

    def train_epoch(
        self,
        dataset: Iterable[GridPairBatch | TokenizedPairBatch],
    ) -> float:
        total_loss = 0.0
        batches = 0
        pending_losses: list[torch.Tensor] = []
        pending_targets: list[torch.Tensor] = []

        def _apply_pending() -> None:
            if not pending_losses:
                return
            self.optimizer.zero_grad(set_to_none=True)
            combined = torch.stack(pending_losses).mean()
            self._step_optimizer(combined)
            if self._use_target_encoder:
                self._update_target_network()
            with torch.no_grad():
                for proj in pending_targets:
                    self.queue.enqueue(proj)
            pending_losses.clear()
            pending_targets.clear()

        for batch in dataset:
            if isinstance(batch, GridPairBatch):
                loss_tensor, _, _, target_proj = self._forward_from_grids(batch.context, batch.target)
            elif isinstance(batch, TokenizedPairBatch):
                moved = batch.to(self.device, non_blocking=self._use_non_blocking)
                loss_tensor, _, _, target_proj = self._forward_from_token_batch(moved)
            else:
                raise TypeError(f"Unsupported batch type: {type(batch)}")

            total_loss += float(loss_tensor.detach().float().cpu().item())
            pending_losses.append(loss_tensor)
            pending_targets.append(self._prepare_queue_projection(target_proj))
            batches += 1

            if len(pending_losses) == self.grad_accum_steps:
                _apply_pending()

        if batches == 0:
            raise ValueError("dataset yielded no batches")

        if pending_losses:
            _apply_pending()

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

    def _encode_grids(
        self,
        grids: Sequence[Grid],
        *,
        object_encoder: ObjectCentricJEPAEncoder | None = None,
    ) -> torch.Tensor:
        encoder = object_encoder or self.trainer.object_encoder
        encoding = encoder.encode(grids, device=self.device)
        return self._masked_mean(encoding.embeddings, encoding.mask.to(self.device))

    def _build_target_network(self) -> None:
        encoder_copy = copy.deepcopy(self.trainer.encoder).to(self.device)
        for param in encoder_copy.parameters():
            param.requires_grad_(False)
        self._target_encoder = encoder_copy
        self._target_object_encoder = ObjectCentricJEPAEncoder(encoder_copy, self.trainer.tokenizer_config)

        projection_copy = copy.deepcopy(self.projection_head).to(self.device)
        for param in projection_copy.parameters():
            param.requires_grad_(False)
        self._target_projection_head = projection_copy

    def _update_target_network(self) -> None:
        if not self._use_target_encoder or self._target_encoder is None or self._target_projection_head is None:
            return
        decay = self.loss_config.target_ema_decay
        self._ema_update(self._target_encoder, self.trainer.encoder, decay)
        self._ema_update(self._target_projection_head, self.projection_head, decay)

    def _ema_update(self, target_module: torch.nn.Module, online_module: torch.nn.Module, decay: float) -> None:
        with torch.no_grad():
            for target_param, online_param in zip(target_module.parameters(), online_module.parameters()):
                target_param.data.mul_(decay).add_(online_param.data, alpha=1 - decay)

    def _init_amp_state(self) -> None:
        if torch is None:
            return
        if not self._amp_requested:
            return
        if self.device.type != "cuda" or not torch.cuda.is_available():
            warnings.warn(
                "AMP requested but CUDA device unavailable; falling back to FP32.",
                RuntimeWarning,
                stacklevel=2,
            )
            return
        if GradScaler is None or cuda_autocast is None:
            warnings.warn(
                "torch.cuda.amp not available; disabling AMP despite training.amp=true",
                RuntimeWarning,
                stacklevel=2,
            )
            return
        self._amp_enabled = True
        self._grad_scaler = GradScaler()

    def _autocast_context(self):
        if not self._amp_enabled or cuda_autocast is None:
            return nullcontext()
        return cuda_autocast(dtype=self._amp_dtype)

    def _step_optimizer(self, loss: torch.Tensor) -> None:
        if self._grad_scaler is not None:
            self._grad_scaler.scale(loss).backward()
            self._grad_scaler.step(self.optimizer)
            self._grad_scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

    def _prepare_queue_projection(self, target_proj: torch.Tensor) -> torch.Tensor:
        proj = target_proj.detach()
        if proj.dtype != torch.float32:
            proj = proj.float()
        return proj

    def _tensor_to_cpu(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach().float().cpu()

    def _token_invariance_loss(
        self,
        batch: TokenizedPairBatch,
        context_repr: torch.Tensor,
        target_repr: torch.Tensor,
    ) -> torch.Tensor | None:
        cfg = self.invariance_config
        if not cfg.enabled:
            return None

        penalties: list[torch.Tensor] = []

        if cfg.color_weight > 0.0:
            augmented = color_permuted_batch(batch, cfg)
            penalties.append(cfg.color_weight * self._invariance_distance(augmented, context_repr, target_repr))

        if cfg.translation_weight > 0.0 and cfg.translation_max_delta > 0.0:
            translated = translated_batch(batch, cfg)
            penalties.append(cfg.translation_weight * self._invariance_distance(translated, context_repr, target_repr))

        if cfg.symmetry_weight > 0.0 and cfg.symmetry_modes:
            flipped = symmetry_flipped_batch(batch, cfg)
            penalties.append(cfg.symmetry_weight * self._invariance_distance(flipped, context_repr, target_repr))

        if not penalties:
            return None
        return torch.stack(penalties).sum()

    def _invariance_distance(
        self,
        batch: TokenizedPairBatch,
        reference_context: torch.Tensor,
        reference_target: torch.Tensor,
    ) -> torch.Tensor:
        context_repr = self._encode_tokenized_context(batch)
        target_repr = self._encode_tokenized_target(batch)
        return torch.nn.functional.mse_loss(context_repr, reference_context) + torch.nn.functional.mse_loss(
            target_repr,
            reference_target,
        )

    def _relational_consistency_loss(
        self,
        context_encoding: ObjectCentricEncoding,
        target_encoding: ObjectCentricEncoding,
        *,
        batch_size: int,
        context_length: int,
    ) -> torch.Tensor | None:
        if not self.relational_config.enabled:
            return None
        return relational_consistency_loss(
            context_encoding,
            target_encoding,
            batch_size=batch_size,
            context_length=context_length,
            config=self.relational_config,
        )
