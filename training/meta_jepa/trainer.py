"""High-level trainer for Meta-JEPA rule-family embeddings."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Mapping, Sequence, Tuple

try:  # pragma: no cover - torch optional
    import torch
    from torch.utils.data import DataLoader
    import torch.utils.data
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    DataLoader = object  # type: ignore
    F = None  # type: ignore

from arcgen import SyntheticTask

from .data import (
    PrimitiveVocabulary,
    RuleFamilyDataset,
    RuleFamilyExample,
    build_rule_family_dataset,
)
from .model import MetaJEPAModel, contrastive_loss
from training.utils import (
    EarlyStopping,
    EarlyStoppingConfig,
    GradientClippingConfig,
    LRSchedulerConfig,
    build_lr_scheduler,
)


def _ensure_torch() -> None:
    if torch is None:  # pragma: no cover - defensive
        raise RuntimeError("PyTorch is required for Meta-JEPA training")


@dataclass(frozen=True)
class TrainingConfig:
    lr: float = 1e-3
    batch_size: int = 16
    epochs: int = 5
    temperature: float = 0.1
    temperature_init: float = 0.1
    temperature_bounds: Tuple[float, float] = (0.03, 0.3)
    learnable_temperature: bool = False
    relational_weight: float = 0.0
    weight_decay: float = 0.0
    device: str = "cpu"
    validation_split: float = 0.0
    split_seed: int | None = None
    early_stopping: EarlyStoppingConfig = EarlyStoppingConfig()
    grad_clip: GradientClippingConfig = field(default_factory=GradientClippingConfig)
    lr_scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)


@dataclass
class TrainingResult:
    history: List[float]
    val_history: List[float]
    vocabulary: PrimitiveVocabulary
    dataset: RuleFamilyDataset
    temperature: float


class MetaJEPATrainer:
    """Convenience wrapper around the Meta-JEPA model."""

    def __init__(
        self,
        dataset: RuleFamilyDataset,
        vocabulary: PrimitiveVocabulary,
        *,
        hidden_dim: int = 128,
        embedding_dim: int = 64,
        dropout: float = 0.1,
        attn_heads: int = 4,
        attn_layers: int = 2,
        relational_decoder: bool = False,
    ) -> None:
        _ensure_torch()
        self.dataset = dataset
        self.vocabulary = vocabulary

        feature_dim = dataset.features.shape[1]
        vocab_size = len(vocabulary)
        stats_dim = feature_dim - vocab_size
        if stats_dim <= 0:
            raise ValueError("dataset features must include primitive stats")
        self.model = MetaJEPAModel(
            vocab_size,
            stats_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            dropout=dropout,
            num_heads=attn_heads,
            num_layers=attn_layers,
            relational_decoder=relational_decoder,
        )

    @classmethod
    def from_tasks(
        cls,
        tasks: Sequence[SyntheticTask],
        *,
        min_family_size: int = 1,
        model_kwargs: Mapping[str, object] | None = None,
    ) -> "MetaJEPATrainer":
        dataset, vocabulary, _ = build_rule_family_dataset(tasks, min_family_size=min_family_size)
        kwargs = dict(model_kwargs or {})
        return cls(dataset, vocabulary, **kwargs)

    def fit(self, config: TrainingConfig) -> TrainingResult:
        _ensure_torch()
        device = torch.device(config.device)
        model = self.model.to(device)

        min_temp, max_temp = config.temperature_bounds
        if min_temp <= 0 or max_temp <= 0:
            raise ValueError("temperature_bounds must be positive")
        if min_temp >= max_temp:
            raise ValueError("temperature_bounds must be an increasing pair")

        base_temperature = config.temperature_init if config.learnable_temperature else config.temperature
        if base_temperature <= 0:
            raise ValueError("temperature must be positive")

        temperature_value = torch.tensor(base_temperature, dtype=torch.float32, device=device)
        if config.learnable_temperature:
            log_temperature = torch.nn.Parameter(torch.log(temperature_value))
        else:
            log_temperature = torch.log(temperature_value).detach()

        params = list(model.parameters())
        if config.learnable_temperature:
            params.append(log_temperature)

        optimizer = torch.optim.AdamW(
            params,
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        train_dataset = self.dataset
        val_dataset = None
        if config.validation_split:
            val_ratio = float(config.validation_split)
            if not 0.0 < val_ratio < 1.0:
                raise ValueError("validation_split must be between 0 and 1")
            dataset_len = len(train_dataset)
            val_size = max(1, int(round(dataset_len * val_ratio)))
            if val_size >= dataset_len:
                raise ValueError("validation_split leaves no training samples")
            train_size = dataset_len - val_size
            generator = None
            if config.split_seed is not None:
                generator = torch.Generator()
                generator.manual_seed(int(config.split_seed))
            train_subset, val_subset = torch.utils.data.random_split(
                train_dataset,
                [train_size, val_size],
                generator=generator,
            )
            train_dataset = train_subset  # type: ignore[assignment]
            val_dataset = val_subset  # type: ignore[assignment]

        loader = DataLoader(
            train_dataset,
            batch_size=min(config.batch_size, len(train_dataset)),
            shuffle=True,
        )
        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=min(config.batch_size, len(val_dataset)),
                shuffle=False,
            )

        early_stopper = EarlyStopping(config.early_stopping)
        if config.early_stopping.enabled and val_loader is None:
            raise ValueError("validation_split must be > 0 when early stopping is enabled")
        try:
            steps_per_epoch = len(loader)
            total_steps = max(1, steps_per_epoch) * max(1, config.epochs)
        except Exception:
            total_steps = None
        scheduler = build_lr_scheduler(optimizer, config.lr_scheduler, total_steps=total_steps)
        grad_clip_cfg = config.grad_clip

        def current_temperature() -> "torch.Tensor":
            temperature = torch.exp(log_temperature)
            if config.learnable_temperature:
                temperature = torch.clamp(temperature, min=min_temp, max=max_temp)
            return temperature

        history: List[float] = []
        val_history: List[float] = []
        model.train()
        use_relational = bool(config.relational_weight > 0 and getattr(self.model, "relational_decoder", None))
        for epoch in range(1, config.epochs + 1):
            epoch_loss = 0.0
            batches = 0
            for features, labels, adjacency in loader:
                features = features.to(device)
                labels = labels.to(device)
                adjacency = adjacency.to(device)
                optimizer.zero_grad()
                embeddings, relational_logits = model(
                    features,
                    adjacency=adjacency,
                    return_relations=use_relational,
                )
                temperature = current_temperature()
                loss = contrastive_loss(embeddings, labels, temperature=temperature)
                if use_relational and relational_logits is not None:
                    relational_loss = F.binary_cross_entropy_with_logits(relational_logits, adjacency)
                    loss = loss + config.relational_weight * relational_loss
                if loss.requires_grad:
                    loss.backward()
                    if grad_clip_cfg.enabled:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            max_norm=grad_clip_cfg.max_norm,
                            norm_type=grad_clip_cfg.norm_type,
                        )
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    epoch_loss += float(loss.item())
                    batches += 1
            history.append(epoch_loss / max(1, batches))
            if val_loader is not None:
                val_loss = self._evaluate_validation(
                    model,
                    val_loader,
                    device,
                    current_temperature,
                    use_relational=use_relational,
                    relational_weight=config.relational_weight,
                )
                val_history.append(val_loss)
                if early_stopper.update(val_loss, step=epoch):
                    break

        final_temperature = float(current_temperature().detach().cpu().item())
        return TrainingResult(
            history=history,
            val_history=val_history,
            vocabulary=self.vocabulary,
            dataset=self.dataset,
            temperature=final_temperature,
        )

    def _evaluate_validation(
        self,
        model: MetaJEPAModel,
        loader: DataLoader,
        device: "torch.device",
        temperature_fn,
        *,
        use_relational: bool,
        relational_weight: float,
    ) -> float:
        state = model.training
        model.eval()
        total_loss = 0.0
        batches = 0
        with torch.no_grad():
            temperature = temperature_fn()
            for features, labels, adjacency in loader:
                features = features.to(device)
                labels = labels.to(device)
                adjacency = adjacency.to(device)
                embeddings, relational_logits = model(
                    features,
                    adjacency=adjacency,
                    return_relations=use_relational,
                )
                loss = contrastive_loss(embeddings, labels, temperature=temperature)
                if use_relational and relational_logits is not None:
                    relational_loss = F.binary_cross_entropy_with_logits(relational_logits, adjacency)
                    loss = loss + relational_weight * relational_loss
                total_loss += float(loss.item())
                batches += 1
        if state:
            model.train()
        return total_loss / max(1, batches)

    def encode(
        self,
        features: "torch.Tensor",
        *,
        adjacency: "torch.Tensor" | None = None,
        device: str | None = None,
    ) -> "torch.Tensor":
        _ensure_torch()
        self.model.eval()
        tensor = features.to(device or "cpu")
        adj = adjacency.to(device or "cpu") if adjacency is not None else None
        with torch.no_grad():
            embeddings, _ = self.model(tensor, adjacency=adj)
            return embeddings
