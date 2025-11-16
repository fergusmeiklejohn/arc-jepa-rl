"""High-level trainer for Meta-JEPA rule-family embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Sequence, Tuple

try:  # pragma: no cover - torch optional
    import torch
    from torch.utils.data import DataLoader
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    DataLoader = object  # type: ignore

from arcgen import SyntheticTask

from .data import (
    PrimitiveVocabulary,
    RuleFamilyDataset,
    RuleFamilyExample,
    build_rule_family_dataset,
)
from .model import MetaJEPAModel, contrastive_loss


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
    weight_decay: float = 0.0
    device: str = "cpu"


@dataclass
class TrainingResult:
    history: List[float]
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

        loader = DataLoader(
            self.dataset,
            batch_size=min(config.batch_size, len(self.dataset)),
            shuffle=True,
        )

        def current_temperature() -> "torch.Tensor":
            temperature = torch.exp(log_temperature)
            if config.learnable_temperature:
                temperature = torch.clamp(temperature, min=min_temp, max=max_temp)
            return temperature

        history: List[float] = []
        model.train()
        for _ in range(config.epochs):
            epoch_loss = 0.0
            batches = 0
            for features, labels, adjacency in loader:
                features = features.to(device)
                labels = labels.to(device)
                adjacency = adjacency.to(device)
                optimizer.zero_grad()
                embeddings = model(features, adjacency=adjacency)
                temperature = current_temperature()
                loss = contrastive_loss(embeddings, labels, temperature=temperature)
                if loss.requires_grad:
                    loss.backward()
                    optimizer.step()
                    epoch_loss += float(loss.item())
                    batches += 1
            history.append(epoch_loss / max(1, batches))

        final_temperature = float(current_temperature().detach().cpu().item())
        return TrainingResult(
            history=history,
            vocabulary=self.vocabulary,
            dataset=self.dataset,
            temperature=final_temperature,
        )

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
            return self.model(tensor, adjacency=adj)
