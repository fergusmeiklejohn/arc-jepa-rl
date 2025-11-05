"""High-level trainer for Meta-JEPA rule-family embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Sequence

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
    weight_decay: float = 0.0
    device: str = "cpu"


@dataclass
class TrainingResult:
    history: List[float]
    vocabulary: PrimitiveVocabulary
    dataset: RuleFamilyDataset


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
    ) -> None:
        _ensure_torch()
        self.dataset = dataset
        self.vocabulary = vocabulary

        feature_dim = dataset.features.shape[1]
        self.model = MetaJEPAModel(
            feature_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            dropout=dropout,
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
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        loader = DataLoader(
            self.dataset,
            batch_size=min(config.batch_size, len(self.dataset)),
            shuffle=True,
        )

        history: List[float] = []
        model.train()
        for _ in range(config.epochs):
            epoch_loss = 0.0
            batches = 0
            for features, labels in loader:
                features = features.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                embeddings = model(features)
                loss = contrastive_loss(embeddings, labels, temperature=config.temperature)
                if loss.requires_grad:
                    loss.backward()
                    optimizer.step()
                    epoch_loss += float(loss.item())
                    batches += 1
            history.append(epoch_loss / max(1, batches))

        return TrainingResult(history=history, vocabulary=self.vocabulary, dataset=self.dataset)

    def encode(self, features: "torch.Tensor", *, device: str | None = None) -> "torch.Tensor":
        _ensure_torch()
        self.model.eval()
        tensor = features.to(device or "cpu")
        with torch.no_grad():
            return self.model(tensor)
