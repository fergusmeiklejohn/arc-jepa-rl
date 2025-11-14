"""Behavioral cloning trainer that reuses the ActorCritic core."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .bc_data import BehavioralCloningDataset
from .models import ActorCriticConfig, ActorCriticCore


@dataclass
class BCTrainingConfig:
    """Hyperparameters for behavioral cloning."""

    batch_size: int = 64
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 0.0
    termination_loss_weight: float = 1.0
    device: str = "cpu"


def _resolve_actor_config(model_cfg: Mapping[str, Any]) -> ActorCriticConfig:
    hidden_dims = tuple(int(x) for x in model_cfg.get("hidden_dims", (256, 256)))
    return ActorCriticConfig(
        hidden_dims=hidden_dims,
        activation=str(model_cfg.get("activation", "relu")),
        layer_norm=bool(model_cfg.get("layer_norm", True)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        include_termination=bool(model_cfg.get("include_termination", True)),
        action_mask_key=None,
    )


def train_behavioral_cloning(
    train_dataset: BehavioralCloningDataset,
    val_dataset: BehavioralCloningDataset,
    *,
    model_cfg: Mapping[str, Any],
    optim_cfg: Mapping[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    """Run supervised option training."""

    output_dir.mkdir(parents=True, exist_ok=True)
    training_cfg = BCTrainingConfig(
        batch_size=int(optim_cfg.get("batch_size", 64)),
        epochs=int(optim_cfg.get("epochs", 5)),
        lr=float(optim_cfg.get("lr", 1e-3)),
        weight_decay=float(optim_cfg.get("weight_decay", 0.0)),
        termination_loss_weight=float(optim_cfg.get("termination_loss_weight", 1.0)),
        device=str(optim_cfg.get("device", "cpu")),
    )

    device = torch.device(training_cfg.device)
    model_config = _resolve_actor_config(model_cfg)
    action_dim = int(model_cfg.get("action_dim", train_dataset.action_dim))
    model = ActorCriticCore(
        obs_dim=train_dataset.feature_dim,
        action_dim=action_dim,
        config=model_config,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_cfg.lr,
        weight_decay=training_cfg.weight_decay,
    )
    ce_loss = nn.CrossEntropyLoss()
    bce_loss = nn.BCELoss()

    train_loader = DataLoader(train_dataset, batch_size=training_cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=training_cfg.batch_size, shuffle=False)

    history: List[Dict[str, float]] = []
    best_acc = float("-inf")
    best_path = output_dir / "bc_model_best.pt"
    latest_path = output_dir / "bc_model_last.pt"

    for epoch in range(1, training_cfg.epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            ce_loss,
            bce_loss,
            optimizer,
            device,
            training_cfg.termination_loss_weight,
            train=True,
        )
        val_metrics = _run_epoch(
            model,
            val_loader,
            ce_loss,
            bce_loss,
            optimizer,
            device,
            training_cfg.termination_loss_weight,
            train=False,
        )
        epoch_metrics = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(epoch_metrics)
        torch.save(model.state_dict(), latest_path)
        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            torch.save(model.state_dict(), best_path)

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    metadata = {
        "feature_dim": train_dataset.feature_dim,
        "action_dim": action_dim,
        "option_id_map": train_dataset.option_id_map,
        "model_config": {
            "hidden_dims": list(model_config.hidden_dims),
            "activation": model_config.activation,
            "layer_norm": model_config.layer_norm,
            "dropout": model_config.dropout,
            "include_termination": model_config.include_termination,
        },
        "training": training_cfg.__dict__,
    }
    metadata_path = output_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return {
        "history": history,
        "best_model": str(best_path),
        "latest_model": str(latest_path),
        "metrics_path": str(metrics_path),
        "metadata_path": str(metadata_path),
    }


def _run_epoch(
    model: ActorCriticCore,
    loader: DataLoader,
    ce_loss: nn.Module,
    bce_loss: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    termination_weight: float,
    *,
    train: bool,
) -> Dict[str, float]:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_action_loss = 0.0
    total_term_loss = 0.0
    total_correct = 0
    total_examples = 0

    for obs, action, termination, term_mask in loader:
        obs = obs.to(device)
        action = action.to(device)
        termination = termination.to(device)
        term_mask = term_mask.to(device)

        with torch.set_grad_enabled(train):
            outputs = model(obs)
            logits = outputs["logits"]
            action_loss = ce_loss(logits, action)
            loss = action_loss

            term_loss_value = torch.tensor(0.0, device=device)
            if model.include_termination and term_mask.any():
                termination_pred = outputs["termination"]
                if termination_pred is None:
                    raise RuntimeError("Model configured with termination head but returned None")
                mask = term_mask.bool()
                term_loss = bce_loss(termination_pred[mask], termination[mask])
                loss = loss + termination_weight * term_loss
                term_loss_value = term_loss.detach()

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        preds = logits.argmax(dim=1)
        total_correct += (preds == action).sum().item()
        total_examples += action.size(0)
        total_loss += loss.detach().item() * action.size(0)
        total_action_loss += action_loss.detach().item() * action.size(0)
        total_term_loss += term_loss_value.item() * action.size(0)

    avg_loss = total_loss / max(1, total_examples)
    avg_action_loss = total_action_loss / max(1, total_examples)
    avg_term_loss = total_term_loss / max(1, total_examples)
    accuracy = total_correct / max(1, total_examples)
    return {
        "loss": avg_loss,
        "action_loss": avg_action_loss,
        "termination_loss": avg_term_loss,
        "accuracy": accuracy,
    }
