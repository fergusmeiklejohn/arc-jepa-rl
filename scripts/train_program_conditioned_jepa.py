"""Train the program-conditioned JEPA counterfactual predictor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.jepa import (
    ObjectCentricJEPATrainer,
    ProgramConditionedJEPA,
    ProgramConditionedModelConfig,
    ProgramTripleBatch,
    ProgramTripleDataset,
    aggregate_object_encoding,
    collate_program_triples,
    load_jepa_config,
    ObjectTokenizerConfig,
    OptimizerConfig,
    InfoNCELossConfig,
)
from training.jepa.object_pipeline import ObjectTokenBatch
from training.jepa.sigreg import SIGRegLoss, SIGRegLossConfig
from training.modules.projection import InfoNCEQueue, ProjectionHead
from training.utils.logging import create_experiment_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a program-conditioned JEPA model")
    parser.add_argument("--config", type=Path, required=True, help="YAML configuration file")
    parser.add_argument("--device", type=str, default=None, help="Optional device override (e.g., cuda:0)")
    parser.add_argument("--dry-run", action="store_true", help="Run a single batch to verify wiring")
    return parser.parse_args()


def _loader_settings(config: Mapping[str, object]) -> dict:
    training_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    if not isinstance(training_cfg, Mapping) or not isinstance(data_cfg, Mapping):
        raise ValueError("config must contain 'training' and 'data' mappings")

    batch_size = int(training_cfg.get("batch_size", 32))
    num_workers = int(training_cfg.get("num_workers", 0))
    pin_memory = bool(training_cfg.get("pin_memory", False))
    seed = config.get("seed", data_cfg.get("seed"))
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
    return {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "generator": generator,
        "seed": seed,
    }


def _build_dataloaders(
    config: Mapping[str, object],
    tokenizer_config: ObjectTokenizerConfig,
) -> tuple[DataLoader, DataLoader | None, ProgramTripleDataset]:
    data_cfg = config.get("data", {})
    if not isinstance(data_cfg, Mapping):
        raise ValueError("config['data'] must be a mapping")
    dataset_path = data_cfg.get("program_dataset") or config.get("program_dataset")
    if not dataset_path:
        raise ValueError("config must define data.program_dataset")
    dataset_path = Path(dataset_path)
    max_program_length = data_cfg.get("max_program_length")
    max_program_length = int(max_program_length) if max_program_length is not None else None

    dataset = ProgramTripleDataset(dataset_path, tokenizer_config, max_program_length=max_program_length)

    loader_cfg = _loader_settings(config)
    batch_size = loader_cfg["batch_size"]
    num_workers = loader_cfg["num_workers"]
    pin_memory = loader_cfg["pin_memory"]
    generator = loader_cfg.get("generator")

    val_fraction = float(data_cfg.get("val_split", 0.1))
    val_count = max(1, int(len(dataset) * val_fraction)) if val_fraction > 0 else 0
    if val_count >= len(dataset):
        val_count = max(1, len(dataset) // 5)
    train_count = len(dataset) - val_count
    if train_count <= 0:
        raise ValueError("validation split too large for dataset size")

    splits = [train_count, val_count] if val_count > 0 else [train_count]
    subsets = random_split(dataset, splits, generator=generator)
    train_dataset = subsets[0]
    val_dataset = subsets[1] if val_count > 0 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_program_triples,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_program_triples,
        )
        if val_dataset is not None
        else None
    )
    return train_loader, val_loader, dataset


def _masked_mean_projection(
    trainer: ObjectCentricJEPATrainer,
    batch: ProgramTripleBatch,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_tokens = ObjectTokenBatch(batch.input_features, batch.input_mask, batch.input_adjacency)
    target_tokens = ObjectTokenBatch(batch.output_features, batch.output_mask, batch.output_adjacency)
    input_encoding = trainer.object_encoder.encode_tokens(input_tokens, device=device, non_blocking=True)
    target_encoding = trainer.object_encoder.encode_tokens(target_tokens, device=device, non_blocking=True)
    return aggregate_object_encoding(input_encoding), aggregate_object_encoding(target_encoding)


def _info_nce_loss(
    predicted_proj: torch.Tensor,
    target_proj: torch.Tensor,
    queue: InfoNCEQueue,
    log_temperature: torch.Tensor,
    bounds: tuple[float, float],
) -> torch.Tensor:
    logits_pos = torch.sum(predicted_proj * target_proj, dim=-1, keepdim=True)
    logits_inbatch = predicted_proj @ target_proj.t()
    batch_size = predicted_proj.size(0)
    if batch_size > 1:
        eye = torch.eye(batch_size, device=logits_inbatch.device, dtype=torch.bool)
        logits_inbatch = logits_inbatch.masked_select(~eye).view(batch_size, -1)
    else:
        logits_inbatch = logits_pos.new_empty((batch_size, 0))

    negatives = queue.get_negatives()
    logits_queue = (
        predicted_proj @ negatives.t() if negatives.size(0) > 0 else logits_pos.new_empty((batch_size, 0))
    )

    logits = torch.cat([logits_pos, logits_inbatch, logits_queue], dim=1)
    temperature = torch.exp(log_temperature)
    temperature = torch.clamp(temperature, min=bounds[0], max=bounds[1])
    logits = logits / temperature
    labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits.float(), labels)


def _step(
    *,
    trainer: ObjectCentricJEPATrainer,
    model: ProgramConditionedJEPA,
    projection: ProjectionHead,
    queue: InfoNCEQueue,
    loss_cfg: InfoNCELossConfig,
    sigreg_cfg: SIGRegLossConfig,
    sigreg: SIGRegLoss | None,
    log_temperature: torch.Tensor,
    batch: ProgramTripleBatch,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    batch = batch.to(device, non_blocking=True)
    current_latent, target_latent = _masked_mean_projection(trainer, batch, device=device)
    predicted_latent = model(
        current_latent,
        batch.program_ids,
        batch.program_params,
        batch.program_mask,
    )
    pred_proj = projection(predicted_latent)
    target_proj = projection(target_latent)
    info_nce = _info_nce_loss(pred_proj, target_proj, queue, log_temperature, tuple(loss_cfg.temperature_bounds))
    with torch.no_grad():
        queue.enqueue(target_proj.detach())
    sigreg_penalty = None
    if sigreg_cfg.enabled and sigreg is not None:
        sigreg_penalty = sigreg(pred_proj)
        total_loss = info_nce + sigreg_cfg.weight * sigreg_penalty
    else:
        total_loss = info_nce
    return total_loss, info_nce, sigreg_penalty


def _evaluate(
    *,
    trainer: ObjectCentricJEPATrainer,
    model: ProgramConditionedJEPA,
    projection: ProjectionHead,
    loader: DataLoader | None,
    device: torch.device,
) -> float | None:
    if loader is None:
        return None
    model.eval()
    projection.eval()
    cosines: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            current_latent, target_latent = _masked_mean_projection(trainer, batch, device=device)
            predicted = model(
                current_latent,
                batch.program_ids,
                batch.program_params,
                batch.program_mask,
            )
            cos_sim = F.cosine_similarity(projection(predicted), projection(target_latent), dim=-1)
            cosines.append(float(cos_sim.mean().item()))
    model.train()
    projection.train()
    if not cosines:
        return None
    return float(sum(cosines) / len(cosines))


def main() -> None:
    args = parse_args()
    config = dict(load_jepa_config(args.config))
    training_cfg = config.get("training", {})
    if not isinstance(training_cfg, Mapping):
        raise ValueError("config['training'] must be a mapping")

    trainer = ObjectCentricJEPATrainer(config)
    train_loader, val_loader, dataset = _build_dataloaders(config, trainer.tokenizer_config)

    loss_cfg = InfoNCELossConfig.from_mapping(config.get("loss"))
    sigreg_cfg = SIGRegLossConfig.from_mapping(config.get("sigreg"))
    program_cfg = ProgramConditionedModelConfig.from_mapping(config.get("program_encoder"))
    opt_cfg = OptimizerConfig.from_mapping(config.get("optimizer"))

    device_str = args.device or training_cfg.get("device") or "cpu"
    device = torch.device(device_str)
    trainer.encoder.to(device)

    model = ProgramConditionedJEPA(
        latent_dim=trainer.encoder_config.hidden_dim,
        vocab_size=dataset.vocab_size,
        parameter_dim=dataset.parameter_dim,
        config=program_cfg,
    ).to(device)

    projection = ProjectionHead(
        input_dim=trainer.encoder_config.hidden_dim,
        output_dim=loss_cfg.projection_dim,
        layers=loss_cfg.projection_layers,
        activation=loss_cfg.projection_activation,
    ).to(device)
    queue = InfoNCEQueue(loss_cfg.projection_dim, loss_cfg.queue_size).to(device)

    temperature_value = torch.tensor(loss_cfg.temperature_init, dtype=torch.float32, device=device)
    if loss_cfg.learnable_temperature:
        log_temperature = torch.nn.Parameter(torch.log(temperature_value))
    else:
        log_temperature = torch.log(temperature_value).detach()

    sigreg = None
    if sigreg_cfg.enabled:
        sigreg = SIGRegLoss(num_slices=sigreg_cfg.num_slices, num_points=sigreg_cfg.num_points).to(device)

    params: list[torch.nn.Parameter] = list(model.parameters()) + list(projection.parameters())
    if isinstance(log_temperature, torch.nn.Parameter):
        params.append(log_temperature)
    optimizer = torch.optim.Adam(params, lr=opt_cfg.lr, weight_decay=opt_cfg.weight_decay)

    checkpoint_dir = Path(training_cfg.get("checkpoint_dir", "artifacts/program_jepa"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger = None
    logging_cfg = config.get("logging")
    if isinstance(logging_cfg, Mapping) and logging_cfg.get("enabled", True):
        logger = create_experiment_logger(
            logging_cfg.get("log_dir", checkpoint_dir / "tensorboard"),
            run_name=logging_cfg.get("run_name"),
            flush_secs=int(logging_cfg.get("flush_secs", 10)),
        )

    if args.dry_run:
        batch = next(iter(train_loader))
        loss, _, _ = _step(
            trainer=trainer,
            model=model,
            projection=projection,
            queue=queue,
            loss_cfg=loss_cfg,
            sigreg_cfg=sigreg_cfg,
            sigreg=sigreg,
            log_temperature=log_temperature,
            batch=batch,
            device=device,
        )
        print(f"Dry-run loss: {float(loss.item()):.6f}")
        return

    epochs = int(training_cfg.get("epochs", 1))
    best_val = None
    history: list[dict[str, float]] = []

    global_step = 0
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_batches = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss, info_nce, sigreg_penalty = _step(
                trainer=trainer,
                model=model,
                projection=projection,
                queue=queue,
                loss_cfg=loss_cfg,
                sigreg_cfg=sigreg_cfg,
                sigreg=sigreg,
                log_temperature=log_temperature,
                batch=batch,
                device=device,
            )
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().item())
            total_batches += 1
            global_step += 1
            if logger is not None:
                scalars = {"loss/info_nce": float(info_nce.detach().item()), "loss/total": float(loss.detach().item())}
                if sigreg_penalty is not None:
                    scalars["loss/sigreg"] = float(sigreg_penalty.detach().item())
                logger.log_scalars("train/batch", scalars, step=global_step)

        avg_loss = total_loss / max(1, total_batches)
        val_cos = _evaluate(
            trainer=trainer,
            model=model,
            projection=projection,
            loader=val_loader,
            device=device,
        )
        history.append({"epoch": epoch, "train_loss": avg_loss, "val_cosine": val_cos or 0.0})
        print(f"Epoch {epoch}/{epochs}: loss={avg_loss:.6f}, val_cosine={val_cos:.4f}" if val_cos is not None else f"Epoch {epoch}/{epochs}: loss={avg_loss:.6f}")
        if logger is not None:
            logger.log_scalar("train/loss", avg_loss, step=epoch)
            if val_cos is not None:
                logger.log_scalar("val/cosine", val_cos, step=epoch)

        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "config": config,
                "model_state": model.state_dict(),
                "projection_state": projection.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "queue_state": queue.state_dict(),
                "log_temperature": float(log_temperature.detach().item())
                if not isinstance(log_temperature, torch.nn.Parameter)
                else log_temperature.detach().cpu(),
            },
            checkpoint_path,
        )
        if val_cos is not None:
            best_val = val_cos if best_val is None else max(best_val, val_cos)

    metrics = {
        "config": str(args.config),
        "epochs": epochs,
        "train_loss": history[-1]["train_loss"] if history else None,
        "best_val_cosine": best_val,
        "history": history,
    }
    metrics_path = checkpoint_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if logger is not None:
        logger.close()


if __name__ == "__main__":
    main()
