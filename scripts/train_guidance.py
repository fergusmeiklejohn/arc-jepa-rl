"""Train neural guidance model for DSL program search."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import yaml

from arcgen import SyntheticARCGenerator, GeneratorConfig
from envs import LatentScorer
from training.dsl import (
    GuidanceDataset,
    GuidanceScorer,
    ProgramEncoder,
    GuidedBeamSearch,
    build_default_primitive_registry,
    build_guidance_examples,
)
from training.jepa import ObjectCentricJEPATrainer
from training.modules.projection import ProjectionHead

try:  # pragma: no cover - optional dependency at runtime
    import torch
    from torch.utils.data import DataLoader
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    DataLoader = object  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train neural guidance model for ARC DSL search")
    parser.add_argument("--jepa-config", type=Path, required=True, help="Path to JEPA YAML config")
    parser.add_argument("--dsl-config", type=Path, required=True, help="Path to DSL guidance YAML config")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device (default: cpu)")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--output", type=Path, default=Path("artifacts/guidance/model.pt"), help="Checkpoint output path")
    return parser.parse_args()


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"YAML at {path} must contain a top-level mapping")
    return data


def build_latent_scorer(jepa_cfg: Dict, device: str) -> LatentScorer:
    trainer = ObjectCentricJEPATrainer(jepa_cfg)
    projection = None
    loss_cfg = jepa_cfg.get("loss") or {}
    if loss_cfg:
        projection = ProjectionHead(
            input_dim=trainer.encoder_config.hidden_dim,
            output_dim=int(loss_cfg.get("projection_dim", trainer.encoder_config.hidden_dim)),
            layers=int(loss_cfg.get("projection_layers", 1)),
            activation=str(loss_cfg.get("projection_activation", "relu")),
        )
    scorer = LatentScorer(trainer.object_encoder, projection_head=projection, device=device)
    return scorer


def sample_tasks(config: Dict) -> list[tuple]:
    generator_cfg = config.get("generator", {})
    gen = SyntheticARCGenerator(
        GeneratorConfig(
            min_grid_size=int(generator_cfg.get("min_grid_size", 5)),
            max_grid_size=int(generator_cfg.get("max_grid_size", 8)),
            min_colors=int(generator_cfg.get("min_colors", 3)),
            max_colors=int(generator_cfg.get("max_colors", 6)),
            background_color=int(generator_cfg.get("background_color", 0)),
            fill_probability=float(generator_cfg.get("fill_probability", 0.75)),
        ),
        seed=generator_cfg.get("seed"),
    )

    schedule = config.get("task_schedule", {"atomic": 8, "sequential": 4})
    tasks = []
    for phase, count in schedule.items():
        for task in gen.sample_many(int(count), phase):
            tasks.append((task.input_grid, task.output_grid))
    return tasks


def main() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required to train the guidance model")

    args = parse_args()
    device = torch.device(args.device)

    jepa_cfg = load_yaml(args.jepa_config)
    dsl_cfg = load_yaml(args.dsl_config)

    latent_scorer = build_latent_scorer(jepa_cfg, device=str(device))
    registry = build_default_primitive_registry(color_constants=dsl_cfg.get("color_constants", [0, 1, 2, 3]))
    program_encoder = ProgramEncoder(registry, embedding_dim=int(dsl_cfg.get("model", {}).get("primitive_embedding_dim", 32)))
    program_encoder.to(device)

    def latent_embedder(grid):
        tensor = latent_scorer.embed(grid)
        return tensor.to(device)

    tasks = sample_tasks(dsl_cfg)
    examples = build_guidance_examples(
        tasks,
        registry,
        latent_embedder,
        program_encoder,
        max_nodes=int(dsl_cfg.get("enumeration", {}).get("max_nodes", 4)),
        max_programs_per_task=dsl_cfg.get("enumeration", {}).get("max_programs_per_task", 32),
    )

    if not examples:
        raise RuntimeError("No guidance examples generated; check configuration")

    dataset = GuidanceDataset(examples)
    feature_dim = dataset[0][0].shape[0]
    train_cfg = dsl_cfg.get("train", {})
    epochs = args.epochs or int(train_cfg.get("epochs", 5))
    batch_size = int(train_cfg.get("batch_size", 32))
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))

    scorer = GuidanceScorer(feature_dim, hidden_dim=int(dsl_cfg.get("model", {}).get("hidden_dim", 128)))
    scorer.to(device)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(scorer.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        scorer.train()
        epoch_loss = 0.0
        batches = 0
        for vectors, labels, _ in dataloader:
            vectors = vectors.to(device)
            labels = labels.to(device)
            preds = scorer(vectors)
            loss = torch.nn.functional.mse_loss(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batches += 1

        avg_loss = epoch_loss / max(1, batches)
        print(f"Epoch {epoch}/{epochs} - loss={avg_loss:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": scorer.state_dict(), "feature_dim": feature_dim}, args.output)
    print(f"Saved guidance model to {args.output}")


if __name__ == "__main__":
    main()
