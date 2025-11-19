"""Benchmark counterfactual latent prediction versus grid execution."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from arcgen import PRIMITIVE_REGISTRY
from training.jepa import (
    ObjectCentricJEPATrainer,
    ProgramConditionedJEPA,
    ProgramConditionedModelConfig,
    ProgramTripleDataset,
    aggregate_object_encoding,
    load_jepa_config,
)
from training.jepa.object_pipeline import ObjectTokenBatch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark latent prediction vs grid execution speed")
    parser.add_argument("--config", type=Path, required=True, help="Training config path")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Program-conditioned JEPA checkpoint")
    parser.add_argument("--samples", type=int, default=8, help="Number of dataset samples to benchmark")
    parser.add_argument("--candidates", type=int, default=1024, help="Simulated candidate programs per sample")
    parser.add_argument("--device", type=str, default=None, help="Optional device override")
    return parser.parse_args()


def _load_checkpoint(model: ProgramConditionedJEPA, checkpoint_path: Path) -> None:
    payload = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(payload["model_state"])


def _encoded_latent(
    trainer: ObjectCentricJEPATrainer,
    sample,
    device: torch.device,
) -> torch.Tensor:
    batch = ObjectTokenBatch(
        features=sample.input_features.unsqueeze(0),
        mask=sample.input_mask.unsqueeze(0),
        adjacency=sample.input_adjacency.unsqueeze(0),
    ).to(device)
    encoding = trainer.object_encoder.encode_tokens(batch, device=device, non_blocking=True)
    return aggregate_object_encoding(encoding)


def _execute_program(record, grid):
    current = grid
    for step in record.program:
        spec = PRIMITIVE_REGISTRY.get(step.primitive)
        current = spec.apply(current, **dict(step.params))
    return current


def main() -> None:
    args = parse_args()
    config = dict(load_jepa_config(args.config))
    training_cfg = config.get("training", {})
    device = torch.device(args.device or training_cfg.get("device") or "cpu")

    trainer = ObjectCentricJEPATrainer(config)
    dataset = ProgramTripleDataset(
        config.get("program_dataset") or config.get("data", {}).get("program_dataset"),
        tokenizer_config=trainer.tokenizer_config,
    )
    program_cfg = ProgramConditionedModelConfig.from_mapping(config.get("program_encoder"))
    model = ProgramConditionedJEPA(
        latent_dim=trainer.encoder_config.hidden_dim,
        vocab_size=dataset.vocab_size,
        parameter_dim=dataset.parameter_dim,
        config=program_cfg,
    ).to(device)
    _load_checkpoint(model, args.checkpoint)
    model.eval()
    trainer.encoder.to(device)

    samples = min(args.samples, len(dataset))
    if samples <= 0:
        raise ValueError("dataset contains no entries to benchmark")

    latent_total = 0.0
    grid_total = 0.0

    with torch.no_grad():
        for idx in range(samples):
            sample = dataset[idx]
            record = dataset.records[idx]
            latent = _encoded_latent(trainer, sample, device)
            latent = latent.expand(args.candidates, -1)
            ids, params, mask = dataset.program_tokenizer.encode_many([record.program] * args.candidates)
            ids = ids.to(device)
            params = params.to(device)
            mask = mask.to(device)

            start = time.perf_counter()
            model.predict_counterfactual(
                latent,
                program_ids=ids,
                program_params=params,
                program_mask=mask,
            )
            latent_total += time.perf_counter() - start

            start = time.perf_counter()
            for _ in range(args.candidates):
                _execute_program(record, record.input_grid)
            grid_total += time.perf_counter() - start

    speedup = grid_total / max(latent_total, 1e-9)
    print(
        f"Benchmarked {samples} samples x {args.candidates} candidates\n"
        f"Latent prediction time: {latent_total:.4f}s\n"
        f"Grid execution time:    {grid_total:.4f}s\n"
        f"Estimated speedup:      {speedup:.1f}x"
    )


if __name__ == "__main__":
    main()

