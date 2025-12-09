#!/usr/bin/env python3
"""Evaluate what a trained JEPA model has learned.

This script analyzes:
1. Embedding similarity patterns (do different inputs produce different embeddings?)
2. Task discrimination (can the model distinguish different ARC task types?)
3. Input-output relationship (are input/output embeddings appropriately related?)
4. Collapse detection (has the model collapsed to trivial solutions?)

Usage:
    python scripts/evaluate_jepa.py --checkpoint artifacts/jepa/arcgen_v7/checkpoint_latest.pt \
        --manifest data/arc_gen_100k/manifest.jsonl --num-samples 1000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
import numpy as np

from training.jepa import ObjectCentricJEPAExperiment


def load_manifest_samples(manifest_path: Path, num_samples: int, seed: int = 42) -> list[dict]:
    """Load random samples from manifest, grouped by task_id."""
    all_samples = []
    with open(manifest_path) as f:
        for line in f:
            if line.strip():
                all_samples.append(json.loads(line))

    random.seed(seed)
    if num_samples < len(all_samples):
        samples = random.sample(all_samples, num_samples)
    else:
        samples = all_samples

    return samples


def compute_embeddings(
    experiment: ObjectCentricJEPAExperiment,
    samples: list[dict],
    embed_type: str = "input",  # "input" or "output"
) -> tuple[torch.Tensor, list[str]]:
    """Compute embeddings for all samples."""
    from arcgen import Grid

    embeddings = []
    task_ids = []

    experiment.trainer.encoder.eval()
    experiment.projection_head.eval()

    with torch.no_grad():
        for sample in samples:
            grid_data = sample[embed_type]
            grid = Grid(grid_data)

            # Encode single grid
            encoding = experiment.trainer.object_encoder.encode([grid], device=experiment.device)

            # Pool over objects
            mask = encoding.mask.to(experiment.device)
            mask_expanded = mask.unsqueeze(-1)
            summed = (encoding.embeddings * mask_expanded).sum(dim=1)
            counts = torch.clamp(mask_expanded.sum(dim=1), min=1.0)
            pooled = summed / counts  # [1, hidden_dim]

            # Project
            projected = experiment.projection_head(pooled)
            projected = F.normalize(projected, dim=-1)

            embeddings.append(projected.cpu())
            task_ids.append(sample.get("task_id", "unknown"))

    return torch.cat(embeddings, dim=0), task_ids


def analyze_collapse(embeddings: torch.Tensor) -> dict:
    """Analyze if embeddings have collapsed."""
    # Compute pairwise cosine similarities
    normed = F.normalize(embeddings, dim=-1)
    similarity_matrix = normed @ normed.t()

    # Exclude diagonal
    n = similarity_matrix.size(0)
    mask = ~torch.eye(n, dtype=torch.bool)
    off_diagonal = similarity_matrix[mask]

    # Compute statistics
    mean_sim = off_diagonal.mean().item()
    std_sim = off_diagonal.std().item()
    min_sim = off_diagonal.min().item()
    max_sim = off_diagonal.max().item()

    # Check for collapse (all embeddings similar)
    is_collapsed = mean_sim > 0.95 and std_sim < 0.05

    # Compute variance of embeddings
    embedding_var = embeddings.var(dim=0).mean().item()

    # Effective rank via singular values
    U, S, V = torch.svd(embeddings - embeddings.mean(dim=0))
    S_norm = S / S.sum()
    entropy = -(S_norm * torch.log(S_norm + 1e-10)).sum()
    effective_rank = torch.exp(entropy).item()

    return {
        "mean_similarity": mean_sim,
        "std_similarity": std_sim,
        "min_similarity": min_sim,
        "max_similarity": max_sim,
        "is_collapsed": is_collapsed,
        "embedding_variance": embedding_var,
        "effective_rank": effective_rank,
    }


def analyze_task_discrimination(
    embeddings: torch.Tensor,
    task_ids: list[str],
) -> dict:
    """Analyze if model can distinguish different tasks."""
    # Group embeddings by task
    task_to_indices = defaultdict(list)
    for i, task_id in enumerate(task_ids):
        task_to_indices[task_id].append(i)

    # Only consider tasks with multiple samples
    valid_tasks = {k: v for k, v in task_to_indices.items() if len(v) >= 2}

    if len(valid_tasks) < 2:
        return {"error": "Need at least 2 tasks with 2+ samples each"}

    # Compute within-task similarity (should be high if task-aware)
    within_sims = []
    for task_id, indices in valid_tasks.items():
        task_embs = embeddings[indices]
        normed = F.normalize(task_embs, dim=-1)
        sim = normed @ normed.t()
        n = sim.size(0)
        mask = ~torch.eye(n, dtype=torch.bool)
        within_sims.extend(sim[mask].tolist())

    # Compute between-task similarity (should be lower)
    between_sims = []
    task_list = list(valid_tasks.keys())
    for i, task1 in enumerate(task_list):
        for task2 in task_list[i+1:]:
            embs1 = F.normalize(embeddings[valid_tasks[task1]], dim=-1)
            embs2 = F.normalize(embeddings[valid_tasks[task2]], dim=-1)
            cross_sim = embs1 @ embs2.t()
            between_sims.extend(cross_sim.flatten().tolist())

    within_mean = np.mean(within_sims)
    between_mean = np.mean(between_sims)
    discrimination_gap = within_mean - between_mean

    return {
        "within_task_similarity": within_mean,
        "between_task_similarity": between_mean,
        "discrimination_gap": discrimination_gap,
        "num_tasks_analyzed": len(valid_tasks),
        "is_task_aware": discrimination_gap > 0.1,
    }


def analyze_input_output_relationship(
    input_embeddings: torch.Tensor,
    output_embeddings: torch.Tensor,
) -> dict:
    """Analyze relationship between input and output embeddings."""
    # Paired similarity (same sample's input and output)
    input_normed = F.normalize(input_embeddings, dim=-1)
    output_normed = F.normalize(output_embeddings, dim=-1)

    paired_sim = (input_normed * output_normed).sum(dim=-1)

    # Cross similarity (different samples)
    cross_sim = input_normed @ output_normed.t()
    n = cross_sim.size(0)
    mask = ~torch.eye(n, dtype=torch.bool)
    unpaired_sim = cross_sim[mask]

    return {
        "paired_similarity_mean": paired_sim.mean().item(),
        "paired_similarity_std": paired_sim.std().item(),
        "unpaired_similarity_mean": unpaired_sim.mean().item(),
        "unpaired_similarity_std": unpaired_sim.std().item(),
        "io_discrimination": paired_sim.mean().item() - unpaired_sim.mean().item(),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate JEPA model")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to manifest.jsonl")
    parser.add_argument("--num-samples", type=int, default=500, help="Number of samples to evaluate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint.get("config", {})

    print("Building experiment...")
    experiment = ObjectCentricJEPAExperiment(config, device=args.device)
    experiment.trainer.encoder.load_state_dict(checkpoint["model_state"])
    experiment.projection_head.load_state_dict(checkpoint["projection_state"])

    print(f"Loading {args.num_samples} samples from {args.manifest}...")
    samples = load_manifest_samples(args.manifest, args.num_samples, args.seed)
    print(f"  Loaded {len(samples)} samples")

    # Count unique tasks
    task_counts = defaultdict(int)
    for s in samples:
        task_counts[s.get("task_id", "unknown")] += 1
    print(f"  Unique tasks: {len(task_counts)}")

    print("\nComputing input embeddings...")
    input_embeddings, task_ids = compute_embeddings(experiment, samples, "input")
    print(f"  Shape: {input_embeddings.shape}")

    print("\nComputing output embeddings...")
    output_embeddings, _ = compute_embeddings(experiment, samples, "output")
    print(f"  Shape: {output_embeddings.shape}")

    print("\n" + "="*60)
    print("COLLAPSE ANALYSIS (Input Embeddings)")
    print("="*60)
    collapse_input = analyze_collapse(input_embeddings)
    for k, v in collapse_input.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    print("\n" + "="*60)
    print("COLLAPSE ANALYSIS (Output Embeddings)")
    print("="*60)
    collapse_output = analyze_collapse(output_embeddings)
    for k, v in collapse_output.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    print("\n" + "="*60)
    print("TASK DISCRIMINATION (Input Embeddings)")
    print("="*60)
    task_disc = analyze_task_discrimination(input_embeddings, task_ids)
    for k, v in task_disc.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    print("\n" + "="*60)
    print("INPUT-OUTPUT RELATIONSHIP")
    print("="*60)
    io_rel = analyze_input_output_relationship(input_embeddings, output_embeddings)
    for k, v in io_rel.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    print("\n" + "="*60)
    print("DIAGNOSIS")
    print("="*60)

    if collapse_input["is_collapsed"]:
        print("  ⚠️  INPUT EMBEDDINGS COLLAPSED - all inputs map to similar vectors")
    else:
        print("  ✓  Input embeddings show diversity")

    if collapse_output["is_collapsed"]:
        print("  ⚠️  OUTPUT EMBEDDINGS COLLAPSED - all outputs map to similar vectors")
    else:
        print("  ✓  Output embeddings show diversity")

    if task_disc.get("is_task_aware", False):
        print("  ✓  Model shows task discrimination")
    else:
        print("  ⚠️  NO TASK DISCRIMINATION - model can't distinguish task types")

    io_disc = io_rel.get("io_discrimination", 0)
    if io_disc > 0.1:
        print(f"  ✓  Input-output pairs are more similar than random ({io_disc:.3f} gap)")
    elif io_disc > 0:
        print(f"  ~  Weak input-output relationship ({io_disc:.3f} gap)")
    else:
        print(f"  ⚠️  NO INPUT-OUTPUT RELATIONSHIP ({io_disc:.3f} gap)")

    # Overall verdict
    print("\n" + "="*60)
    print("VERDICT")
    print("="*60)

    if collapse_input["is_collapsed"] and collapse_output["is_collapsed"]:
        print("  The model has COMPLETELY COLLAPSED to a trivial solution.")
        print("  All embeddings are nearly identical regardless of input.")
        print("  The 0.0202 loss plateau is pure SIGReg regularization.")
    elif collapse_input["mean_similarity"] > 0.9:
        print("  The model shows SEVERE COLLAPSE - embeddings are too similar.")
        print("  It has likely found a shortcut that ignores input content.")
    elif not task_disc.get("is_task_aware", False):
        print("  The model encodes SOME information but cannot distinguish tasks.")
        print("  It may be learning low-level features, not ARC transformations.")
    else:
        print("  The model shows signs of meaningful learning!")
        print("  Consider training longer or with more data.")


if __name__ == "__main__":
    main()
