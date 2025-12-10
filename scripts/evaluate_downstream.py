#!/usr/bin/env python3
"""Comprehensive downstream evaluation for JEPA representations.

This script evaluates how well JEPA representations transfer to actual ARC tasks
following the L-JEPA paper's evaluation methodology:

1. Linear Probing: Train a linear classifier to predict transformation types
2. Few-Shot Matching: Use latent deltas to predict outputs from few examples
3. Representation Quality: Covariance analysis, effective rank, isotropy

Usage:
    python scripts/evaluate_downstream.py \
        --checkpoint temp/lejepa_v14/checkpoint_epoch_0400.pt \
        --num-samples 2000 \
        --device cpu
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from arcgen import Grid, SyntheticARCGenerator, GeneratorConfig
from training.jepa import ObjectCentricJEPAExperiment


@dataclass
class EvaluationResults:
    """Container for all evaluation metrics."""

    # Representation quality
    effective_rank: float
    covariance_condition_number: float
    embedding_variance: float
    isotropy_score: float

    # Linear probe (transformation classification)
    linear_probe_accuracy: Optional[float] = None
    linear_probe_per_class: Optional[Dict[str, float]] = None

    # Few-shot matching
    few_shot_top1_accuracy: Optional[float] = None
    few_shot_top3_accuracy: Optional[float] = None
    few_shot_mean_rank: Optional[float] = None

    # Task discrimination
    within_task_similarity: float = 0.0
    between_task_similarity: float = 0.0
    discrimination_gap: float = 0.0

    def to_dict(self) -> dict:
        return {
            "representation_quality": {
                "effective_rank": self.effective_rank,
                "covariance_condition_number": self.covariance_condition_number,
                "embedding_variance": self.embedding_variance,
                "isotropy_score": self.isotropy_score,
            },
            "linear_probe": {
                "accuracy": self.linear_probe_accuracy,
                "per_class": self.linear_probe_per_class,
            },
            "few_shot": {
                "top1_accuracy": self.few_shot_top1_accuracy,
                "top3_accuracy": self.few_shot_top3_accuracy,
                "mean_rank": self.few_shot_mean_rank,
            },
            "task_discrimination": {
                "within_task_similarity": self.within_task_similarity,
                "between_task_similarity": self.between_task_similarity,
                "discrimination_gap": self.discrimination_gap,
            },
        }


class JEPAEmbedder:
    """Helper to compute JEPA embeddings for grids."""

    def __init__(self, experiment: ObjectCentricJEPAExperiment):
        self.experiment = experiment
        self.device = experiment.device
        experiment.trainer.encoder.eval()
        experiment.projection_head.eval()

    @torch.no_grad()
    def embed(self, grid: Grid) -> torch.Tensor:
        """Embed a single grid, returning normalized projection."""
        encoding = self.experiment.trainer.object_encoder.encode([grid], device=self.device)

        # Pool over objects
        mask = encoding.mask.to(self.device)
        mask_expanded = mask.unsqueeze(-1)
        summed = (encoding.embeddings * mask_expanded).sum(dim=1)
        counts = torch.clamp(mask_expanded.sum(dim=1), min=1.0)
        pooled = summed / counts  # [1, hidden_dim]

        # Project
        projected = self.experiment.projection_head(pooled)
        projected = F.normalize(projected, dim=-1)
        return projected.squeeze(0)

    @torch.no_grad()
    def embed_batch(self, grids: List[Grid]) -> torch.Tensor:
        """Embed multiple grids efficiently."""
        embeddings = []
        for grid in grids:
            embeddings.append(self.embed(grid))
        return torch.stack(embeddings)


def generate_labeled_data(
    num_samples: int,
    seed: int = 42,
) -> Tuple[List[Tuple[Grid, Grid]], List[str], List[str]]:
    """Generate synthetic data with transformation labels.

    Returns:
        samples: List of (input_grid, output_grid) tuples
        labels: List of primary transformation names
        task_ids: List of task identifiers (grouped by transformation type for discrimination)
    """
    config = GeneratorConfig(
        min_grid_size=5,
        max_grid_size=12,
        min_colors=3,
        max_colors=6,
    )
    generator = SyntheticARCGenerator(
        config,
        seed=seed,
        program_length_schedule={1: 1.0},  # Single primitive for classification
    )

    samples = []
    labels = []
    task_ids = []

    for i in range(num_samples):
        try:
            task = generator.sample_task("atomic")
            samples.append((task.input_grid, task.output_grid))
            # Use first primitive as label
            if task.rule_trace:
                prim = task.rule_trace[0].primitive
                labels.append(prim)
                # Use transformation type as task_id for discrimination analysis
                # This groups samples by their transformation type
                task_ids.append(prim)
            else:
                labels.append("unknown")
                task_ids.append("unknown")
        except RuntimeError:
            continue

    return samples, labels, task_ids


def compute_representation_quality(embeddings: torch.Tensor) -> Dict[str, float]:
    """Compute representation quality metrics following L-JEPA paper."""
    # Center embeddings
    centered = embeddings - embeddings.mean(dim=0)

    # Covariance matrix
    n = embeddings.size(0)
    cov = (centered.T @ centered) / (n - 1)

    # Eigenvalues for effective rank
    eigenvalues = torch.linalg.eigvalsh(cov)
    eigenvalues = torch.clamp(eigenvalues, min=1e-10)

    # Effective rank (entropy-based)
    p = eigenvalues / eigenvalues.sum()
    entropy = -(p * torch.log(p + 1e-10)).sum()
    effective_rank = torch.exp(entropy).item()

    # Condition number (ratio of largest to smallest eigenvalue)
    condition_number = (eigenvalues.max() / eigenvalues.min()).item()

    # Variance
    variance = embeddings.var(dim=0).mean().item()

    # Isotropy score (how uniform the eigenvalue spectrum is)
    # Perfect isotropy = all eigenvalues equal = max entropy
    max_entropy = np.log(len(eigenvalues))
    isotropy_score = entropy.item() / max_entropy if max_entropy > 0 else 0

    return {
        "effective_rank": effective_rank,
        "covariance_condition_number": condition_number,
        "embedding_variance": variance,
        "isotropy_score": isotropy_score,
    }


def train_linear_probe(
    embeddings: torch.Tensor,
    labels: List[str],
    train_ratio: float = 0.8,
    epochs: int = 100,
    lr: float = 0.01,
) -> Tuple[float, Dict[str, float]]:
    """Train a linear classifier on frozen embeddings.

    Returns:
        accuracy: Overall classification accuracy
        per_class_accuracy: Per-class accuracy dict
    """
    # Create label mapping
    unique_labels = sorted(set(labels))
    if len(unique_labels) < 2:
        return 0.0, {}

    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    label_indices = torch.tensor([label_to_idx[l] for l in labels])

    # Train/test split
    n = len(labels)
    indices = list(range(n))
    random.shuffle(indices)
    split = int(n * train_ratio)
    train_idx, test_idx = indices[:split], indices[split:]

    if len(test_idx) < 10:
        return 0.0, {}

    train_X = embeddings[train_idx]
    train_y = label_indices[train_idx]
    test_X = embeddings[test_idx]
    test_y = label_indices[test_idx]

    # Simple linear classifier
    num_classes = len(unique_labels)
    classifier = nn.Linear(embeddings.size(1), num_classes)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Training
    classifier.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = classifier(train_X)
        loss = criterion(logits, train_y)
        loss.backward()
        optimizer.step()

    # Evaluation
    classifier.eval()
    with torch.no_grad():
        logits = classifier(test_X)
        preds = logits.argmax(dim=1)
        accuracy = (preds == test_y).float().mean().item()

        # Per-class accuracy
        per_class = {}
        for label, idx in label_to_idx.items():
            mask = test_y == idx
            if mask.sum() > 0:
                class_acc = (preds[mask] == test_y[mask]).float().mean().item()
                per_class[label] = class_acc

    return accuracy, per_class


def evaluate_few_shot_matching(
    embedder: JEPAEmbedder,
    samples: List[Tuple[Grid, Grid]],
    task_ids: List[str],
    k_shot: int = 3,
) -> Dict[str, float]:
    """Evaluate few-shot transfer using latent delta matching.

    For each task:
    1. Use k examples as "support" to compute average transformation delta
    2. Apply delta to query input and find nearest output in candidates
    """
    # Group by task
    task_to_samples = defaultdict(list)
    for i, (inp, out) in enumerate(samples):
        task_to_samples[task_ids[i]].append((inp, out))

    # Filter to tasks with enough examples
    valid_tasks = {k: v for k, v in task_to_samples.items() if len(v) >= k_shot + 1}

    if len(valid_tasks) < 5:
        return {"top1_accuracy": 0.0, "top3_accuracy": 0.0, "mean_rank": float('inf')}

    ranks = []
    top1_correct = 0
    top3_correct = 0
    total = 0

    for task_id, task_samples in valid_tasks.items():
        if len(task_samples) < k_shot + 1:
            continue

        # Use first k as support, rest as queries
        support = task_samples[:k_shot]
        queries = task_samples[k_shot:]

        # Compute support deltas
        deltas = []
        for inp, out in support:
            inp_emb = embedder.embed(inp)
            out_emb = embedder.embed(out)
            deltas.append(out_emb - inp_emb)
        avg_delta = torch.stack(deltas).mean(dim=0)

        # Collect all outputs as candidates (from all tasks for harder test)
        all_outputs = [out for s in samples for inp, out in [s]]
        candidate_embs = embedder.embed_batch(all_outputs)

        for inp, expected_out in queries:
            inp_emb = embedder.embed(inp)
            predicted_emb = inp_emb + avg_delta
            predicted_emb = F.normalize(predicted_emb, dim=0)

            # Get target embedding
            target_emb = embedder.embed(expected_out)

            # Find rank of correct output
            similarities = (candidate_embs @ predicted_emb).cpu().numpy()
            target_sim = (target_emb @ predicted_emb).item()

            rank = (similarities > target_sim).sum() + 1
            ranks.append(rank)

            if rank == 1:
                top1_correct += 1
            if rank <= 3:
                top3_correct += 1
            total += 1

    if total == 0:
        return {"top1_accuracy": 0.0, "top3_accuracy": 0.0, "mean_rank": float('inf')}

    return {
        "top1_accuracy": top1_correct / total,
        "top3_accuracy": top3_correct / total,
        "mean_rank": np.mean(ranks),
    }


def evaluate_task_discrimination(
    embeddings: torch.Tensor,
    task_ids: List[str],
) -> Dict[str, float]:
    """Evaluate if embeddings cluster by task."""
    # Group by task
    task_to_indices = defaultdict(list)
    for i, task_id in enumerate(task_ids):
        task_to_indices[task_id].append(i)

    # Filter to tasks with multiple samples
    valid_tasks = {k: v for k, v in task_to_indices.items() if len(v) >= 2}

    if len(valid_tasks) < 2:
        return {"within_task_similarity": 0.0, "between_task_similarity": 0.0, "discrimination_gap": 0.0}

    # Within-task similarity
    within_sims = []
    for task_id, indices in valid_tasks.items():
        task_embs = embeddings[indices]
        normed = F.normalize(task_embs, dim=-1)
        sim = normed @ normed.t()
        n = sim.size(0)
        mask = ~torch.eye(n, dtype=torch.bool)
        within_sims.extend(sim[mask].tolist())

    # Between-task similarity
    between_sims = []
    task_list = list(valid_tasks.keys())
    for i, task1 in enumerate(task_list):
        for task2 in task_list[i+1:]:
            embs1 = F.normalize(embeddings[valid_tasks[task1]], dim=-1)
            embs2 = F.normalize(embeddings[valid_tasks[task2]], dim=-1)
            cross_sim = embs1 @ embs2.t()
            between_sims.extend(cross_sim.flatten().tolist())

    within_mean = np.mean(within_sims) if within_sims else 0
    between_mean = np.mean(between_sims) if between_sims else 0

    return {
        "within_task_similarity": within_mean,
        "between_task_similarity": between_mean,
        "discrimination_gap": within_mean - between_mean,
    }


def main():
    parser = argparse.ArgumentParser(description="Downstream evaluation for JEPA")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint")
    parser.add_argument("--num-samples", type=int, default=2000, help="Samples for evaluation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path")
    parser.add_argument("--skip-few-shot", action="store_true", help="Skip slow few-shot eval")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})

    print("Building experiment...")
    experiment = ObjectCentricJEPAExperiment(config, device=args.device)
    experiment.trainer.encoder.load_state_dict(checkpoint["model_state"])
    experiment.projection_head.load_state_dict(checkpoint["projection_state"])

    embedder = JEPAEmbedder(experiment)

    print(f"\nGenerating {args.num_samples} labeled samples...")
    samples, labels, task_ids = generate_labeled_data(args.num_samples, seed=args.seed)
    print(f"  Generated {len(samples)} samples")
    print(f"  Unique transformations: {len(set(labels))}")
    print(f"  Label distribution: {dict(sorted([(l, labels.count(l)) for l in set(labels)], key=lambda x: -x[1]))}")

    print("\nComputing embeddings...")
    input_embeddings = embedder.embed_batch([inp for inp, _ in samples])
    output_embeddings = embedder.embed_batch([out for _, out in samples])
    print(f"  Input shape: {input_embeddings.shape}")
    print(f"  Output shape: {output_embeddings.shape}")

    # 1. Representation Quality
    print("\n" + "="*60)
    print("REPRESENTATION QUALITY")
    print("="*60)
    quality = compute_representation_quality(input_embeddings)
    for k, v in quality.items():
        print(f"  {k}: {v:.4f}")

    # 2. Linear Probe
    print("\n" + "="*60)
    print("LINEAR PROBE (Transformation Classification)")
    print("="*60)
    accuracy, per_class = train_linear_probe(input_embeddings, labels)
    print(f"  Overall accuracy: {accuracy:.4f}")
    if per_class:
        print("  Per-class accuracy:")
        for label, acc in sorted(per_class.items(), key=lambda x: -x[1])[:10]:
            print(f"    {label}: {acc:.4f}")

    # 3. Task Discrimination
    print("\n" + "="*60)
    print("TASK DISCRIMINATION")
    print("="*60)
    task_disc = evaluate_task_discrimination(input_embeddings, task_ids)
    for k, v in task_disc.items():
        print(f"  {k}: {v:.4f}")

    # 4. Few-Shot Matching (slow, can skip)
    few_shot_results = {"top1_accuracy": None, "top3_accuracy": None, "mean_rank": None}
    if not args.skip_few_shot:
        print("\n" + "="*60)
        print("FEW-SHOT LATENT MATCHING")
        print("="*60)
        few_shot_results = evaluate_few_shot_matching(embedder, samples, task_ids, k_shot=3)
        for k, v in few_shot_results.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Compile results
    results = EvaluationResults(
        effective_rank=quality["effective_rank"],
        covariance_condition_number=quality["covariance_condition_number"],
        embedding_variance=quality["embedding_variance"],
        isotropy_score=quality["isotropy_score"],
        linear_probe_accuracy=accuracy,
        linear_probe_per_class=per_class,
        few_shot_top1_accuracy=few_shot_results.get("top1_accuracy"),
        few_shot_top3_accuracy=few_shot_results.get("top3_accuracy"),
        few_shot_mean_rank=few_shot_results.get("mean_rank"),
        within_task_similarity=task_disc["within_task_similarity"],
        between_task_similarity=task_disc["between_task_similarity"],
        discrimination_gap=task_disc["discrimination_gap"],
    )

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  Effective Rank: {results.effective_rank:.1f} (higher = using more dimensions)")
    print(f"  Isotropy Score: {results.isotropy_score:.3f} (1.0 = perfectly uniform)")
    print(f"  Linear Probe Accuracy: {results.linear_probe_accuracy:.1%}")
    print(f"  Task Discrimination Gap: {results.discrimination_gap:.3f} (positive = good)")
    if results.few_shot_top1_accuracy is not None:
        print(f"  Few-Shot Top-1: {results.few_shot_top1_accuracy:.1%}")
        print(f"  Few-Shot Mean Rank: {results.few_shot_mean_rank:.1f}")

    # Interpretation
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)

    if results.effective_rank > 100:
        print("  ✓ High effective rank - representations use embedding space well")
    else:
        print("  ⚠ Low effective rank - may be underutilizing capacity")

    if results.isotropy_score > 0.7:
        print("  ✓ Good isotropy - SIGReg is working as intended")
    else:
        print("  ⚠ Low isotropy - embeddings may be collapsed in some dimensions")

    if results.linear_probe_accuracy and results.linear_probe_accuracy > 0.5:
        print("  ✓ Linear probe shows representations capture transformation semantics")
    elif results.linear_probe_accuracy and results.linear_probe_accuracy > 0.3:
        print("  ~ Linear probe shows weak transformation awareness")
    else:
        print("  ⚠ Linear probe fails - representations may not encode transformations")

    if results.discrimination_gap > 0.1:
        print("  ✓ Good task discrimination - same-task examples cluster together")
    else:
        print("  ⚠ Weak task discrimination - model may not distinguish tasks well")

    # Save results
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results.to_dict(), f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
