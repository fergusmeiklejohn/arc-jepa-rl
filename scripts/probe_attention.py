#!/usr/bin/env python3
"""Probe relational attention patterns in trained JEPA model.

This script analyzes what the attention layers are attending to:
- Do different transformations produce different attention patterns?
- Are attention heads specializing (e.g., one for symmetry, one for color)?
- Is attention focusing on transformation-relevant features?
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from arcgen import Grid, SyntheticARCGenerator, GeneratorConfig
from training.jepa import ObjectCentricJEPAExperiment


def generate_samples_by_transform(
    num_per_transform: int = 50,
    seed: int = 42,
) -> Dict[str, List[Tuple[Grid, Grid]]]:
    """Generate samples grouped by transformation type."""
    config = GeneratorConfig(
        min_grid_size=5,
        max_grid_size=10,
        min_colors=3,
        max_colors=5,
    )
    generator = SyntheticARCGenerator(
        config,
        seed=seed,
        program_length_schedule={1: 1.0},
    )

    samples_by_transform: Dict[str, List[Tuple[Grid, Grid]]] = defaultdict(list)
    max_attempts = num_per_transform * 20

    for _ in range(max_attempts):
        try:
            task = generator.sample_task("atomic")
            if task.rule_trace:
                prim = task.rule_trace[0].primitive
                if len(samples_by_transform[prim]) < num_per_transform:
                    samples_by_transform[prim].append((task.input_grid, task.output_grid))
        except RuntimeError:
            continue

        # Check if we have enough for all transforms
        if all(len(v) >= num_per_transform for v in samples_by_transform.values()):
            break

    return dict(samples_by_transform)


def extract_attention_weights(
    experiment: ObjectCentricJEPAExperiment,
    grid: Grid,
) -> List[torch.Tensor]:
    """Extract attention weights from all layers for a single grid.

    Returns list of attention tensors, one per layer, each of shape [num_heads, num_objects, num_objects]
    """
    # Hook to capture attention weights
    attention_weights = []
    hooks = []

    def make_hook(layer_idx):
        def hook(module, input, output):
            # MultiheadAttention returns (attn_output, attn_weights)
            if isinstance(output, tuple) and len(output) >= 2:
                attn_weights = output[1]  # [batch, num_heads, seq_len, seq_len] or [batch, seq, seq]
                if attn_weights is not None:
                    attention_weights.append((layer_idx, attn_weights.detach()))
        return hook

    # Register hooks on attention layers
    encoder = experiment.trainer.object_encoder
    if hasattr(encoder, 'relational_layers'):
        for i, layer in enumerate(encoder.relational_layers):
            if hasattr(layer, 'self_attn'):
                h = layer.self_attn.register_forward_hook(make_hook(i))
                hooks.append(h)

    try:
        experiment.trainer.encoder.eval()
        with torch.no_grad():
            encoding = experiment.trainer.object_encoder.encode([grid], device=experiment.device)
    finally:
        for h in hooks:
            h.remove()

    return [w for _, w in sorted(attention_weights, key=lambda x: x[0])]


def analyze_attention_entropy(attention_weights: List[torch.Tensor]) -> Dict[str, float]:
    """Compute attention entropy per head - high entropy = diffuse attention, low = focused."""
    results = {}

    for layer_idx, attn in enumerate(attention_weights):
        # attn shape: [batch, num_heads, seq_len, seq_len] or similar
        if attn.dim() == 3:
            attn = attn.unsqueeze(1)  # Add head dimension if missing

        batch, num_heads, seq_len, _ = attn.shape

        for head_idx in range(num_heads):
            head_attn = attn[0, head_idx]  # [seq_len, seq_len]
            # Compute entropy for each query position, average
            entropy = -(head_attn * torch.log(head_attn + 1e-10)).sum(dim=-1).mean()
            results[f"layer{layer_idx}_head{head_idx}_entropy"] = entropy.item()

    return results


def analyze_attention_by_transform(
    experiment: ObjectCentricJEPAExperiment,
    samples_by_transform: Dict[str, List[Tuple[Grid, Grid]]],
) -> Dict[str, Dict[str, float]]:
    """Analyze if attention patterns differ by transformation type."""

    transform_attention_stats = {}

    for transform, samples in samples_by_transform.items():
        all_entropies = defaultdict(list)

        for inp, out in samples[:20]:  # Limit for speed
            weights = extract_attention_weights(experiment, inp)
            if weights:
                entropies = analyze_attention_entropy(weights)
                for k, v in entropies.items():
                    all_entropies[k].append(v)

        # Average across samples
        transform_attention_stats[transform] = {
            k: np.mean(v) for k, v in all_entropies.items()
        }

    return transform_attention_stats


def compute_attention_discrimination(
    stats: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    """Check if attention patterns can discriminate between transforms."""

    transforms = list(stats.keys())
    if len(transforms) < 2:
        return {}

    # For each attention head, compute variance across transforms
    head_discrimination = {}

    # Get all heads
    sample_stats = next(iter(stats.values()))
    heads = list(sample_stats.keys())

    for head in heads:
        values = [stats[t].get(head, 0) for t in transforms]
        # Higher variance = more discrimination
        head_discrimination[head] = np.var(values)

    return head_discrimination


def main():
    parser = argparse.ArgumentParser(description="Probe JEPA attention patterns")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint")
    parser.add_argument("--num-samples", type=int, default=30, help="Samples per transform")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})

    print("Building experiment...")
    experiment = ObjectCentricJEPAExperiment(config, device=args.device)
    experiment.trainer.encoder.load_state_dict(checkpoint["model_state"])
    experiment.projection_head.load_state_dict(checkpoint["projection_state"])

    print(f"\nGenerating {args.num_samples} samples per transformation...")
    samples = generate_samples_by_transform(args.num_samples, args.seed)
    print(f"  Generated samples for {len(samples)} transformations")
    for t, s in samples.items():
        print(f"    {t}: {len(s)} samples")

    print("\nAnalyzing attention patterns...")

    # Check if model has attention layers we can probe
    # The ObjectCentricJEPAEncoder wraps ObjectTokenEncoder, which has 'relational'
    obj_encoder = experiment.trainer.object_encoder
    inner_encoder = obj_encoder.encoder if hasattr(obj_encoder, 'encoder') else obj_encoder
    has_relational = hasattr(inner_encoder, 'relational') and inner_encoder.relational is not None

    if not has_relational:
        print("  Model does not have relational attention layers to probe")
        print("  Falling back to embedding similarity analysis...")

    # Always do embedding similarity analysis - most informative for downstream transfer
    print("\n" + "="*60)
    print("EMBEDDING SIMILARITY BY TRANSFORM")
    print("="*60)

    from scripts.evaluate_downstream import JEPAEmbedder
    embedder = JEPAEmbedder(experiment)

    # Compute mean embedding per transform
    transform_means = {}
    transform_embeddings = {}
    for transform, trans_samples in samples.items():
        embeddings = []
        for inp, _ in trans_samples[:20]:
            emb = embedder.embed(inp)
            embeddings.append(emb)
        emb_stack = torch.stack(embeddings)
        transform_means[transform] = emb_stack.mean(dim=0)
        transform_embeddings[transform] = emb_stack

    # Compute pairwise similarities between transform centroids
    print("\nCentroid similarities between transforms (lower = better separation):")
    transforms = list(transform_means.keys())
    all_between_sims = []
    for i, t1 in enumerate(transforms):
        for t2 in transforms[i+1:]:
            sim = F.cosine_similarity(
                transform_means[t1].unsqueeze(0),
                transform_means[t2].unsqueeze(0)
            ).item()
            all_between_sims.append(sim)
            # Only print a few
    print(f"  Mean between-transform centroid similarity: {np.mean(all_between_sims):.3f}")
    print(f"  Min: {np.min(all_between_sims):.3f}, Max: {np.max(all_between_sims):.3f}")

    # Compute within-transform vs between-transform similarity
    print("\n" + "="*60)
    print("WITHIN-TRANSFORM COHERENCE")
    print("="*60)
    print("\nWithin-transform embedding similarity (higher = more coherent):")
    all_within_sims = []
    for transform in transforms:
        emb_stack = transform_embeddings[transform]
        emb_norm = F.normalize(emb_stack, dim=-1)
        sim_matrix = emb_norm @ emb_norm.t()

        # Off-diagonal mean
        n = sim_matrix.size(0)
        mask = ~torch.eye(n, dtype=torch.bool)
        within_sim = sim_matrix[mask].mean().item()
        all_within_sims.append(within_sim)
        print(f"  {transform}: {within_sim:.3f}")

    print(f"\nMean within-transform similarity: {np.mean(all_within_sims):.3f}")
    print(f"Mean between-transform centroid similarity: {np.mean(all_between_sims):.3f}")
    gap = np.mean(all_within_sims) - np.mean(all_between_sims)
    print(f"Discrimination gap: {gap:.3f} (positive = good clustering)")

    # Compute input-output delta similarity by transform
    print("\n" + "="*60)
    print("TRANSFORMATION DELTA ANALYSIS")
    print("="*60)
    print("\nAre input->output deltas consistent within transforms?")

    transform_deltas = {}
    for transform, trans_samples in samples.items():
        deltas = []
        for inp, out in trans_samples[:20]:
            inp_emb = embedder.embed(inp)
            out_emb = embedder.embed(out)
            delta = out_emb - inp_emb
            deltas.append(delta)
        transform_deltas[transform] = torch.stack(deltas)

    # Within-transform delta similarity
    print("\nWithin-transform delta coherence (higher = consistent transformation direction):")
    delta_coherences = []
    for transform in transforms:
        deltas = transform_deltas[transform]
        deltas_norm = F.normalize(deltas, dim=-1)
        sim_matrix = deltas_norm @ deltas_norm.t()
        n = sim_matrix.size(0)
        mask = ~torch.eye(n, dtype=torch.bool)
        coherence = sim_matrix[mask].mean().item()
        delta_coherences.append(coherence)
        print(f"  {transform}: {coherence:.3f}")

    print(f"\nMean delta coherence: {np.mean(delta_coherences):.3f}")
    print("(Higher coherence means the model learns consistent transformation directions)")

    # Between-transform delta similarity
    mean_deltas = {t: transform_deltas[t].mean(dim=0) for t in transforms}
    between_delta_sims = []
    for i, t1 in enumerate(transforms):
        for t2 in transforms[i+1:]:
            sim = F.cosine_similarity(
                mean_deltas[t1].unsqueeze(0),
                mean_deltas[t2].unsqueeze(0)
            ).item()
            between_delta_sims.append(sim)

    print(f"\nMean between-transform delta similarity: {np.mean(between_delta_sims):.3f}")
    delta_gap = np.mean(delta_coherences) - np.mean(between_delta_sims)
    print(f"Delta discrimination gap: {delta_gap:.3f}")

    print("\n" + "="*60)
    print("DIAGNOSIS")
    print("="*60)
    if gap > 0.05:
        print("  ✓ Inputs cluster somewhat by transformation type")
    else:
        print("  ⚠ Inputs do NOT cluster by transformation type")

    if np.mean(delta_coherences) > 0.3:
        print("  ✓ Transformation deltas are somewhat consistent within transforms")
    else:
        print("  ⚠ Transformation deltas are NOT consistent - model doesn't learn transformation direction")

    if delta_gap > 0.1:
        print("  ✓ Different transforms have distinguishable deltas")
    else:
        print("  ⚠ Different transforms have similar deltas - can't distinguish transformations")

    return

    print(f"  Found {len(weights)} attention layers")
    for i, w in enumerate(weights):
        print(f"    Layer {i}: shape {w.shape}")

    # Full analysis
    print("\n" + "="*60)
    print("ATTENTION ENTROPY BY TRANSFORMATION")
    print("="*60)

    stats = analyze_attention_by_transform(experiment, samples)

    # Print summary
    print("\nMean attention entropy per transform (lower = more focused):")
    for transform, entropy_dict in stats.items():
        mean_entropy = np.mean(list(entropy_dict.values()))
        print(f"  {transform}: {mean_entropy:.3f}")

    print("\n" + "="*60)
    print("HEAD DISCRIMINATION SCORES")
    print("="*60)

    discrimination = compute_attention_discrimination(stats)
    sorted_heads = sorted(discrimination.items(), key=lambda x: -x[1])

    print("\nHeads ranked by discrimination power (higher = better):")
    for head, score in sorted_heads[:10]:
        print(f"  {head}: {score:.6f}")

    if sorted_heads:
        best_head = sorted_heads[0][0]
        print(f"\nMost discriminative head: {best_head}")
        print(f"  Values by transform:")
        for transform in stats:
            val = stats[transform].get(best_head, 0)
            print(f"    {transform}: {val:.3f}")


if __name__ == "__main__":
    main()
