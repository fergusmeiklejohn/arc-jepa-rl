"""Run LeJEPA ablations (baseline, +VQ, +relational, +invariance, +SIGReg) on a shared manifest."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys  # noqa: E402

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from arcgen import Grid

from scripts._jepa_loader import build_jepa_dataloader
from training.jepa import ObjectCentricJEPAExperiment
from training.jepa.ablation import AblationVariant, apply_variant, build_ablation_variants, load_base_config
from training.jepa.dataset import GridPairBatch, TokenizedPairBatch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/training/jepa_tiny.yaml"),
        help="Base JEPA YAML config to clone for all variants.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/jepa/ablations"),
        help="Where to write JSON/Markdown summaries.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device to run ablation sweeps on.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Epochs per variant (keep small for smoke runs).",
    )
    parser.add_argument(
        "--max-variants",
        type=int,
        default=None,
        help="Optional cap on number of variants (debugging).",
    )
    return parser.parse_args()


def _cosine_alignment_rate(pairs: Iterable[tuple[torch.Tensor, torch.Tensor]], threshold: float = 0.5) -> float:
    total = 0
    aligned = 0
    for context, target in pairs:
        sims = F.cosine_similarity(context, target)
        aligned += int((sims > threshold).sum().item())
        total += sims.numel()
    return aligned / total if total else 0.0


def _collect_alignment_pairs(
    experiment: ObjectCentricJEPAExperiment, loader: Iterable[GridPairBatch | TokenizedPairBatch]
) -> List[tuple[torch.Tensor, torch.Tensor]]:
    pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    device = experiment.device
    for batch in loader:
        if isinstance(batch, GridPairBatch):
            context_repr = experiment._encode_context_sequences(batch.context).to(device)
            target_repr = experiment._encode_grids(batch.target).to(device)
        else:
            moved = batch.to(device, non_blocking=True)
            context_repr = experiment._encode_tokenized_context(moved).to(device)
            target_repr = experiment._encode_tokenized_target(moved).to(device)
        pairs.append((context_repr.detach(), target_repr.detach()))
    return pairs


def run_ablation_variant(
    variant: AblationVariant,
    base_config: Mapping[str, object],
    *,
    device: str,
    epochs: int,
) -> Dict[str, Any]:
    cfg = apply_variant(base_config, variant)
    experiment = ObjectCentricJEPAExperiment(cfg, device=device)
    train_loader, _, _, _ = build_jepa_dataloader(cfg, experiment.trainer.tokenizer_config)

    losses: List[float] = []
    for _ in range(max(1, epochs)):
        losses.append(float(experiment.train_epoch(train_loader)))

    alignment_pairs = _collect_alignment_pairs(experiment, train_loader)
    alignment = _cosine_alignment_rate(alignment_pairs)

    codebook_usage = None
    embedding_events = experiment.consume_embedding_metrics(flush=True)
    for event in reversed(embedding_events):
        if "vq_usage_ratio" in event:
            codebook_usage = float(event["vq_usage_ratio"])
            break

    return {
        "variant": asdict(variant),
        "loss": losses[-1] if losses else None,
        "loss_history": losses,
        "alignment_rate": alignment,
        "codebook_usage": codebook_usage,
    }


def write_markdown_summary(results: Sequence[Mapping[str, Any]], path: Path) -> None:
    lines = [
        "# LeJEPA Ablation Summary",
        "",
        "| Variant | Description | Loss | Alignment Rate | Codebook Usage |",
        "| --- | --- | --- | --- | --- |",
    ]
    for entry in results:
        variant = entry["variant"]
        name = variant["name"]
        desc = variant["description"]
        loss = entry.get("loss")
        align = entry.get("alignment_rate")
        codebook = entry.get("codebook_usage")
        lines.append(
            f"| {name} | {desc} | "
            f"{loss:.4f} | {align:.3f} | {codebook:.3f} |"
            if loss is not None and align is not None and codebook is not None
            else f"| {name} | {desc} | {loss or 'n/a'} | {align or 'n/a'} | {codebook or 'n/a'} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    base_config = load_base_config(args.base_config)
    variants = build_ablation_variants()
    if args.max_variants is not None:
        variants = variants[: args.max_variants]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for variant in variants:
        print(f"[ablation] running variant: {variant.name}")
        summary = run_ablation_variant(variant, base_config, device=args.device, epochs=args.epochs)
        results.append(summary)

    json_path = args.output_dir / "ablation_results.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    md_path = args.output_dir / "ablation_summary.md"
    write_markdown_summary(results, md_path)
    print(f"Wrote ablation results to {json_path} and {md_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
