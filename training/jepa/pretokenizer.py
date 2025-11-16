"""Utilities to convert manifest grids into pre-tokenized JEPA shards."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, MutableSequence

from arcgen import Grid

try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

from training.jepa.dataset import iter_manifest_examples
from training.jepa.object_pipeline import ObjectTokenizerConfig
from training.modules.object_tokenizer import tokenize_grid_objects


def _ensure_torch_available() -> None:
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required to pre-tokenize JEPA manifests")


@dataclass
class _ShardBuffer:
    context_features: List[List[List[List[float]]]]
    context_mask: List[List[List[float]]]
    context_adjacency: List[List[List[List[float]]]]
    target_features: List[List[List[float]]]
    target_mask: List[List[float]]
    target_adjacency: List[List[List[float]]]
    metadata: MutableSequence[Mapping[str, object] | None]

    def __init__(self) -> None:
        self.context_features = []
        self.context_mask = []
        self.context_adjacency = []
        self.target_features = []
        self.target_mask = []
        self.target_adjacency = []
        self.metadata = []

    def __len__(self) -> int:
        return len(self.target_features)

    def append(
        self,
        *,
        context_features: List[List[List[float]]],
        context_mask: List[List[float]],
        context_adjacency: List[List[List[float]]],
        target_features: List[List[float]],
        target_mask: List[float],
        target_adjacency: List[List[float]],
        metadata: Mapping[str, object] | None,
    ) -> None:
        self.context_features.append(context_features)
        self.context_mask.append(context_mask)
        self.context_adjacency.append(context_adjacency)
        self.target_features.append(target_features)
        self.target_mask.append(target_mask)
        self.target_adjacency.append(target_adjacency)
        self.metadata.append(metadata)

    def flush(self, shard_path: Path) -> int:
        if not self.target_features:
            return 0

        payload = {
            "context_features": torch.tensor(self.context_features, dtype=torch.float32),
            "context_mask": torch.tensor(self.context_mask, dtype=torch.float32),
            "context_adjacency": torch.tensor(self.context_adjacency, dtype=torch.float32),
            "target_features": torch.tensor(self.target_features, dtype=torch.float32),
            "target_mask": torch.tensor(self.target_mask, dtype=torch.float32),
            "target_adjacency": torch.tensor(self.target_adjacency, dtype=torch.float32),
            "metadata": list(self.metadata),
        }
        torch.save(payload, shard_path)
        count = len(self.target_features)
        self.__init__()
        return count


def _tokenize_grid(grid: Grid, cfg: ObjectTokenizerConfig):
    tokens = tokenize_grid_objects(grid, **cfg.as_kwargs())
    return tokens.features, tokens.mask, tokens.adjacency


def pretokenize_manifest(
    manifest_path: Path | str,
    output_dir: Path | str,
    *,
    tokenizer_cfg: ObjectTokenizerConfig,
    context_window: int,
    target_offset: int,
    shard_size: int = 2048,
    limit: int | None = None,
    overwrite: bool = False,
) -> Mapping[str, object]:
    """Convert ``manifest_path`` into tensor shards ready for fast JEPA training."""

    _ensure_torch_available()

    manifest = Path(manifest_path)
    if not manifest.exists():
        raise FileNotFoundError(f"manifest not found: {manifest}")

    out_dir = Path(output_dir)
    if out_dir.exists():
        if not overwrite and any(out_dir.iterdir()):
            raise FileExistsError(f"output directory {out_dir} is not empty (pass overwrite=True to replace)")
        if overwrite:
            for child in out_dir.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    shards_meta: list[dict] = []
    buffer = _ShardBuffer()
    processed = 0
    shard_index = 0

    for example in iter_manifest_examples(
        manifest,
        context_window=context_window,
        target_offset=target_offset,
    ):
        if example.context_sequence is None:
            raise ValueError("manifest example is missing context")
        context_tokens: List[List[List[float]]] = []
        context_masks: List[List[List[float]]] = []
        context_adjacency: List[List[List[List[float]]]] = []
        for grid in example.context_sequence:
            features, mask, adjacency = _tokenize_grid(grid, tokenizer_cfg)
            context_tokens.append(features)
            context_masks.append([float(value) for value in mask])
            context_adjacency.append(adjacency)

        target_features, target_mask, target_adj = _tokenize_grid(example.target, tokenizer_cfg)

        buffer.append(
            context_features=context_tokens,
            context_mask=context_masks,
            context_adjacency=context_adjacency,
            target_features=target_features,
            target_mask=[float(value) for value in target_mask],
            target_adjacency=target_adj,
            metadata=example.metadata,
        )
        processed += 1

        if len(buffer) >= shard_size:
            shard_index += 1
            shard_name = f"shard_{shard_index:05d}.pt"
            shard_path = out_dir / shard_name
            count = buffer.flush(shard_path)
            shards_meta.append({"path": shard_name, "num_samples": count})

        if limit is not None and processed >= limit:
            break

    if len(buffer) > 0:
        shard_index += 1
        shard_name = f"shard_{shard_index:05d}.pt"
        shard_path = out_dir / shard_name
        count = buffer.flush(shard_path)
        shards_meta.append({"path": shard_name, "num_samples": count})

    total_samples = sum(entry["num_samples"] for entry in shards_meta)
    if total_samples == 0:
        raise RuntimeError("no samples were produced during pre-tokenization")

    summary = {
        "manifest": str(manifest),
        "context_length": context_window,
        "target_offset": target_offset,
        "max_objects": tokenizer_cfg.max_objects,
        "feature_dim": tokenizer_cfg.feature_dim,
        "tokenizer": tokenizer_cfg.as_kwargs(),
        "shard_size": shard_size,
        "total_samples": total_samples,
        "shards": shards_meta,
    }
    metadata_path = out_dir / "metadata.json"
    metadata_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
