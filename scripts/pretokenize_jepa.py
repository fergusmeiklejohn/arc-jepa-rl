"""Pre-tokenize JEPA manifest pairs into tensor shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.jepa import load_jepa_config
from training.jepa.object_pipeline import build_object_tokenizer_config
from training.jepa.pretokenizer import pretokenize_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-tokenize JEPA manifests into .pt shards")
    parser.add_argument("--config", type=Path, required=True, help="JEPA YAML config to read defaults from")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional override for dataset manifest path")
    parser.add_argument("--output", type=Path, required=True, help="Directory to write tokenized shards into")
    parser.add_argument("--context-window", type=int, default=None, help="Override context window length")
    parser.add_argument("--target-offset", type=int, default=None, help="Override prediction offset")
    parser.add_argument("--shard-size", type=int, default=2048, help="Number of samples per .pt shard")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of samples to tokenize")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing output directory contents")
    return parser.parse_args()


def _resolve_data_settings(config: Mapping[str, object], args: argparse.Namespace) -> tuple[int, int, Path]:
    data_cfg = config.get("data", {})
    if not isinstance(data_cfg, Mapping):
        raise ValueError("config['data'] must be a mapping")

    context_window = int(
        args.context_window
        or data_cfg.get("context_window")
        or data_cfg.get("context_length", 3),
    )
    target_offset = int(args.target_offset or data_cfg.get("target_offset", 1))
    if context_window <= 0 or target_offset <= 0:
        raise ValueError("context window and target offset must be positive")

    manifest_value = args.manifest or config.get("dataset_manifest")
    if not manifest_value:
        raise ValueError("either --manifest must be provided or config.dataset_manifest must be set")
    manifest_path = Path(manifest_value)
    return context_window, target_offset, manifest_path


def main() -> None:
    args = parse_args()
    config = load_jepa_config(args.config)
    tokenizer_cfg = build_object_tokenizer_config(config.get("tokenizer"))
    context_window, target_offset, manifest_path = _resolve_data_settings(config, args)

    summary = pretokenize_manifest(
        manifest_path,
        args.output,
        tokenizer_cfg=tokenizer_cfg,
        context_window=context_window,
        target_offset=target_offset,
        shard_size=int(args.shard_size),
        limit=args.limit,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
