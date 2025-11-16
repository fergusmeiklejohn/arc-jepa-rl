import json

import torch

from arcgen import Grid
from training.jepa.dataset import TokenizedPairDataset, collate_tokenized_samples
from training.jepa.object_pipeline import ObjectTokenizerConfig
from training.jepa.pretokenizer import pretokenize_manifest
from training.modules.object_tokenizer import tokenize_grid_objects


def _write_manifest(tmp_path):
    manifest_path = tmp_path / "manifest.jsonl"
    records = [
        {
            "id": "sample_1",
            "input": [[0, 1], [0, 0]],
            "output": [[1, 0], [0, 0]],
        },
        {
            "id": "sample_2",
            "input": [[0, 0], [1, 1]],
            "output": [[1, 1], [0, 0]],
        },
    ]
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return manifest_path, records


def test_pretokenize_manifest_round_trip(tmp_path):
    manifest, records = _write_manifest(tmp_path)
    output_dir = tmp_path / "tokens"
    tokenizer_cfg = ObjectTokenizerConfig(max_objects=4, max_color_features=2, normalize=False)

    summary = pretokenize_manifest(
        manifest,
        output_dir,
        tokenizer_cfg=tokenizer_cfg,
        context_window=1,
        target_offset=1,
        shard_size=1,
    )
    assert summary["total_samples"] == len(records)

    dataset = TokenizedPairDataset(output_dir)
    assert len(dataset) == len(records)

    sample = dataset[0]
    assert sample.context_features.shape[0] == 1
    input_tokens = tokenize_grid_objects(Grid(records[0]["input"]), **tokenizer_cfg.as_kwargs())
    target_tokens = tokenize_grid_objects(Grid(records[0]["output"]), **tokenizer_cfg.as_kwargs())

    assert torch.allclose(
        sample.context_features[0],
        torch.tensor(input_tokens.features, dtype=torch.float32),
    )
    assert torch.allclose(
        sample.target_features,
        torch.tensor(target_tokens.features, dtype=torch.float32),
    )
    assert torch.equal(
        sample.target_mask,
        torch.tensor(target_tokens.mask, dtype=torch.float32),
    )


def test_collate_tokenized_samples_builds_batch(tmp_path):
    manifest, _ = _write_manifest(tmp_path)
    output_dir = tmp_path / "tokens"
    tokenizer_cfg = ObjectTokenizerConfig(max_objects=4, max_color_features=2)

    pretokenize_manifest(
        manifest,
        output_dir,
        tokenizer_cfg=tokenizer_cfg,
        context_window=1,
        target_offset=1,
        shard_size=8,
    )

    dataset = TokenizedPairDataset(output_dir)
    batch = collate_tokenized_samples([dataset[0], dataset[1]])

    assert batch.context_features.shape[0] == 2
    assert batch.context_length == 1
    assert len(batch.metadata) == 2
