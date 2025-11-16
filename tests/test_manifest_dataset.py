import json
from pathlib import Path

import pytest

from arcgen import Grid
from training.jepa import (
    AugmentationConfig,
    GridPairBatch,
    ManifestGridPairDataset,
    ManifestTokenizedPairDataset,
)


def _write_manifest(tmp_path: Path, records: list[dict]) -> Path:
    path = tmp_path / "dataset.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            json.dump(record, handle)
            handle.write("\n")
    return path


def test_manifest_dataset_batches_context_target(tmp_path):
    manifest = [
        {
            "id": "sample-1",
            "context": [
                [
                    [0, 1],
                    [0, 0],
                ]
            ],
            "target": [
                [1, 1],
                [0, 0],
            ],
        },
        {
            "id": "sample-2",
            "context": [
                [
                    [0, 2],
                    [0, 2],
                ]
            ],
            "target": [
                [2, 2],
                [0, 0],
            ],
        },
    ]
    path = _write_manifest(tmp_path, manifest)

    dataset = ManifestGridPairDataset(path, batch_size=2, shuffle=False)
    batches = list(dataset)

    assert len(batches) == 1
    batch = batches[0]
    assert isinstance(batch, GridPairBatch)
    assert len(batch.context) == 2
    assert len(batch.target) == 2
    assert batch.context_length == 3
    assert all(isinstance(sequence, tuple) for sequence in batch.context)
    assert all(len(sequence) == batch.context_length for sequence in batch.context)
    assert all(isinstance(item, Grid) for sequence in batch.context for item in sequence)
    assert all(isinstance(item, Grid) for item in batch.target)


def test_manifest_dataset_supports_frame_windowing(tmp_path):
    frames = [
        [
            [0, 0],
            [0, 1],
        ],
        [
            [0, 0],
            [0, 2],
        ],
        [
            [0, 0],
            [0, 3],
        ],
        [
            [0, 0],
            [0, 4],
        ],
    ]
    manifest = [{"id": "seq-1", "frames": frames}]
    path = _write_manifest(tmp_path, manifest)

    dataset = ManifestGridPairDataset(
        path,
        batch_size=1,
        shuffle=False,
        context_window=2,
        target_offset=1,
    )
    batches = list(dataset)

    assert len(batches) == 2
    assert batches[0].context_length == 2
    # contexts should correspond to the second and third frames in the sequence
    context_values = [batch.context[0][-1].cells[1][1] for batch in batches]
    assert context_values == [2, 3]
    target_values = [batch.target[0].cells[1][1] for batch in batches]
    assert target_values == [3, 4]


def test_manifest_dataset_applies_masking_augmentation(tmp_path):
    manifest = [
        {
            "context": [
                [
                    [0, 1],
                    [2, 3],
                ]
            ],
            "target": [
                [3, 3],
                [3, 3],
            ],
        }
    ]
    path = _write_manifest(tmp_path, manifest)

    augmentations = AugmentationConfig(
        mask_ratio=1.0,
        random_crop_radius=0,
        palette_permutation=False,
        gaussian_noise_std=0.0,
        min_grid_size_for_crop=0,
        background_color=0,
    )
    dataset = ManifestGridPairDataset(
        path,
        batch_size=1,
        shuffle=False,
        augmentations=augmentations,
        seed=123,
    )

    batch = next(iter(dataset))
    masked_context_sequence = batch.context[0]
    assert all(
        value == 0
        for grid in masked_context_sequence
        for row in grid.cells
        for value in row
    )


def test_manifest_tokenized_dataset_returns_tokenized_samples(tmp_path):
    torch = pytest.importorskip("torch")

    manifest = [
        {
            "context": [
                [
                    [0, 1],
                    [0, 0],
                ],
                [
                    [0, 2],
                    [0, 2],
                ],
            ],
            "target": [
                [1, 1],
                [0, 0],
            ],
            "metadata": {"sample_id": "tok-1"},
        }
    ]
    path = _write_manifest(tmp_path, manifest)

    dataset = ManifestTokenizedPairDataset(
        path,
        context_window=2,
        target_offset=1,
        augmentations=None,
        tokenizer_config={
            "max_objects": 2,
            "max_color_features": 1,
        },
        seed=17,
    )

    sample = dataset[0]
    assert sample.context_features.shape == (2, dataset.max_objects, dataset.feature_dim)
    assert sample.target_features.shape == (dataset.max_objects, dataset.feature_dim)
    assert sample.metadata == {"sample_id": "tok-1"}
    assert torch.all(sample.context_mask >= 0)
