import json
from pathlib import Path

from arcgen import Grid
from training.jepa import AugmentationConfig, GridPairBatch, ManifestGridPairDataset


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
    assert all(isinstance(item, Grid) for item in batch.context)
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
    # contexts should correspond to the second and third frames in the sequence
    context_values = [batch.context[0].cells[1][1] for batch in batches]
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
    masked_context = batch.context[0]
    assert all(value == 0 for row in masked_context.cells for value in row)
