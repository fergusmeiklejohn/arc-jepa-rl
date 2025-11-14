from pathlib import Path

import torch

from training.rllib_utils.bc_data import BehavioralCloningDataset, TraceStepRecord
from training.rllib_utils.bc_trainer import train_behavioral_cloning


def _build_records():
    records = []
    for idx in range(8):
        obs = {
            "current": [[0, 1], [1, 0]],
            "target": [[1, 0], [0, 1]],
            "steps": [idx],
        }
        records.append(
            TraceStepRecord(
                observation=obs,
                action=idx % 2,
                option_name=f"opt_{idx % 2}",
                termination=bool(idx % 2),
            )
        )
    return records


def test_train_behavioral_cloning(tmp_path):
    records = _build_records()
    train_ds = BehavioralCloningDataset(records[:6])
    val_ds = BehavioralCloningDataset(records[6:])
    result = train_behavioral_cloning(
        train_ds,
        val_ds,
        model_cfg={"hidden_dims": [16], "include_termination": True},
        optim_cfg={"epochs": 1, "batch_size": 2, "device": "cpu"},
        output_dir=tmp_path / "bc",
    )

    assert Path(result["best_model"]).exists()
    assert Path(result["metadata_path"]).exists()
