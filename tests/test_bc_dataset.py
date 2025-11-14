import json
from pathlib import Path

import pytest
import torch

from training.rllib_utils.bc_data import BehavioralCloningDataset, load_option_traces, split_records


def _write_trace(tmp_path: Path, entries):
    path = tmp_path / "traces.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")
    return path


def test_load_option_traces_and_dataset(tmp_path):
    entries = [
        {
            "task_id": "t1",
            "steps": [
                {
                    "observation": {
                        "current": [[0, 1], [2, 3]],
                        "target": [[3, 2], [1, 0]],
                        "steps": [0],
                    },
                    "action": 1,
                    "option_name": "mirror_x",
                    "termination": True,
                },
                {
                    "observation": {
                        "current": [[1, 0], [2, 3]],
                        "target": [[3, 2], [1, 0]],
                        "steps": [1],
                    },
                    "action": 0,
                    "option_name": "mirror_y",
                    "termination": False,
                },
            ],
        }
    ]
    path = _write_trace(tmp_path, entries)
    records = load_option_traces(path)
    assert len(records) == 2
    dataset = BehavioralCloningDataset(records)
    sample = dataset[0]
    obs, action, termination, mask = sample
    assert obs.shape[0] == dataset.feature_dim
    assert action.item() == 1
    assert termination.item() in (0.0, 1.0)
    assert mask.item() in (0, 1)
    assert dataset.action_dim == 2
    assert dataset.option_id_map[1] == "mirror_x"


def test_split_records(tmp_path):
    entries = [
        {
            "task_id": f"t{idx}",
            "steps": [
                {
                    "observation": {"current": [[idx]], "target": [[idx]], "steps": [idx]},
                    "action": idx % 2,
                }
            ],
        }
        for idx in range(6)
    ]
    path = _write_trace(tmp_path, entries)
    records = load_option_traces(path)
    train, val = split_records(records, val_ratio=0.3, seed=1)
    assert len(train) + len(val) == len(records)
    assert len(train) > 0
    assert len(val) > 0
