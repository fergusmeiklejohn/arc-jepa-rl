"""Data loading utilities for behavioral cloning on latent option traces."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from .observation import flatten_observation


@dataclass(frozen=True)
class TraceStepRecord:
    """Single state-action example extracted from an option rollout."""

    observation: Mapping[str, Any]
    action: int
    option_name: Optional[str] = None
    termination: Optional[bool] = None
    task_id: Optional[str] = None


def load_option_traces(path: Path, *, max_steps: Optional[int] = None) -> List[TraceStepRecord]:
    """Load JSONL episodes and flatten into per-step records."""

    records: List[TraceStepRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            steps = data.get("steps")
            if not isinstance(steps, Sequence):
                raise ValueError("each JSONL entry must contain a 'steps' list")
            task_id = data.get("task_id")
            for idx, step in enumerate(steps):
                if max_steps is not None and idx >= max_steps:
                    break
                observation = step.get("observation")
                if not isinstance(observation, Mapping):
                    raise ValueError("each step must contain an 'observation' mapping")
                action = int(step["action"])
                option_name = step.get("option") or step.get("option_name")
                termination = step.get("termination")
                if termination is not None:
                    termination = bool(termination)
                records.append(
                    TraceStepRecord(
                        observation=observation,
                        action=action,
                        option_name=option_name,
                        termination=termination,
                        task_id=task_id,
                    )
                )
    if not records:
        raise ValueError(f"No steps found in {path}")
    return records


def split_records(
    records: Sequence[TraceStepRecord],
    *,
    val_ratio: float,
    seed: int = 0,
) -> Tuple[List[TraceStepRecord], List[TraceStepRecord]]:
    """Shuffle and split records into train/val partitions."""

    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must lie in [0, 1)")
    indices = list(range(len(records)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_size = int(len(records) * val_ratio)
    val_indices = set(indices[:val_size])
    train_records: List[TraceStepRecord] = []
    val_records: List[TraceStepRecord] = []
    for idx, record in enumerate(records):
        if idx in val_indices:
            val_records.append(record)
        else:
            train_records.append(record)
    if not train_records:
        raise ValueError("train split is empty; decrease val_ratio")
    if not val_records:
        val_records.append(train_records.pop())
    return train_records, val_records


class BehavioralCloningDataset(Dataset):
    """Torch dataset that prepares flattened tensors for BC training."""

    def __init__(self, records: Sequence[TraceStepRecord]):
        if not records:
            raise ValueError("records must not be empty")
        features = []
        actions = []
        terminations = []
        masks = []
        option_lookup: MutableMapping[int, str] = {}

        for record in records:
            vector = flatten_observation(record.observation)
            features.append(vector)
            actions.append(int(record.action))
            if record.option_name:
                option_lookup.setdefault(int(record.action), record.option_name)
            if record.termination is None:
                terminations.append(0.0)
                masks.append(False)
            else:
                terminations.append(float(record.termination))
                masks.append(True)

        self.features = torch.stack(features, dim=0)
        self.actions = torch.tensor(actions, dtype=torch.long)
        self.terminations = torch.tensor(terminations, dtype=torch.float32)
        self.termination_mask = torch.tensor(masks, dtype=torch.bool)
        self.action_dim = int(self.actions.max().item()) + 1
        self.feature_dim = self.features.size(1)
        self.option_id_map = {idx: name for idx, name in option_lookup.items()}

    def __len__(self) -> int:
        return self.features.size(0)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.features[index],
            self.actions[index],
            self.terminations[index],
            self.termination_mask[index],
        )

    @property
    def has_termination(self) -> bool:
        return bool(self.termination_mask.any())
