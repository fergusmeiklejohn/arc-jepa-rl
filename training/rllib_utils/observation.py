"""Helpers for flattening latent option observations into tensors."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


def flatten_value(value: Any) -> torch.Tensor:
    """Flatten arbitrary nested structures (scalars/sequences/mappings) to 1D tensor."""

    if isinstance(value, torch.Tensor):
        return value.reshape(-1).float()

    if isinstance(value, np.ndarray):
        return torch.from_numpy(value).reshape(-1).float()

    if isinstance(value, (bytes, bytearray)):
        return torch.tensor(list(value), dtype=torch.float32)

    if isinstance(value, Mapping):
        parts = [flatten_value(value[key]) for key in sorted(value.keys())]
        if not parts:
            raise ValueError("mapping observation is empty")
        return torch.cat(parts, dim=0)

    if isinstance(value, (list, tuple)):
        parts = [flatten_value(item) for item in value]
        if not parts:
            return torch.zeros(0, dtype=torch.float32)
        return torch.cat(parts, dim=0)

    if isinstance(value, (int, float, bool)):
        return torch.tensor([float(value)], dtype=torch.float32)

    raise TypeError(f"Unsupported observation value type: {type(value)}")


def flatten_observation(obs: Any) -> torch.Tensor:
    """Return flattened 1D tensor for a single observation."""

    tensor = flatten_value(obs)
    if tensor.dim() != 1:
        tensor = tensor.reshape(-1)
    return tensor


def flatten_batch_observation(obs: Mapping[str, Any]) -> torch.Tensor:
    """Flatten a dict observation assumed to already contain batched tensors."""

    if not isinstance(obs, Mapping):
        raise TypeError("batched observations must be a mapping")

    keys = sorted(obs.keys())
    parts = []
    batch_size = None
    for key in keys:
        value = obs[key]
        tensor = flatten_value(value)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        if batch_size is None:
            batch_size = tensor.size(0)
        elif tensor.size(0) != batch_size:
            raise ValueError("inconsistent batch dimensions in observation")
        parts.append(tensor)

    if not parts:
        raise ValueError("observation dict is empty")
    return torch.cat(parts, dim=-1)
