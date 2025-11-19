"""Dataset utilities for program-conditioned JEPA training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from arcgen import Grid

from .object_pipeline import ObjectTokenizerConfig
from training.modules.object_tokenizer import tokenize_grid_objects

try:  # pragma: no cover - PyTorch optional at import time
    import torch
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    Dataset = object  # type: ignore[misc]


def _ensure_torch() -> None:
    if torch is None:  # pragma: no cover - defensive fallback
        raise RuntimeError("PyTorch is required for program-conditioned JEPA datasets")


def _grid_from_data(value: object) -> Grid:
    if isinstance(value, Grid):
        return value
    if isinstance(value, Sequence) and value and not isinstance(value, (str, bytes)):
        first = value[0]
        if isinstance(first, Sequence):
            return Grid(value)  # type: ignore[arg-type]
    raise ValueError("grid data must be a 2D sequence of integers")


@dataclass(frozen=True)
class ProgramStepRecord:
    primitive: str
    params: Mapping[str, object]


@dataclass(frozen=True)
class ProgramTripleRecord:
    input_grid: Grid
    output_grid: Grid
    program: Sequence[ProgramStepRecord]
    metadata: Mapping[str, object] | None = None


@dataclass(frozen=True)
class ProgramTripleSample:
    input_features: "torch.Tensor"
    input_mask: "torch.Tensor"
    input_adjacency: "torch.Tensor"
    output_features: "torch.Tensor"
    output_mask: "torch.Tensor"
    output_adjacency: "torch.Tensor"
    program_ids: "torch.Tensor"
    program_params: "torch.Tensor"
    program_mask: "torch.Tensor"
    metadata: Mapping[str, object] | None


@dataclass(frozen=True)
class ProgramTripleBatch:
    input_features: "torch.Tensor"
    input_mask: "torch.Tensor"
    input_adjacency: "torch.Tensor"
    output_features: "torch.Tensor"
    output_mask: "torch.Tensor"
    output_adjacency: "torch.Tensor"
    program_ids: "torch.Tensor"
    program_params: "torch.Tensor"
    program_mask: "torch.Tensor"
    metadata: Sequence[Mapping[str, object] | None]

    def to(
        self,
        device: "torch.device | str",
        *,
        non_blocking: bool = False,
    ) -> "ProgramTripleBatch":  # pragma: no cover - thin convenience wrapper
        return ProgramTripleBatch(
            input_features=self.input_features.to(device, non_blocking=non_blocking),
            input_mask=self.input_mask.to(device, non_blocking=non_blocking),
            input_adjacency=self.input_adjacency.to(device, non_blocking=non_blocking),
            output_features=self.output_features.to(device, non_blocking=non_blocking),
            output_mask=self.output_mask.to(device, non_blocking=non_blocking),
            output_adjacency=self.output_adjacency.to(device, non_blocking=non_blocking),
            program_ids=self.program_ids.to(device, non_blocking=non_blocking),
            program_params=self.program_params.to(device, non_blocking=non_blocking),
            program_mask=self.program_mask.to(device, non_blocking=non_blocking),
            metadata=self.metadata,
        )


class ProgramTraceTokenizer:
    """Encodes primitive sequences (with parameters) into padded tensors."""

    def __init__(
        self,
        primitives: Sequence[str],
        parameter_names: Sequence[str],
        *,
        max_length: int,
    ) -> None:
        _ensure_torch()
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        unique_primitives = sorted(set(primitives))
        if not unique_primitives:
            raise ValueError("primitive vocabulary must not be empty")

        self._pad_id = 0
        self._primitive_to_index: dict[str, int] = {name: idx + 1 for idx, name in enumerate(unique_primitives)}
        self._index_to_primitive = {idx: name for name, idx in self._primitive_to_index.items()}
        self._parameter_to_index: dict[str, int] = {name: idx for idx, name in enumerate(sorted(set(parameter_names)))}
        self.max_length = max_length

    @property
    def vocab_size(self) -> int:
        return len(self._primitive_to_index) + 1  # padding id

    @property
    def parameter_dim(self) -> int:
        return len(self._parameter_to_index)

    def encode(self, steps: Sequence[ProgramStepRecord]) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        ids = torch.full((self.max_length,), self._pad_id, dtype=torch.long)
        params = torch.zeros((self.max_length, max(1, self.parameter_dim)), dtype=torch.float32)
        mask = torch.zeros((self.max_length,), dtype=torch.float32)

        for index, step in enumerate(steps[: self.max_length]):
            ids[index] = self._primitive_to_index.get(step.primitive, self._pad_id)
            params[index] = self._encode_params(step.params)
            mask[index] = 1.0
        return ids, params, mask

    def encode_many(
        self,
        sequences: Sequence[Sequence[ProgramStepRecord]],
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        stacked_ids = []
        stacked_params = []
        stacked_mask = []
        for steps in sequences:
            ids, params, mask = self.encode(steps)
            stacked_ids.append(ids)
            stacked_params.append(params)
            stacked_mask.append(mask)
        return (
            torch.stack(stacked_ids, dim=0),
            torch.stack(stacked_params, dim=0),
            torch.stack(stacked_mask, dim=0),
        )

    def decode(self, ids: Sequence[int]) -> list[str]:  # pragma: no cover - convenience
        return [self._index_to_primitive.get(i, "<pad>") for i in ids]

    def _encode_params(self, params: Mapping[str, object]) -> "torch.Tensor":
        if self.parameter_dim == 0:
            return torch.zeros((1,), dtype=torch.float32)
        vec = torch.zeros((self.parameter_dim,), dtype=torch.float32)
        for name, value in params.items():
            idx = self._parameter_to_index.get(name)
            if idx is None:
                continue
            vec[idx] = self._normalize_value(value)
        return vec

    @staticmethod
    def _normalize_value(value: object) -> float:
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, Sequence) and value and not isinstance(value, (str, bytes)):
            return float(len(value))
        return 0.0


class ProgramTripleDataset(Dataset):  # type: ignore[misc]
    """Dataset of (input grid, program trace, output grid) triples."""

    def __init__(
        self,
        path: str | Path,
        tokenizer_config: ObjectTokenizerConfig,
        *,
        max_program_length: int | None = None,
    ) -> None:
        _ensure_torch()
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"program triple dataset not found: {path}")
        self.tokenizer_config = tokenizer_config
        self.records = self._load_records(self.path)
        if not self.records:
            raise ValueError(f"dataset at {path} contained no valid entries")

        primitives = [step.primitive for record in self.records for step in record.program]
        parameters = [name for record in self.records for step in record.program for name in step.params.keys()]
        max_length_from_data = max((len(record.program) for record in self.records), default=1)
        maximum = max_program_length or max_length_from_data
        self.program_tokenizer = ProgramTraceTokenizer(primitives, parameters, max_length=maximum)

    @staticmethod
    def _load_records(path: Path) -> list[ProgramTripleRecord]:
        records: list[ProgramTripleRecord] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                raw = line.strip()
                if not raw:
                    continue
                data = json.loads(raw)
                input_grid = _grid_from_data(data.get("input"))
                output_grid = _grid_from_data(data.get("output"))
                program_steps = ProgramTripleDataset._parse_program(data)
                metadata = data.get("metadata")
                records.append(
                    ProgramTripleRecord(
                        input_grid=input_grid,
                        output_grid=output_grid,
                        program=program_steps,
                        metadata=metadata if isinstance(metadata, Mapping) else None,
                    )
                )
        return records

    @staticmethod
    def _parse_program(data: Mapping[str, object]) -> Sequence[ProgramStepRecord]:
        candidate = data.get("program")
        if candidate is None:
            candidate = data.get("program_steps") or data.get("rule_trace")
        if candidate is None:
            raise ValueError("program triple record must include 'program', 'program_steps', or 'rule_trace'")
        if isinstance(candidate, Mapping) and "steps" in candidate:
            steps = candidate["steps"]
        else:
            steps = candidate
        if not isinstance(steps, Sequence):
            raise ValueError("program specification must be a sequence of steps")
        normalized: list[ProgramStepRecord] = []
        for step in steps:
            if not isinstance(step, Mapping):
                raise ValueError("each program step must be a mapping with primitive+params")
            primitive = step.get("primitive")
            if not isinstance(primitive, str):
                raise ValueError("program step missing 'primitive' string")
            params = step.get("params", {})
            if params is None:
                params = {}
            if not isinstance(params, Mapping):
                raise ValueError("program step 'params' must be a mapping")
            normalized.append(ProgramStepRecord(primitive=primitive, params=params))
        return tuple(normalized)

    def __len__(self) -> int:
        return len(self.records)

    @property
    def vocab_size(self) -> int:
        return self.program_tokenizer.vocab_size

    @property
    def parameter_dim(self) -> int:
        return max(1, self.program_tokenizer.parameter_dim)

    @property
    def max_program_length(self) -> int:
        return self.program_tokenizer.max_length

    def __getitem__(self, idx: int) -> ProgramTripleSample:
        record = self.records[idx]
        in_feat, in_mask, in_adj = self._tokenize_grid(record.input_grid)
        out_feat, out_mask, out_adj = self._tokenize_grid(record.output_grid)
        program_ids, program_params, program_mask = self.program_tokenizer.encode(record.program)
        return ProgramTripleSample(
            input_features=in_feat,
            input_mask=in_mask,
            input_adjacency=in_adj,
            output_features=out_feat,
            output_mask=out_mask,
            output_adjacency=out_adj,
            program_ids=program_ids,
            program_params=program_params,
            program_mask=program_mask,
            metadata=record.metadata,
        )

    def _tokenize_grid(self, grid: Grid) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        tokens = tokenize_grid_objects(grid, **self.tokenizer_config.as_kwargs())
        features = torch.tensor(tokens.features, dtype=torch.float32)
        mask = torch.tensor(tokens.mask, dtype=torch.float32)
        adjacency = torch.tensor(tokens.adjacency, dtype=torch.float32)
        return features, mask, adjacency


def collate_program_triples(samples: Sequence[ProgramTripleSample]) -> ProgramTripleBatch:
    if not samples:
        raise ValueError("collate_program_triples received an empty batch")
    return ProgramTripleBatch(
        input_features=torch.stack([sample.input_features for sample in samples], dim=0),
        input_mask=torch.stack([sample.input_mask for sample in samples], dim=0),
        input_adjacency=torch.stack([sample.input_adjacency for sample in samples], dim=0),
        output_features=torch.stack([sample.output_features for sample in samples], dim=0),
        output_mask=torch.stack([sample.output_mask for sample in samples], dim=0),
        output_adjacency=torch.stack([sample.output_adjacency for sample in samples], dim=0),
        program_ids=torch.stack([sample.program_ids for sample in samples], dim=0),
        program_params=torch.stack([sample.program_params for sample in samples], dim=0),
        program_mask=torch.stack([sample.program_mask for sample in samples], dim=0),
        metadata=[sample.metadata for sample in samples],
    )

