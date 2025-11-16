"""Data utilities for Meta-JEPA rule-family training."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

try:  # pragma: no cover - torch optional
    import torch
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    Dataset = object  # type: ignore

from arcgen import ProgramStep, SyntheticTask


def _ensure_torch() -> None:
    if torch is None:  # pragma: no cover - defensive
        raise RuntimeError("PyTorch is required for Meta-JEPA datasets")


def _program_signature(trace: Sequence[ProgramStep]) -> Tuple[str, ...]:
    return tuple(step.primitive for step in trace)


@dataclass(frozen=True)
class RuleFamilyExample:
    """Aggregated information for a single rule family."""

    family_id: str
    signature: Tuple[str, ...]
    primitive_counts: Mapping[str, int]
    mean_program_length: float
    mean_changed_cells: float
    family_size: int
    phases: Tuple[str, ...]

    def feature_stats(self) -> Dict[str, float]:
        return {
            "mean_program_length": self.mean_program_length,
            "mean_changed_cells": self.mean_changed_cells,
            "family_size": float(self.family_size),
        }


def build_rule_family_examples(
    tasks: Iterable[SyntheticTask],
    *,
    include_phase: bool = True,
    min_family_size: int = 1,
) -> List[RuleFamilyExample]:
    """Group tasks by program signature and compute aggregate statistics."""

    families: Dict[Tuple[str, ...], List[SyntheticTask]] = defaultdict(list)
    for task in tasks:
        signature = _program_signature(task.rule_trace)
        if not signature:
            # Skip degenerate tasks that did not transform the grid.
            continue
        families[signature].append(task)

    examples: List[RuleFamilyExample] = []
    for signature, group in families.items():
        if len(group) < min_family_size:
            continue

        primitive_counts: Counter[str] = Counter()
        total_length = 0.0
        total_changed = 0.0
        phases: List[str] = []

        for task in group:
            for step in task.rule_trace:
                primitive_counts[step.primitive] += 1
            total_length += float(len(task.rule_trace))
            total_changed += float(task.metadata.get("changed_cells", 0))
            if include_phase:
                phases.append(str(task.metadata.get("phase", "")))

        family_id = "::".join(signature)
        family_size = len(group)
        examples.append(
            RuleFamilyExample(
                family_id=family_id,
                signature=signature,
                primitive_counts=dict(primitive_counts),
                mean_program_length=total_length / family_size,
                mean_changed_cells=total_changed / family_size if family_size else 0.0,
                family_size=family_size,
                phases=tuple(sorted(set(phases))) if phases else (),
            )
        )

    examples.sort(key=lambda item: item.family_size, reverse=True)
    return examples


class PrimitiveVocabulary:
    """Simple vocabulary mapping primitives to feature indices."""

    def __init__(self, primitives: Sequence[str]) -> None:
        if not primitives:
            raise ValueError("primitive vocabulary must contain at least one entry")
        unique = sorted(set(primitives))
        self._index = {name: idx for idx, name in enumerate(unique)}
        self._items = unique

    @classmethod
    def from_examples(cls, examples: Iterable[RuleFamilyExample]) -> "PrimitiveVocabulary":
        names = []
        for example in examples:
            names.extend(example.primitive_counts.keys())
        return cls(names)

    def __len__(self) -> int:
        return len(self._items)

    def items(self) -> Tuple[str, ...]:
        return tuple(self._items)

    def index(self, name: str) -> int:
        return self._index[name]

    def __contains__(self, name: str) -> bool:  # pragma: no cover - trivial wrapper
        return name in self._index

    def encode(self, counts: Mapping[str, int]) -> "torch.Tensor":
        _ensure_torch()
        vector = torch.zeros(len(self._items), dtype=torch.float32)
        for name, value in counts.items():
            if name not in self._index:
                continue
            vector[self._index[name]] = float(value)
        total = vector.sum()
        if total > 0:
            vector = vector / total
        return vector


class RuleFamilyDataset(Dataset):  # type: ignore[misc]
    """Dataset wrapping rule family features and labels."""

    def __init__(
        self,
        examples: Sequence[RuleFamilyExample],
        vocabulary: PrimitiveVocabulary,
    ) -> None:
        _ensure_torch()
        self.examples = list(examples)
        if not self.examples:
            raise ValueError("RuleFamilyDataset requires at least one example")

        self.vocabulary = vocabulary
        self.family_to_index = {example.family_id: idx for idx, example in enumerate(self.examples)}
        self.features = torch.stack([self._encode_example(example) for example in self.examples], dim=0)
        self.labels = torch.arange(len(self.examples), dtype=torch.long)
        self.adjacency = torch.stack([self._encode_adjacency(example) for example in self.examples], dim=0)

    def _encode_example(self, example: RuleFamilyExample) -> "torch.Tensor":
        primitive_vec = self.vocabulary.encode(example.primitive_counts)
        stats = example.feature_stats()
        stats_vec = torch.tensor(
            [stats["mean_program_length"], stats["mean_changed_cells"], stats["family_size"]],
            dtype=torch.float32,
        )
        return torch.cat([primitive_vec, stats_vec], dim=0)

    def __len__(self) -> int:
        return len(self.examples)

    def _encode_adjacency(self, example: RuleFamilyExample) -> "torch.Tensor":
        size = len(self.vocabulary)
        adjacency = torch.zeros((size, size), dtype=torch.float32)
        signature = example.signature
        if not signature:
            return adjacency

        def _index(name: str) -> int | None:
            try:
                return self.vocabulary.index(name)
            except KeyError:
                return None

        for primitive in signature:
            idx = _index(primitive)
            if idx is not None:
                adjacency[idx, idx] = 1.0

        for src, dst in zip(signature, signature[1:]):
            src_idx = _index(src)
            dst_idx = _index(dst)
            if src_idx is None or dst_idx is None:
                continue
            adjacency[src_idx, dst_idx] += 1.0
            adjacency[dst_idx, src_idx] += 1.0

        total = adjacency.sum()
        if total > 0:
            adjacency = adjacency / total
        return adjacency

    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx], self.adjacency[idx]


def build_rule_family_dataset(
    tasks: Sequence[SyntheticTask],
    *,
    min_family_size: int = 1,
) -> Tuple[RuleFamilyDataset, PrimitiveVocabulary, List[RuleFamilyExample]]:
    examples = build_rule_family_examples(tasks, min_family_size=min_family_size)
    vocabulary = PrimitiveVocabulary.from_examples(examples)
    dataset = RuleFamilyDataset(examples, vocabulary)
    return dataset, vocabulary, examples
