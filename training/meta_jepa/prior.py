"""Meta-JEPA prior scoring utilities for DSL search."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping, Tuple

try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

from arcgen import Grid
from training.meta_jepa.trainer import MetaJEPATrainer
from training.utils import count_changed_cells

if torch:
    Tensor = torch.Tensor
else:  # pragma: no cover
    Tensor = object  # type: ignore[misc]


class MetaJEPAPrior:
    """Compute similarity-based priors over DSL programs using Meta-JEPA."""

    def __init__(
        self,
        trainer: MetaJEPATrainer,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        if torch is None:  # pragma: no cover - defensive
            raise RuntimeError("PyTorch is required for Meta-JEPA priors")

        self.trainer = trainer
        self.device = torch.device(device)
        self.vocabulary = trainer.vocabulary
        self.dataset = trainer.dataset

        features = trainer.dataset.features.to(self.device)
        adjacency = trainer.dataset.adjacency.to(self.device)
        self._reference_embeddings = trainer.encode(
            features,
            adjacency=adjacency,
            device=str(self.device),
        )
        self._reference_embeddings = self._reference_embeddings.to(self.device)
        self._examples = trainer.dataset.examples

    def score_program(
        self,
        program,
        context: Grid,
        candidate: Grid,
    ) -> float:
        """Return cosine similarity between a program and the closest rule family."""

        if torch is None:  # pragma: no cover - defensive
            return 0.0

        counts = self._primitive_counts(program)
        if not counts:
            return 0.0

        changed = count_changed_cells(context, candidate)
        features, adjacency = self._encode_features(
            counts,
            changed_cells=float(changed),
            program_length=float(len(program)),
            program=program,
        )
        adjacency_tensor = adjacency.unsqueeze(0) if adjacency is not None else None
        embedding = self.trainer.encode(
            features.unsqueeze(0),
            adjacency=adjacency_tensor,
            device=str(self.device),
        ).squeeze(0)
        embedding = embedding.to(self.device)

        similarities = torch.matmul(self._reference_embeddings, embedding)
        if similarities.numel() == 0:
            return 0.0
        return float(similarities.max().item())

    def weighted_score(
        self,
        program,
        context: Grid,
        candidate: Grid,
        *,
        weight: float = 0.2,
    ) -> float:
        """Convenience helper returning a scaled similarity."""

        return weight * self.score_program(program, context, candidate)

    def _primitive_counts(self, program) -> Mapping[str, int]:
        from training.dsl.enumerator import Program  # local import to avoid cycle

        if not isinstance(program, Program):
            return {}

        primitives: Iterable[str] = (
            expr.primitive.name
            for expr in program.traverse()
            if getattr(expr, "primitive", None) is not None
        )
        counter = Counter(primitives)
        return {name: count for name, count in counter.items() if count > 0}

    def _encode_features(
        self,
        counts: Mapping[str, int],
        *,
        changed_cells: float,
        program_length: float,
        family_size: float = 1.0,
        program=None,
    ) -> Tuple[Tensor, Tensor | None]:
        primitive_vec = self.vocabulary.encode(counts).to(self.device)
        stats = torch.tensor(
            [float(program_length), float(changed_cells), float(family_size)],
            dtype=torch.float32,
            device=self.device,
        )
        features = torch.cat([primitive_vec, stats], dim=0)
        adjacency = self._build_program_adjacency(program) if program is not None else None
        return features, adjacency

    def _build_program_adjacency(self, program) -> Tensor | None:
        from training.dsl.enumerator import Program  # local import to avoid cycle

        if torch is None or not isinstance(program, Program):
            return None

        size = len(self.vocabulary)
        adjacency = torch.zeros((size, size), dtype=torch.float32, device=self.device)

        def _index(name: str) -> int | None:
            try:
                return self.vocabulary.index(name)
            except KeyError:
                return None

        for expr in program.traverse():
            primitive = getattr(expr, "primitive", None)
            if primitive is None:
                continue
            src_idx = _index(primitive.name)
            if src_idx is None:
                continue
            adjacency[src_idx, src_idx] = 1.0
            for child in getattr(expr, "args", ()):  # type: ignore[attr-defined]
                child_primitive = getattr(child, "primitive", None)
                if child_primitive is None:
                    continue
                dst_idx = _index(child_primitive.name)
                if dst_idx is None:
                    continue
                adjacency[src_idx, dst_idx] += 1.0
                adjacency[dst_idx, src_idx] += 1.0

        total = float(adjacency.sum().item())
        if total > 0:
            adjacency = adjacency / total
            return adjacency
        return None


__all__ = ["MetaJEPAPrior"]
