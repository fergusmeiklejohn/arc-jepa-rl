"""Meta-JEPA prior scoring utilities for DSL search."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping

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
        self._reference_embeddings = trainer.encode(features, device=str(self.device))
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
        features = self._encode_features(counts, changed_cells=float(changed), program_length=float(len(program)))
        embedding = self.trainer.encode(features.unsqueeze(0), device=str(self.device)).squeeze(0)
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
    ) -> Tensor:
        primitive_vec = self.vocabulary.encode(counts).to(self.device)
        stats = torch.tensor(
            [float(program_length), float(changed_cells), float(family_size)],
            dtype=torch.float32,
            device=self.device,
        )
        return torch.cat([primitive_vec, stats], dim=0)


__all__ = ["MetaJEPAPrior"]
