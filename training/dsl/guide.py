"""Neural guidance model and utilities for DSL program search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Sequence, Tuple, TYPE_CHECKING

import math

try:  # pragma: no cover - torch optional at import time
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore

from arcgen import Grid

from .enumerator import Expression, InputVar, Program, ProgramEnumerator, ProgramInterpreter
from .metrics import description_length
from .primitives import PrimitiveRegistry
from .types import Grid as GridType

if TYPE_CHECKING:  # pragma: no cover
    from training.meta_jepa.prior import MetaJEPAPrior


class GuidanceModuleUnavailable(RuntimeError):
    pass


def _ensure_torch() -> None:
    if torch is None:  # pragma: no cover
        raise GuidanceModuleUnavailable("PyTorch is required for guidance model training")


@dataclass(frozen=True)
class ProgramFeatures:
    latent_context: torch.Tensor
    latent_target: torch.Tensor
    latent_candidate: torch.Tensor
    program_embedding: torch.Tensor
    meta: Mapping[str, float]

    def stacked(self) -> torch.Tensor:
        device = self.latent_context.device
        scalars = [torch.tensor([value], dtype=torch.float32, device=device) for value in self.meta.values()]
        return torch.cat(
            [
                self.latent_context,
                self.latent_target,
                self.latent_candidate,
                self.program_embedding,
                *scalars,
            ],
            dim=-1,
        )


class PrimitiveEmbedding(nn.Module if nn is not None else object):  # type: ignore[misc]
    def __init__(self, registry: PrimitiveRegistry, embedding_dim: int) -> None:
        _ensure_torch()
        super().__init__()
        self.registry = registry
        self.embedding = nn.Embedding(len(registry.list()), embedding_dim)
        name_to_idx = {primitive.name: idx for idx, primitive in enumerate(registry.list())}
        self.register_buffer("_indices", torch.tensor([name_to_idx[primitive.name] for primitive in registry.list()], dtype=torch.long))
        self._name_to_idx = name_to_idx

    def forward(self, primitive_name: str) -> torch.Tensor:
        idx = self._name_to_idx[primitive_name]
        return self.embedding.weight[idx]


class ProgramEncoder(nn.Module if nn is not None else object):  # type: ignore[misc]
    def __init__(self, registry: PrimitiveRegistry, embedding_dim: int = 32) -> None:
        _ensure_torch()
        super().__init__()
        self.primitive_embeddings = PrimitiveEmbedding(registry, embedding_dim)
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
        )

    def forward(self, program: Program) -> torch.Tensor:
        embeddings = []
        for expr in program.traverse():
            if expr.primitive is not None:
                embeddings.append(self.primitive_embeddings(expr.primitive.name))
        if not embeddings:
            return torch.zeros(self.primitive_embeddings.embedding.embedding_dim, device=self.primitive_embeddings.embedding.weight.device)
        stacked = torch.stack(embeddings, dim=0)
        pooled = stacked.mean(dim=0)
        return self.linear(pooled)


class GuidanceScorer(nn.Module if nn is not None else object):  # type: ignore[misc]
    def __init__(self, feature_dim: int, hidden_dim: int = 128) -> None:
        _ensure_torch()
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


class GuidedBeamSearch:
    def __init__(
        self,
        registry: PrimitiveRegistry,
        scorer: GuidanceScorer,
        program_encoder: ProgramEncoder,
        interpreter: ProgramInterpreter,
        latent_encoder,
        beam_width: int = 5,
        length_penalty: float = 0.05,
        meta_prior: "MetaJEPAPrior | None" = None,
        meta_weight: float = 0.2,
    ) -> None:
        self.registry = registry
        self.scorer = scorer
        self.program_encoder = program_encoder
        self.interpreter = interpreter
        self.latent_encoder = latent_encoder
        self.beam_width = beam_width
        self.length_penalty = length_penalty
        self.meta_prior = meta_prior
        self.meta_weight = float(meta_weight)

    def search(
        self,
        latent_context: torch.Tensor,
        latent_target: torch.Tensor,
        enumerator: ProgramEnumerator,
        input_grid: Grid,
        *,
        cache=None,
        mdl_weight: float = 0.0,
    ) -> List[Tuple[Program, float]]:
        candidates = list(enumerator.enumerate())

        if not candidates:
            return []

        device = latent_context.device
        scores: List[Tuple[Program, float]] = []
        error_marker = getattr(cache, "ERROR", None) if cache is not None else None

        for program in candidates:
            prog_embedding = self.program_encoder(program)
            length = len(program)
            output_grid = None
            if cache is not None:
                cached = cache.get(program, input_grid)
                if cached is error_marker:
                    continue
                if cached is not None:
                    output_grid = cached

            if output_grid is None:
                try:
                    output_grid = self.interpreter.evaluate(program, {"grid": input_grid})
                except Exception:
                    if cache is not None:
                        cache.store_failure(program, input_grid)
                    continue
                if not isinstance(output_grid, Grid):
                    if cache is not None:
                        cache.store_failure(program, input_grid)
                    continue
                if cache is not None:
                    cache.store_success(program, input_grid, output_grid)

            if not isinstance(output_grid, Grid):
                continue

            candidate_embedding = self.latent_encoder(output_grid)
            if candidate_embedding.device != device:
                candidate_embedding = candidate_embedding.to(device)
            features = torch.cat(
                [
                    latent_context,
                    latent_target,
                    candidate_embedding,
                    prog_embedding,
                    torch.tensor([length], dtype=torch.float32, device=device),
                ]
            )
            neural_score = float(self.scorer(features.unsqueeze(0)).item())
            meta_bonus = 0.0
            if self.meta_prior is not None:
                meta_bonus = self.meta_weight * self.meta_prior.score_program(program, input_grid, output_grid)
            mdl_penalty = float(mdl_weight) * description_length(program)
            total_score = neural_score - self.length_penalty * length - mdl_penalty + meta_bonus
            scores.append((program, total_score))

        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[: self.beam_width]


def encode_program_features(
    program: Program,
    latent_context: torch.Tensor,
    latent_target: torch.Tensor,
    latent_candidate: torch.Tensor,
    program_encoder: ProgramEncoder,
) -> ProgramFeatures:
    _ensure_torch()
    prog_embedding = program_encoder(program)
    meta = {
        "length": float(len(program)),
    }
    return ProgramFeatures(
        latent_context=latent_context,
        latent_target=latent_target,
        latent_candidate=latent_candidate,
        program_embedding=prog_embedding,
        meta=meta,
    )
