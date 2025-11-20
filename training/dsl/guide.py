"""Neural guidance model and utilities for DSL program search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple, TYPE_CHECKING

import hashlib
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
    from training.solver.constraints import ConstraintChecker


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


class ChildSumTreeLSTM(nn.Module if nn is not None else object):  # type: ignore[misc]
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        _ensure_torch()
        super().__init__()
        self.hidden_dim = hidden_dim
        self.W_i = nn.Linear(input_dim, hidden_dim)
        self.W_o = nn.Linear(input_dim, hidden_dim)
        self.W_u = nn.Linear(input_dim, hidden_dim)
        self.W_f = nn.Linear(input_dim, hidden_dim)

        self.U_i = nn.Linear(hidden_dim, hidden_dim)
        self.U_o = nn.Linear(hidden_dim, hidden_dim)
        self.U_u = nn.Linear(hidden_dim, hidden_dim)
        self.U_f = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        node_input: torch.Tensor,
        child_states: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = node_input.device
        dtype = node_input.dtype
        if child_states:
            child_hidden = torch.stack([state[0] for state in child_states], dim=0)
            child_cells = torch.stack([state[1] for state in child_states], dim=0)
            child_sum = torch.sum(child_hidden, dim=0)
        else:
            child_hidden = torch.zeros(0, self.hidden_dim, device=device, dtype=dtype)
            child_cells = torch.zeros(0, self.hidden_dim, device=device, dtype=dtype)
            child_sum = torch.zeros(self.hidden_dim, device=device, dtype=dtype)

        i = torch.sigmoid(self.W_i(node_input) + self.U_i(child_sum))
        o = torch.sigmoid(self.W_o(node_input) + self.U_o(child_sum))
        u = torch.tanh(self.W_u(node_input) + self.U_u(child_sum))

        if child_states:
            forget_input = self.W_f(node_input)
            forget_terms = []
            for h_child, c_child in zip(child_hidden, child_cells):
                f_child = torch.sigmoid(forget_input + self.U_f(h_child))
                forget_terms.append(f_child * c_child)
            forget_sum = torch.sum(torch.stack(forget_terms, dim=0), dim=0)
        else:
            forget_sum = torch.zeros(self.hidden_dim, device=device, dtype=dtype)

        cell = i * u + forget_sum
        hidden = o * torch.tanh(cell)
        return hidden, cell


class ProgramEncoder(nn.Module if nn is not None else object):  # type: ignore[misc]
    def __init__(
        self,
        registry: PrimitiveRegistry,
        embedding_dim: int = 32,
        hidden_dim: int | None = None,
        max_input_vars: int = 8,
    ) -> None:
        _ensure_torch()
        super().__init__()
        if max_input_vars <= 0:
            raise ValueError("max_input_vars must be positive")
        self.primitive_embeddings = PrimitiveEmbedding(registry, embedding_dim)
        self.hidden_dim = hidden_dim or embedding_dim
        self.tree_lstm = ChildSumTreeLSTM(embedding_dim, self.hidden_dim)
        self.output = nn.Sequential(
            nn.Linear(self.hidden_dim, embedding_dim),
            nn.ReLU(),
        )
        self.var_embeddings = nn.Embedding(max_input_vars, embedding_dim)
        self._var_to_index: Dict[str, int] = {}
        self._max_input_vars = max_input_vars

    def forward(self, program: Program) -> torch.Tensor:
        hidden, _ = self._encode_expression(program.root)
        return self.output(hidden)

    def _encode_expression(self, expr: Expression) -> Tuple[torch.Tensor, torch.Tensor]:
        if expr.var is not None:
            node_input = self._embed_variable(expr.var.name)
            return self.tree_lstm(node_input, ())

        if expr.primitive is None:  # pragma: no cover - defensive guard
            raise ValueError("expression must have a primitive or variable reference")

        child_states = [self._encode_expression(child) for child in expr.args]
        node_input = self.primitive_embeddings(expr.primitive.name)
        return self.tree_lstm(node_input, child_states)

    def _embed_variable(self, name: str) -> torch.Tensor:
        idx = self._var_to_index.get(name)
        if idx is None:
            digest = hashlib.sha1(name.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], byteorder="little") % self._max_input_vars
            self._var_to_index[name] = idx
        index_tensor = torch.tensor(
            idx,
            dtype=torch.long,
            device=self.var_embeddings.weight.device,
        )
        return self.var_embeddings(index_tensor)


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
        constraint_checker: "ConstraintChecker | None" = None,
        parallel: bool = True,
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
        self.constraint_checker = constraint_checker
        self.parallel = parallel

    def search(
        self,
        latent_context: torch.Tensor,
        latent_target: torch.Tensor,
        enumerator: ProgramEnumerator,
        input_grid: Grid,
        target_grid: Grid,
        *,
        cache=None,
        mdl_weight: float = 0.0,
        constraint_checker: "ConstraintChecker | None" = None,
    ) -> List[Tuple[Program, float]]:
        candidates = list(enumerator.enumerate())

        if not candidates:
            return []

        device = latent_context.device
        scores: List[Tuple[Program, float]] = []
        error_marker = getattr(cache, "ERROR", None) if cache is not None else None
        checker = constraint_checker or self.constraint_checker

        batch_features: List[torch.Tensor] = []
        candidate_meta: List[Tuple[Program, float, Grid]] = []

        for program in candidates:
            if checker is not None and checker.pre_check(program, input_grid, target_grid):
                continue
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

            if checker is not None and checker.post_check(program, input_grid, output_grid, target_grid):
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
            batch_features.append(features.unsqueeze(0))
            candidate_meta.append((program, float(length), output_grid))

        if not batch_features:
            return []

        neural_scores: torch.Tensor
        if self.parallel:
            features_tensor = torch.cat(batch_features, dim=0)
            neural_scores = self.scorer(features_tensor).view(-1)
            if neural_scores.numel() < len(candidate_meta):
                pad_value = neural_scores[-1] if neural_scores.numel() > 0 else torch.tensor(0.0, device=device)
                pad = pad_value.expand(len(candidate_meta) - neural_scores.numel())
                neural_scores = torch.cat([neural_scores, pad], dim=0)
        else:
            score_list = []
            for feats in batch_features:
                score = self.scorer(feats).view(-1)
                score_list.append(score[0] if score.numel() else torch.tensor(0.0, device=device))
            neural_scores = torch.stack(score_list, dim=0) if score_list else torch.zeros(0, device=device)

        for idx, (program, length, output_grid) in enumerate(candidate_meta):
            if neural_scores.numel() == 0:
                neural_score = 0.0
            else:
                # Clamp index to last available score if scorer returned fewer entries.
                neural_score = float(neural_scores[min(idx, neural_scores.numel() - 1)].item())
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
