"""Few-shot solver orchestrating DSL enumeration and neural guidance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING

try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

from arcgen import Grid

from training.dsl.enumerator import InputVar, Program, ProgramEnumerator, ProgramInterpreter
from training.dsl.metrics import description_length
from training.dsl.guide import GuidedBeamSearch
from training.dsl.primitives import PrimitiveRegistry
from training.dsl.types import Grid as GridType
from training.options import DiscoveredOption, discover_option_sequences, promote_discovered_option
from training.options.discovery import OptionEpisode
from training.solver.cache import EVALUATION_ERROR, EvaluationCache
from training.solver.constraints import ConstraintChecker

if TYPE_CHECKING:  # pragma: no cover
    from training.meta_jepa.prior import MetaJEPAPrior


@dataclass(frozen=True)
class FewShotResult:
    """Outcome of the few-shot solving pipeline."""

    program: Optional[Program]
    score: Optional[float]
    evaluated: int
    candidates: Sequence[Tuple[Program, float]]

    def solved(self) -> bool:
        return self.program is not None


def augment_registry_with_options(
    registry: PrimitiveRegistry,
    episodes: Iterable[OptionEpisode],
    *,
    min_support: int = 2,
    max_sequence_length: int = 3,
    min_success_rate: float = 0.6,
) -> List[DiscoveredOption]:
    """Discover and promote option sequences into the provided registry."""

    discovered = discover_option_sequences(
        episodes,
        min_support=min_support,
        max_sequence_length=max_sequence_length,
        min_success_rate=min_success_rate,
    )
    for option in discovered:
        promote_discovered_option(option, registry)
    return discovered


class FewShotSolver:
    """Coordinate enumeration, neural guidance, and consistency checking."""

    def __init__(
        self,
        registry: PrimitiveRegistry,
        *,
        interpreter: Optional[ProgramInterpreter] = None,
        mdl_weight: float = 0.1,
        constraint_checker: Optional[ConstraintChecker] = None,
    ) -> None:
        self.registry = registry
        self.interpreter = interpreter or ProgramInterpreter()
        self.mdl_weight = float(mdl_weight)
        self.constraint_checker = constraint_checker or ConstraintChecker()
        self._cache = EvaluationCache()

    def solve(
        self,
        examples: Sequence[Tuple[Grid, Grid]],
        *,
        max_nodes: int = 4,
        beam_search: Optional[GuidedBeamSearch] = None,
        latent_context: Optional[torch.Tensor] = None,
        latent_target: Optional[torch.Tensor] = None,
        meta_prior: "MetaJEPAPrior | None" = None,
    ) -> FewShotResult:
        if not examples:
            raise ValueError("examples must contain at least one (input, output) pair")
        if max_nodes <= 0:
            raise ValueError("max_nodes must be positive")

        enumerator = ProgramEnumerator(
            self.registry,
            inputs=[InputVar("grid", GridType)],
            target_type=GridType,
            max_nodes=max_nodes,
        )

        first_input, first_target = examples[0]

        if beam_search is None:
            if meta_prior is not None:
                ranked: List[Tuple[Program, float]] = []
                for program in enumerator.enumerate():
                    if self.constraint_checker.pre_check(program, first_input, first_target):
                        continue
                    candidate = self._evaluate_with_cache(program, first_input)
                    if not isinstance(candidate, Grid):
                        continue
                    if self.constraint_checker.post_check(program, first_input, candidate, first_target):
                        continue
                    score = meta_prior.score_program(program, first_input, candidate)
                    score -= self.mdl_weight * description_length(program)
                    ranked.append((program, score))

                ranked.sort(key=lambda item: item[1], reverse=True)

                for idx, (program, score) in enumerate(ranked, 1):
                    if self._consistent(program, examples):
                        return FewShotResult(
                            program=program,
                            score=score,
                            evaluated=idx,
                            candidates=tuple(ranked),
                        )
                return FewShotResult(program=None, score=None, evaluated=len(ranked), candidates=tuple(ranked))

            evaluated = 0
            for program in enumerator.enumerate():
                if self.constraint_checker.pre_check(program, first_input, first_target):
                    continue
                evaluated += 1
                if self._consistent(program, examples):
                    mdl_score = -self.mdl_weight * description_length(program)
                    return FewShotResult(program=program, score=mdl_score, evaluated=evaluated, candidates=())
            return FewShotResult(program=None, score=None, evaluated=evaluated, candidates=())

        if torch is None:
            raise RuntimeError("Guided beam search requires PyTorch but it is not available")

        if latent_context is None or latent_target is None:
            raise ValueError("latent_context and latent_target must be provided for guided search")

        candidates = beam_search.search(
            latent_context,
            latent_target,
            enumerator,
            first_input,
            first_target,
            cache=self._cache,
            mdl_weight=self.mdl_weight,
            constraint_checker=self.constraint_checker,
        )
        for program, score in candidates:
            if self._consistent(program, examples):
                return FewShotResult(program=program, score=score, evaluated=len(candidates), candidates=candidates)
        return FewShotResult(program=None, score=None, evaluated=len(candidates), candidates=candidates)

    def _consistent(self, program: Program, examples: Sequence[Tuple[Grid, Grid]]) -> bool:
        for idx, (input_grid, target_grid) in enumerate(examples):
            if idx == 0 and self.constraint_checker.pre_check(program, input_grid, target_grid):
                return False

            output = self._evaluate_with_cache(program, input_grid)
            if not isinstance(output, Grid):
                return False

            if self.constraint_checker.post_check(program, input_grid, output, target_grid):
                return False

            if output.cells != target_grid.cells:
                return False
        return True

    def _evaluate_with_cache(self, program: Program, grid: Grid) -> object:
        cached = self._cache.get(program, grid)
        if cached is EVALUATION_ERROR:
            return None
        if cached is not None:
            return cached

        try:
            result = self.interpreter.evaluate(program, {"grid": grid})
        except Exception:
            self._cache.store_failure(program, grid)
            return None

        if isinstance(result, Grid):
            self._cache.store_success(program, grid, result)
        else:
            self._cache.store_failure(program, grid)
        return result
