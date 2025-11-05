"""Utilities for building training data for DSL guidance models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

try:  # pragma: no cover - optional dependency
    import torch
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    Dataset = object  # type: ignore

from arcgen import Grid

from training.dsl.enumerator import InputVar, ProgramEnumerator, ProgramInterpreter
from training.dsl.guide import ProgramEncoder, encode_program_features, ProgramFeatures
from training.dsl.primitives import PrimitiveRegistry
from training.dsl.types import Grid as GridType


class GuidanceDataUnavailable(RuntimeError):
    pass


def _ensure_torch() -> None:
    if torch is None:  # pragma: no cover - defensive
        raise GuidanceDataUnavailable("PyTorch is required for guidance dataset")


@dataclass(frozen=True)
class GuidanceExample:
    features: ProgramFeatures
    improvement: float
    success: bool


class GuidanceDataset(Dataset):  # type: ignore[misc]
    def __init__(self, examples: Sequence[GuidanceExample]) -> None:
        _ensure_torch()
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):  # pragma: no cover - thin wrapper
        example = self.examples[idx]
        vector = example.features.stacked().detach()
        label = torch.tensor(example.improvement, dtype=torch.float32, device=vector.device)
        return vector, label, example.success


def build_guidance_examples(
    tasks: Iterable[tuple[Grid, Grid]],
    registry: PrimitiveRegistry,
    latent_embedder,
    program_encoder: ProgramEncoder,
    *,
    max_nodes: int = 4,
    max_programs_per_task: int | None = 32,
) -> List[GuidanceExample]:
    _ensure_torch()
    interpreter = ProgramInterpreter()
    examples: List[GuidanceExample] = []

    for context_grid, target_grid in tasks:
        latent_context = latent_embedder(context_grid)
        latent_target = latent_embedder(target_grid)

        enumerator = ProgramEnumerator(
            registry,
            inputs=[InputVar("grid", GridType)],
            target_type=GridType,
            max_nodes=max_nodes,
        )

        count = 0
        dist_before = torch.norm(latent_context - latent_target, p=2).item()
        for program in enumerator.enumerate():
            if max_programs_per_task is not None and count >= max_programs_per_task:
                break
            try:
                output_grid = interpreter.evaluate(program, {"grid": context_grid})
            except Exception:
                continue

            if not isinstance(output_grid, Grid):
                continue

            latent_candidate = latent_embedder(output_grid)
            dist_after = torch.norm(latent_candidate - latent_target, p=2).item()
            improvement = dist_before - dist_after
            success = output_grid.cells == target_grid.cells
            features = encode_program_features(
                program,
                latent_context,
                latent_target,
                latent_candidate,
                program_encoder,
            )
            examples.append(
                GuidanceExample(
                    features=features,
                    improvement=improvement,
                    success=success,
                )
            )
            count += 1

    return examples
