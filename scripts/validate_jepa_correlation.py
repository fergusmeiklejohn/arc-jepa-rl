"""Correlate JEPA validation loss with downstream solver success."""

from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable, List, Mapping, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:  # pragma: no cover - torch required at runtime only
    import torch
    from torch.utils.data import DataLoader
except Exception as exc:  # pragma: no cover
    raise RuntimeError("PyTorch is required to run the correlation study") from exc

from arcgen import Grid, SyntheticTask
from envs import LatentScorer

from training.dsl.metrics import description_length
from training.dsl.primitives import build_default_primitive_registry
from training.eval import ArcTask, load_arc_dev_tasks, load_synthetic_tasks_jsonl
from training.jepa import ObjectCentricJEPAExperiment, ObjectTokenizerConfig, load_jepa_config
from training.jepa.dataset import AugmentationConfig, ManifestTokenizedPairDataset, collate_tokenized_samples
from training.modules.projection import ProjectionHead
from training.solver import FewShotSolver


@dataclass(frozen=True)
class NormalizedTask:
    """Unified view over ARC dev or synthetic tasks."""

    task_id: str
    train_examples: Sequence[Tuple[Grid, Grid]]
    test_examples: Sequence[Tuple[Grid, Grid | None]]


class LatentDistanceBeam:
    """Minimal beam scorer that ranks programs by JEPA latent distance."""

    def __init__(
        self,
        scorer: LatentScorer,
        *,
        interpreter,
        beam_width: int,
        length_penalty: float,
        metric: str,
        context_bonus: float,
    ) -> None:
        self.scorer = scorer
        self.interpreter = interpreter
        self.beam_width = max(1, beam_width)
        self.length_penalty = length_penalty
        self.metric = metric
        self.context_bonus = context_bonus

    def search(
        self,
        latent_context: torch.Tensor,
        latent_target: torch.Tensor,
        enumerator,
        input_grid: Grid,
        *,
        cache=None,
        mdl_weight: float = 0.0,
    ) -> List[Tuple[object, float]]:
        scored: List[Tuple[object, float]] = []
        error_marker = getattr(cache, "ERROR", None) if cache is not None else None
        for program in enumerator.enumerate():
            output_grid = None
            if cache is not None:
                cached = cache.get(program, input_grid)
                if cached is error_marker:
                    continue
                if cached is not None:
                    output_grid = cached
            if output_grid is None:
                try:
                    output = self.interpreter.evaluate(program, {"grid": input_grid})
                except Exception:
                    if cache is not None:
                        cache.store_failure(program, input_grid)
                    continue
                if not isinstance(output, Grid):
                    if cache is not None:
                        cache.store_failure(program, input_grid)
                    continue
                output_grid = output
                if cache is not None:
                    cache.store_success(program, input_grid, output_grid)

            candidate_embedding = self.scorer.embed(output_grid).to(latent_target.device)
            target_distance = self.scorer.distance(candidate_embedding, latent_target, metric=self.metric)
            context_distance = self.scorer.distance(candidate_embedding, latent_context, metric=self.metric)
            score = -float(target_distance.detach().item())
            score += self.context_bonus * float(context_distance.detach().item())
            score -= self.length_penalty * len(program)
            score -= mdl_weight * description_length(program)
            scored.append((program, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[: self.beam_width]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Correlate JEPA loss with solver success")
    parser.add_argument("--jepa-config", type=Path, required=True, help="Path to JEPA YAML config")
    parser.add_argument(
        "--checkpoints",
        type=Path,
        nargs="+",
        required=True,
        help="List of checkpoint paths to evaluate",
    )
    parser.add_argument(
        "--val-manifest",
        type=Path,
        default=None,
        help="Optional override for validation manifest (defaults to config.dataset_manifest)",
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        default=None,
        help="Synthetic JSONL manifest for solver evaluation",
    )
    parser.add_argument(
        "--arc-dev-root",
        type=Path,
        default=None,
        help="ARC dev directory/JSON file for solver evaluation",
    )
    parser.add_argument(
        "--limit-tasks",
        type=int,
        default=None,
        help="Optional cap on the number of evaluation tasks (for smoke tests)",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Torch device to use")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for JEPA loss eval")
    parser.add_argument(
        "--beam-width",
        type=int,
        default=16,
        help="Number of candidate programs retained by the latent-distance beam scorer",
    )
    parser.add_argument(
        "--solver-max-nodes",
        type=int,
        default=3,
        help="Maximum DSL nodes considered during solver enumeration",
    )
    parser.add_argument(
        "--length-penalty",
        type=float,
        default=0.05,
        help="Length penalty applied when ranking programs",
    )
    parser.add_argument(
        "--context-bonus",
        type=float,
        default=0.0,
        help="Weight encouraging candidates to stay close to the context embedding",
    )
    parser.add_argument(
        "--latent-metric",
        type=str,
        default="cosine",
        choices=("cosine", "l2"),
        help="Distance metric used when comparing latent embeddings",
    )
    parser.add_argument(
        "--mdl-weight",
        type=float,
        default=0.1,
        help="MDL penalty forwarded to the few-shot solver",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/eval/jepa_correlation.json"),
        help="Path to write JSON summary",
    )
    parser.add_argument(
        "--keep-augmentations",
        action="store_true",
        help="Use augmentations defined in the JEPA config during validation (default: disabled)",
    )
    args = parser.parse_args()
    if bool(args.tasks) == bool(args.arc_dev_root):
        parser.error("Specify exactly one of --tasks or --arc-dev-root for solver evaluation")
    return args


def build_eval_loader(
    config: Mapping[str, object],
    tokenizer_config: ObjectTokenizerConfig,
    manifest_path: Path,
    *,
    batch_size: int,
    keep_augmentations: bool,
) -> DataLoader:
    data_cfg = config.get("data", {})
    if not isinstance(data_cfg, Mapping):
        raise ValueError("config['data'] must be a mapping")
    context_window = int(data_cfg.get("context_window", data_cfg.get("context_length", 3)))
    target_offset = int(data_cfg.get("target_offset", 1))
    augmentations = config.get("augmentations") if keep_augmentations else AugmentationConfig()

    dataset = ManifestTokenizedPairDataset(
        manifest_path,
        context_window=context_window,
        target_offset=target_offset,
        augmentations=augmentations,
        tokenizer_config=tokenizer_config,
        seed=int(config.get("seed", 0)) if config.get("seed") is not None else None,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_tokenized_samples,
    )


def load_checkpoint(experiment: ObjectCentricJEPAExperiment, checkpoint_path: Path) -> None:
    payload = torch.load(checkpoint_path, map_location=experiment.device)

    model_state = payload.get("model_state")
    if model_state is None:
        raise ValueError(f"checkpoint {checkpoint_path} missing 'model_state'")
    experiment.trainer.encoder.load_state_dict(model_state)

    projection_state = payload.get("projection_state")
    if projection_state is not None:
        experiment.projection_head.load_state_dict(projection_state, strict=False)

    queue_state = payload.get("queue_state")
    if queue_state is not None:
        experiment.queue.load_state_dict(queue_state, strict=False)
    else:
        experiment.queue.queue.zero_()
        experiment.queue.ptr.zero_()
        experiment.queue.filled.zero_()

    log_temperature = payload.get("log_temperature")
    if log_temperature is not None and getattr(experiment, "log_temperature", None) is not None:
        if isinstance(log_temperature, torch.Tensor):
            experiment.log_temperature.data.copy_(log_temperature.to(experiment.device))
        else:
            value = torch.tensor(float(log_temperature), device=experiment.device)
            experiment.log_temperature.data.copy_(value)


def evaluate_loss(experiment: ObjectCentricJEPAExperiment, loader: DataLoader) -> float:
    training_state = experiment.trainer.encoder.training
    projection_state = experiment.projection_head.training
    experiment.trainer.encoder.eval()
    experiment.projection_head.eval()
    try:
        return experiment.evaluate_epoch(loader)
    finally:
        if training_state:
            experiment.trainer.encoder.train()
        if projection_state:
            experiment.projection_head.train()


def build_latent_scorer(experiment: ObjectCentricJEPAExperiment, device: str) -> LatentScorer:
    projection_copy = copy.deepcopy(experiment.projection_head)
    projection_copy.to(device)
    scorer = LatentScorer(
        experiment.trainer.object_encoder,
        projection_head=projection_copy,
        device=device,
    )
    return scorer


def normalize_tasks(tasks: Sequence[ArcTask | SyntheticTask]) -> List[NormalizedTask]:
    normalized: List[NormalizedTask] = []
    for idx, task in enumerate(tasks):
        train = []
        test = []
        if isinstance(task, ArcTask):
            source_id = task.task_id
            for example in task.train_examples:
                if example.output_grid is None:
                    raise ValueError(f"Task {task.task_id} missing output grid in training example")
                train.append((example.input_grid, example.output_grid))
            for example in task.test_examples:
                test.append((example.input_grid, example.output_grid))
        else:
            source_id = getattr(task, "task_id", f"task_{idx}")
            train.append((task.input_grid, task.output_grid))

        if not train:
            raise ValueError(f"Task {source_id} does not contain training examples")
        normalized.append(NormalizedTask(task_id=source_id, train_examples=tuple(train), test_examples=tuple(test)))
    return normalized


def evaluate_on_tests(program, solver: FewShotSolver, test_examples: Sequence[Tuple[Grid, Grid | None]]) -> bool:
    if not program:
        return False
    interpreter = solver.interpreter
    for grid, expected in test_examples:
        try:
            result = interpreter.evaluate(program, {"grid": grid})
        except Exception:
            return False
        if not isinstance(result, Grid):
            return False
        if expected is not None and result.cells != expected.cells:
            return False
    return True


def evaluate_solver_with_latents(
    tasks: Sequence[NormalizedTask],
    scorer: LatentScorer,
    *,
    max_nodes: int,
    beam_width: int,
    length_penalty: float,
    mdl_weight: float,
    metric: str,
    context_bonus: float,
) -> dict:
    registry = build_default_primitive_registry()
    solver = FewShotSolver(registry, mdl_weight=mdl_weight)
    beam = LatentDistanceBeam(
        scorer,
        interpreter=solver.interpreter,
        beam_width=beam_width,
        length_penalty=length_penalty,
        metric=metric,
        context_bonus=context_bonus,
    )

    successes = 0
    total_programs = 0
    for task in tasks:
        first_input, first_target = task.train_examples[0]
        latent_context = scorer.embed(first_input)
        latent_target = scorer.embed(first_target)
        result = solver.solve(
            task.train_examples,
            max_nodes=max_nodes,
            beam_search=beam,
            latent_context=latent_context,
            latent_target=latent_target,
        )
        total_programs += result.evaluated
        solved = result.solved() and evaluate_on_tests(result.program, solver, task.test_examples)
        successes += int(solved)

    total_tasks = len(tasks)
    return {
        "task_count": total_tasks,
        "successes": successes,
        "success_rate": successes / total_tasks if total_tasks else 0.0,
        "avg_programs_tested": total_programs / total_tasks if total_tasks else 0.0,
    }


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    denom = denom_x * denom_y
    if denom == 0.0:
        return 0.0
    return num / denom


def main() -> None:
    args = parse_args()
    config = dict(load_jepa_config(args.jepa_config))

    manifest_path = args.val_manifest or config.get("dataset_manifest")
    if manifest_path is None:
        raise ValueError("Validation manifest not provided and config is missing 'dataset_manifest'")
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Validation manifest not found: {manifest_path}")

    tokenizer_cfg = ObjectTokenizerConfig.from_mapping(config.get("tokenizer"))
    eval_loader = build_eval_loader(
        config,
        tokenizer_cfg,
        manifest_path,
        batch_size=args.batch_size,
        keep_augmentations=args.keep_augmentations,
    )

    if args.tasks:
        raw_tasks = load_synthetic_tasks_jsonl(args.tasks)
        dataset_label = str(args.tasks)
    else:
        raw_tasks = load_arc_dev_tasks(args.arc_dev_root)
        dataset_label = str(args.arc_dev_root)

    normalized_tasks = normalize_tasks(raw_tasks)
    if args.limit_tasks is not None:
        normalized_tasks = normalized_tasks[: args.limit_tasks]
    if not normalized_tasks:
        raise RuntimeError("No tasks available for solver evaluation")

    results = []
    losses: List[float] = []
    success_rates: List[float] = []

    for checkpoint in args.checkpoints:
        experiment = ObjectCentricJEPAExperiment(config, device=args.device)
        load_checkpoint(experiment, checkpoint)
        val_loss = evaluate_loss(experiment, eval_loader)
        scorer = build_latent_scorer(experiment, device=args.device)
        solver_metrics = evaluate_solver_with_latents(
            normalized_tasks,
            scorer,
            max_nodes=args.solver_max_nodes,
            beam_width=args.beam_width,
            length_penalty=args.length_penalty,
            mdl_weight=args.mdl_weight,
            metric=args.latent_metric,
            context_bonus=args.context_bonus,
        )
        record = {
            "checkpoint": str(checkpoint),
            "val_manifest": str(manifest_path),
            "jepa_loss": val_loss,
            "task_source": dataset_label,
            **solver_metrics,
        }
        results.append(record)
        losses.append(val_loss)
        success_rates.append(solver_metrics["success_rate"])

        print(
            f"[{checkpoint}] val_loss={val_loss:.6f} "
            f"success_rate={solver_metrics['success_rate']:.3f} "
            f"avg_programs={solver_metrics['avg_programs_tested']:.2f}",
        )

    correlation = pearson(losses, success_rates)
    summary = {
        "jepa_config": str(args.jepa_config),
        "val_manifest": str(manifest_path),
        "task_source": dataset_label,
        "task_count": len(normalized_tasks),
        "results": results,
        "pearson_correlation": correlation,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if correlation is None:
        print("Pearson correlation requires at least two checkpoints")
    else:
        print(f"Pearson correlation (loss vs solve rate): {correlation:.3f}")


if __name__ == "__main__":
    main()
