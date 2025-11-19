"""Profile JEPA training throughput across loader/optimizer settings."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:  # pragma: no cover - optional dependency in CI
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:  # pragma: no cover - optional dependency in CI
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None  # type: ignore

from scripts._jepa_loader import build_jepa_dataloader
from training.jepa import ObjectCentricJEPAExperiment, load_jepa_config
from training.jepa.dataset import GridPairBatch, TokenizedPairBatch, build_dummy_dataset


DEFAULT_MAX_BATCHES = 24
DEFAULT_WARMUP_BATCHES = 4


@dataclass(frozen=True)
class ProfileCombination:
    batch_size: int
    num_workers: int
    grad_accum_steps: int
    amp: bool


@dataclass(frozen=True)
class ProfileResult:
    combination: ProfileCombination
    samples_per_sec: float
    batches_per_sec: float
    avg_step_time: float
    samples_processed: int
    measured_batches: int
    elapsed_seconds: float
    cpu_percent: float | None
    cpu_cores: float | None
    system_cpu_percent: float | None
    gpu_sm_util: float | None
    gpu_mem_util: float | None
    data_source: str
    data_path: str | None
    notes: list[str]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["combination"] = asdict(self.combination)
        return payload


class NvidiaSMISampler:
    """Lightweight helper that samples GPU utilisation via nvidia-smi."""

    def __init__(self, device_index: int = 0) -> None:
        self._binary = shutil.which("nvidia-smi")
        self._device_index = device_index
        self._samples: list[tuple[float, float]] = []

    @property
    def available(self) -> bool:
        return self._binary is not None

    def reset(self) -> None:
        self._samples.clear()

    def sample(self) -> None:
        if not self.available:
            return
        assert self._binary is not None
        try:
            result = subprocess.run(
                [
                    self._binary,
                    "-i",
                    str(self._device_index),
                    "--query-gpu=utilization.gpu,utilization.memory",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            line = result.stdout.strip().splitlines()
            if not line:
                return
            values = line[0].split(",")
            if len(values) < 2:
                return
            gpu_util = float(values[0].strip())
            mem_util = float(values[1].strip())
            self._samples.append((gpu_util, mem_util))
        except Exception:
            # Stop querying after the first failure to avoid noisy logs.
            self._binary = None
            self._samples.clear()

    def summary(self) -> tuple[float | None, float | None]:
        if not self._samples:
            return None, None
        gpu_vals = [entry[0] for entry in self._samples]
        mem_vals = [entry[1] for entry in self._samples]
        return (sum(gpu_vals) / len(gpu_vals), sum(mem_vals) / len(mem_vals))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile JEPA throughput on a manifest or dummy dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, required=True, help="YAML configuration file")
    parser.add_argument("--device", type=str, default=None, help="Torch device override")
    parser.add_argument("--manifest", type=Path, help="Optional manifest override path")
    parser.add_argument("--batch-sizes", type=int, nargs="+", help="Batch sizes to sweep")
    parser.add_argument("--num-workers", type=int, nargs="+", help="DataLoader worker counts to sweep")
    parser.add_argument("--grad-accum", type=int, nargs="+", help="Gradient accumulation steps to sweep")
    parser.add_argument(
        "--amp-options",
        choices=("on", "off"),
        nargs="+",
        help="AMP modes to sweep (on/off)",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=DEFAULT_MAX_BATCHES,
        help="Number of batches to measure per configuration",
    )
    parser.add_argument(
        "--warmup-batches",
        type=int,
        default=DEFAULT_WARMUP_BATCHES,
        help="Warmup batches before measurement",
    )
    parser.add_argument("--gpu-index", type=int, default=0, help="GPU index for utilisation sampling")
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path to store raw profiling results as JSON",
    )
    parser.add_argument(
        "--use-dummy-data",
        action="store_true",
        help="Fall back to a synthetic in-memory dataset if the manifest/tokenized data is unavailable",
    )
    parser.add_argument(
        "--dummy-batches",
        type=int,
        default=64,
        help="Number of batches for the dummy dataset when --use-dummy-data is set",
    )
    return parser.parse_args()


def _unique_sorted(values: Iterable[int], *, allow_zero: bool = False) -> list[int]:
    min_value = 0 if allow_zero else 1
    unique = sorted({int(v) for v in values if int(v) >= min_value})
    if not unique:
        raise ValueError("no positive values provided for sweep")
    return unique


def _default_batch_sizes(training_cfg: Mapping[str, object]) -> list[int]:
    base = int(training_cfg.get("batch_size", 512))
    min_batch = max(32, base // 2)
    mid_batch = base
    high_batch = base + max(64, base // 4)
    return _unique_sorted([min_batch, mid_batch, high_batch])


def _default_workers(training_cfg: Mapping[str, object]) -> list[int]:
    base = int(training_cfg.get("num_workers", 4))
    return _unique_sorted([max(2, base), max(4, base + 2), max(6, base + 4)])


def _default_grad_steps(training_cfg: Mapping[str, object]) -> list[int]:
    base = int(training_cfg.get("grad_accum_steps", 1))
    candidates = {1, base, max(2, base * 2)}
    return _unique_sorted(candidates)


def _default_amp_modes(training_cfg: Mapping[str, object]) -> list[bool]:
    base = bool(training_cfg.get("amp", True))
    return sorted({base, not base})


def _prepare_manifest_override(config: dict, manifest: Path | None) -> None:
    if manifest is None:
        return
    config["dataset_manifest"] = str(manifest)


def _try_build_loader(
    config: Mapping[str, object],
    experiment: ObjectCentricJEPAExperiment,
    *,
    allow_dummy: bool,
    dummy_batches: int,
    batch_size: int,
) -> tuple[Iterable[object], str, str | None]:
    try:
        dataloader, _, manifest_path, tokenized_path = build_jepa_dataloader(
            config,
            experiment.trainer.tokenizer_config,
        )
        if manifest_path is not None:
            return dataloader, "manifest", str(manifest_path)
        if tokenized_path is not None:
            return dataloader, "tokenized", str(tokenized_path)
        return dataloader, "unknown", None
    except FileNotFoundError:
        if not allow_dummy:
            raise
    if dummy_batches <= 0:
        raise ValueError("--dummy-batches must be positive when --use-dummy-data is set")
    dummy_dataset = build_dummy_dataset(
        num_batches=dummy_batches,
        context_length=experiment.context_length,
        batch_size=batch_size,
    )
    return dummy_dataset, "dummy", None


def _train_single_batch(
    experiment: ObjectCentricJEPAExperiment,
    batch: GridPairBatch | TokenizedPairBatch,
    pending_losses: list["torch.Tensor"],
    pending_targets: list["torch.Tensor"],
) -> int:
    if isinstance(batch, GridPairBatch):
        loss_tensor, _, _, target_proj = experiment._forward_from_grids(batch.context, batch.target)
        batch_size = len(batch.target)
    elif isinstance(batch, TokenizedPairBatch):
        moved = batch.to(experiment.device, non_blocking=getattr(experiment, "_use_non_blocking", False))
        loss_tensor, _, _, target_proj = experiment._forward_from_token_batch(moved)
        batch_size = moved.context_features.size(0)
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    pending_losses.append(loss_tensor)
    pending_targets.append(experiment._prepare_queue_projection(target_proj))
    return batch_size


def _apply_pending(
    experiment: ObjectCentricJEPAExperiment,
    pending_losses: list["torch.Tensor"],
    pending_targets: list["torch.Tensor"],
) -> None:
    if not pending_losses:
        return
    experiment.optimizer.zero_grad(set_to_none=True)
    combined = torch.stack(pending_losses).mean()
    experiment._step_optimizer(combined)
    if getattr(experiment, "_use_target_encoder", False):
        experiment._update_target_network()
    with torch.no_grad():
        for proj in pending_targets:
            experiment.queue.enqueue(proj)
    pending_losses.clear()
    pending_targets.clear()


def _finalise_cpu_stats(process: "psutil.Process | None", psutil_module) -> tuple[float | None, float | None, float | None]:
    if process is None or psutil_module is None:  # pragma: no cover - optional dependency
        return None, None, None
    proc_percent = process.cpu_percent(interval=None)
    system_percent = psutil_module.cpu_percent(interval=None)
    cpu_count = psutil_module.cpu_count(logical=True) or 0
    cores = (proc_percent / 100.0) * cpu_count if cpu_count else None
    return proc_percent, cores, system_percent


def profile_configuration(
    experiment: ObjectCentricJEPAExperiment,
    data_iterable: Iterable[object],
    *,
    warmup_batches: int,
    measure_batches: int,
    gpu_sampler: NvidiaSMISampler,
) -> tuple[int, int, float, float | None, float | None, float | None, float | None, float | None, list[str]]:
    iterator = iter(data_iterable)
    pending_losses: list["torch.Tensor"] = []
    pending_targets: list["torch.Tensor"] = []
    notes: list[str] = []

    # Warmup
    for _ in range(max(0, warmup_batches)):
        try:
            batch = next(iterator)
        except StopIteration:
            notes.append("dataset exhausted during warmup")
            return 0, 0, 0.0, None, None, None, None, None, notes
        _train_single_batch(experiment, batch, pending_losses, pending_targets)
    _apply_pending(experiment, pending_losses, pending_targets)

    if measure_batches <= 0:
        notes.append("measure_batches set to zero, no profiling performed")
        return 0, 0, 0.0, None, None, None, None, None, notes

    device_type = experiment.device.type if torch is not None else "cpu"
    if torch is not None and device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()

    process = None
    psutil_module = None
    if psutil is not None:  # pragma: no branch - optional dependency
        process = psutil.Process(os.getpid())
        process.cpu_percent(interval=None)
        psutil_module = psutil
        psutil_module.cpu_percent(interval=None)

    gpu_sampler.reset()
    gpu_sampler.sample()

    measured_batches = 0
    samples_processed = 0
    start_time = time.perf_counter()

    while measured_batches < measure_batches:
        try:
            batch = next(iterator)
        except StopIteration:
            notes.append("dataset exhausted before reaching max_batches")
            break
        batch_samples = _train_single_batch(experiment, batch, pending_losses, pending_targets)
        samples_processed += batch_samples
        measured_batches += 1
        gpu_sampler.sample()
        if measured_batches % experiment.grad_accum_steps == 0:
            _apply_pending(experiment, pending_losses, pending_targets)

    _apply_pending(experiment, pending_losses, pending_targets)

    if torch is not None and device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start_time
    proc_cpu_percent, cpu_cores, system_cpu_percent = _finalise_cpu_stats(process, psutil_module)
    gpu_sm, gpu_mem = gpu_sampler.summary()
    return (
        samples_processed,
        measured_batches,
        elapsed,
        gpu_sm,
        gpu_mem,
        proc_cpu_percent,
        cpu_cores,
        system_cpu_percent,
        notes + ([] if measured_batches else ["no batches profiled"]),
    )


def main() -> None:
    if torch is None:
        raise RuntimeError("PyTorch must be installed to profile JEPA training")
    if psutil is None:
        print("psutil not available; CPU metrics will be reported as None", file=sys.stderr)

    args = parse_args()
    config = dict(load_jepa_config(args.config))
    training_cfg = config.setdefault("training", {})
    _prepare_manifest_override(config, args.manifest)

    device = args.device or training_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, str) and device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA device requested but unavailable; falling back to CPU", file=sys.stderr)
        device = "cpu"

    batch_sizes = _unique_sorted(args.batch_sizes) if args.batch_sizes else _default_batch_sizes(training_cfg)
    worker_counts = (
        _unique_sorted(args.num_workers, allow_zero=True) if args.num_workers else _default_workers(training_cfg)
    )
    grad_steps = _unique_sorted(args.grad_accum) if args.grad_accum else _default_grad_steps(training_cfg)
    if args.amp_options:
        amp_modes = [choice == "on" for choice in args.amp_options]
    else:
        amp_modes = _default_amp_modes(training_cfg)

    combinations = [
        ProfileCombination(bs, workers, grad, amp)
        for bs in batch_sizes
        for workers in worker_counts
        for grad in grad_steps
        for amp in amp_modes
    ]

    print(f"Profiling {len(combinations)} JEPA settings on device={device}")
    results: list[ProfileResult] = []
    gpu_sampler = NvidiaSMISampler(device_index=args.gpu_index)

    for idx, combo in enumerate(combinations, 1):
        variant = copy.deepcopy(config)
        variant_training = variant.setdefault("training", {})
        variant_training["batch_size"] = combo.batch_size
        variant_training["num_workers"] = combo.num_workers
        variant_training["grad_accum_steps"] = combo.grad_accum_steps
        variant_training["amp"] = combo.amp

        experiment = ObjectCentricJEPAExperiment(variant, device=device)
        data_iterable, source_name, source_path = _try_build_loader(
            variant,
            experiment,
            allow_dummy=args.use_dummy_data,
            dummy_batches=args.dummy_batches,
            batch_size=combo.batch_size,
        )
        (
            samples,
            measured_batches,
            elapsed,
            gpu_sm,
            gpu_mem,
            cpu_percent,
            cpu_cores,
            system_cpu_percent,
            notes,
        ) = profile_configuration(
            experiment,
            data_iterable,
            warmup_batches=args.warmup_batches,
            measure_batches=args.max_batches,
            gpu_sampler=gpu_sampler,
        )
        batches_per_sec = samples_per_sec = avg_step_time = 0.0
        if elapsed > 0 and measured_batches > 0:
            samples_per_sec = samples / elapsed
            batches_per_sec = measured_batches / elapsed
            avg_step_time = elapsed / measured_batches

        result = ProfileResult(
            combination=combo,
            samples_per_sec=samples_per_sec,
            batches_per_sec=batches_per_sec,
            avg_step_time=avg_step_time,
            samples_processed=samples,
            measured_batches=measured_batches,
            elapsed_seconds=elapsed,
            cpu_percent=cpu_percent,
            cpu_cores=cpu_cores,
            system_cpu_percent=system_cpu_percent,
            gpu_sm_util=gpu_sm,
            gpu_mem_util=gpu_mem,
            data_source=source_name,
            data_path=source_path,
            notes=notes,
        )
        results.append(result)
        print(
            f"[{idx}/{len(combinations)}] "
            f"bs={combo.batch_size:<4} workers={combo.num_workers:<2} grad={combo.grad_accum_steps:<2} "
            f"amp={'on' if combo.amp else 'off':<3} "
            f"samples/s={samples_per_sec:7.1f} gpu%={gpu_sm or 0:4.1f} "
            f"cpu%={(cpu_percent or 0):4.1f} notes={'; '.join(notes) if notes else 'ok'}",
        )

    if args.output_json:
        payload = [entry.to_dict() for entry in results]
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote raw profiling data to {args.output_json}")

    if results:
        best = max(results, key=lambda item: item.samples_per_sec)
        combo = best.combination
        print(
            "\nFastest configuration:"
            f"\n  batch_size={combo.batch_size}"
            f"\n  num_workers={combo.num_workers}"
            f"\n  grad_accum_steps={combo.grad_accum_steps}"
            f"\n  amp={'on' if combo.amp else 'off'}"
            f"\n  samples_per_sec={best.samples_per_sec:.1f}"
            f"\n  batches_per_sec={best.batches_per_sec:.2f}"
        )
    else:
        print("No profiling results collected.")


if __name__ == "__main__":
    main()
