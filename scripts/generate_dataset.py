"""Generate synthetic ARC datasets from YAML configuration."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, Mapping, Sequence

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from arcgen import GeneratorConfig, SyntheticARCGenerator, SyntheticTask

ROMAN_TO_PHASE = {
    "i": "atomic",
    "ii": "sequential",
}


@dataclass
class ExportOptions:
    include_png: bool = False
    compress_json: bool = False
    cell_size: int = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic ARC data")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config")
    parser.add_argument("--output", type=Path, default=None, help="Optional override for output root")
    parser.add_argument("--summary", type=Path, default=None, help="Optional path for JSON summary")
    return parser.parse_args()


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, Mapping):
        raise ValueError(f"configuration must be a mapping: {path}")
    return dict(data)


def normalise_phase(phase: str) -> str:
    key = str(phase).strip().lower()
    if key in ROMAN_TO_PHASE:
        return ROMAN_TO_PHASE[key]
    if key in {"atomic", "sequential"}:
        return key
    raise ValueError(f"unsupported curriculum phase '{phase}'")


def build_generator_config(config: Mapping) -> GeneratorConfig:
    generator_cfg = config.get("generator", {})
    if not isinstance(generator_cfg, Mapping):
        raise ValueError("generator configuration must be a mapping")

    grid_cfg = generator_cfg.get("grid_size") or {}
    colors_cfg = generator_cfg.get("colors")

    min_grid = int(grid_cfg.get("min", generator_cfg.get("min_grid_size", 6)))
    max_grid = int(grid_cfg.get("max", generator_cfg.get("max_grid_size", min_grid)))
    if max_grid < min_grid:
        raise ValueError("max grid size cannot be smaller than min grid size")

    if isinstance(colors_cfg, Mapping):
        min_colors = int(colors_cfg.get("min", 3))
        max_colors = int(colors_cfg.get("max", min_colors))
    elif colors_cfg is not None:
        min_colors = max_colors = int(colors_cfg)
    else:
        min_colors, max_colors = 3, 6

    cfg = GeneratorConfig(
        min_grid_size=min_grid,
        max_grid_size=max_grid,
        min_colors=min_colors,
        max_colors=max_colors,
        background_color=int(generator_cfg.get("background_color", 0)),
        fill_probability=float(generator_cfg.get("fill_probability", 0.75)),
        max_parameter_retries=int(generator_cfg.get("max_parameter_retries", 8)),
        max_task_retries=int(generator_cfg.get("max_task_retries", 16)),
    )
    return cfg


def extract_schedule(config: Mapping) -> Dict[str, int]:
    if "task_schedule" in config:
        raw = config["task_schedule"]
        if not isinstance(raw, Mapping):
            raise ValueError("task_schedule must be a mapping of phase->count")
        return {normalise_phase(phase): int(count) for phase, count in raw.items()}

    phase = config.get("curriculum_phase", "atomic")
    count = int(config.get("generator", {}).get("samples", config.get("samples", 0)))
    if count <= 0:
        raise ValueError("number of samples must be positive")
    return {normalise_phase(phase): count}


def parse_export_options(config: Mapping) -> ExportOptions:
    raw = config.get("export", {})
    if not isinstance(raw, Mapping):
        raw = {}
    return ExportOptions(
        include_png=bool(raw.get("include_png", False)),
        compress_json=bool(raw.get("compress_json", False)),
        cell_size=int(raw.get("cell_size", 20)),
    )


def build_output_root(args: argparse.Namespace, config: Mapping) -> Path:
    if args.output:
        return args.output
    output_root = config.get("output_root") or config.get("generator", {}).get("output_root")
    if output_root is None:
        raise ValueError("config must provide an output_root or supply --output")
    return Path(output_root)


def generate_tasks(config: Mapping) -> Sequence[SyntheticTask]:
    generator = SyntheticARCGenerator(build_generator_config(config), seed=config.get("seed"))
    schedule = extract_schedule(config)

    tasks: list[SyntheticTask] = []
    for phase, count in schedule.items():
        tasks.extend(generator.sample_many(count, phase))
    return tasks


def export_manifest(root: Path, tasks: Sequence[SyntheticTask], compress: bool) -> Path:
    manifest_path = root / ("manifest.jsonl.gz" if compress else "manifest.jsonl")
    root.mkdir(parents=True, exist_ok=True)

    if compress:
        import gzip

        with gzip.open(manifest_path, "wt", encoding="utf-8") as handle:
            for task in tasks:
                json.dump(task.to_json_record(), handle)
                handle.write("\n")
    else:
        with manifest_path.open("w", encoding="utf-8") as handle:
            for task in tasks:
                json.dump(task.to_json_record(), handle)
                handle.write("\n")
    return manifest_path


def grid_to_image(grid: "Grid", cell_size: int) -> "Image.Image":  # type: ignore[name-defined]
    from PIL import Image

    palette = [
        (0, 0, 0),
        (255, 0, 0),
        (0, 0, 255),
        (0, 255, 0),
        (255, 255, 0),
        (255, 140, 0),
        (128, 0, 128),
        (0, 255, 255),
        (255, 255, 255),
    ]

    height, width = grid.shape
    img = Image.new("RGB", (width * cell_size, height * cell_size), (0, 0, 0))
    pixels = img.load()

    for y, row in enumerate(grid.to_lists()):
        for x, value in enumerate(row):
            color = palette[value % len(palette)]
            for dy in range(cell_size):
                for dx in range(cell_size):
                    pixels[x * cell_size + dx, y * cell_size + dy] = color
    return img


def export_images(root: Path, tasks: Sequence[SyntheticTask], cell_size: int) -> None:
    try:
        from PIL import Image  # noqa: F401
    except Exception as exc:
        print(f"[generate_dataset] Pillow not installed, skipping PNG export ({exc})")
        return

    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    for task in tasks:
        base = images_dir / task.task_id
        input_png = grid_to_image(task.input_grid, cell_size)
        output_png = grid_to_image(task.output_grid, cell_size)
        input_png.save(base.with_name(f"{task.task_id}_input.png"))
        output_png.save(base.with_name(f"{task.task_id}_output.png"))


def describe(values: Iterable[int]) -> Dict[str, float]:
    seq = list(values)
    if not seq:
        return {}
    seq.sort()
    length = len(seq)
    def percentile(p: float) -> float:
        idx = min(length - 1, max(0, int(round(p * (length - 1)))))
        return float(seq[idx])

    return {
        "min": float(seq[0]),
        "max": float(seq[-1]),
        "mean": float(mean(seq)),
        "median": float(median(seq)),
        "p10": percentile(0.10),
        "p90": percentile(0.90),
    }


def summarise(tasks: Sequence[SyntheticTask]) -> Dict[str, object]:
    palette_counts = Counter()
    program_lengths = []
    changed_cells: list[int] = []
    heights = []
    widths = []
    phase_counts = Counter()

    for task in tasks:
        metadata = task.metadata or {}
        palette = metadata.get("palette") or task.input_grid.palette()
        palette_counts[len(palette)] += 1
        program_lengths.append(len(task.rule_trace))
        changed = metadata.get("changed_cells")
        if isinstance(changed, int):
            changed_cells.append(changed)
        heights.append(task.input_grid.height)
        widths.append(task.input_grid.width)
        phase_counts[metadata.get("phase_name", metadata.get("phase", task.phase))] += 1

    summary: Dict[str, object] = {
        "num_tasks": len(tasks),
        "phases": dict(phase_counts),
        "palette_size_counts": dict(palette_counts),
        "program_length": describe(program_lengths),
        "changed_cells": describe(changed_cells),
        "grid_height": describe(heights),
        "grid_width": describe(widths),
    }
    return summary


def write_summary(path: Path, summary: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    output_root = build_output_root(args, config)
    export_opts = parse_export_options(config)

    tasks = generate_tasks(config)
    if not tasks:
        raise RuntimeError("generator produced no tasks")

    manifest_path = export_manifest(output_root, tasks, export_opts.compress_json)
    print(f"[generate_dataset] Wrote {len(tasks)} tasks to {manifest_path}")

    if export_opts.include_png:
        export_images(output_root, tasks, export_opts.cell_size)

    summary = summarise(tasks)
    summary_path = args.summary or output_root / "summary.json"
    write_summary(summary_path, summary)

    print("[generate_dataset] Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
