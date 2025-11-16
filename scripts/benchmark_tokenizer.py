"""Micro-benchmark for the object tokenizer implementations."""

from __future__ import annotations

import argparse
import statistics
import time

from arcgen import Grid, SeededRNG
from training.modules import tokenize_grid_objects
from training.modules.object_tokenizer_legacy import tokenize_grid_objects_legacy


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark vectorized vs legacy tokenizer")
    parser.add_argument("--samples", type=int, default=256, help="Number of random grids to tokenize")
    parser.add_argument("--height", type=int, default=18, help="Grid height")
    parser.add_argument("--width", type=int, default=18, help="Grid width")
    parser.add_argument("--palette", type=int, default=6, help="Number of colors in random palette")
    parser.add_argument("--max-objects", type=int, default=16, help="Tokenizer max_objects parameter")
    parser.add_argument("--max-color-features", type=int, default=10, help="Tokenizer max_color_features parameter")
    parser.add_argument("--seed", type=int, default=7, help="Base RNG seed")
    parser.add_argument("--respect-colors", action="store_true", help="Respect per-color connected components")
    parser.add_argument("--connectivity", type=int, default=4, choices=(4, 8))
    parser.add_argument("--normalize", action="store_true")
    return parser.parse_args()


def _generate_grids(args: argparse.Namespace) -> list[Grid]:
    rng = SeededRNG(args.seed)
    palette = list(range(args.palette))
    grids = []
    for idx in range(args.samples):
        grids.append(
            Grid.random(
                args.height,
                args.width,
                palette,
                fill_prob=0.7,
                background=0,
                rng=rng.spawn(idx + 1),
            )
        )
    return grids


def _time_function(func, grids, kwargs) -> float:
    warmup = min(5, len(grids))
    for grid in grids[:warmup]:
        func(grid, **kwargs)

    start = time.perf_counter()
    for grid in grids:
        func(grid, **kwargs)
    end = time.perf_counter()
    return (end - start) / max(len(grids), 1)


def _validate_equivalence(grids, kwargs) -> None:
    for grid in grids[:5]:
        new_tokens = tokenize_grid_objects(grid, **kwargs)
        old_tokens = tokenize_grid_objects_legacy(grid, **kwargs)
        if new_tokens.mask != old_tokens.mask or new_tokens.adjacency != old_tokens.adjacency:
            raise AssertionError("Tokenizer outputs diverged during benchmark validation")
        for a_row, b_row in zip(new_tokens.features, old_tokens.features):
            if any(abs(a - b) > 1e-6 for a, b in zip(a_row, b_row)):
                raise AssertionError("Tokenizer feature mismatch detected")


def main() -> None:
    args = _parse_args()
    grids = _generate_grids(args)
    kwargs = {
        "max_objects": args.max_objects,
        "max_color_features": args.max_color_features,
        "respect_colors": args.respect_colors,
        "connectivity": args.connectivity,
        "normalize": args.normalize,
    }

    _validate_equivalence(grids, kwargs)

    legacy_times = [_time_function(tokenize_grid_objects_legacy, grids, kwargs) for _ in range(3)]
    vector_times = [_time_function(tokenize_grid_objects, grids, kwargs) for _ in range(3)]

    legacy_avg = statistics.mean(legacy_times)
    vector_avg = statistics.mean(vector_times)
    speedup = legacy_avg / vector_avg if vector_avg > 0 else float("inf")

    print("Tokenizer benchmark results")
    print(f"Samples           : {args.samples}")
    print(f"Grid size         : {args.height}x{args.width}")
    print(f"Legacy avg (ms)   : {legacy_avg * 1e3:.3f}")
    print(f"Vectorized avg(ms): {vector_avg * 1e3:.3f}")
    print(f"Speedup           : {speedup:.2f}x")


if __name__ == "__main__":
    main()
