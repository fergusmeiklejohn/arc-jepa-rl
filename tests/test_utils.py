"""Tests for seeding and logging utilities."""

from __future__ import annotations

import importlib
import random
from types import SimpleNamespace

import training.utils.seed as seed_mod
from training.utils import create_experiment_logger, set_global_seeds


def test_set_global_seeds_reproducible(monkeypatch) -> None:
    monkeypatch.setenv("ARC_SKIP_NUMPY_SEED", "1")

    set_global_seeds(1234)
    samples_a = [random.random() for _ in range(5)]

    set_global_seeds(1234)
    samples_b = [random.random() for _ in range(5)]

    assert samples_a == samples_b

    captured: list[int] = []
    original_import = importlib.import_module

    def fake_import(name: str, package: str | None = None):
        if name == "numpy":
            return SimpleNamespace(random=SimpleNamespace(seed=lambda value: captured.append(value)))
        return original_import(name, package)

    monkeypatch.delenv("ARC_SKIP_NUMPY_SEED", raising=False)
    monkeypatch.setattr(seed_mod.importlib, "import_module", fake_import)

    set_global_seeds(4321)
    assert captured == [4321]

    captured.clear()
    set_global_seeds(9876, enable_numpy=False)
    assert captured == []


def test_experiment_logger_creates_run_directory(tmp_path) -> None:
    run_name = "test-run"
    with create_experiment_logger(tmp_path, run_name=run_name) as logger:
        logger.log_scalar("loss", 1.0, step=0)
        backend = logger.backend

    run_dir = tmp_path / run_name
    assert run_dir.exists()

    contents = list(run_dir.iterdir())
    assert contents, "run directory should contain at least one artifact"

    if backend == "jsonl":
        assert any(path.name == "events.jsonl" for path in contents)
