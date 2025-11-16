import json
import subprocess
import sys
from pathlib import Path

from arcgen import Grid
from envs import make_primitive_option
from training.options.traces import load_option_episodes_from_traces


def _write_trace(tmp_path: Path, steps):
    trace_path = tmp_path / "traces.jsonl"
    with trace_path.open("w", encoding="utf-8") as handle:
        json.dump({"task_id": "task_1", "steps": steps}, handle)
        handle.write("\n")
        json.dump({"task_id": "task_2", "steps": steps}, handle)
        handle.write("\n")
    return trace_path


def test_load_option_episodes_from_traces(tmp_path):
    option_a = make_primitive_option("mirror_x")
    option_b = make_primitive_option("mirror_y")
    grid = Grid([[0, 1], [2, 3]])
    mid = option_a.apply(grid)
    final = option_b.apply(mid)

    steps = [
        {
            "observation": {"current": grid.to_lists()},
            "action": 0,
            "option_name": option_a.name,
            "reward": 0.1,
            "success": False,
            "grid_before": grid.to_lists(),
            "grid_after": mid.to_lists(),
        },
        {
            "observation": {"current": mid.to_lists()},
            "action": 1,
            "option_name": option_b.name,
            "reward": 1.0,
            "success": True,
            "grid_before": mid.to_lists(),
            "grid_after": final.to_lists(),
        },
    ]

    trace_path = _write_trace(tmp_path, steps)
    episodes = load_option_episodes_from_traces(trace_path, (option_a, option_b))

    assert len(episodes) == 2
    for episode in episodes:
        assert episode.steps[0].option.name == option_a.name
        assert episode.steps[1].option.name == option_b.name
        assert episode.steps[0].after.cells == mid.cells
        assert episode.steps[1].after.cells == final.cells


def test_cli_discovers_options_from_traces(tmp_path):
    option_a = make_primitive_option("mirror_x")
    option_b = make_primitive_option("mirror_y")
    grid = Grid([[0, 1], [2, 3]])
    mid = option_a.apply(grid)
    final = option_b.apply(mid)

    steps = [
        {
            "observation": {"current": grid.to_lists()},
            "action": 0,
            "option_name": option_a.name,
            "reward": 0.1,
            "success": False,
            "grid_before": grid.to_lists(),
            "grid_after": mid.to_lists(),
        },
        {
            "observation": {"current": mid.to_lists()},
            "action": 1,
            "option_name": option_b.name,
            "reward": 1.0,
            "success": True,
            "grid_before": mid.to_lists(),
            "grid_after": final.to_lists(),
        },
    ]

    trace_path = _write_trace(tmp_path, steps)

    env_cfg = tmp_path / "env.yaml"
    env_cfg.write_text(
        """
options:
  include_defaults: false
  extra:
    - primitive: mirror_x
    - primitive: mirror_y
"""
    )

    output_path = tmp_path / "summary.json"
    script = Path("scripts/discover_options.py").resolve()
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--env-config",
            str(env_cfg),
            "--traces",
            str(trace_path),
            "--output",
            str(output_path),
            "--min-support",
            "2",
            "--max-length",
            "2",
            "--allow-singleton",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    summary = json.loads(output_path.read_text(encoding="utf-8"))
    assert summary["discovered_options"], "expected discoveries to be recorded"
    first = summary["discovered_options"][0]
    assert first["sequence"] == [option_a.name, option_b.name]
