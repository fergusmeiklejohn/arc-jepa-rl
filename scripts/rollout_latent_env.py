"""Run random rollouts in the latent option environment and export traces."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys
from typing import Dict, List

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from arcgen import Grid
from envs import ArcLatentOptionEnv
from training.rllib_utils import (
    LatentTaskSampler,
    build_env_reward_config,
    build_generator_from_config,
    build_latent_scorer_from_config,
    build_options_from_config,
)

try:  # pragma: no cover - script guard
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-config", type=Path, required=True, help="Environment YAML configuration")
    parser.add_argument("--jepa-config", type=Path, required=True, help="JEPA configuration for encoder")
    parser.add_argument("--episodes", type=int, default=5, help="Number of random episodes to simulate")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for encoder (default: cpu)")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/latent_option_traces.jsonl"),
        help="Path to JSONL file where traces will be written",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"YAML file {path} must contain a mapping at top level")
    return data


def _grid_to_array(grid: Grid, pad: int, background: int) -> List[List[int]]:
    cells = grid.to_lists()
    padded = [[background for _ in range(pad)] for _ in range(pad)]
    height = min(len(cells), pad)
    for r in range(height):
        row = cells[r]
        width = min(len(row), pad)
        padded[r][:width] = row[:width]
    return padded


def _build_observation(grid: Grid, target: Grid, steps: int, pad: int, background: int) -> Dict[str, object]:
    return {
        "current": _grid_to_array(grid, pad, background),
        "target": _grid_to_array(target, pad, background),
        "steps": [steps],
    }


def main() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required for rollouts but is not available")

    args = parse_args()
    env_cfg = load_yaml(args.env_config)
    jepa_cfg = load_yaml(args.jepa_config)

    reward_cfg = build_env_reward_config(env_cfg.get("reward"))
    options = build_options_from_config(env_cfg.get("options"))
    scorer = build_latent_scorer_from_config(jepa_cfg, device=args.device)

    env = ArcLatentOptionEnv(
        options=options,
        scorer=scorer,
        reward_config=reward_cfg,
        max_steps=int(env_cfg.get("max_steps", 8)),
    )

    generator = build_generator_from_config(env_cfg.get("generator"), seed=env_cfg.get("seed"))
    schedule = env_cfg.get("task_schedule", {"atomic": 1})
    sampler = LatentTaskSampler(generator, schedule)
    pad = int(generator.config.max_grid_size)
    background = generator.config.background_color

    total_rewards: List[float] = []
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for idx in range(1, args.episodes + 1):
            task = sampler.sample_task()
            current_grid = env.reset(task=(task.input_grid, task.output_grid))
            obs = _build_observation(current_grid, task.output_grid, env.steps, pad, background)
            done = False
            episode_reward = 0.0
            steps: List[Dict[str, object]] = []

            while not done:
                action = random.randrange(env.action_space_n)
                next_grid, reward, done, info = env.step(action)
                episode_reward += reward
                step_record = {
                    "observation": obs,
                    "action": int(action),
                    "option_name": info.get("option"),
                    "reward": float(reward),
                    "success": bool(info.get("success")),
                    "termination": bool(info.get("success")) if info.get("success") is not None else None,
                }
                steps.append(step_record)

                obs = _build_observation(next_grid, task.output_grid, env.steps, pad, background)
                current_grid = next_grid

            episode = {
                "task_id": task.task_id,
                "phase": task.phase,
                "steps": steps,
                "episode_reward": episode_reward,
                "success": bool(info.get("success")),
            }
            json.dump(episode, handle)
            handle.write("\n")

            total_rewards.append(episode_reward)
            print(f"Episode {idx}: reward={episode_reward:.3f} success={info.get('success')} steps={env.steps}")

    print(f"Wrote {len(total_rewards)} episodes to {output_path}")

    if total_rewards:
        avg_reward = sum(total_rewards) / len(total_rewards)
        print(f"Average reward over {len(total_rewards)} episodes: {avg_reward:.3f}")
    else:
        print("No tasks sampled; check task_schedule in config.")


if __name__ == "__main__":
    main()
