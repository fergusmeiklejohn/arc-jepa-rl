"""Run random rollouts in the latent option environment."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List

import yaml

from arcgen import GeneratorConfig, SyntheticARCGenerator
from envs import (
    ArcLatentOptionEnv,
    LatentScorer,
    RewardConfig,
    default_options,
    make_primitive_option,
)
from training.jepa import ObjectCentricJEPATrainer
from training.modules.projection import ProjectionHead

try:  # pragma: no cover - script guard
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Roll out the latent option environment with random policy")
    parser.add_argument("--env-config", type=Path, required=True, help="Environment YAML configuration")
    parser.add_argument("--jepa-config", type=Path, required=True, help="JEPA configuration for encoder")
    parser.add_argument("--episodes", type=int, default=5, help="Number of random episodes to simulate")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for encoder (default: cpu)")
    return parser.parse_args()


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"YAML file {path} must contain a mapping at top level")
    return data


def build_reward_config(data: Dict) -> RewardConfig:
    return RewardConfig(**data)


def build_options(config: Dict) -> List:
    include_defaults = bool(config.get("include_defaults", True))
    options = list(default_options()) if include_defaults else []
    for entry in config.get("extra", []):
        primitive = entry["primitive"]
        params = entry.get("params", {})
        options.append(make_primitive_option(primitive, **params))
    return options


def build_scorer(jepa_config: Dict, device: str) -> LatentScorer:
    trainer = ObjectCentricJEPATrainer(jepa_config)
    projection = None
    loss_cfg = jepa_config.get("loss") or {}
    if loss_cfg:
        projection = ProjectionHead(
            input_dim=trainer.encoder_config.hidden_dim,
            output_dim=int(loss_cfg.get("projection_dim", trainer.encoder_config.hidden_dim)),
            layers=int(loss_cfg.get("projection_layers", 1)),
            activation=str(loss_cfg.get("projection_activation", "relu")),
        )
    scorer = LatentScorer(
        trainer.object_encoder,
        projection_head=projection,
        device=device,
    )
    return scorer


def main() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required for rollouts but is not available")

    args = parse_args()
    env_cfg = load_yaml(args.env_config)
    jepa_cfg = load_yaml(args.jepa_config)

    reward_cfg = build_reward_config(env_cfg.get("reward", {}))
    options = build_options(env_cfg.get("options", {}))
    scorer = build_scorer(jepa_cfg, args.device)

    env = ArcLatentOptionEnv(
        options=options,
        scorer=scorer,
        reward_config=reward_cfg,
        max_steps=int(env_cfg.get("max_steps", 8)),
    )

    generator_cfg = env_cfg.get("generator", {})
    generator = SyntheticARCGenerator(
        GeneratorConfig(
            min_grid_size=int(generator_cfg.get("min_grid_size", 5)),
            max_grid_size=int(generator_cfg.get("max_grid_size", 10)),
            min_colors=int(generator_cfg.get("min_colors", 3)),
            max_colors=int(generator_cfg.get("max_colors", 6)),
            fill_probability=float(generator_cfg.get("fill_probability", 0.75)),
            background_color=int(generator_cfg.get("background_color", 0)),
        ),
        seed=env_cfg.get("seed"),
    )

    schedule = env_cfg.get("task_schedule", {"atomic": 1})
    tasks = []
    for phase, count in schedule.items():
        tasks.extend(generator.sample_many(int(count), phase))

    total_rewards = []
    for idx, task in enumerate(tasks[: args.episodes], 1):
        env.reset(task=(task.input_grid, task.output_grid))
        done = False
        episode_reward = 0.0

        while not done:
            action = random.randrange(env.action_space_n)
            _, reward, done, info = env.step(action)
            episode_reward += reward
        total_rewards.append(episode_reward)
        print(f"Episode {idx}: reward={episode_reward:.3f} success={info.get('success')} steps={env.steps}")

    if total_rewards:
        avg_reward = sum(total_rewards) / len(total_rewards)
        print(f"Average reward over {len(total_rewards)} episodes: {avg_reward:.3f}")
    else:
        print("No tasks sampled; check task_schedule in config.")


if __name__ == "__main__":
    main()
