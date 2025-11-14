"""Train option policies in the latent JEPA environment using RLlib."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import yaml

from envs import RewardConfig
from training.jepa import load_jepa_config
from training.rllib_utils import LatentOptionRLLibEnv, register_hierarchical_models

try:  # pragma: no cover - optional heavy dependency
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.algorithms.appo import APPOConfig
    from ray.tune.registry import register_env

    RLLIB_AVAILABLE = True
except Exception:  # pragma: no cover
    ray = None
    PPOConfig = None
    APPOConfig = None
    register_env = None
    RLLIB_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="RL configuration YAML")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/rl"), help="Where to store metrics")
    parser.add_argument("--stop-iters", type=int, default=None, help="Override training_iteration stop")
    parser.add_argument("--local-mode", action="store_true", help="Run Ray in local-mode for easier debugging")
    return parser.parse_args()


def load_rl_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("RL config must be a mapping")
    return data


def resolve_jepa_config(config: Dict[str, Any], root: Path) -> Dict[str, Any]:
    if "jepa_config" in config:
        jepa_cfg = config["jepa_config"]
        if not isinstance(jepa_cfg, dict):
            raise ValueError("jepa_config must be an inline mapping")
        return jepa_cfg
    if "jepa_config_path" in config:
        path = Path(config["jepa_config_path"])
        if not path.is_absolute():
            path = (root / path).resolve()
        return load_jepa_config(path)
    raise ValueError("config must provide jepa_config or jepa_config_path")


def prepare_env_config(config: Dict[str, Any], config_dir: Path) -> Dict[str, Any]:
    env_cfg = dict(config.get("env", {}))
    jepa_cfg = resolve_jepa_config(config, config_dir)
    env_cfg.setdefault("reward", asdict(RewardConfig()))
    env_cfg["jepa_config"] = jepa_cfg
    env_cfg.setdefault("device", config.get("device", "cpu"))
    return env_cfg


def build_algorithm_config(env_name: str, env_config: Dict[str, Any], trainer_cfg: Dict[str, Any]):
    algorithm = str(trainer_cfg.get("algorithm", "ppo")).lower()
    algo_map = {}
    if PPOConfig is not None:
        algo_map["ppo"] = PPOConfig
    if APPOConfig is not None:
        algo_map["appo"] = APPOConfig

    config_cls = algo_map.get(algorithm)
    if config_cls is None:
        raise ValueError(
            f"Unsupported or unavailable trainer.algorithm '{algorithm}'."
        )
    config = config_cls()

    config = config.api_stack(
        enable_rl_module_and_learner=False,
        enable_env_runner_and_connector_v2=False,
    )
    config = config.environment(env=env_name, env_config=env_config)
    rollout_cfg = trainer_cfg.get("rollouts", {})
    num_workers = int(trainer_cfg.get("num_workers", rollout_cfg.get("num_env_runners", 0)))
    fragment_length = int(rollout_cfg.get("fragment_length", trainer_cfg.get("rollout_fragment_length", 64)))
    num_envs_per_worker = int(rollout_cfg.get("num_envs_per_worker", 1))
    config = config.env_runners(
        num_env_runners=num_workers,
        num_envs_per_env_runner=num_envs_per_worker,
        rollout_fragment_length=fragment_length,
    )
    config = config.resources(num_gpus=float(trainer_cfg.get("num_gpus", 0)))
    config = config.framework(str(trainer_cfg.get("framework", "torch")))
    lr = float(trainer_cfg.get("lr", 3e-4))
    train_batch_size = int(trainer_cfg.get("train_batch_size", 256))
    if algorithm == "ppo":
        config = config.training(
            train_batch_size=train_batch_size,
            lr=lr,
        )
    elif algorithm == "appo":
        config = config.training(
            lr=lr,
            train_batch_size=train_batch_size,
            use_kl_loss=bool(trainer_cfg.get("use_kl_loss", False)),
            kl_coeff=float(trainer_cfg.get("kl_coeff", 0.5)),
        )
    model_cfg = trainer_cfg.get("model")
    if model_cfg:
        merged = dict(getattr(config, "model", {}) or {})
        merged.update(model_cfg)
        config.model = merged
    return config


def run_training(algo_config, stop_iters: int | None) -> Dict[str, Any]:
    algo = algo_config.build()
    iterations = stop_iters if stop_iters is not None else 1
    results: list[Dict[str, Any]] = []
    for idx in range(1, iterations + 1):
        result = algo.train()
        summary = {
            "iteration": result.get("training_iteration", idx),
            "episode_reward_mean": result.get("episode_reward_mean"),
            "episodes_total": result.get("episodes_total"),
            "timesteps_total": result.get("timesteps_total"),
        }
        results.append(summary)
        print(f"Iteration {summary['iteration']}: reward_mean={summary['episode_reward_mean']}")
    checkpoint = algo.save()
    checkpoint_path = str(checkpoint)
    return {"results": results, "checkpoint": checkpoint_path}


def main() -> None:
    if not RLLIB_AVAILABLE:  # pragma: no cover - exercised in runtime
        raise RuntimeError("ray[rllib] is required for this script. Install via `pip install 'ray[rllib]'`")

    args = parse_args()
    config = load_rl_config(args.config)
    env_config = prepare_env_config(config, args.config.parent)
    trainer_cfg = dict(config.get("trainer", {}))
    algorithm = str(trainer_cfg.get("algorithm", "ppo")).lower()

    env_name = "arc_latent_option_env"
    register_env(env_name, lambda cfg: LatentOptionRLLibEnv(cfg))
    register_hierarchical_models()

    algo_config = build_algorithm_config(env_name, env_config, trainer_cfg)

    stop_iters = args.stop_iters or trainer_cfg.get("stop", {}).get("training_iteration")

    ray.init(local_mode=args.local_mode, ignore_reinit_error=True)
    try:
        metrics = run_training(algo_config, stop_iters)
    finally:
        ray.shutdown()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / f"{algorithm}_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
