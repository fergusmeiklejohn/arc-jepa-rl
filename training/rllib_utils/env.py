"""Gym/Gymnasium wrapper for :class:`envs.ArcLatentOptionEnv`."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Mapping, MutableMapping

import numpy as np

try:  # pragma: no cover - prefer gymnasium when available
    import gymnasium as gym
    from gymnasium import spaces
    GYMNASIUM_API = True
    GYM_AVAILABLE = True
except Exception:  # pragma: no cover
    try:
        import gym
        from gym import spaces
        GYMNASIUM_API = False
        GYM_AVAILABLE = True
    except Exception:  # pragma: no cover - neither gym API available
        gym = None  # type: ignore
        spaces = None  # type: ignore
        GYMNASIUM_API = False
        GYM_AVAILABLE = False

from arcgen import GeneratorConfig, Grid, SyntheticARCGenerator, SyntheticTask
from envs import ArcLatentOptionEnv

from .builders import (
    build_env_reward_config,
    build_latent_scorer_from_config,
    build_options_from_config,
)


@dataclass
class LatentTaskSampler:
    """Weighted curriculum sampler for ARC tasks."""

    generator: SyntheticARCGenerator
    schedule: Mapping[str, int]

    def __post_init__(self) -> None:
        weights = [(str(phase), int(weight)) for phase, weight in self.schedule.items() if weight > 0]
        if not weights:
            raise ValueError("task schedule must contain at least one positive weight")
        self._phases, raw_weights = zip(*weights)
        self._weights = list(raw_weights)

    def sample_task(self) -> SyntheticTask:
        phase = random.choices(self._phases, weights=self._weights, k=1)[0]
        return self.generator.sample_task(phase)


if not GYM_AVAILABLE:

    class LatentOptionRLLibEnv:  # type: ignore[misc]
        """Fallback stub when gym/gymnasium is unavailable."""

        def __init__(self, *_args, **_kwargs) -> None:  # pragma: no cover - depends on env setup
            raise RuntimeError("gym or gymnasium is required to instantiate LatentOptionRLLibEnv")

else:

    class LatentOptionRLLibEnv(gym.Env):
        """Expose :class:`ArcLatentOptionEnv` through a Gym-compatible API."""

        metadata = {"render.modes": ["ansi"]}

        def __init__(self, config: MutableMapping[str, object]):
            # Generator/task sampling -------------------------------------------------
            generator_cfg = GeneratorConfig(**(config.get("generator", {}) or {}))
            self._generator = SyntheticARCGenerator(
                generator_cfg,
                seed=config.get("seed"),
                allowed_primitives=config.get("allowed_primitives"),
            )
            schedule = config.get("task_schedule") or {"atomic": 1}
            if not isinstance(schedule, Mapping):
                raise TypeError("task_schedule must be a mapping of phase -> weight")
            self._task_sampler = LatentTaskSampler(self._generator, schedule)

            # Latent scorer + base env ----------------------------------------------
            jepa_cfg = config.get("jepa_config")
            if jepa_cfg is None:
                raise ValueError("env_config missing 'jepa_config'")
            scorer = build_latent_scorer_from_config(jepa_cfg, device=str(config.get("device", "cpu")))
            options = build_options_from_config(config.get("options", {}))
            reward_cfg = build_env_reward_config(config.get("reward", {}))
            max_steps = int(config.get("max_steps", 8))
            terminate_on_exact = bool(config.get("terminate_on_exact_match", True))

            self._base_env = ArcLatentOptionEnv(
                options=options,
                scorer=scorer,
                reward_config=reward_cfg,
                max_steps=max_steps,
                terminate_on_exact_match=terminate_on_exact,
            )

            # Observation/action spaces ---------------------------------------------
            pad_size = int(generator_cfg.max_grid_size)
            self._pad_shape = (pad_size, pad_size)
            self._background = generator_cfg.background_color
            high_value = max(generator_cfg.max_colors, self._background + 1)

            grid_box = spaces.Box(
                low=0,
                high=high_value,
                shape=self._pad_shape,
                dtype=np.int32,
            )

            self.observation_space = spaces.Dict(
                {
                    "current": grid_box,
                    "target": grid_box,
                    "steps": spaces.Box(low=0, high=max_steps, shape=(1,), dtype=np.int32),
                }
            )
            self.action_space = spaces.Discrete(self._base_env.action_space_n)

            self._last_task: Dict[str, object] | None = None

        # ------------------------------------------------------------------ helpers
        def _grid_to_array(self, grid: Grid) -> np.ndarray:
            array = np.full(self._pad_shape, self._background, dtype=np.int32)
            cells = grid.to_lists()
            height = min(len(cells), self._pad_shape[0])
            for row_idx in range(height):
                width = min(len(cells[row_idx]), self._pad_shape[1])
                array[row_idx, :width] = cells[row_idx][:width]
            return array

        def _build_observation(self, current: Grid, target: Grid) -> Dict[str, np.ndarray]:
            obs = {
                "current": self._grid_to_array(current),
                "target": self._grid_to_array(target),
                "steps": np.array([self._base_env.steps], dtype=np.int32),
            }
            return obs

        # --------------------------------------------------------------------- gym
        def reset(self, *, seed: int | None = None, options: Dict | None = None):  # type: ignore[override]
            if seed is not None:  # pragma: no cover - deterministic seeds unused
                random.seed(seed)
            task = self._task_sampler.sample_task()
            self._base_env.reset(task=(task.input_grid, task.output_grid))
            self._last_task = {"task_id": task.task_id, "target": task.output_grid, "start": task.input_grid}
            obs = self._build_observation(task.input_grid, task.output_grid)
            info = {"task_id": task.task_id, "phase": task.phase}
            if GYMNASIUM_API:
                return obs, info
            return obs

        def step(self, action):  # type: ignore[override]
            grid, reward, done, info = self._base_env.step(int(action))
            target_grid = self._last_task["target"] if self._last_task else grid
            obs = self._build_observation(grid, target_grid)
            success = bool(info.get("success"))
            terminated = success
            truncated = done and not success
            if GYMNASIUM_API:
                return obs, reward, terminated, truncated, info
            done_flag = done
            return obs, reward, done_flag, info

        def render(self):  # pragma: no cover - delegated to base env
            return self._base_env.render()


__all__ = ["LatentOptionRLLibEnv", "LatentTaskSampler", "GYMNASIUM_API", "GYM_AVAILABLE"]
