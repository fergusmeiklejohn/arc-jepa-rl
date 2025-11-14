import pytest

torch = pytest.importorskip("torch")

from training.rllib_utils.env import GYMNASIUM_API, GYM_AVAILABLE, LatentOptionRLLibEnv

pytestmark = pytest.mark.skipif(not GYM_AVAILABLE, reason="gym/gymnasium not installed")


def _tiny_jepa_config() -> dict:
    return {
        "tokenizer": {
            "max_objects": 4,
            "max_color_features": 2,
            "background": 0,
        },
        "object_encoder": {
            "hidden_dim": 8,
            "num_embeddings": 16,
            "commitment_cost": 0.25,
            "ema_decay": 0.9,
        },
    }


def _env_config() -> dict:
    return {
        "generator": {
            "min_grid_size": 3,
            "max_grid_size": 3,
            "min_colors": 2,
            "max_colors": 3,
            "background_color": 0,
            "fill_probability": 0.8,
        },
        "task_schedule": {"atomic": 1},
        "options": {"include_defaults": True},
        "reward": {
            "success_threshold": 1.0,
            "success_bonus": 0.5,
            "step_penalty": 0.0,
            "invalid_penalty": 0.0,
            "distance_scale": 1.0,
            "metric": "cosine",
        },
        "jepa_config": _tiny_jepa_config(),
        "max_steps": 2,
    }


def test_rllib_env_reset_and_step_shapes():
    env = LatentOptionRLLibEnv(_env_config())
    reset_output = env.reset()
    if GYMNASIUM_API:
        obs, info = reset_output
        assert "task_id" in info
    else:
        obs = reset_output
    assert set(obs.keys()) == {"current", "target", "steps"}
    assert obs["current"].shape == (3, 3)
    assert obs["target"].shape == (3, 3)

    action = 0 if env.action_space.n > 0 else env.action_space.sample()
    step_output = env.step(action)
    if GYMNASIUM_API:
        next_obs, reward, terminated, truncated, info = step_output
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
    else:
        next_obs, reward, done, info = step_output
        assert isinstance(done, bool)
    assert next_obs["current"].shape == (3, 3)
    assert isinstance(reward, float)
