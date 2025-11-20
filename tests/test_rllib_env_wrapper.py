import pytest

torch = pytest.importorskip("torch")

from arcgen import Grid
from envs.arc_latent_env import HierarchicalArcOptionEnv, Option
from training.rllib_utils.env import (
    GYMNASIUM_API,
    GYM_AVAILABLE,
    HierarchicalOptionRLLibEnv,
    LatentOptionRLLibEnv,
)

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


def test_rllib_env_reset_and_step_shapes(monkeypatch):
    class DummyScorer:
        def embed(self, grid):
            return torch.zeros(4)
        def distance(self, a, b, metric="cosine"):
            return torch.zeros(1)

    def _dummy_scorer(_cfg, device="cpu"):
        return DummyScorer()

    monkeypatch.setattr("training.rllib_utils.env.build_latent_scorer_from_config", _dummy_scorer)

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


def test_hierarchical_env_termination_action(monkeypatch):
    class DummyScorer:
        def embed(self, grid):
            return torch.zeros(4)
        def distance(self, a, b, metric="cosine"):
            return torch.zeros(1)

    monkeypatch.setattr("training.rllib_utils.env.build_latent_scorer_from_config", lambda *_args, **_kwargs: DummyScorer())

    env = HierarchicalOptionRLLibEnv(_env_config())
    reset_output = env.reset()
    obs = reset_output[0] if GYMNASIUM_API else reset_output
    assert "action_mask" in obs

    terminate_action = env.action_space.n - 1
    step_output = env.step(terminate_action)
    if GYMNASIUM_API:
        _, _, terminated, truncated, info = step_output
        assert terminated or truncated
    else:
        _, _, done, info = step_output
        assert done
    assert info.get("terminated", True)


def test_hierarchical_arc_env_executes_manager_option():
    class DummyScorer:
        def embed(self, grid):
            value = float(sum(sum(row) for row in grid.cells))
            return torch.tensor([value])

        def distance(self, a, b, metric="cosine"):
            return torch.abs(a - b)

    target_grid = Grid([[1, 1], [1, 1]])
    start_grid = Grid([[0, 0], [0, 0]])

    options = (
        Option(name="noop", apply=lambda g: g),
        Option(name="to_target", apply=lambda _g: target_grid),
    )
    env = HierarchicalArcOptionEnv(
        options=options,
        scorer=DummyScorer(),
        reward_config={"success_threshold": 0.01, "step_penalty": 0.0, "success_bonus": 1.0},
    )
    env.reset(task=(start_grid, target_grid))
    grid, reward, done, info = env.step(1)
    assert grid.cells == target_grid.cells
    assert done
    assert info.get("success")
