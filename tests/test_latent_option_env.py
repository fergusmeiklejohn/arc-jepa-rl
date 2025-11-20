import pytest

torch = pytest.importorskip("torch")

from arcgen import Grid
from envs import ArcLatentOptionEnv, LatentScorer, Option, RewardConfig, default_options
from training.jepa import ObjectCentricJEPATrainer


def build_scorer():
    config = {
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
    trainer = ObjectCentricJEPATrainer(config)
    return LatentScorer(trainer.object_encoder, device="cpu", return_codes=False)


def test_latent_env_successful_option_finishes_episode():
    scorer = build_scorer()
    reward_cfg = RewardConfig(
        success_threshold=1.0,
        success_bonus=1.0,
        step_penalty=0.0,
        invalid_penalty=0.0,
        distance_scale=1.0,
        metric="cosine",
    )
    env = ArcLatentOptionEnv(default_options(), scorer, reward_config=reward_cfg, max_steps=3)

    start = Grid([[0, 1], [0, 0]])
    target = Grid([[1, 0], [0, 0]])  # mirror_y produces target
    env.reset(task=(start, target))

    mirror_index = next(idx for idx, opt in enumerate(env.options) if opt.name.startswith("mirror_y"))
    _, reward, done, info = env.step(mirror_index)

    assert done is True
    assert info["success"] is True
    assert reward > 0.5  # reward should include success bonus
    assert env.steps == 1


def test_latent_env_penalises_invalid_option():
    scorer = build_scorer()
    reward_cfg = RewardConfig(
        success_threshold=0.1,
        success_bonus=0.0,
        step_penalty=0.05,
        invalid_penalty=0.2,
        distance_scale=1.0,
        metric="cosine",
    )

    def failing(_grid: Grid) -> Grid:
        raise ValueError("invalid")

    option = Option(name="fail", apply=failing, description="always fails")
    env = ArcLatentOptionEnv([option], scorer, reward_config=reward_cfg, max_steps=1)

    grid = Grid([[0, 0], [0, 0]])
    env.reset(task=(grid, grid))

    _, reward, done, info = env.step(0)

    assert done is True  # max_steps reached
    assert info["applied"] is False
    assert reward == pytest.approx(-(reward_cfg.step_penalty + reward_cfg.invalid_penalty))


def test_reward_config_validation():
    cfg = RewardConfig()
    cfg.validate()  # should not raise

    with pytest.raises(ValueError):
        RewardConfig(success_threshold=0.0).validate()


def test_curiosity_bonus_increases_reward_when_state_changes():
    class DummyScorer:
        def embed(self, grid: Grid):
            return torch.tensor([float(sum(sum(row) for row in grid.cells))])

        def distance(self, a, b, metric="l2"):
            return torch.abs(a - b)

    scorer = DummyScorer()
    reward_cfg = RewardConfig(
        success_threshold=0.001,
        success_bonus=0.0,
        step_penalty=0.0,
        invalid_penalty=0.0,
        curiosity_weight=1.0,
        novelty_weight=0.0,
        metric="l2",
    )

    def increment(grid: Grid) -> Grid:
        cells = [[value + 1 for value in row] for row in grid.cells]
        return Grid(cells)

    option = Option(name="inc", apply=increment)
    env = ArcLatentOptionEnv([option], scorer, reward_config=reward_cfg, max_steps=2)

    start = Grid([[0]])
    target = Grid([[10]])
    env.reset(task=(start, target))

    _, reward, done, info = env.step(0)
    assert not done
    assert info["intrinsic_reward"] > 0.0
    assert reward > 0.0


def test_novelty_bonus_decreases_on_revisit():
    class DummyScorer:
        def embed(self, grid: Grid):
            return torch.tensor([float(sum(sum(row) for row in grid.cells))])

        def distance(self, a, b, metric="l2"):
            return torch.abs(a - b)

    scorer = DummyScorer()
    reward_cfg = RewardConfig(
        success_threshold=0.001,
        success_bonus=0.0,
        step_penalty=0.0,
        invalid_penalty=0.0,
        curiosity_weight=0.0,
        novelty_weight=1.0,
        metric="l2",
    )

    def toggle(grid: Grid) -> Grid:
        cells = [[1 - value for value in row] for row in grid.cells]
        return Grid(cells)

    option = Option(name="toggle", apply=toggle)
    env = ArcLatentOptionEnv([option], scorer, reward_config=reward_cfg, max_steps=3)

    start = Grid([[0]])
    target = Grid([[5]])
    env.reset(task=(start, target))

    _, reward1, done1, info1 = env.step(0)
    assert not done1
    _, reward2, done2, info2 = env.step(0)

    assert reward2 <= reward1
    assert info2["novelty"] <= info1["novelty"]


def test_latent_scorer_can_return_codes():
    scorer = build_scorer()
    grid = Grid([[1, 0], [0, 1]])
    embedding, codes = scorer.embed(grid, include_codes=True)
    assert embedding.shape[0] > 0
    assert codes is not None
    assert codes.shape[0] == 1
