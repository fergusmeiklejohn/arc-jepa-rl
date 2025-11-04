import pytest

torch = pytest.importorskip("torch")

from arcgen import Grid
from training.jepa import ObjectCentricJEPAExperiment


def config_with_optimizer():
    return {
        "tokenizer": {
            "max_objects": 2,
            "max_color_features": 2,
        },
        "object_encoder": {
            "hidden_dim": 8,
            "num_embeddings": 16,
        },
        "optimizer": {
            "lr": 1e-3,
        },
    }


def build_sample_grids():
    grid = Grid(
        [
            [0, 1, 1],
            [0, 0, 2],
            [0, 0, 2],
        ]
    )
    return [grid, grid], [grid, grid]


def test_experiment_train_step_runs_and_returns_loss():
    config = config_with_optimizer()
    experiment = ObjectCentricJEPAExperiment(config)

    context, target = build_sample_grids()
    result = experiment.train_step(context, target)

    assert isinstance(result.loss, float)
    assert result.encoded_context.shape[0] == len(context)
    assert result.encoded_target.shape[0] == len(target)
