import pytest

torch = pytest.importorskip("torch")

from arcgen import Grid
from training.jepa import (
    ObjectCentricJEPATrainer,
    build_trainer_from_config,
    load_jepa_config,
)


def sample_config():
    return {
        "tokenizer": {
            "max_objects": 2,
            "max_color_features": 2,
            "background": 0,
        },
        "object_encoder": {
            "hidden_dim": 12,
            "num_embeddings": 8,
            "commitment_cost": 0.75,
            "ema_decay": 0.95,
            "activation": "relu",
        },
    }


def test_trainer_builds_encoder_from_config():
    config = sample_config()
    trainer = ObjectCentricJEPATrainer(config)

    grid = Grid([[0, 1], [0, 1]])
    pair = trainer.encode_batch([grid], [grid])

    assert pair.context.embeddings.shape[-1] == config["object_encoder"]["hidden_dim"]


def test_build_trainer_from_config():
    trainer = build_trainer_from_config(sample_config())
    assert isinstance(trainer, ObjectCentricJEPATrainer)


def test_load_and_build_trainer_from_yaml(tmp_path):
    yaml = pytest.importorskip("yaml")
    config = sample_config()
    path = tmp_path / "config.yaml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle)

    loaded = load_jepa_config(path)
    trainer = ObjectCentricJEPATrainer(loaded)

    grid = Grid([[0, 1], [0, 1]])
    pair = trainer.encode_batch([grid], [grid])
    assert pair.context.embeddings.shape[0] == 1
