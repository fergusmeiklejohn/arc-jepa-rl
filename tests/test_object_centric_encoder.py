import pytest

torch = pytest.importorskip("torch")

from arcgen import Grid
from training.jepa import (
    ObjectCentricJEPAEncoder,
    ObjectTokenizerConfig,
    build_object_token_batch,
)
from training.modules import ObjectTokenEncoder


def test_build_object_token_batch_shapes():
    grid = Grid([[0, 1], [0, 1]])
    cfg = ObjectTokenizerConfig(max_objects=2, max_color_features=1)

    batch = build_object_token_batch([grid, grid], cfg)

    assert batch.features.shape == (2, 2, cfg.feature_dim)
    assert batch.mask.shape == (2, 2)
    assert batch.adjacency.shape == (2, 2, 2)


def test_object_centric_encoder_forward_pass():
    grid = Grid(
        [
            [0, 1, 1],
            [0, 0, 2],
            [0, 0, 2],
        ]
    )

    cfg = ObjectTokenizerConfig(max_objects=3, max_color_features=2)
    encoder = ObjectTokenEncoder(
        feature_dim=cfg.feature_dim,
        hidden_dim=8,
        num_embeddings=4,
        commitment_cost=0.5,
    )

    wrapper = ObjectCentricJEPAEncoder(encoder, cfg)
    output = wrapper.encode([grid, grid])

    assert output.embeddings.shape == (2, 3, 8)
    assert output.mask.shape == (2, 3)
    assert output.adjacency.shape == (2, 3, 3)
    assert output.vq_loss.shape == ()
    assert output.vq_indices.shape == (2, 3)
