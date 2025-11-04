import pytest

torch = pytest.importorskip("torch", reason="object encoder requires torch")

from arcgen import Grid
from training.modules import ObjectTokenEncoder, tokenize_grid_objects


def test_object_token_encoder_forward_shapes():
    grid = Grid(
        [
            [0, 1, 1],
            [0, 0, 2],
            [0, 0, 2],
        ]
    )

    tokens = tokenize_grid_objects(grid, max_objects=3, max_color_features=2)
    features = torch.tensor(tokens.features, dtype=torch.float32).unsqueeze(0)
    mask = torch.tensor(tokens.mask, dtype=torch.float32).unsqueeze(0)

    encoder = ObjectTokenEncoder(feature_dim=features.size(-1), hidden_dim=8, num_embeddings=4, commitment_cost=0.5)
    output = encoder(features, mask)

    assert output["embeddings"].shape == (1, 3, 8)
    assert output["mask"].shape == (1, 3)
    assert output["vq_loss"].shape == ()
    assert output["vq_indices"].shape == (1, 3)


def test_object_token_encoder_without_vq_returns_none_loss():
    grid = Grid([[0, 1], [0, 1]])
    tokens = tokenize_grid_objects(grid, max_objects=1, max_color_features=1)
    features = torch.tensor(tokens.features, dtype=torch.float32).unsqueeze(0)

    encoder = ObjectTokenEncoder(feature_dim=features.size(-1), hidden_dim=4, num_embeddings=None)
    output = encoder(features)

    assert output["vq_loss"] is None
    assert output["vq_indices"] is None
