import pytest

torch = pytest.importorskip("torch")

from training.rllib_utils import (
    ActorCriticConfig,
    ActorCriticCore,
    RLLibNotInstalledError,
    register_hierarchical_models,
)


def test_actor_critic_core_shapes_and_sampling():
    config = ActorCriticConfig(hidden_dims=(32,), include_termination=True, dropout=0.0)
    core = ActorCriticCore(obs_dim=6, action_dim=4, config=config)

    batch = torch.randn(3, 6)
    outputs = core(batch)

    assert outputs["logits"].shape == (3, 4)
    assert outputs["value"].shape == (3,)

    termination = outputs["termination"]
    assert termination is not None
    assert termination.shape == (3,)
    assert torch.all((termination >= 0.0) & (termination <= 1.0))

    actions = core.sample_actions(outputs["logits"])
    assert actions.shape == (3,)


def test_actor_critic_core_without_termination():
    config = ActorCriticConfig(hidden_dims=(16,), include_termination=False)
    core = ActorCriticCore(obs_dim=4, action_dim=3, config=config)

    outputs = core(torch.randn(1, 4))
    assert outputs["termination"] is None


def test_register_models_dependency_guard():
    try:
        import ray  # type: ignore
    except Exception:
        with pytest.raises(RLLibNotInstalledError):
            register_hierarchical_models()
        return

    # If ray is available the registration should succeed without error.
    register_hierarchical_models()
