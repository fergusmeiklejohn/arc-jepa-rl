import torch

from training.reasoner import ActiveReasonerPolicy, ActiveReasonerPolicyConfig, HypothesisSearchEnv


def _maybe_unwrap_obs(reset_result):
    if isinstance(reset_result, tuple):
        return reset_result[0]
    return reset_result


def _obs_to_tensors(obs):
    return {
        "current_latent": torch.from_numpy(obs["current_latent"]).unsqueeze(0),
        "target_latent": torch.from_numpy(obs["target_latent"]).unsqueeze(0),
        "program_ids": torch.from_numpy(obs["program_ids"]).unsqueeze(0),
        "program_params": torch.from_numpy(obs["program_params"]).unsqueeze(0),
        "program_mask": torch.from_numpy(obs["program_mask"]).unsqueeze(0),
        "steps": torch.from_numpy(obs["steps"]).unsqueeze(0),
    }


def test_hypothesis_env_reset_and_step():
    env = HypothesisSearchEnv(
        {
            "dataset_path": "tests/data/program_triples_tiny.jsonl",
            "jepa_config_path": "configs/training/jepa_program_conditioned.yaml",
            "max_chain_length": 2,
            "reward": {"success_distance": 2.0},
            "max_actions": 8,
        }
    )

    obs = _maybe_unwrap_obs(env.reset())
    assert obs["current_latent"].shape[-1] == env.latent_dim
    result = env.step(0)
    if len(result) == 5:
        next_obs, reward, terminated, truncated, info = result
        done = terminated or truncated
    else:
        next_obs, reward, done, info = result
    assert isinstance(reward, float)
    assert "success" in info
    assert next_obs["program_ids"].shape == (env.max_steps,)
    assert done in {True, False}


def test_active_reasoner_policy_shapes():
    env = HypothesisSearchEnv(
        {
            "dataset_path": "tests/data/program_triples_tiny.jsonl",
            "jepa_config_path": "configs/training/jepa_program_conditioned.yaml",
            "max_chain_length": 2,
            "reward": {"success_distance": 2.0},
            "max_actions": 8,
        }
    )
    obs = _obs_to_tensors(_maybe_unwrap_obs(env.reset()))
    policy_cfg = ActiveReasonerPolicyConfig(
        latent_dim=env.latent_dim,
        action_dim=env.action_space.n,
        program_embedding_dim=32,
        program_hidden_dim=32,
        hidden_dims=(64,),
        dropout=0.0,
    )
    policy = ActiveReasonerPolicy(env.vocab_size, env.parameter_dim, config=policy_cfg)
    logits, values = policy(obs)
    assert logits.shape[-1] == env.action_space.n
    assert values.shape[0] == 1
