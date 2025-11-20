import json
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import argparse

from scripts import train_active_reasoner, train_jepa
from training.reasoner import ActiveReasonerPolicy, ActiveReasonerPolicyConfig


def _dummy_obs(batch: int, latent_dim: int, seq_len: int, parameter_dim: int):
    return {
        "current_latent": torch.zeros((batch, latent_dim)),
        "target_latent": torch.ones((batch, latent_dim)),
        "program_ids": torch.zeros((batch, seq_len), dtype=torch.long),
        "program_params": torch.zeros((batch, seq_len, parameter_dim)),
        "program_mask": torch.ones((batch, seq_len)),
        "steps": torch.zeros((batch, 1)),
    }


def test_active_reasoner_policy_shapes_and_actions():
    config = ActiveReasonerPolicyConfig(
        latent_dim=4,
        action_dim=3,
        program_embedding_dim=8,
        program_hidden_dim=8,
        program_layers=1,
        hidden_dims=(16,),
        dropout=0.0,
    )
    policy = ActiveReasonerPolicy(vocab_size=5, parameter_dim=1, config=config)

    obs = _dummy_obs(batch=2, latent_dim=config.latent_dim, seq_len=2, parameter_dim=1)
    logits, values = policy(obs)
    assert logits.shape == (2, config.action_dim)
    assert values.shape == (2,)

    action, log_prob, value, entropy = policy.act(obs)
    assert action.shape == (2,)
    assert log_prob.shape == (2,)
    assert value.shape == (2,)
    assert entropy.shape == (2,)

    selected = torch.tensor([0, 1])
    log_prob_eval, entropy_eval, values_eval = policy.evaluate_actions(obs, selected)
    assert log_prob_eval.shape == (2,)
    assert entropy_eval.shape == (2,)
    assert values_eval.shape == (2,)


def test_active_reasoner_policy_rejects_invalid_config():
    with pytest.raises(ValueError):
        ActiveReasonerPolicy(vocab_size=3, parameter_dim=1, config=ActiveReasonerPolicyConfig(latent_dim=0, action_dim=2))
    with pytest.raises(ValueError):
        ActiveReasonerPolicy(
            vocab_size=3,
            parameter_dim=1,
            config=ActiveReasonerPolicyConfig(latent_dim=2, action_dim=2, activation="bogus"),
        )


class _StubEnv:
    def __init__(self, latent_dim: int = 4, seq_len: int = 2):
        self.device = torch.device("cpu")
        self.latent_dim = latent_dim
        self.vocab_size = 5
        self.parameter_dim = 1
        self.action_space = SimpleNamespace(n=2)
        self._seq_len = seq_len

    def reset(self):
        return self._obs()

    def step(self, action: int):
        # Terminate after a single step with a success flag
        return self._obs(), 1.0, True, {"success": True, "sigreg_penalty": 0.0}

    def _obs(self):
        return {
            "current_latent": np.zeros((1, self.latent_dim), dtype=np.float32),
            "target_latent": np.ones((1, self.latent_dim), dtype=np.float32),
            "program_ids": np.zeros((1, self._seq_len), dtype=np.int64),
            "program_params": np.zeros((1, self._seq_len, self.parameter_dim), dtype=np.float32),
            "program_mask": np.ones((1, self._seq_len), dtype=np.float32),
            "steps": np.zeros((1, 1), dtype=np.float32),
        }


def test_train_active_reasoner_cli_smoke(monkeypatch, tmp_path):
    # Stub out the heavy HypothesisSearchEnv with a minimal CPU-only env
    monkeypatch.setattr(train_active_reasoner, "HypothesisSearchEnv", lambda cfg: _StubEnv())

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        json.dumps(
            {
                "env": {},
                "trainer": {
                    "episodes_per_iter": 1,
                    "iterations": 1,
                    "entropy_coeff": 0.0,
                    "value_coeff": 0.5,
                },
                "policy": {
                    "device": "cpu",
                    "program_embedding_dim": 4,
                    "program_hidden_dim": 4,
                    "hidden_dims": [8],
                    "activation": "relu",
                    "dropout": 0.0,
                },
            }
        )
    )

    output_dir = tmp_path / "run"
    monkeypatch.setattr(
        train_active_reasoner,
        "parse_args",
        lambda: argparse.Namespace(config=config_path, output_dir=output_dir, iterations=1),
    )

    train_active_reasoner.main()

    metrics_path = output_dir / "metrics.jsonl"
    checkpoint_path = output_dir / "policy.pt"
    assert metrics_path.exists()
    metrics = metrics_path.read_text().strip().splitlines()
    assert metrics, "expected at least one metrics line"
    assert checkpoint_path.exists()


def test_train_jepa_dry_run_cli(tmp_path, monkeypatch):
    config = {
        "tokenizer": {"max_objects": 2, "max_color_features": 2},
        "object_encoder": {"hidden_dim": 4, "num_embeddings": 8},
        "optimizer": {"lr": 1e-3},
        "training": {"device": "cpu"},
    }
    config_path = tmp_path / "jepa.yaml"
    config_path.write_text(json.dumps(config))

    monkeypatch.setattr(
        train_jepa,
        "parse_args",
        lambda: argparse.Namespace(
            config=config_path,
            dry_run=True,
            device="cpu",
            mixed_precision=None,
            ddp=False,
            ddp_backend="nccl",
            ddp_world_size=None,
            ddp_rank=None,
        ),
    )

    train_jepa.main()
