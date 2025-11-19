"""Actor–critic policy for the hypothesis search environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.jepa.program_conditioned import ProgramSequenceEncoder


_ACTIVATIONS: Mapping[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
}


@dataclass(frozen=True)
class ActiveReasonerPolicyConfig:
    """Configurable hyper-parameters for :class:`ActiveReasonerPolicy`."""

    latent_dim: int
    action_dim: int
    program_embedding_dim: int = 128
    program_hidden_dim: int = 128
    program_layers: int = 1
    dropout: float = 0.1
    hidden_dims: Sequence[int] = (512, 256)
    activation: str = "relu"


class ActiveReasonerPolicy(nn.Module):
    """Actor–critic network consuming JEPA latents + partial program traces."""

    def __init__(
        self,
        vocab_size: int,
        parameter_dim: int,
        *,
        config: ActiveReasonerPolicyConfig,
    ) -> None:
        super().__init__()
        if config.latent_dim <= 0 or config.action_dim <= 0:
            raise ValueError("latent_dim and action_dim must be positive")
        if parameter_dim < 0:
            raise ValueError("parameter_dim cannot be negative")

        activation_cls = _ACTIVATIONS.get(config.activation.lower())
        if activation_cls is None:
            raise ValueError(f"Unsupported activation '{config.activation}'")

        self.config = config
        self.program_encoder = ProgramSequenceEncoder(
            vocab_size=vocab_size,
            parameter_dim=parameter_dim,
            embedding_dim=config.program_embedding_dim,
            hidden_dim=config.program_hidden_dim,
            num_layers=config.program_layers,
            dropout=config.dropout,
        )

        input_dim = config.latent_dim * 2 + config.program_embedding_dim + 1
        layers: list[nn.Module] = []
        dim = input_dim
        for width in config.hidden_dims:
            if width <= 0:
                continue
            layers.append(nn.Linear(dim, width))
            layers.append(nn.LayerNorm(width))
            layers.append(activation_cls())
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            dim = width

        self.state_encoder = nn.Sequential(*layers) if layers else nn.Identity()
        self.actor_head = nn.Linear(dim, config.action_dim)
        self.value_head = nn.Linear(dim, 1)

    def forward(self, obs: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return action logits and value estimates for a batch of observations."""

        state_vec = self._encode_state(obs)
        features = self.state_encoder(state_vec)
        logits = self.actor_head(features)
        values = self.value_head(features).squeeze(-1)
        return logits, values

    def act(
        self,
        obs: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action and return (action, log_prob, value, entropy)."""

        logits, values = self(obs)
        distribution = torch.distributions.Categorical(logits=logits)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        entropy = distribution.entropy()
        return action, log_prob, values, entropy

    def evaluate_actions(
        self,
        obs: Mapping[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate log-probs/entropy/value for provided actions."""

        logits, values = self(obs)
        distribution = torch.distributions.Categorical(logits=logits)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return log_prob, entropy, values

    # ------------------------------------------------------------------ helpers
    def _encode_state(self, obs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        current = self._ensure_batch(obs["current_latent"].float())
        target = self._ensure_batch(obs["target_latent"].float())
        ids = self._ensure_batch(obs["program_ids"]).long()
        params = self._ensure_batch(obs["program_params"]).float()
        mask = self._ensure_batch(obs["program_mask"]).float()
        steps = self._ensure_batch(obs.get("steps", torch.zeros_like(current[:, :1]))).float()

        program_embedding = self.program_encoder(ids, params, mask)
        state_vec = torch.cat([current, target, program_embedding, steps], dim=-1)
        return state_vec

    @staticmethod
    def _ensure_batch(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 1:
            return tensor.unsqueeze(0)
        return tensor


__all__ = ["ActiveReasonerPolicy", "ActiveReasonerPolicyConfig"]
