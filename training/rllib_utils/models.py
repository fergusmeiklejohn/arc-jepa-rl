"""Actor–critic policy builders for hierarchical RLlib integrations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .observation import flatten_batch_observation, flatten_observation

try:  # pragma: no cover - optional dependency
    import gymnasium as gym
except Exception:  # pragma: no cover
    gym = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from ray.rllib.models import ModelCatalog
    from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
    from ray.rllib.utils.typing import ModelConfigDict, TensorType

    _HAS_RLLIB = True
except Exception:  # pragma: no cover
    ModelCatalog = None  # type: ignore
    ModelConfigDict = Dict[str, Any]  # type: ignore
    TensorType = torch.Tensor  # type: ignore

    class TorchModelV2(nn.Module):  # type: ignore[misc]
        """Lightweight stub used when RLlib is unavailable."""

        def __init__(self, obs_space, action_space, num_outputs, model_config, name):  # pragma: no cover - stub
            super().__init__()
            self.obs_space = obs_space
            self.action_space = action_space
            self.num_outputs = num_outputs
            self.model_config = model_config
            self.name = name

        def value_function(self) -> TensorType:  # pragma: no cover - stub
            raise RuntimeError("RLlib is required for this model")

    _HAS_RLLIB = False


class RLLibNotInstalledError(RuntimeError):
    """Raised when RLlib-specific helpers are invoked without the dependency."""


_ACTIVATIONS: Mapping[str, nn.Module] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
}


@dataclass(frozen=True)
class ActorCriticConfig:
    """Configurable hyper-parameters for the shared trunk."""

    hidden_dims: Sequence[int] = (256, 256)
    activation: str = "relu"
    layer_norm: bool = True
    dropout: float = 0.0
    include_termination: bool = True
    action_mask_key: Optional[str] = "action_mask"


class ActorCriticCore(nn.Module):
    """Torch module that powers both option and manager policies."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        config: Optional[ActorCriticConfig] = None,
    ) -> None:
        super().__init__()
        if obs_dim <= 0:
            raise ValueError("obs_dim must be positive")
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")

        cfg = config or ActorCriticConfig()
        activation_cls = _ACTIVATIONS.get(cfg.activation.lower())
        if activation_cls is None:
            raise ValueError(f"Unsupported activation '{cfg.activation}'")
        if cfg.dropout < 0 or cfg.dropout >= 1:
            raise ValueError("dropout must be in [0, 1)")

        layers: List[nn.Module] = []
        last_dim = obs_dim
        for width in cfg.hidden_dims:
            if width <= 0:
                raise ValueError("hidden_dims entries must be positive")
            layers.append(nn.Linear(last_dim, width))
            if cfg.layer_norm:
                layers.append(nn.LayerNorm(width))
            layers.append(activation_cls())
            if cfg.dropout > 0:
                layers.append(nn.Dropout(p=cfg.dropout))
            last_dim = width

        self.trunk = nn.Sequential(*layers) if layers else nn.Identity()
        self.policy_head = nn.Linear(last_dim, action_dim)
        self.value_head = nn.Linear(last_dim, 1)
        self.termination_head = nn.Linear(last_dim, 1) if cfg.include_termination else None
        self.include_termination = cfg.include_termination

        self._reset_parameters()

    def forward(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        features = self.trunk(obs.float())
        logits = self.policy_head(features)
        values = self.value_head(features).squeeze(-1)
        termination: Optional[torch.Tensor] = None
        if self.termination_head is not None:
            termination = torch.sigmoid(self.termination_head(features)).squeeze(-1)
        return {
            "logits": logits,
            "value": values,
            "termination": termination,
            "features": features,
        }

    @torch.no_grad()
    def sample_actions(self, logits: torch.Tensor, *, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sample discrete actions respecting optional boolean masks."""

        masked_logits = _apply_action_mask(logits, mask)
        distribution = torch.distributions.Categorical(logits=masked_logits)
        return distribution.sample()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)
        if self.termination_head is not None:
            nn.init.orthogonal_(self.termination_head.weight, gain=0.01)
            nn.init.zeros_(self.termination_head.bias)


class _BaseRLLibActorCritic(TorchModelV2):
    """Shared glue for RLlib Torch models."""

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        *,
        include_termination: bool,
    ) -> None:
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        obs_dim = _infer_obs_dim(obs_space)
        action_dim = getattr(action_space, "n", num_outputs)
        cfg, custom_cfg = _build_config(model_config, include_termination=include_termination)
        self.core = ActorCriticCore(
            obs_dim,
            action_dim,
            config=cfg,
        )
        self._last_value: Optional[torch.Tensor] = None
        self._last_termination: Optional[torch.Tensor] = None
        self._action_mask_key = cfg.action_mask_key
        self._load_pretrained(custom_cfg.get("pretrained_weights"))

    def forward(  # type: ignore[override]
        self,
        input_dict: Mapping[str, Any],
        state: List[torch.Tensor],
        seq_lens: torch.Tensor,
    ) -> Tuple[TensorType, List[torch.Tensor]]:
        obs_tensor = _resolve_obs_tensor(input_dict)
        outputs = self.core(obs_tensor)
        self._last_value = outputs["value"]
        self._last_termination = outputs["termination"]

        mask = _extract_action_mask(input_dict, self._action_mask_key)
        logits = _apply_action_mask(outputs["logits"], mask)
        return logits, state

    def value_function(self) -> TensorType:  # type: ignore[override]
        if self._last_value is None:
            raise RuntimeError("value_function called before forward pass")
        return self._last_value

    def termination_probability(self) -> Optional[torch.Tensor]:
        return self._last_termination

    def _load_pretrained(self, path_value: Any) -> None:
        if not path_value:
            return
        path = Path(path_value).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Pretrained weights not found: {path}")
        state = torch.load(path, map_location="cpu")
        missing, unexpected = self.core.load_state_dict(state, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected keys in pretrained weights: {unexpected}")
        if missing:
            print(f"[rllib_utils] Loaded pretrained weights with missing keys: {missing}")


class OptionActorCriticModel(_BaseRLLibActorCritic):
    """Custom RLlib model for option-level policies with termination head."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
            include_termination=True,
        )


class ManagerActorCriticModel(_BaseRLLibActorCritic):
    """Custom RLlib model for high-level option selection."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
            include_termination=False,
        )


def register_hierarchical_models() -> None:
    """Register custom models so RLlib configs can reference them by name."""

    if not _HAS_RLLIB:
        raise RLLibNotInstalledError("RLlib is required to register custom models")

    assert ModelCatalog is not None  # for type-checkers
    ModelCatalog.register_custom_model("arc_option_actor_critic", OptionActorCriticModel)
    ModelCatalog.register_custom_model("arc_manager_actor_critic", ManagerActorCriticModel)


def _build_config(model_config: ModelConfigDict, *, include_termination: bool) -> Tuple[ActorCriticConfig, Mapping[str, Any]]:
    custom_cfg = model_config.get("custom_model_config", {}) if isinstance(model_config, Mapping) else {}
    hidden_dims = tuple(int(x) for x in custom_cfg.get("hidden_dims", (256, 256)))
    activation = str(custom_cfg.get("activation", "relu"))
    layer_norm = bool(custom_cfg.get("layer_norm", True))
    dropout = float(custom_cfg.get("dropout", 0.0))
    action_mask_key = custom_cfg.get("action_mask_key", "action_mask")
    include_term = bool(custom_cfg.get("include_termination", include_termination))
    config = ActorCriticConfig(
        hidden_dims=hidden_dims,
        activation=activation,
        layer_norm=layer_norm,
        dropout=dropout,
        include_termination=include_term,
        action_mask_key=action_mask_key,
    )
    return config, custom_cfg


def _infer_obs_dim(space: Any) -> int:
    if hasattr(space, "shape") and space.shape is not None:  # Box-like
        size = int(np.prod(space.shape))
        if size <= 0:
            raise ValueError("observation space shape must contain elements")
        return size

    if hasattr(space, "n"):  # Discrete
        return int(space.n)

    if gym is not None:
        if isinstance(space, gym.spaces.Dict):
            return sum(_infer_obs_dim(sub_space) for sub_space in space.spaces.values())
        if isinstance(space, gym.spaces.Tuple):
            return sum(_infer_obs_dim(sub_space) for sub_space in space.spaces)

    nested = getattr(space, "spaces", None)
    if isinstance(nested, Mapping):
        return sum(_infer_obs_dim(sub_space) for sub_space in nested.values())
    if isinstance(nested, (list, tuple)):
        return sum(_infer_obs_dim(sub_space) for sub_space in nested)

    raise TypeError("Unsupported observation space for ActorCriticModel")


def _resolve_obs_tensor(input_dict: Mapping[str, Any]) -> torch.Tensor:
    obs = input_dict.get("obs_flat")
    if isinstance(obs, torch.Tensor):
        return obs.float()

    obs = input_dict.get("obs")
    if isinstance(obs, torch.Tensor):
        return obs.view(obs.size(0), -1).float()
    if isinstance(obs, Mapping):
        obs = {key: value for key, value in obs.items() if key != "action_mask"}
        return flatten_batch_observation(obs)

    raise TypeError("Unsupported observation payload")


def _extract_action_mask(input_dict: Mapping[str, Any], action_mask_key: Optional[str]) -> Optional[torch.Tensor]:
    if not action_mask_key:
        return None
    obs = input_dict.get("obs")
    mask_source: Any = None
    if isinstance(obs, Mapping) and action_mask_key in obs:
        mask_source = obs[action_mask_key]
    elif action_mask_key in input_dict:
        mask_source = input_dict[action_mask_key]
    if mask_source is None:
        return None
    tensor = torch.as_tensor(mask_source).float()
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


def _apply_action_mask(logits: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return logits
    if mask.shape != logits.shape:
        raise ValueError("action mask must match logits shape")
    # clamp to avoid log(0)
    masked = logits + torch.log(mask.clamp(min=1e-6))
    return masked
