"""Utilities for wiring the latent option environment into RLlib."""

from .builders import (
    build_env_reward_config,
    build_generator_from_config,
    build_latent_scorer_from_config,
    build_options_from_config,
)
from .bc_data import BehavioralCloningDataset, load_option_traces, split_records
from .bc_trainer import train_behavioral_cloning
from .env import LatentOptionRLLibEnv, LatentTaskSampler
from .models import (
    ActorCriticConfig,
    ActorCriticCore,
    ManagerActorCriticModel,
    OptionActorCriticModel,
    RLLibNotInstalledError,
    register_hierarchical_models,
)

__all__ = [
    "LatentOptionRLLibEnv",
    "LatentTaskSampler",
    "build_env_reward_config",
    "build_latent_scorer_from_config",
    "build_options_from_config",
    "build_generator_from_config",
    "ActorCriticConfig",
    "ActorCriticCore",
    "ManagerActorCriticModel",
    "OptionActorCriticModel",
    "RLLibNotInstalledError",
    "register_hierarchical_models",
    "BehavioralCloningDataset",
    "load_option_traces",
    "split_records",
    "train_behavioral_cloning",
]
