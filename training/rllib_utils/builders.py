"""Helper utilities shared between RLlib scripts and env wrappers."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from arcgen import GeneratorConfig, SyntheticARCGenerator
from envs import LatentScorer, RewardConfig, default_options, make_primitive_option
from envs.arc_latent_env import Option
from training.jepa import ObjectCentricJEPATrainer, load_jepa_config
from training.modules.projection import ProjectionHead


def build_env_reward_config(config: Mapping[str, object] | None) -> RewardConfig:
    """Return a ``RewardConfig`` from a (possibly empty) mapping."""

    if config is None:
        return RewardConfig()
    if isinstance(config, RewardConfig):
        return config
    return RewardConfig(**config)


def build_options_from_config(config: Mapping[str, object] | None) -> Sequence[Option]:
    """Construct option list based on config (mirrors rollout helper)."""

    if config is None:
        return default_options()

    include_defaults = bool(config.get("include_defaults", True))
    options = list(default_options()) if include_defaults else []
    for entry in config.get("extra", []):
        primitive = entry["primitive"]
        params = entry.get("params", {})
        options.append(make_primitive_option(primitive, **params))
    return tuple(options)


def build_latent_scorer_from_config(
    config_or_path: Mapping[str, object] | str | Path,
    *,
    device: str = "cpu",
) -> LatentScorer:
    """Instantiate :class:`LatentScorer` (encoder + projection head)."""

    if isinstance(config_or_path, (str, Path)):
        jepa_cfg = load_jepa_config(config_or_path)
    else:
        jepa_cfg = config_or_path

    trainer = ObjectCentricJEPATrainer(jepa_cfg)
    projection = None
    loss_cfg = jepa_cfg.get("loss") if isinstance(jepa_cfg, Mapping) else None
    if loss_cfg:
        projection = ProjectionHead(
            input_dim=trainer.encoder_config.hidden_dim,
            output_dim=int(loss_cfg.get("projection_dim", trainer.encoder_config.hidden_dim)),
            layers=int(loss_cfg.get("projection_layers", 1)),
            activation=str(loss_cfg.get("projection_activation", "relu")),
        )
    scorer = LatentScorer(
        trainer.object_encoder,
        projection_head=projection,
        device=device,
    )
    return scorer


def build_generator_from_config(
    config: Mapping[str, object] | None,
    *,
    seed: int | None = None,
    allowed_primitives: Sequence[str] | None = None,
) -> SyntheticARCGenerator:
    cfg = GeneratorConfig(**(config or {}))
    return SyntheticARCGenerator(cfg, seed=seed, allowed_primitives=allowed_primitives)


__all__ = [
    "build_env_reward_config",
    "build_latent_scorer_from_config",
    "build_options_from_config",
    "build_generator_from_config",
]
