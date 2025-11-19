"""Gym environment for latent hypothesis search using ProgramConditioned JEPA."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Sequence

import numpy as np

try:  # pragma: no cover - prefer gymnasium when available
    import gymnasium as gym
    from gymnasium import spaces

    GYMNASIUM_API = True
    GYM_AVAILABLE = True
except Exception:  # pragma: no cover
    try:
        import gym  # type: ignore
        from gym import spaces  # type: ignore

        GYMNASIUM_API = False
        GYM_AVAILABLE = True
    except Exception:  # pragma: no cover - neither API is present
        gym = None  # type: ignore
        spaces = None  # type: ignore
        GYMNASIUM_API = False
        GYM_AVAILABLE = False

import torch
import torch.nn.functional as F

from training.jepa import (
    ObjectCentricJEPATrainer,
    ProgramConditionedJEPA,
    ProgramConditionedModelConfig,
    ProgramTripleDataset,
    ProgramTripleRecord,
    aggregate_object_encoding,
    load_jepa_config,
)
from training.jepa.object_pipeline import build_object_token_batch


class GymUnavailableError(RuntimeError):
    """Raised when the environment is instantiated without gym/gymnasium installed."""


@dataclass(frozen=True)
class HypothesisRewardConfig:
    """Reward shaping parameters for :class:`HypothesisSearchEnv`."""

    success_distance: float = 0.05
    success_bonus: float = 1.0
    step_penalty: float = 0.05
    invalid_penalty: float = 0.1
    distance_scale: float = 1.0
    simplicity_weight: float = 0.02

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None) -> "HypothesisRewardConfig":
        if data is None:
            return cls()
        return cls(
            success_distance=float(data.get("success_distance", cls.success_distance)),
            success_bonus=float(data.get("success_bonus", cls.success_bonus)),
            step_penalty=float(data.get("step_penalty", cls.step_penalty)),
            invalid_penalty=float(data.get("invalid_penalty", cls.invalid_penalty)),
            distance_scale=float(data.get("distance_scale", cls.distance_scale)),
            simplicity_weight=float(data.get("simplicity_weight", cls.simplicity_weight)),
        )

    def validate(self) -> None:
        if self.success_distance <= 0:
            raise ValueError("success_distance must be positive")
        if self.distance_scale <= 0:
            raise ValueError("distance_scale must be positive")
        if self.step_penalty < 0 or self.invalid_penalty < 0 or self.simplicity_weight < 0:
            raise ValueError("penalties must be non-negative")


@dataclass(frozen=True)
class _ActionEntry:
    """Single primitive+parameter combination available to the policy."""

    primitive: str
    params: Mapping[str, object]
    token_id: int
    param_vector: torch.Tensor


def _resolve_path(path_value: str | Path | None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = path.resolve()
    return path


def _load_checkpoint(model: ProgramConditionedJEPA, checkpoint_path: Path) -> None:
    payload = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(payload, Mapping) and "model_state" in payload:
        state = payload["model_state"]
    else:
        state = payload
    model.load_state_dict(state, strict=False)


def _normalize(observation: torch.Tensor) -> torch.Tensor:
    return F.normalize(observation, dim=-1)


def _latent_from_grid(
    trainer: ObjectCentricJEPATrainer,
    grid,
    *,
    device: torch.device,
) -> torch.Tensor:
    batch = build_object_token_batch([grid], trainer.tokenizer_config, device=device)
    encoding = trainer.object_encoder.encode_tokens(batch, device=device, non_blocking=True)
    latent = aggregate_object_encoding(encoding)
    return _normalize(latent.squeeze(0))


class HypothesisSearchEnv(gym.Env if GYM_AVAILABLE else object):  # type: ignore[misc]
    """Latent-program RL environment driven by ProgramConditioned JEPA counterfactuals."""

    metadata = {"render.modes": ["human"]}

    def __init__(self, config: MutableMapping[str, object] | None):
        if not GYM_AVAILABLE:  # pragma: no cover - import guard
            raise GymUnavailableError("gym or gymnasium must be installed to use HypothesisSearchEnv")
        if torch is None:  # pragma: no cover - defensive guard
            raise RuntimeError("PyTorch is required for HypothesisSearchEnv")

        cfg = dict(config or {})
        dataset_path = _resolve_path(cfg.get("dataset_path") or cfg.get("program_dataset"))
        if dataset_path is None:
            raise ValueError("env_config missing 'dataset_path' pointing to program triples JSONL")

        jepa_config = cfg.get("jepa_config")
        if jepa_config is None:
            jepa_path = _resolve_path(cfg.get("jepa_config_path"))
            if jepa_path is None:
                raise ValueError("env_config must provide 'jepa_config' inline or via 'jepa_config_path'")
            jepa_config = load_jepa_config(jepa_path)

        self.device = torch.device(cfg.get("device") or cfg.get("training_device") or "cpu")
        self._rng = random.Random(int(cfg.get("seed", 0)))

        self._trainer = ObjectCentricJEPATrainer(jepa_config)
        self._trainer.encoder.to(self.device)

        dataset = ProgramTripleDataset(
            dataset_path,
            tokenizer_config=self._trainer.tokenizer_config,
            max_program_length=cfg.get("max_program_length"),
        )
        self._dataset = dataset
        self._records: Sequence[ProgramTripleRecord] = dataset.records
        if not self._records:
            raise ValueError("program dataset contained no usable records for HypothesisSearchEnv")

        self._latent_dim = self._trainer.encoder_config.hidden_dim
        self._parameter_dim = dataset.parameter_dim
        self._max_length = dataset.max_program_length

        program_cfg = ProgramConditionedModelConfig.from_mapping(cfg.get("program_encoder"))
        self._model = ProgramConditionedJEPA(
            latent_dim=self._latent_dim,
            vocab_size=dataset.vocab_size,
            parameter_dim=self._parameter_dim,
            config=program_cfg,
        ).to(self.device)
        checkpoint_path = _resolve_path(cfg.get("program_checkpoint") or cfg.get("checkpoint"))
        if checkpoint_path is not None:
            _load_checkpoint(self._model, checkpoint_path)
        self._model.eval()

        self.max_steps = int(cfg.get("max_chain_length") or cfg.get("max_steps") or self._max_length)
        self.max_steps = max(1, min(self.max_steps, self._max_length))

        self.reward_cfg = HypothesisRewardConfig.from_mapping(cfg.get("reward"))
        self.reward_cfg.validate()

        self._phase_indices = self._build_phase_index(self._records)
        self._phase_schedule = self._normalize_phase_schedule(cfg.get("phase_schedule"))

        max_actions = int(cfg.get("max_actions", 256))
        self._actions = self._build_action_library(self._records, max_entries=max_actions)
        if not self._actions:
            raise ValueError("action library is empty; dataset may lack primitive traces")

        self.action_space = spaces.Discrete(len(self._actions))
        self.observation_space = spaces.Dict(
            {
                "current_latent": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self._latent_dim,),
                    dtype=np.float32,
                ),
                "target_latent": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self._latent_dim,),
                    dtype=np.float32,
                ),
                "program_ids": spaces.Box(
                    low=0,
                    high=float(dataset.vocab_size),
                    shape=(self.max_steps,),
                    dtype=np.float32,
                ),
                "program_params": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.max_steps, self._parameter_dim),
                    dtype=np.float32,
                ),
                "program_mask": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.max_steps,),
                    dtype=np.float32,
                ),
                "steps": spaces.Box(
                    low=0.0,
                    high=float(self.max_steps),
                    shape=(1,),
                    dtype=np.float32,
                ),
            }
        )

        self._current_record: ProgramTripleRecord | None = None
        self._start_latent: torch.Tensor | None = None
        self._target_latent: torch.Tensor | None = None
        self._current_latent: torch.Tensor | None = None
        self._current_distance: float = 1.0
        self._program_ids = torch.zeros((self.max_steps,), dtype=torch.float32, device=self.device)
        self._program_params = torch.zeros(
            (self.max_steps, self._parameter_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self._program_mask = torch.zeros((self.max_steps,), dtype=torch.float32, device=self.device)
        self._step_count = 0

    # ---------------------------------------------------------------- sampling
    def _build_phase_index(
        self,
        records: Sequence[ProgramTripleRecord],
    ) -> Dict[str, list[int]]:
        buckets: Dict[str, list[int]] = {}
        for idx, record in enumerate(records):
            metadata = record.metadata or {}
            phase = str(metadata.get("phase") or metadata.get("phase_name") or "unknown").lower()
            buckets.setdefault(phase, []).append(idx)
        return buckets

    def _normalize_phase_schedule(self, schedule: Mapping[str, object] | None) -> Mapping[str, float]:
        if not schedule:
            return {}
        weights: Dict[str, float] = {}
        for raw_phase, magnitude in schedule.items():
            try:
                weight = float(magnitude)
            except (TypeError, ValueError) as exc:  # pragma: no cover - config guard
                raise ValueError("phase weights must be numeric") from exc
            if weight <= 0:
                continue
            weights[str(raw_phase).lower()] = weight
        total = sum(weights.values())
        if total <= 0:
            return {}
        return {phase: weight / total for phase, weight in weights.items()}

    def _sample_record(self) -> ProgramTripleRecord:
        if not self._phase_schedule:
            idx = self._rng.randrange(len(self._records))
            return self._records[idx]

        phases = list(self._phase_schedule.keys())
        weights = [self._phase_schedule[p] for p in phases]
        phase = random.choices(phases, weights=weights, k=1)[0]
        candidates = self._phase_indices.get(phase)
        if not candidates:
            idx = self._rng.randrange(len(self._records))
        else:
            idx = self._rng.choice(candidates)
        return self._records[idx]

    def _build_action_library(
        self,
        records: Sequence[ProgramTripleRecord],
        *,
        max_entries: int,
    ) -> Sequence[_ActionEntry]:
        seen: Dict[tuple[str, tuple[tuple[str, object], ...]], _ActionEntry] = {}
        for record in records:
            for step in record.program:
                key = (step.primitive, tuple(sorted(step.params.items())))
                if key in seen:
                    continue
                ids, params, _ = self._dataset.program_tokenizer.encode([step])
                entry = _ActionEntry(
                    primitive=step.primitive,
                    params=step.params,
                    token_id=int(ids[0].item()),
                    param_vector=params[0].to(self.device),
                )
                seen[key] = entry
                if 0 < max_entries <= len(seen):
                    return list(seen.values())
        return list(seen.values())

    # ------------------------------------------------------------------- gym API
    def reset(self, *, seed: int | None = None, options: Mapping[str, object] | None = None):  # type: ignore[override]
        if seed is not None:
            self._rng.seed(seed)
        _ = options  # unused placeholder

        record = self._sample_record()
        self._current_record = record
        self._program_ids.zero_()
        self._program_mask.zero_()
        self._program_params.zero_()
        self._step_count = 0

        self._start_latent = _latent_from_grid(self._trainer, record.input_grid, device=self.device)
        self._target_latent = _latent_from_grid(self._trainer, record.output_grid, device=self.device)
        self._current_latent = self._start_latent.clone()
        self._current_distance = self._distance(self._current_latent, self._target_latent)

        obs = self._build_observation()
        info = {
            "task_phase": (record.metadata or {}).get("phase"),
            "program_length": len(record.program),
        }
        if GYMNASIUM_API:
            return obs, info
        return obs

    def step(self, action):  # type: ignore[override]
        if self._current_record is None:
            raise RuntimeError("environment must be reset before stepping")

        idx = int(action)
        if idx < 0 or idx >= len(self._actions):
            raise IndexError("action index out of range")

        info: Dict[str, object] = {
            "action_index": idx,
            "chain_length": self._step_count,
        }

        if self._step_count >= self.max_steps:
            reward = -self.reward_cfg.invalid_penalty
            done = True
            success = False
        else:
            entry = self._actions[idx]
            slot = self._step_count
            self._program_ids[slot] = float(entry.token_id)
            self._program_params[slot] = entry.param_vector
            self._program_mask[slot] = 1.0
            self._step_count += 1
            prev_distance = self._current_distance

            new_latent = self._predict_latent()
            self._current_latent = new_latent
            self._current_distance = self._distance(self._current_latent, self._target_latent)
            progress = prev_distance - self._current_distance

            reward = float(progress * self.reward_cfg.distance_scale)
            reward -= self.reward_cfg.step_penalty
            reward -= self.reward_cfg.simplicity_weight * (self._step_count - 1)

            success = self._current_distance <= self.reward_cfg.success_distance
            if success:
                reward += self.reward_cfg.success_bonus

            done = success or self._step_count >= self.max_steps
            info.update(
                {
                    "distance": self._current_distance,
                    "progress": float(progress),
                    "success": success,
                    "chain_length": self._step_count,
                }
            )

        info.setdefault("success", success)

        obs = self._build_observation()
        if GYMNASIUM_API:
            return obs, float(reward), done, False, info
        return obs, float(reward), done, info

    # ----------------------------------------------------------------- helpers
    def _predict_latent(self) -> torch.Tensor:
        assert self._start_latent is not None
        with torch.inference_mode():
            ids = self._program_ids.unsqueeze(0).long()
            params = self._program_params.unsqueeze(0)
            mask = self._program_mask.unsqueeze(0)
            latent = self._model.predict_counterfactual(
                self._start_latent.unsqueeze(0),
                program_ids=ids,
                program_params=params,
                program_mask=mask,
            )
        return _normalize(latent.squeeze(0))

    def _distance(self, a: torch.Tensor, b: torch.Tensor) -> float:
        value = 1.0 - F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1)
        return float(torch.nan_to_num(value.squeeze(0), nan=1.0, posinf=1.0, neginf=1.0).item())

    def _build_observation(self) -> Dict[str, np.ndarray]:
        assert self._current_latent is not None
        assert self._target_latent is not None
        return {
            "current_latent": self._current_latent.detach().cpu().numpy().astype(np.float32, copy=False),
            "target_latent": self._target_latent.detach().cpu().numpy().astype(np.float32, copy=False),
            "program_ids": self._program_ids.detach().cpu().numpy().astype(np.float32, copy=False),
            "program_params": self._program_params.detach().cpu().numpy().astype(np.float32, copy=False),
            "program_mask": self._program_mask.detach().cpu().numpy().astype(np.float32, copy=False),
            "steps": np.array([float(self._step_count)], dtype=np.float32),
        }

    # -------------------------------------------------------------- diagnostics
    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    @property
    def parameter_dim(self) -> int:
        return self._parameter_dim

    @property
    def vocab_size(self) -> int:
        return self._dataset.vocab_size


__all__ = ["HypothesisRewardConfig", "HypothesisSearchEnv"]
