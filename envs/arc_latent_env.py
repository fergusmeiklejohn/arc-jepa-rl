"""Arc latent option environment driven by JEPA embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Mapping, Optional, Sequence, Tuple

from arcgen import Grid, PRIMITIVE_REGISTRY

try:  # pragma: no cover - torch optional
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    F = None  # type: ignore

from training.jepa import ObjectCentricJEPAEncoder
from training.modules.projection import ProjectionHead


class TorchUnavailableError(RuntimeError):
    """Raised when torch-dependent functionality is requested without torch."""


def _ensure_torch() -> None:
    if torch is None:  # pragma: no cover - safety net
        raise TorchUnavailableError("PyTorch is required for latent reward computations")


@dataclass(frozen=True)
class Option:
    """Discrete action that transforms an ARC grid."""

    name: str
    apply: Callable[[Grid], Grid]
    description: str = ""


def make_primitive_option(name: str, **kwargs: int) -> Option:
    spec = PRIMITIVE_REGISTRY.get(name)

    def apply(grid: Grid) -> Grid:
        return spec.apply(grid, **kwargs)

    if kwargs:
        params = ", ".join(f"{key}={value}" for key, value in kwargs.items())
        option_name = f"{name}[{params}]"
        description = f"{name} with {params}"
    else:
        option_name = name
        description = name
    return Option(name=option_name, apply=apply, description=description)


def default_options() -> Tuple[Option, ...]:
    """Return a curated set of options suitable for early experiments."""

    options: List[Option] = [
        make_primitive_option("mirror_x"),
        make_primitive_option("mirror_y"),
        make_primitive_option("rotate90", k=1),
        make_primitive_option("rotate90", k=2),
        make_primitive_option("rotate90", k=3),
    ]

    for dx in (-1, 1):
        options.append(make_primitive_option("translate", dx=dx, dy=0, fill=0))
    for dy in (-1, 1):
        options.append(make_primitive_option("translate", dx=0, dy=dy, fill=0))

    return tuple(options)


@dataclass
class RewardConfig:
    """Parameters controlling latent-space shaping."""

    success_threshold: float = 0.05
    success_bonus: float = 1.0
    step_penalty: float = 0.05
    invalid_penalty: float = 0.1
    distance_scale: float = 1.0
    metric: str = "cosine"  # or "l2"

    def validate(self) -> None:
        if self.success_threshold <= 0:
            raise ValueError("success_threshold must be positive")
        if self.step_penalty < 0:
            raise ValueError("step_penalty must be non-negative")
        if self.invalid_penalty < 0:
            raise ValueError("invalid_penalty must be non-negative")
        if self.distance_scale <= 0:
            raise ValueError("distance_scale must be positive")
        if self.metric not in {"cosine", "l2"}:
            raise ValueError("metric must be 'cosine' or 'l2'")


class LatentScorer:
    """Encodes grids with JEPA encoder and provides distance computations."""

    def __init__(
        self,
        encoder: ObjectCentricJEPAEncoder,
        *,
        projection_head: Optional[ProjectionHead] = None,
        device: str | torch.device | None = None,
        detach: bool = True,
    ) -> None:
        _ensure_torch()
        self.encoder = encoder
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.projection_head = projection_head
        if self.projection_head is not None:
            self.projection_head.to(self.device)

        self.detach = detach

    def embed(self, grid: Grid) -> torch.Tensor:
        batch = self.encoder.encode([grid], device=self.device)
        embedding = batch.embeddings
        mask = batch.mask

        pooled = (embedding * mask.unsqueeze(-1)).sum(dim=1) / torch.clamp(mask.sum(dim=1, keepdim=True), min=1.0)
        pooled = pooled.squeeze(0)

        if self.projection_head is not None:
            pooled = self.projection_head(pooled)
        else:
            pooled = F.normalize(pooled, dim=-1)

        if self.detach:
            pooled = pooled.detach()
        return pooled

    def distance(self, a: torch.Tensor, b: torch.Tensor, metric: str = "cosine") -> torch.Tensor:
        if metric == "cosine":
            a_norm = F.normalize(a.unsqueeze(0), dim=-1)
            b_norm = F.normalize(b.unsqueeze(0), dim=-1)
            return 1.0 - torch.sum(a_norm * b_norm, dim=-1)
        if metric == "l2":
            return torch.norm(a - b, p=2, dim=-1, keepdim=False)
        raise ValueError(f"unsupported metric '{metric}'")


class ArcLatentOptionEnv:
    """Object-centric ARC environment with discrete options and latent shaping."""

    metadata = {"render.modes": ["ansi"]}

    def __init__(
        self,
        options: Sequence[Option],
        scorer: LatentScorer,
        *,
        reward_config: RewardConfig | Mapping[str, object],
        max_steps: int = 8,
        terminate_on_exact_match: bool = True,
    ) -> None:
        if not options:
            raise ValueError("options must contain at least one entry")
        self.options = tuple(options)
        self.scorer = scorer
        self.max_steps = max_steps
        self.terminate_on_exact_match = terminate_on_exact_match

        if isinstance(reward_config, RewardConfig):
            reward_cfg = reward_config
        else:
            reward_cfg = RewardConfig(**reward_config)
        reward_cfg.validate()
        self.reward_cfg = reward_cfg

        self._target_embedding: Optional[torch.Tensor] = None
        self._current_embedding: Optional[torch.Tensor] = None
        self._target_grid: Optional[Grid] = None
        self._state: Optional[Grid] = None
        self._steps = 0

    def seed(self, seed: int | None = None) -> None:  # pragma: no cover - deterministic RNG handled elsewhere
        pass

    def reset(self, *, task: Tuple[Grid, Grid]) -> Grid:
        """Reset environment with (initial_grid, target_grid)."""

        _ensure_torch()

        start, target = task
        if not isinstance(start, Grid) or not isinstance(target, Grid):
            raise TypeError("task must be tuple of Grid objects")

        self._state = start
        self._target_grid = target
        self._target_embedding = self.scorer.embed(target)
        self._current_embedding = self.scorer.embed(start)
        self._steps = 0
        return self._state

    def step(self, action_index: int) -> Tuple[Grid, float, bool, dict]:
        if self._state is None or self._target_grid is None or self._target_embedding is None:
            raise RuntimeError("environment must be reset before stepping")

        if not 0 <= action_index < len(self.options):
            raise IndexError("action index out of range")

        option = self.options[action_index]
        self._steps += 1
        info: dict[str, object] = {"option": option.name}

        prev_grid = self._state
        try:
            next_grid = option.apply(prev_grid)
            info["applied"] = True
        except Exception as exc:
            next_grid = prev_grid
            info["applied"] = False
            info["error"] = str(exc)

        self._state = next_grid
        new_embedding = self.scorer.embed(next_grid)

        prev_distance = self.scorer.distance(self._current_embedding, self._target_embedding, metric=self.reward_cfg.metric)
        current_distance = self.scorer.distance(new_embedding, self._target_embedding, metric=self.reward_cfg.metric)

        progress = (prev_distance - current_distance) * self.reward_cfg.distance_scale
        reward = float(progress.item())

        if not info.get("applied", True):
            reward -= self.reward_cfg.invalid_penalty

        reward -= self.reward_cfg.step_penalty

        done = False
        success = False

        if current_distance.item() <= self.reward_cfg.success_threshold:
            reward += self.reward_cfg.success_bonus
            done = True
            success = True

        if self.terminate_on_exact_match and self._state.cells == self._target_grid.cells:
            if not done:
                reward += self.reward_cfg.success_bonus
            done = True
            success = True

        if self._steps >= self.max_steps:
            done = True

        self._current_embedding = new_embedding.detach()
        info.update(
            {
                "prev_distance": float(prev_distance.item()),
                "current_distance": float(current_distance.item()),
                "success": success,
            }
        )
        return self._state, reward, done, info

    @property
    def action_space_n(self) -> int:
        return len(self.options)

    @property
    def steps(self) -> int:
        return self._steps

    def render(self) -> str:
        if self._state is None:
            return "Environment not initialised"
        lines = ["Current grid:"]
        for row in self._state.cells:
            lines.append(" ".join(str(value) for value in row))
        return "\n".join(lines)


class HierarchicalArcOptionEnv:
    """Wrap :class:`ArcLatentOptionEnv` with a manager termination action."""

    def __init__(
        self,
        options: Sequence[Option],
        scorer: LatentScorer,
        *,
        reward_config: RewardConfig | Mapping[str, object],
        max_steps: int = 8,
        terminate_on_exact_match: bool = True,
    ) -> None:
        self._base = ArcLatentOptionEnv(
            options=options,
            scorer=scorer,
            reward_config=reward_config,
            max_steps=max_steps,
            terminate_on_exact_match=terminate_on_exact_match,
        )
        self._terminate_index = len(self._base.options)

    def reset(self, *, task: Tuple[Grid, Grid]) -> Grid:
        return self._base.reset(task=task)

    def step(self, manager_action: int) -> Tuple[Grid, float, bool, dict]:
        if manager_action < 0 or manager_action > self._terminate_index:
            raise IndexError("manager action index out of range")
        if manager_action == self._terminate_index:
            if self._base._state is None or self._base._target_grid is None or self._base._target_embedding is None:
                raise RuntimeError("environment must be reset before stepping")
            reward = -self._base.reward_cfg.step_penalty
            current_distance = self._base.scorer.distance(
                self._base._current_embedding,
                self._base._target_embedding,
                metric=self._base.reward_cfg.metric,
            )
            success = bool(current_distance.item() <= self._base.reward_cfg.success_threshold)
            if self._base._state.cells == self._base._target_grid.cells:
                success = True
            if success:
                reward += self._base.reward_cfg.success_bonus
            info = {
                "terminated": True,
                "success": success,
                "current_distance": float(current_distance.item()),
            }
            return self._base._state, float(reward), True, info

        grid, reward, done, info = self._base.step(manager_action)
        info["terminated"] = done
        info["manager_action"] = manager_action
        return grid, reward, done, info

    @property
    def action_space_n(self) -> int:
        return len(self._base.options) + 1
