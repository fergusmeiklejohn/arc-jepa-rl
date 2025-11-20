"""Optimization helpers shared across training loops."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

try:  # pragma: no cover - torch is optional at import time
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


@dataclass(frozen=True)
class GradientClippingConfig:
    """Configuration for gradient clipping."""

    max_norm: float = 0.0
    norm_type: float = 2.0

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None) -> "GradientClippingConfig":
        if data is None:
            return cls()
        max_norm_value = float(data.get("max_norm", cls.max_norm))
        if max_norm_value < 0:
            raise ValueError("grad_clip.max_norm must be non-negative")
        norm_type_value = float(data.get("norm_type", cls.norm_type))
        if norm_type_value <= 0:
            raise ValueError("grad_clip.norm_type must be positive")
        return cls(max_norm=max_norm_value, norm_type=norm_type_value)

    @property
    def enabled(self) -> bool:
        return self.max_norm > 0


@dataclass(frozen=True)
class LRSchedulerConfig:
    """Simple scheduler configuration supporting warmup + cosine/linear decay."""

    name: str = "none"
    warmup_steps: int = 0
    min_lr_scale: float = 0.0
    total_steps: int | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None) -> "LRSchedulerConfig":
        if data is None:
            return cls()
        name = str(data.get("name", cls.name)).lower()
        warmup_steps = int(data.get("warmup_steps", cls.warmup_steps))
        if warmup_steps < 0:
            raise ValueError("lr_scheduler.warmup_steps must be non-negative")
        min_lr_scale = float(data.get("min_lr_scale", cls.min_lr_scale))
        if not 0.0 <= min_lr_scale <= 1.0:
            raise ValueError("lr_scheduler.min_lr_scale must be in [0, 1]")
        total_steps = data.get("total_steps", cls.total_steps)
        resolved_steps = None
        if total_steps is not None:
            resolved_steps = int(total_steps)
            if resolved_steps <= 0:
                raise ValueError("lr_scheduler.total_steps must be positive when provided")
        return cls(
            name=name,
            warmup_steps=warmup_steps,
            min_lr_scale=min_lr_scale,
            total_steps=resolved_steps,
        )

    @property
    def enabled(self) -> bool:
        return self.name not in {"", "none", None}


@dataclass(frozen=True)
class MixedPrecisionConfig:
    """Simple mixed-precision selector."""

    mode: str = "none"  # one of: none, fp16, bf16

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, object] | str | None,
        *,
        legacy_amp: bool | None = None,
    ) -> "MixedPrecisionConfig":
        if isinstance(data, Mapping):
            raw_mode = data.get("mode", cls.mode)
        else:
            raw_mode = data if data is not None else cls.mode
        if isinstance(raw_mode, bool):
            mode = "fp16" if raw_mode else "none"
        else:
            mode = str(raw_mode).lower().strip()
        aliases = {
            "float16": "fp16",
            "half": "fp16",
            "amp": "fp16",
            "bfloat16": "bf16",
        }
        mode = aliases.get(mode, mode)
        if legacy_amp and mode == cls.mode:
            mode = "fp16"
        if mode not in {"none", "fp16", "bf16"}:
            raise ValueError("mixed_precision mode must be one of: none, fp16, bf16")
        return cls(mode=mode)

    @property
    def enabled(self) -> bool:
        return self.mode != "none"

    @property
    def torch_dtype(self):
        if torch is None:  # pragma: no cover - torch optional
            return None
        if self.mode == "fp16":
            return torch.float16
        if self.mode == "bf16":
            return torch.bfloat16
        return None

    @property
    def use_grad_scaler(self) -> bool:
        return self.mode == "fp16"

    def supported_on_device(self, device_type: str) -> bool:
        if torch is None:  # pragma: no cover - torch optional
            return False
        if device_type != "cuda":
            return False
        if not torch.cuda.is_available():
            return False
        if self.mode == "bf16":
            checker = getattr(torch.cuda, "is_bf16_supported", None)
            if checker is not None:
                return bool(checker())
        return True


def _resolve_total_steps(config: LRSchedulerConfig, total_steps: int | None) -> int:
    steps = total_steps or config.total_steps
    if steps is None:
        raise ValueError("total_steps must be provided when lr_scheduler.name is set")
    if steps <= 0:
        raise ValueError("total_steps must be positive")
    return steps


def build_lr_scheduler(
    optimizer,
    config: LRSchedulerConfig,
    *,
    total_steps: int | None = None,
):
    """Construct a learning-rate scheduler or return None when disabled."""
    if not config.enabled:
        return None
    if torch is None:  # pragma: no cover - defensive
        raise RuntimeError("PyTorch is required to build LR schedulers")

    steps = _resolve_total_steps(config, total_steps)
    warmup = max(0, config.warmup_steps)
    min_scale = max(0.0, min(1.0, config.min_lr_scale))
    name = config.name.lower()

    def _warmup_scale(step: int) -> float:
        if warmup == 0:
            return 1.0
        return max(1e-8, (step + 1) / float(warmup))

    if name == "cosine":

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return _warmup_scale(step)
            progress = (step - warmup) / max(1, steps - warmup)
            decay = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
            return max(min_scale, float(decay))

    elif name == "linear":

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return _warmup_scale(step)
            progress = (step - warmup) / max(1, steps - warmup)
            decay = 1.0 - min(1.0, progress)
            return max(min_scale, float(decay))

    else:
        raise ValueError(f"Unsupported lr_scheduler.name: {config.name}")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
