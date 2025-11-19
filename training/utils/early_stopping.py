"""Generic early-stopping utilities for training loops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class EarlyStoppingConfig:
    """Configuration bundle for early-stopping helpers."""

    enabled: bool = False
    patience: int = 5
    min_delta: float = 0.0
    mode: str = "min"

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None) -> "EarlyStoppingConfig":
        if data is None:
            return cls()
        mode_value = str(data.get("mode", cls.mode)).lower()
        if mode_value not in {"min", "max"}:
            raise ValueError("early_stopping.mode must be 'min' or 'max'")
        patience_value = int(data.get("patience", cls.patience))
        if patience_value <= 0:
            raise ValueError("early_stopping.patience must be positive")
        min_delta_value = float(data.get("min_delta", cls.min_delta))
        if min_delta_value < 0:
            raise ValueError("early_stopping.min_delta must be non-negative")
        return cls(
            enabled=bool(data.get("enabled", cls.enabled)),
            patience=patience_value,
            min_delta=min_delta_value,
            mode=mode_value,
        )


class EarlyStopping:
    """Track a metric and determine when to stop training early."""

    def __init__(self, config: EarlyStoppingConfig) -> None:
        self.config = config
        self.best_metric: float | None = None
        self.best_step: int | None = None
        self.stopped_step: int | None = None
        self._wait = 0

    def update(self, metric: float, *, step: int | None = None) -> bool:
        """Record a metric value and return True when patience is exceeded."""

        improved = False
        if self.best_metric is None:
            improved = True
        else:
            delta = metric - self.best_metric
            if self.config.mode == "min":
                improved = delta < -self.config.min_delta
            else:
                improved = delta > self.config.min_delta

        if improved:
            self.best_metric = metric
            self.best_step = step
            self._wait = 0
            return False

        self._wait += 1
        if not self.config.enabled:
            return False
        if self._wait >= self.config.patience:
            self.stopped_step = step
            return True
        return False

    def reset(self) -> None:
        self.best_metric = None
        self.best_step = None
        self.stopped_step = None
        self._wait = 0
