"""Utilities for deterministic experiment reproducibility."""

from __future__ import annotations

import importlib
import os
import random
from typing import Any


def _seed_torch(seed: int, deterministic_cudnn: bool) -> None:
    """Seed PyTorch if it is installed.

    Any ImportError or AttributeError is swallowed so the utility works even
    when torch is not present (e.g., during lightweight unit tests).
    """

    if importlib.util.find_spec("torch") is None:
        return

    try:
        torch = importlib.import_module("torch")
    except Exception:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    try:
        torch.use_deterministic_algorithms(deterministic_cudnn)
    except (AttributeError, RuntimeError):
        # Fall back to cuDNN flags if deterministic algorithms helper missing.
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = deterministic_cudnn
            torch.backends.cudnn.benchmark = not deterministic_cudnn


def _seed_optional_backend(module_name: str, seed: int) -> None:
    """Attempt to seed an optional backend that exposes a ``seed`` function."""

    if importlib.util.find_spec(module_name) is None:
        return

    try:
        module = importlib.import_module(module_name)
    except Exception:
        return

    seed_fn: Any = getattr(module, "seed", None)
    if callable(seed_fn):
        try:
            seed_fn(seed)
        except Exception:
            # Intentionally swallow errors from optional libraries.
            pass


def set_global_seeds(
    seed: int,
    *,
    deterministic_cudnn: bool = True,
    enable_numpy: bool = True,
) -> None:
    """Seed all known random number generators for reproducibility.

    Parameters
    ----------
    seed:
        The seed value to apply globally.
    deterministic_cudnn:
        When ``True`` (default), attempts to turn on deterministic behaviour in
        PyTorch/cuDNN if available.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    if enable_numpy and os.environ.get("ARC_SKIP_NUMPY_SEED") != "1":
        try:
            numpy_module = importlib.import_module("numpy")
        except Exception:
            numpy_module = None

        if numpy_module is not None:
            try:
                numpy_module.random.seed(seed)  # type: ignore[attr-defined]
            except Exception:
                # Some environments ship minimal NumPy builds; swallow runtime errors.
                pass

    _seed_optional_backend("jax", seed)
    _seed_optional_backend("tensorflow", seed)
    _seed_torch(seed, deterministic_cudnn)
