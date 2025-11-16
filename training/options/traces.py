"""Helpers for converting RL trace logs into option discovery episodes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence

from arcgen import Grid
from envs import Option

from .discovery import OptionApplication, OptionEpisode


def load_option_episodes_from_traces(
    trace_path: str | Path,
    options: Sequence[Option],
    *,
    max_steps: int | None = None,
) -> List[OptionEpisode]:
    """Parse a JSON/JSONL trace file into :class:`OptionEpisode` objects.

    Each trace entry is expected to contain a ``steps`` list mirroring the
    structure produced by :mod:`scripts/rollout_latent_env`. The loader accepts
    both JSONL (one entry per line) and JSON arrays of entries.
    """

    entries = list(_iter_trace_entries(Path(trace_path)))
    if not entries:
        raise ValueError(f"No episodes found in {trace_path}")

    lookup = {option.name: option for option in options}
    episodes: List[OptionEpisode] = []

    for entry in entries:
        steps = entry.get("steps")
        if not isinstance(steps, Sequence):
            raise ValueError("each trace entry must contain a 'steps' list")

        applications: List[OptionApplication] = []
        for idx, step in enumerate(steps):
            if max_steps is not None and idx >= max_steps:
                break

            option = _resolve_option(step, lookup, options)
            before_grid = _extract_grid(step, "grid_before")
            if before_grid is None:
                before_grid = _grid_from_observation(step.get("observation"))
            if before_grid is None:
                raise ValueError("step missing grid_before/observation data")

            after_grid = _extract_grid(step, "grid_after")
            if after_grid is None:
                after_grid = _apply_option_safe(option, before_grid)

            reward = float(step.get("reward", 0.0))
            success_field = step.get("success")
            success = None if success_field is None else bool(success_field)

            applications.append(
                OptionApplication(
                    option=option,
                    before=before_grid,
                    after=after_grid,
                    reward=reward,
                    success=success,
                )
            )

        if applications:
            episodes.append(OptionEpisode(steps=tuple(applications), task_id=entry.get("task_id")))

    if not episodes:
        raise ValueError(f"No usable option steps recovered from {trace_path}")
    return episodes


def _iter_trace_entries(path: Path) -> Iterable[Mapping[str, object]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, Mapping):
        return [parsed]

    entries: List[Mapping[str, object]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        entries.append(json.loads(stripped))
    return entries


def _resolve_option(step: Mapping[str, object], lookup: Mapping[str, Option], options: Sequence[Option]) -> Option:
    name = step.get("option_name") or step.get("option")
    if isinstance(name, str) and name in lookup:
        return lookup[name]

    action = step.get("action")
    if action is not None:
        idx = int(action)
        if 0 <= idx < len(options):
            option = options[idx]
            lookup.setdefault(option.name, option)
            return option
    raise KeyError(f"unable to resolve option for step: {step}")


def _extract_grid(step: Mapping[str, object], key: str) -> Grid | None:
    raw = step.get(key)
    if raw is None:
        return None
    return _grid_from_data(raw, key)


def _grid_from_observation(obs: object) -> Grid | None:
    if not isinstance(obs, Mapping):
        return None
    current = obs.get("current")
    if current is None:
        return None
    return _grid_from_data(current, "observation.current")


def _grid_from_data(data: object, context: str) -> Grid:
    try:
        return Grid(data)  # type: ignore[arg-type]
    except Exception as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"invalid grid structure in {context}") from exc


def _apply_option_safe(option: Option, grid: Grid) -> Grid:
    try:
        return option.apply(grid)
    except Exception:
        return grid
