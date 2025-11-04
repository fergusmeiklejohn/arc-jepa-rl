"""High-level training scaffolding for object-centric JEPA."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from arcgen import Grid

from .object_pipeline import (
    EncodedPairBatch,
    ObjectCentricJEPAEncoder,
    ObjectEncoderConfig,
    build_object_encoder,
    build_object_tokenizer_config,
    encode_context_target,
)

try:  # pragma: no cover - optional dependency
    import yaml
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


class ObjectCentricJEPATrainer:
    """Minimal trainer wiring object tokenizer and encoder from config."""

    def __init__(self, config: Mapping[str, object]) -> None:
        self._config = config
        self.tokenizer_config = build_object_tokenizer_config(config.get("tokenizer"))
        self.encoder_config = ObjectEncoderConfig.from_mapping(config.get("object_encoder"))
        self.encoder = build_object_encoder(self.tokenizer_config, self.encoder_config)
        self.object_encoder = ObjectCentricJEPAEncoder(self.encoder, self.tokenizer_config)

    @property
    def raw_config(self) -> Mapping[str, object]:
        return self._config

    def encode_batch(
        self,
        context_grids: Sequence[Grid],
        target_grids: Sequence[Grid],
        *,
        device=None,
    ) -> EncodedPairBatch:
        return encode_context_target(self.object_encoder, context_grids, target_grids, device=device)


def load_jepa_config(path: str | Path) -> Mapping[str, object]:
    if yaml is None:  # pragma: no cover
        raise RuntimeError("PyYAML is required to load JEPA configs")

    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, Mapping):
        raise ValueError("JEPA config must be a mapping")
    return data


def build_trainer_from_config(config: Mapping[str, object]) -> ObjectCentricJEPATrainer:
    return ObjectCentricJEPATrainer(config)


def build_trainer_from_file(path: str | Path) -> ObjectCentricJEPATrainer:
    config = load_jepa_config(path)
    return build_trainer_from_config(config)
