"""JEPA-specific helpers and object-centric encoders."""

from .object_pipeline import (
    EncodedPairBatch,
    ObjectCentricEncoding,
    ObjectCentricJEPAEncoder,
    ObjectEncoderConfig,
    ObjectTokenizerConfig,
    build_object_centric_encoder_from_config,
    build_object_encoder,
    build_object_token_batch,
    build_object_tokenizer_config,
    encode_context_target,
)
from .trainer import (
    ObjectCentricJEPATrainer,
    build_trainer_from_config,
    build_trainer_from_file,
    load_jepa_config,
)
from .loop import (
    ObjectCentricJEPAExperiment,
    OptimizerConfig,
    TrainStepResult,
)
from .relational_loss import RelationalConsistencyConfig
from .invariance import InvarianceLossConfig
from .dataset import (
    AugmentationConfig,
    GridPairBatch,
    InMemoryGridPairDataset,
    ManifestGridPairDataset,
    ManifestTokenizedPairDataset,
    build_dummy_dataset,
)

__all__ = [
    "AugmentationConfig",
    "EncodedPairBatch",
    "ObjectCentricEncoding",
    "ObjectCentricJEPAEncoder",
    "ObjectEncoderConfig",
    "ObjectTokenizerConfig",
    "ObjectCentricJEPATrainer",
    "ObjectCentricJEPAExperiment",
    "OptimizerConfig",
    "TrainStepResult",
    "RelationalConsistencyConfig",
    "GridPairBatch",
    "InMemoryGridPairDataset",
    "ManifestGridPairDataset",
    "ManifestTokenizedPairDataset",
    "build_object_centric_encoder_from_config",
    "build_object_encoder",
    "build_object_token_batch",
    "build_object_tokenizer_config",
    "InvarianceLossConfig",
    "build_trainer_from_config",
    "build_trainer_from_file",
    "build_dummy_dataset",
    "encode_context_target",
    "load_jepa_config",
]
