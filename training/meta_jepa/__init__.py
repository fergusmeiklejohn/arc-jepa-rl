"""Meta-JEPA rule-family training utilities."""

from .data import (
    PrimitiveVocabulary,
    RuleFamilyDataset,
    RuleFamilyExample,
    build_rule_family_examples,
    build_rule_family_dataset,
)
from .model import MetaJEPAModel, contrastive_loss
from .trainer import MetaJEPATrainer, TrainingConfig, TrainingResult
from .prior import MetaJEPAPrior

__all__ = [
    "PrimitiveVocabulary",
    "RuleFamilyDataset",
    "RuleFamilyExample",
    "build_rule_family_examples",
    "build_rule_family_dataset",
    "MetaJEPAModel",
    "contrastive_loss",
    "MetaJEPATrainer",
    "TrainingConfig",
    "TrainingResult",
    "MetaJEPAPrior",
]
