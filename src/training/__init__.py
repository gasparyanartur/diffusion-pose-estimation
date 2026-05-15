"""Training utilities: Lightning module, config, losses, augmentations."""

from src.training.augmentations import (
    image_dropout,
    joint_dropout,
    joint_randomize,
    keypoint_jitter,
    pose_dropout,
)
from src.training.backbone_probe import BackboneProbeModule, ProbeHead
from src.training.diagnostics import compute_probe_diagnostics
from src.training.losses import (
    SetPredictionLoss,
    SetPredictionLossWeights,
    compute_bone_loss,
    compute_joint_loss,
    compute_oks,
    hungarian_match,
)
from src.training.set_prediction_module import SetPredictionTrainingModule
from src.training.training_config import (
    BackboneProbeConfig,
    DiffusionTrainingModuleConfig,
    SetPredictionLossWeightsConfig,
    SetPredictionTrainingModuleConfig,
    TrainingModuleConfig,
)
from src.training.training_module import (
    AugmentationConfig,
    LossConfig,
    PoseEstimatorTrainingModule,
    StepOutputs,
)

__all__ = [
    "AugmentationConfig",
    "BackboneProbeConfig",
    "BackboneProbeModule",
    "DiffusionTrainingModuleConfig",
    "LossConfig",
    "PoseEstimatorTrainingModule",
    "ProbeHead",
    "SetPredictionLoss",
    "SetPredictionLossWeights",
    "SetPredictionLossWeightsConfig",
    "SetPredictionTrainingModule",
    "SetPredictionTrainingModuleConfig",
    "StepOutputs",
    "TrainingModuleConfig",
    "compute_bone_loss",
    "compute_joint_loss",
    "compute_oks",
    "compute_probe_diagnostics",
    "hungarian_match",
    "image_dropout",
    "joint_dropout",
    "joint_randomize",
    "keypoint_jitter",
    "pose_dropout",
]
