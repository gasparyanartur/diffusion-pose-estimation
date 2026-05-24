"""Training utilities: Lightning module, config, losses, augmentations."""

from src.training.augmentations import (
    image_dropout,
    joint_dropout,
    joint_randomize,
    keypoint_jitter,
    pose_dropout,
    random_hflip_image_and_boxes,
)
from src.training.backbone_probe import BackboneProbeModule, ConvProbe
from src.training.base_module import BaseTrainingModule
from src.training.box_diffusion_module import (
    BoxSetDiffusionModule,
    BoxSetDiffusionModuleConfig,
    SetBoxDiffusionLossWeightsConfig,
)
from src.training.diagnostics import compute_probe_diagnostics
from src.training.losses import (
    SetBoxDiffusionLoss,
    SetBoxDiffusionLossWeights,
    SetPredictionLoss,
    SetPredictionLossWeights,
    box_cxcywh_to_xyxy,
    box_matching_cost,
    compute_bone_loss,
    compute_joint_loss,
    compute_oks,
    hungarian_box_match,
    hungarian_match,
    pairwise_giou,
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
    "BaseTrainingModule",
    "BoxSetDiffusionModule",
    "BoxSetDiffusionModuleConfig",
    "DiffusionTrainingModuleConfig",
    "LossConfig",
    "PoseEstimatorTrainingModule",
    "ConvProbe",
    "SetBoxDiffusionLoss",
    "SetBoxDiffusionLossWeights",
    "SetBoxDiffusionLossWeightsConfig",
    "SetPredictionLoss",
    "SetPredictionLossWeights",
    "SetPredictionLossWeightsConfig",
    "SetPredictionTrainingModule",
    "SetPredictionTrainingModuleConfig",
    "StepOutputs",
    "TrainingModuleConfig",
    "box_cxcywh_to_xyxy",
    "box_matching_cost",
    "compute_bone_loss",
    "compute_joint_loss",
    "compute_oks",
    "compute_probe_diagnostics",
    "hungarian_box_match",
    "hungarian_match",
    "image_dropout",
    "joint_dropout",
    "joint_randomize",
    "keypoint_jitter",
    "pairwise_giou",
    "pose_dropout",
    "random_hflip_image_and_boxes",
]
