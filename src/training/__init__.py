"""Training utilities: Lightning module, config, losses, augmentations.

Exports are resolved lazily so importing a training submodule does not import
every training module. This avoids a cycle when decoder modules import
``src.training.diffusion_modules`` while ``src.pose_estimator`` is loading.
"""

from importlib import import_module


_EXPORTS = {
    "AugmentationConfig": ("src.training.training_module", "AugmentationConfig"),
    "BackboneProbeConfig": ("src.training.training_config", "BackboneProbeConfig"),
    "BackboneProbeModule": ("src.training.backbone_probe", "BackboneProbeModule"),
    "BaseTrainingModule": ("src.training.base_module", "BaseTrainingModule"),
    "BoxSetDiffusionModule": ("src.training.box_diffusion_module", "BoxSetDiffusionModule"),
    "BoxSetDiffusionModuleConfig": (
        "src.training.box_diffusion_module",
        "BoxSetDiffusionModuleConfig",
    ),
    "DiffusionTrainingModuleConfig": (
        "src.training.training_config",
        "DiffusionTrainingModuleConfig",
    ),
    "GraphDiffusionRefinerModule": (
        "src.training.training_module",
        "GraphDiffusionRefinerModule",
    ),
    "GraphDiffusionRefinerModuleConfig": (
        "src.training.training_config",
        "GraphDiffusionRefinerModuleConfig",
    ),
    "GraphDiffusionOnestageModule": (
        "src.training.graph_pipeline_module",
        "GraphDiffusionOnestageModule",
    ),
    "GraphDiffusionOnestageModuleConfig": (
        "src.training.graph_pipeline_module",
        "GraphDiffusionOnestageModuleConfig",
    ),
    "LossConfig": ("src.training.training_module", "LossConfig"),
    "PoseEstimatorTrainingModule": (
        "src.training.training_module",
        "PoseEstimatorTrainingModule",
    ),
    "ConvProbe": ("src.training.backbone_probe", "ConvProbe"),
    "SetBoxDiffusionLoss": ("src.training.losses", "SetBoxDiffusionLoss"),
    "SetBoxDiffusionLossWeights": (
        "src.training.losses",
        "SetBoxDiffusionLossWeights",
    ),
    "SetBoxDiffusionLossWeightsConfig": (
        "src.training.box_diffusion_module",
        "SetBoxDiffusionLossWeightsConfig",
    ),
    "SetPredictionLoss": ("src.training.losses", "SetPredictionLoss"),
    "SetPredictionLossWeights": ("src.training.losses", "SetPredictionLossWeights"),
    "SetPredictionLossWeightsConfig": (
        "src.training.training_config",
        "SetPredictionLossWeightsConfig",
    ),
    "SetPredictionTrainingModule": (
        "src.training.set_prediction_module",
        "SetPredictionTrainingModule",
    ),
    "SetPredictionTrainingModuleConfig": (
        "src.training.training_config",
        "SetPredictionTrainingModuleConfig",
    ),
    "StepOutputs": ("src.training.training_module", "StepOutputs"),
    "TrainingModuleConfig": ("src.training.training_config", "TrainingModuleConfig"),
    "box_cxcywh_to_xyxy": ("src.training.losses", "box_cxcywh_to_xyxy"),
    "box_matching_cost": ("src.training.losses", "box_matching_cost"),
    "compute_bone_loss": ("src.training.losses", "compute_bone_loss"),
    "compute_joint_loss": ("src.training.losses", "compute_joint_loss"),
    "compute_oks": ("src.training.losses", "compute_oks"),
    "compute_probe_diagnostics": ("src.training.diagnostics", "compute_probe_diagnostics"),
    "hungarian_box_match": ("src.training.losses", "hungarian_box_match"),
    "hungarian_match": ("src.training.losses", "hungarian_match"),
    "image_dropout": ("src.training.augmentations", "image_dropout"),
    "joint_dropout": ("src.training.augmentations", "joint_dropout"),
    "joint_randomize": ("src.training.augmentations", "joint_randomize"),
    "keypoint_jitter": ("src.training.augmentations", "keypoint_jitter"),
    "pairwise_giou": ("src.training.losses", "pairwise_giou"),
    "pose_dropout": ("src.training.augmentations", "pose_dropout"),
    "random_hflip_image_and_boxes": (
        "src.training.augmentations",
        "random_hflip_image_and_boxes",
    ),
}


def __getattr__(name: str):
    """Load a public training symbol only when it is requested."""
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    attribute = getattr(import_module(module_name), attribute_name)
    globals()[name] = attribute
    return attribute

__all__ = [
    "AugmentationConfig",
    "BackboneProbeConfig",
    "BackboneProbeModule",
    "BaseTrainingModule",
    "BoxSetDiffusionModule",
    "BoxSetDiffusionModuleConfig",
    "DiffusionTrainingModuleConfig",
    "GraphDiffusionOnestageModule",
    "GraphDiffusionOnestageModuleConfig",
    "GraphDiffusionRefinerModule",
    "GraphDiffusionRefinerModuleConfig",
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
