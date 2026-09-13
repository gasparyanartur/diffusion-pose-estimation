"""Native DiffusionRegPose model components."""

from src.models.pipelines.pose_estimators.diffusionregpose.backbone import (
    DiffusionRegPoseResNetBackbone,
    FrozenBatchNorm2d,
    PositionEmbeddingSineHW,
)
from src.models.pipelines.pose_estimators.diffusionregpose.diffusion import (
    DiffusionRegPoseDiffusion,
    cosine_beta_schedule,
    extract,
)
from src.models.pipelines.pose_estimators.diffusionregpose.evaluation import (
    DiffusionRegPoseAPMetric,
    DiffusionRegPoseBBoxAPMetric,
    box_cxcywh_to_xyxy,
    postprocess_crowdpose,
)
from src.models.pipelines.pose_estimators.diffusionregpose.model import (
    DiffusionRegPoseModel,
    TransformerInputs,
    gen_encoder_output_proposals,
)
from src.models.pipelines.pose_estimators.diffusionregpose.training import (
    CROWDPOSE_PERSON_LABEL,
    DiffusionRegPoseCriterion,
    DiffusionRegPoseLossWeights,
    DiffusionRegPoseLossWeightsConfig,
    DiffusionRegPoseMatcher,
    DiffusionRegPoseOTAMatcher,
    DiffusionRegPosePipeline,
    DiffusionRegPosePipelineConfig,
    build_diffusionregpose_targets,
    diffusionregpose_ap_epoch_active,
    remap_diffusionregpose_checkpoint_layers,
    update_diffusionregpose_ap_metric,
    update_diffusionregpose_bbox_ap_metric,
)
from src.models.pipelines.pose_estimators.diffusionregpose.transformer import (
    DeformableTransformerDecoder,
    DeformableTransformerDecoderLayer,
    DeformableTransformerEncoder,
    DeformableTransformerEncoderLayer,
    DiffusionRegPosePredictionHeads,
    DiffusionRegPoseTimeConditioning,
    MultiScaleDeformableAttention,
    build_input_projections,
)

__all__ = [
    "DiffusionRegPoseDiffusion",
    "DiffusionRegPoseResNetBackbone",
    "FrozenBatchNorm2d",
    "cosine_beta_schedule",
    "extract",
    "box_cxcywh_to_xyxy",
    "postprocess_crowdpose",
    "DiffusionRegPoseAPMetric",
    "DiffusionRegPoseBBoxAPMetric",
    "DeformableTransformerDecoder",
    "DeformableTransformerDecoderLayer",
    "DeformableTransformerEncoder",
    "DeformableTransformerEncoderLayer",
    "DiffusionRegPosePredictionHeads",
    "DiffusionRegPoseTimeConditioning",
    "MultiScaleDeformableAttention",
    "PositionEmbeddingSineHW",
    "build_input_projections",
    "DiffusionRegPoseModel",
    "TransformerInputs",
    "gen_encoder_output_proposals",
    "DiffusionRegPoseCriterion",
    "DiffusionRegPoseLossWeights",
    "DiffusionRegPoseLossWeightsConfig",
    "DiffusionRegPoseMatcher",
    "DiffusionRegPoseOTAMatcher",
    "DiffusionRegPosePipeline",
    "DiffusionRegPosePipelineConfig",
    "build_diffusionregpose_targets",
    "diffusionregpose_ap_epoch_active",
    "remap_diffusionregpose_checkpoint_layers",
    "update_diffusionregpose_ap_metric",
    "update_diffusionregpose_bbox_ap_metric",
    "CROWDPOSE_PERSON_LABEL",
]
