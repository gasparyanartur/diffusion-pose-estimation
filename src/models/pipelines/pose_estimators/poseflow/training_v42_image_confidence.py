from typing import Literal

from .training_v42_paired_proposal import PoseFlowV42PairedProposalPipelineConfig


class PoseFlowV42ImageConfidencePipelineConfig(PoseFlowV42PairedProposalPipelineConfig):
    confidence_mode: Literal['image_only'] = 'image_only'
