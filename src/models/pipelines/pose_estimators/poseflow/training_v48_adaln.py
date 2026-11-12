from typing import Literal

from ..diffusionregpose.model import DiffusionRegPoseModel
from .model_v48_adaln import PoseFlowV48AdaLNModel
from .training_v44 import NATIVE_MODEL_FIELDS
from .training_v48 import PoseFlowV48Pipeline, PoseFlowV48PipelineConfig


class PoseFlowV48AdaLNPipelineConfig(PoseFlowV48PipelineConfig):
    variant: Literal['image_cross_attention_generation_step_adaln'] = 'image_cross_attention_generation_step_adaln'

    def create_model(self, native=None):
        config = self.native_config()
        if native is None:
            native = DiffusionRegPoseModel(backbone_pretrained=False,
                separate_expansion_pose_head=config.pretrained_keep_decoder_layers is not None,
                **{name: getattr(config, name) for name in NATIVE_MODEL_FIELDS})
        return PoseFlowV48AdaLNModel(native=native, capacity=self.capacity,
            hidden_dim=config.hidden_dim, num_heads=config.num_heads, feedforward_dim=config.feedforward_dim,
            codec_width=self.codec_width, latent_dim=self.latent_dim,
            activation_checkpointing=self.activation_checkpointing)

    def build(self, *, max_epochs=80, load_pretrained_checkpoint=True, **kwargs):
        base = super().build(max_epochs=max_epochs, load_pretrained_checkpoint=load_pretrained_checkpoint, **kwargs)
        base.model.initialization_report.update(variant=self.variant,
            modulation='zero_scale_shift_gate_deltas;identity_cross_attention;common_V48_weights_and_RNG_preserved')
        return PoseFlowV48AdaLNPipeline(config=self, model=base.model, max_epochs=max_epochs)


class PoseFlowV48AdaLNPipeline(PoseFlowV48Pipeline):
    def __init__(self, *, config, model=None, max_epochs=80):
        specification = config if isinstance(config, PoseFlowV48AdaLNPipelineConfig) else PoseFlowV48AdaLNPipelineConfig(**config)
        super().__init__(config=specification, model=model, max_epochs=max_epochs)

    def checkpoint_contract(self):
        contract = super().checkpoint_contract()
        contract.update(variant=self.config.variant,
            generation_step_conditioning=dict(source='valid_prefix_people_count;no_GT_total_count;not_row_index',
                embedding='learned_shared_0_to_capacity', scope='image_cross_attention_query_LN_and_residual_only',
                modulation='per_layer_SiLU_Linear_width_to_3width;scale_shift_gate_deltas',
                initial_scale=1., initial_shift=0., initial_gate=1.,
                initialization='zero_modulation_linear;baseline_common_weights_and_RNG_exact'))
        return contract
