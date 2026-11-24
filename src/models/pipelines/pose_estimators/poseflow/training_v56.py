import hashlib
from pathlib import Path

from ..diffusionregpose.training import DiffusionRegPosePipelineConfig
from .model_v56 import PoseFlowV56Model
from .training_v55 import PoseFlowV55Pipeline, PoseFlowV55PipelineConfig


EDPOSE_SHA256 = '0f50dae126a73ad2a15c41906c9ba6d8efa3421ef6cc2fbdc878238bd26b8f93'


class PoseFlowV56PipelineConfig(PoseFlowV55PipelineConfig):
    def native_config(self):
        defaults = dict(backbone_pretrained=False, num_encoder_layers=6, num_decoder_layers=6,
            num_box_decoder_layers=2, num_queries=200, num_proposals=50, num_group=50, dn_number=50,
            pretrained_minimum_critical_coverage=.99, freeze_backbone_stem_and_layer1=True)
        config = DiffusionRegPosePipelineConfig(**dict(defaults, **self.native,
            pretrained_checkpoint=self.pretrained_checkpoint))
        if ((config.num_encoder_layers, config.num_decoder_layers, config.num_box_decoder_layers) != (6, 6, 2)
                or config.pretrained_keep_encoder_layers is not None or config.pretrained_keep_decoder_layers is not None):
            raise ValueError('V56 requires full native depth without sparse layer remapping')
        return config

    def create_model(self, native=None):
        if native is None:
            native = self.native_config().build(max_epochs=80, load_pretrained_checkpoint=False).model
        return PoseFlowV56Model(native=native, capacity=self.capacity,
            activation_checkpointing=self.activation_checkpointing)

    def build(self, *, max_epochs=80, load_pretrained_checkpoint=True, **kwargs):
        native = None
        if load_pretrained_checkpoint and self.pretrained_checkpoint:
            if hashlib.sha256(Path(self.pretrained_checkpoint).read_bytes()).hexdigest() != EDPOSE_SHA256:
                raise ValueError('V56 original ED-Pose asset hash mismatch')
            native = self.native_config().build(max_epochs=max_epochs).model
        model = self.create_model(native)
        model.initialization_report = dict(source=self.pretrained_checkpoint if native is not None else None,
            source_sha256=EDPOSE_SHA256 if native is not None else None,
            frontend='original_EDPose_full_depth', encoder_source_layers=list(range(1, 7)),
            decoder_source_layers=list(range(1, 7)), box_source_layers=[1, 2], pose_source_layers=[3, 4, 5, 6],
            expansion_head='tied_to_first_pose_head', prefix_correction='zero_output_projection',
            decoder='verified_full_depth_native_birth;no_direct_prefix_tokens;no_refinement')
        return PoseFlowV56Pipeline(config=self, model=model, max_epochs=max_epochs)


class PoseFlowV56Pipeline(PoseFlowV55Pipeline):
    def __init__(self, *, config, model=None, max_epochs=80):
        config = config if isinstance(config, PoseFlowV56PipelineConfig) else PoseFlowV56PipelineConfig(**config)
        super().__init__(config=config, model=model, max_epochs=max_epochs)
        self.model.architecture = 'V56'

    def checkpoint_contract(self):
        contract = super().checkpoint_contract()
        contract.update(version=1, architecture='V56',
            decoder='full_native_pose3_4_5_6;cached_image_values;independent_groups;no_direct_prefix_tokens',
            initialization='SHA_pinned_original_EDPose_all6encoder_2box_4pose;tied_expansion_first_pose_head;fresh_selector_prefix',
            pretrained_sha256=EDPOSE_SHA256,
            parameter_sharing='expansion_head_is_pose_head0;original_native_extent_head_sharing')
        contract['objectives']['normalization'] = 'all_four_pose_layers_then_people;equal_normalized_pose_box;balanced_per_image_person_and_firstEOS'
        contract['proposal_context']['detach'] = 'raw_candidates_and_image_identity_detached;prefix_encoder_trainable;corrected_output_differentiable'
        contract['gradients']['routes'] = 'classification_and_geometry_train_context_and_prefix_encoder;native_feature_residual_gradient_preserved'
        return contract

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        for key in ('contract', 'initialization', 'calibration', 'global_rng', 'prefix_bank'):
            checkpoint[f'v56_{key}'] = checkpoint.pop(f'v55_{key}')

    def on_load_checkpoint(self, checkpoint):
        if checkpoint.get('v56_contract') != self.checkpoint_contract():
            raise ValueError('Checkpoint does not satisfy V56 contract')
        translated = dict(checkpoint)
        for key in ('contract', 'initialization', 'calibration', 'global_rng', 'prefix_bank'):
            translated[f'v55_{key}'] = checkpoint.get(f'v56_{key}')
        super().on_load_checkpoint(translated)
