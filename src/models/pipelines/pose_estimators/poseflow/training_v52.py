from copy import deepcopy
from pathlib import Path

import torch
from pydantic import Field

from .balance_v51 import clip_families, gradient_families
from .calibration_v49_balanced import capture_rng, fixed_scales, restore_rng
from .losses_v49_balanced import TASK_NAMES
from .model_v52 import PoseFlowV52Model
from .training_v50 import PoseFlowV50Pipeline, PoseFlowV50PipelineConfig


class PoseFlowV52PipelineConfig(PoseFlowV50PipelineConfig):
    latent_dim: int = 256
    calibration_seed: int = 42052
    sampling_points: int = Field(default=4, ge=1)

    def create_model(self, native=None):
        from ..diffusionregpose.model import DiffusionRegPoseModel
        from .training_v44 import NATIVE_MODEL_FIELDS
        config = self.native_config()
        if native is None:
            native = DiffusionRegPoseModel(backbone_pretrained=False,
                separate_expansion_pose_head=config.pretrained_keep_decoder_layers is not None,
                **{name: getattr(config, name) for name in NATIVE_MODEL_FIELDS})
        return PoseFlowV52Model(native=native, rows=self.rows, capacity=self.capacity, scale=self.scale,
            hidden_dim=config.hidden_dim, num_heads=config.num_heads, feedforward_dim=config.feedforward_dim,
            codec_width=config.hidden_dim, latent_dim=config.hidden_dim, sampling_points=self.sampling_points,
            activation_checkpointing=self.activation_checkpointing)

    def build(self, *, max_epochs=80, load_pretrained_checkpoint=True, **kwargs):
        base = super().build(max_epochs=max_epochs, load_pretrained_checkpoint=load_pretrained_checkpoint, **kwargs)
        base.model.initialization_report.update(codec='absent;direct_large_features',
            decoder='fresh_two_layer_time_AdaLN_deformable;direct_joint_geometry_heads')
        return PoseFlowV52Pipeline(config=self, model=base.model, max_epochs=max_epochs)


class PoseFlowV52Pipeline(PoseFlowV50Pipeline):
    def __init__(self, *, config, model=None, max_epochs=80):
        specification = config if isinstance(config, PoseFlowV52PipelineConfig) else PoseFlowV52PipelineConfig(**config)
        super().__init__(config=specification, model=model, max_epochs=max_epochs)

    def _forward_losses(self, batch, generator, scales):
        losses = super()._forward_losses(batch, generator, scales)
        for index, name in enumerate(TASK_NAMES):
            losses[f'{name}_contribution'] = .25*losses[name]/scales[index]
        losses['total'] = sum(losses[f'{name}_contribution'] for name in TASK_NAMES)
        losses['total'] = losses['total']+losses['eos']+losses['selection']+losses['frontend']
        return losses

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None):
        if gradient_clip_val is not None and abs(float(gradient_clip_val)-.1) > 1e-8:
            raise ValueError('V52 combined clipping contract requires .1')
        values = clip_families(gradient_families(self.model))
        if self._trainer is not None:
            self.log_dict({f'train/gradients/{name}': value for name, value in values.items()}, on_step=True, on_epoch=False)
        return values

    def log_detection_report(self, report, *, step=None):
        from torch.utils.tensorboard import SummaryWriter
        from .quick_eval_v52 import tensorboard_metrics
        if step is None:
            self.log_dict(tensorboard_metrics(report))
        else:
            for logger in self.trainer.loggers:
                logger.log_metrics(tensorboard_metrics(report), step=step)
                if hasattr(logger.experiment, 'flush'):
                    logger.experiment.flush()
        callback = self.trainer.checkpoint_callback
        root = Path(callback.dirpath).parent if callback else Path(self.trainer.default_root_dir)
        for label in ('birth', 'first_eos'):
            writer = SummaryWriter(str(root/f'{label}_tb_logs'))
            try:
                for name, value in report[f'{label}_metrics'].items():
                    writer.add_scalar(f'quick-test/{name}', value, self.global_step if step is None else step)
                writer.flush()
            finally:
                writer.close()

    def on_validation_epoch_end(self):
        if not self.config.quick_eval_enabled or self.trainer.sanity_checking:
            return
        from .quick_eval_v52 import run_v52_evaluation
        callback = self.trainer.checkpoint_callback
        root = Path(callback.dirpath).parent if callback else Path(self.trainer.default_root_dir)
        report = run_v52_evaluation(self.model, self.trainer.datamodule, root/'quick_test', epoch=self.current_epoch,
            count=self.config.quick_eval_images, seed=self.config.seed)
        self.log_detection_report(report)
        self.log('val/selection_AP', min(report['metrics']['pose_AP'], report['metrics']['bbox_AP']))

    def checkpoint_contract(self):
        contract = super().checkpoint_contract()
        contract.pop('codec', None)
        contract.pop('residual', None)
        contract.update(architecture='V52', variant='joint_clean_large_feature_time_deformable_v1',
            representation='shared_D_person_feature;direct32_to_D_to_D_input;no_coordinate_codec',
            readout='per_layer_LN_D_to_D_GELU_to32;linear_center_relative_joints;sigmoid_center;softplus_wh+1e-6;clean_prediction_all_sigmas',
            encoder_input='normalized_noisy_jointXY_minus_noisy_center;noisy_boxCXCYWH;all_valid_tokens',
            attention=dict(kind='row_independent_self;shared_per_image_projected_values_deformable',
                levels=self.model.num_feature_levels, points=self.model.sampling_points,
                references='prefix_deattenuated_noisy_current_box_wh_clamp1e-4_2;birth_selected_proposal;next_layer_clean_box_detached;valid_ratio_scaled'),
            time_conditioning='per_token_sinusoidal_sigma999_MLP_RMScap1;additive_and_image_query_AdaLN',
            identity='V50_detached_content_and_reference;selector_detached_prefix;keys_trainable',
            supervision='both_layers;per_person_birth_and_refine_half;present_age_buckets_equal;fixed_pose_box_equal',
            calibration=dict(tasks=list(TASK_NAMES), batches=self.config.calibration_batches,
                seed=self.config.calibration_seed, rule='frozen_TRAIN_median_two_layer_decoder_task_L2_floor1e-8',
                coefficients=[.25]*4, adaptive=False),
            initialization='original_EDPose_native_frontend;fresh256D_person_decoder_and_readouts',
            gradient_clipping='after_accumulation;disjoint_native_selector_geometry;each_.1/sqrt3;combined<=.1')
        contract['objectives']['priorities'] = dict(pose_box=[.5, .5], birth_refine=[.5, .5], timestep_dependent=False)
        return contract

    def on_save_checkpoint(self, checkpoint):
        if not self.calibration_ready:
            raise ValueError('Cannot save uncalibrated V52 checkpoint')
        checkpoint.update(v52_contract=self.checkpoint_contract(),
            v52_initialization=self.model.initialization_report, v52_global_rng=capture_rng(),
            v52_calibration=deepcopy(self.calibration_provenance))

    def on_load_checkpoint(self, checkpoint):
        if checkpoint.get('v52_contract') != self.checkpoint_contract():
            raise ValueError('Checkpoint does not satisfy the V52 contract')
        provenance = checkpoint.get('v52_calibration')
        state = checkpoint['state_dict']
        if (not provenance or provenance.get('task_names') != list(TASK_NAMES)
                or len(provenance.get('norms', [])) != self.config.calibration_batches
                or not state.get('calibration_ready', False)
                or int(state.get('calibration_samples', 0)) != self.config.calibration_batches):
            raise ValueError('Invalid V52 calibration provenance')
        expected = fixed_scales(provenance['norms'])
        if not torch.equal(state['gradient_scales'].cpu(), expected) or provenance.get('scales') != expected.tolist():
            raise ValueError('Corrupted V52 fixed calibration scales')
        self.model.initialization_report = checkpoint['v52_initialization']
        self.calibration_provenance = deepcopy(provenance)
        restore_rng(checkpoint['v52_global_rng'])
