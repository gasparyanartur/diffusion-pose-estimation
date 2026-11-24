from copy import deepcopy
from pathlib import Path

import pytorch_lightning as pl
import torch
from pydantic import Field
from torch.nn import functional as F

from .balance_v51 import clip_families, gradient_families
from .calibration_v49_balanced import capture_rng, restore_rng
from .calibration_v53 import fixed_scales
from .codec_v43 import geometry_from_batch
from .coordinates_v47 import ordered_coordinates
from .losses_v53 import TASK_NAMES, row_losses
from .model_v53 import PoseFlowV53Model
from .rows_v53 import training_rows
from .training_v44 import NATIVE_MODEL_FIELDS, PoseFlowV44Pipeline
from .training_v48 import PoseFlowV48Pipeline, PoseFlowV48PipelineConfig
from .training_v50 import PoseFlowV50Pipeline
from ..diffusionregpose.model import DiffusionRegPoseModel
from ..diffusionregpose.training import DiffusionRegPoseCriterion, DiffusionRegPoseMatcher


class PoseFlowV53PipelineConfig(PoseFlowV48PipelineConfig):
    capacity: int = 49
    rows: int = 50
    calibration_batches: int = Field(default=32, ge=1)
    calibration_seed: int = 42053

    def create_model(self, native=None):
        config = self.native_config()
        if native is None:
            native = DiffusionRegPoseModel(backbone_pretrained=False,
                separate_expansion_pose_head=config.pretrained_keep_decoder_layers is not None,
                **{name: getattr(config, name) for name in NATIVE_MODEL_FIELDS})
        return PoseFlowV53Model(native=native, capacity=self.capacity, activation_checkpointing=self.activation_checkpointing)

    def build(self, *, max_epochs=80, load_pretrained_checkpoint=True, **kwargs):
        native = None
        if load_pretrained_checkpoint and self.pretrained_checkpoint:
            native = self.native_config().build(max_epochs=max_epochs).model
        model = self.create_model(native)
        model.initialization_report = dict(source=self.pretrained_checkpoint if native is not None else None,
            frontend='original_EDPose' if native is not None else 'random;initializer_disabled',
            decoder='native_keypoint_expansion_and_pose_heads;no_refinement',
            encoder_source_layers=[2, 4, 6], decoder_source_layers=[2, 4, 6])
        return PoseFlowV53Pipeline(config=self, model=model, max_epochs=max_epochs)


class PoseFlowV53Pipeline(pl.LightningModule):
    configure_optimizers = PoseFlowV50Pipeline.configure_optimizers
    on_before_optimizer_step = PoseFlowV44Pipeline.on_before_optimizer_step
    training_step = PoseFlowV48Pipeline.training_step
    validation_step = PoseFlowV48Pipeline.validation_step
    _frontend_loss = PoseFlowV44Pipeline._frontend_loss

    def __init__(self, *, config, model=None, max_epochs=80):
        super().__init__()
        self.config = config if isinstance(config, PoseFlowV53PipelineConfig) else PoseFlowV53PipelineConfig(**config)
        self.max_epochs = max_epochs
        self.save_hyperparameters(dict(config=self.config.model_dump(), max_epochs=max_epochs))
        self.model = self.config.create_model() if model is None else model
        native = self.config.native_config()
        weights = native.weights.to_dataclass()
        self.frontend_criterion = DiffusionRegPoseCriterion(num_body_points=14, num_box_decoder_layers=native.num_box_decoder_layers,
            weights=weights, dn_label_coef=native.dn_label_coef, dn_bbox_coef=native.dn_bbox_coef,
            interm_loss_coef=native.interm_loss_coef, no_interm_box_loss=native.no_interm_box_loss)
        self.slot_matcher = DiffusionRegPoseMatcher(num_body_points=14, weights=weights)
        self.register_buffer('gradient_scales', torch.ones(2))
        self.register_buffer('calibration_ready', torch.tensor(False))
        self.register_buffer('calibration_samples', torch.tensor(0, dtype=torch.long))
        self.calibration_provenance = None

    def matched_rows(self, batch, context, *, generator=None):
        from .rows_v53 import matched_training_rows
        return matched_training_rows(self.model, self.slot_matcher, batch, context, generator=generator)

    def _forward_losses(self, batch, generator, scales):
        from .latent_v43 import geometry_targets
        _, targets = geometry_targets(batch)
        if any(len(target['labels']) > self.model.capacity for target in targets):
            raise ValueError('Person count exceeds V53 capacity')
        context = self.model.prepare_context(batch['image'], batch.get('padding_mask'), targets=targets)
        rows, sources, geometry, targets = self.matched_rows(batch, context, generator=generator)
        output = self.model.forward_rows(context, **{name: rows[name] for name in self.model.row_input_fields})
        losses = row_losses(output, rows, sources, geometry, batch['img_hw'].flip(-1), scales=scales,
            coordinate_weight=self.config.coordinate_loss_weight, geometry_weight=self.config.geometry_loss_weight)
        birth = rows['birth_rows']
        losses['selection'] = (F.cross_entropy(output['selection_logits'][birth], rows['selection_targets'][birth])
                               if birth.any() else output['selection_logits'].sum()*0)
        losses['frontend'] = self._frontend_loss(context, targets)
        losses['total'] = losses['total']+losses['selection']+losses['frontend']
        losses['debug/people'] = sources['counts'].sum().to(losses['total'])
        losses.update(rows.get('augmentation_statistics', {}))
        if birth.any():
            losses['selection_accuracy'] = (output['selection_logits'].argmax(-1)[birth] == rows['selection_targets'][birth]).float().mean()
        return losses

    def set_calibration(self, norms, provenance):
        if self.calibration_ready or len(norms) != self.config.calibration_batches:
            raise ValueError('Invalid or already frozen V53 calibration')
        scales = fixed_scales(norms)
        self.gradient_scales.copy_(scales.to(self.gradient_scales))
        self.calibration_ready.fill_(True)
        self.calibration_samples.fill_(len(norms))
        self.calibration_provenance = dict(deepcopy(provenance), norms=deepcopy(norms), scales=scales.tolist(), task_names=list(TASK_NAMES))

    def calibration_losses(self, batch, batch_idx):
        return self._forward_losses(batch, torch.Generator().manual_seed(self.config.calibration_seed+batch_idx), torch.ones_like(self.gradient_scales))

    def _losses_for_batch(self, batch, *, training=True, batch_idx=0):
        if not self.calibration_ready:
            raise RuntimeError('V53 requires frozen TRAIN-only gradient calibration')
        generator = None if training else torch.Generator().manual_seed(self.config.seed+batch_idx)
        losses = self._forward_losses(batch, generator, self.gradient_scales)
        losses.update({f'calibration_scale_{name}': scale.detach() for name, scale in zip(TASK_NAMES, self.gradient_scales)})
        return losses

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None):
        if gradient_clip_val is not None and abs(float(gradient_clip_val)-.1) > 1e-8:
            raise ValueError('V53 requires combined clipping bound .1')
        values = clip_families(gradient_families(self.model))
        if self._trainer is not None:
            self.log_dict({f'train/gradients/{name}': value for name, value in values.items()}, on_step=True, on_epoch=False)
        return values

    def log_detection_report(self, report, *, step=None):
        from .quick_eval_v53 import tensorboard_metrics
        if step is None:
            self.log_dict(tensorboard_metrics(report))
        else:
            for logger in self.trainer.loggers:
                logger.log_metrics(tensorboard_metrics(report), step=step)
                if hasattr(logger.experiment, 'flush'):
                    logger.experiment.flush()

    def on_validation_epoch_end(self):
        if not self.config.quick_eval_enabled or self.trainer.sanity_checking:
            return
        from .quick_eval_v53 import run_v53_evaluation
        callback = self.trainer.checkpoint_callback
        root = Path(callback.dirpath).parent if callback else Path(self.trainer.default_root_dir)
        report = run_v53_evaluation(self.model, self.trainer.datamodule, root/'quick_test', epoch=self.current_epoch,
                                    count=self.config.quick_eval_images, seed=self.config.seed)
        self.log_detection_report(report)
        self.log('val/selection_AP', min(report['metrics']['pose_AP'], report['metrics']['bbox_AP']))

    def checkpoint_contract(self):
        return dict(version=1, architecture='V53', rows=self.model.rows, capacity=self.model.capacity,
            native=self.config.native_config().model_dump(mode='json'), expansion='one_selected_person+14native_joint_queries;explicit_no_DN_no_topk',
            decoder='native_pose_layers_and_norm;cached_image_values;row_independent_self_attention;native_reference_updates',
            state='clean_fixed_completed_coordinates;detached_image_local_identity;no_hidden_recurrence',
            order='random_once_per_forward;one_selected_slot_Hungarian_assignment', noise='none', time_conditioning='none',
            sampling='birth_only;immediate_per_image_EOS>=.5_or_capacity;native_objectness_times_birth_nonEOS',
            objectives=dict(tasks=list(TASK_NAMES), weights=[self.config.coordinate_loss_weight, self.config.geometry_loss_weight],
                normalization='both_pose_layers_then_people;equal_normalized_pose_box;balanced_per_image_person_and_firstEOS',
                calibration_batches=self.config.calibration_batches, calibration_seed=self.config.calibration_seed),
            gradients='disjoint_native_selector_geometry;each_.1/sqrt3;selector_prefix_detached',
            initialization='original_EDPose_encoder2_4_6_box2_pose4_6_and_expansion_heads;fresh_clean_prefix_and_selector')

    def on_save_checkpoint(self, checkpoint):
        if not self.calibration_ready:
            raise ValueError('Cannot save uncalibrated V53 checkpoint')
        checkpoint.update(v53_contract=self.checkpoint_contract(), v53_initialization=self.model.initialization_report,
                          v53_calibration=deepcopy(self.calibration_provenance), v53_global_rng=capture_rng())

    def on_load_checkpoint(self, checkpoint):
        if checkpoint.get('v53_contract') != self.checkpoint_contract():
            raise ValueError('Checkpoint does not satisfy V53 contract')
        provenance, state = checkpoint.get('v53_calibration'), checkpoint['state_dict']
        if (not provenance or provenance.get('task_names') != list(TASK_NAMES)
                or len(provenance.get('norms', [])) != self.config.calibration_batches
                or not state.get('calibration_ready', False) or int(state.get('calibration_samples', 0)) != self.config.calibration_batches):
            raise ValueError('Invalid V53 calibration provenance')
        scales = fixed_scales(provenance['norms'])
        if not torch.equal(state['gradient_scales'].cpu(), scales) or provenance.get('scales') != scales.tolist():
            raise ValueError('Corrupted V53 calibration scales')
        self.calibration_provenance = deepcopy(provenance)
        self.model.initialization_report = checkpoint['v53_initialization']
        restore_rng(checkpoint['v53_global_rng'])
