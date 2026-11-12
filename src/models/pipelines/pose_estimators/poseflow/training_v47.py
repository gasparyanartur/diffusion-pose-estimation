import random
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pydantic import Field, model_validator

from src.configs.base_config import BaseConfig
from ..diffusionregpose.model import DiffusionRegPoseModel
from .codec_v43 import geometry_from_batch
from .coordinates_v47 import ordered_coordinates, training_rows
from .losses_v47 import row_losses
from .model_v47 import PoseFlowV47Model
from .training_v44 import NATIVE_MODEL_FIELDS, PoseFlowV44Pipeline, PoseFlowV44PipelineConfig


class PoseFlowV47PipelineConfig(BaseConfig):
    native: dict = Field(default_factory=dict)
    rows: int = 64
    capacity: int = 50
    codec_width: int = 256
    latent_dim: int = 32
    coordinate_scale: float = 6.
    pretrained_checkpoint: str | None = 'external/DiffusionRegPose/pretrain_models/edpose_r50_crowdpose.pth'
    coordinate_loss_weight: float = 10.
    geometry_loss_weight: float = .5
    activation_checkpointing: bool = True
    seed: int = 42
    quick_eval_enabled: bool = True
    quick_eval_images: int = 32
    lr: float = 2e-4
    lr_backbone: float = 1e-4
    weight_decay: float = 1e-4

    native_config = PoseFlowV44PipelineConfig.native_config

    @model_validator(mode='after')
    def validate_contract(self):
        values = (self.coordinate_scale, self.coordinate_loss_weight, self.geometry_loss_weight)
        if (not 0 < self.capacity < self.rows or not np.isfinite(values).all() or self.coordinate_scale <= 0
                or min(values[1:]) < 0 or min(self.codec_width, self.latent_dim, self.quick_eval_images) < 1):
            raise ValueError('Invalid V47 dimensions, scale or objective')
        return self

    def create_model(self, native=None):
        config = self.native_config()
        if native is None:
            native = DiffusionRegPoseModel(backbone_pretrained=False,
                separate_expansion_pose_head=config.pretrained_keep_decoder_layers is not None,
                **{name: getattr(config, name) for name in NATIVE_MODEL_FIELDS})
        return PoseFlowV47Model(native=native, rows=self.rows, capacity=self.capacity,
            hidden_dim=config.hidden_dim, num_heads=config.num_heads, feedforward_dim=config.feedforward_dim,
            codec_width=self.codec_width, latent_dim=self.latent_dim, scale=self.coordinate_scale,
            activation_checkpointing=self.activation_checkpointing)

    def build(self, *, max_epochs=80, load_pretrained_checkpoint=True, **kwargs):
        native = None
        initialization = dict(frontend='random; initializer disabled', codec='fresh learned coordinate encoder/reader',
                              decoder='fresh dense two-layer')
        if load_pretrained_checkpoint and self.pretrained_checkpoint:
            native = self.native_config().build(max_epochs=max_epochs).model
            initialization.update(frontend='original_EDPose', source=self.pretrained_checkpoint,
                retained_encoder_source_layers=[2, 4, 6], critical_minimum_coverage=.99)
        model = self.create_model(native)
        initialization['retained_tensor_count'] = sum(name.startswith(('backbone.', 'input_proj.', 'transformer.'))
                                                      for name in model.state_dict())
        model.initialization_report = initialization
        return PoseFlowV47Pipeline(config=self, model=model, max_epochs=max_epochs)


class PoseFlowV47Pipeline(pl.LightningModule):
    configure_optimizers = PoseFlowV44Pipeline.configure_optimizers
    on_before_optimizer_step = PoseFlowV44Pipeline.on_before_optimizer_step

    def __init__(self, *, config, model=None, max_epochs=80):
        super().__init__()
        self.config = config if isinstance(config, PoseFlowV47PipelineConfig) else PoseFlowV47PipelineConfig(**config)
        self.max_epochs = max_epochs
        self.save_hyperparameters(dict(config=self.config.model_dump(), max_epochs=max_epochs))
        self.model = self.config.create_model() if model is None else model
        self.register_buffer('training_rng_state', torch.Generator().manual_seed(self.config.seed).get_state())

    def _losses_for_batch(self, batch, *, training=True, batch_idx=0):
        geometry, locations = geometry_from_batch(batch)
        sources = ordered_coordinates(geometry, locations, batch_size=len(batch['image']), capacity=self.model.capacity)
        generator = torch.Generator()
        if training:
            generator.set_state(self.training_rng_state.cpu())
        else:
            generator.manual_seed(self.config.seed+batch_idx)
        rows = training_rows(sources['coordinates'], sources['counts'], rows=self.model.rows,
                             scale=self.model.scale, generator=generator)
        if training:
            self.training_rng_state.copy_(generator.get_state().to(self.training_rng_state.device))
        context = self.model.prepare_context(batch['image'], batch.get('padding_mask'))
        output = self.model.forward_rows(context, **{name: rows[name] for name in self.model.row_input_fields})
        losses = row_losses(output, rows, sources, geometry, batch['img_hw'].flip(-1),
            coordinate_weight=self.config.coordinate_loss_weight, geometry_weight=self.config.geometry_loss_weight)
        losses['debug/people'] = sources['counts'].sum().to(losses['total'])
        return losses

    def training_step(self, batch, batch_idx):
        losses = self._losses_for_batch(batch, batch_idx=batch_idx)
        if not torch.isfinite(losses['total']):
            raise FloatingPointError('Nonfinite V47 training loss')
        self.log_dict({f'train/{name}': value for name, value in losses.items()},
            on_step=True, on_epoch=True, batch_size=len(batch['image']))
        return losses['total']

    def validation_step(self, batch, batch_idx):
        losses = self._losses_for_batch(batch, training=False, batch_idx=batch_idx)
        self.log_dict({f'val/{name}': value for name, value in losses.items()},
            on_step=False, on_epoch=True, batch_size=len(batch['image']))
        return losses['total']

    def on_validation_epoch_end(self):
        if not self.config.quick_eval_enabled or self.trainer.sanity_checking:
            return
        from .quick_eval_v47 import run_v47_evaluation, tensorboard_metrics
        callback = self.trainer.checkpoint_callback
        root = Path(callback.dirpath).parent if callback else Path(self.trainer.default_root_dir)
        report = run_v47_evaluation(self.model, self.trainer.datamodule, root/'quick_test',
            epoch=self.current_epoch, count=self.config.quick_eval_images, seed=self.config.seed)
        self.log_dict(tensorboard_metrics(report))
        self.log('val/ordered_coordinate_selection_AP', min(report['metrics']['pose_AP'], report['metrics']['bbox_AP']))

    def checkpoint_contract(self):
        return dict(version=1, architecture='V47', order='512_columns_cy_cx_wh_masked_joints_stable',
            rows=self.model.rows, capacity=self.model.capacity, state='noisy_coordinates', scale=self.model.scale,
            coordinates='image_normalized_jointXY28_boxCXCYWH4;unobserved_joint_box_center;no_mask_input',
            codec=dict(width=self.model.codec_width, latent_dim=self.model.latent_dim,
                encoder='32_linear_GELU_linear_nonaffineLN', reader='linear_GELU_linear_GELU_linear;softplus_wh+1e-6',
                training='joint_2D_objectives_only'),
            decoder=dict(layers=2, width=self.model.hidden_dim, heads=self.model.num_heads,
                         ffn=self.model.feedforward_dim, norm='pre', dropout=0., activation='GELU'),
            frontend=self.config.native, attention='bidirectional_independent_rows_cached_image_KV',
            noise='row_cosine_s.008_exact_endpoints', sampling='coordinate_DDIM_eta0;newborn_forward_noise',
            eos=dict(threshold=.5, latch=True, confidence='birth_person_probability'),
            supervision='equal_birth_refine_tasks_per_person_both_layers;per_image_balanced_EOS',
            loss_weights=[self.config.coordinate_loss_weight, self.config.geometry_loss_weight, 1.],
            initialization='original_EDPose_frontend_fresh_codec_and_decoder')

    def on_save_checkpoint(self, checkpoint):
        checkpoint['v47_contract'] = self.checkpoint_contract()
        checkpoint['v47_initialization'] = self.model.initialization_report
        checkpoint['v47_global_rng'] = dict(python=random.getstate(), numpy=np.random.get_state(),
            torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])

    def on_load_checkpoint(self, checkpoint):
        if checkpoint.get('v47_contract') != self.checkpoint_contract():
            raise ValueError('Checkpoint does not satisfy the V47 contract')
        self.model.initialization_report = checkpoint['v47_initialization']
        rng = checkpoint['v47_global_rng']
        random.setstate(rng['python'])
        np.random.set_state(rng['numpy'])
        torch.set_rng_state(rng['torch'].cpu())
        if rng['cuda'] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([value.cpu() for value in rng['cuda']])
