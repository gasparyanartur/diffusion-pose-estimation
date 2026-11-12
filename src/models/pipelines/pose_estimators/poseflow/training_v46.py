import random
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pydantic import Field, model_validator

from src.configs.base_config import BaseConfig
from src.models.pipelines.pose_vae.fidelity import FidelityPoseVAE
from ..diffusionregpose.model import DiffusionRegPoseModel
from .codec_v43 import CODEC_CHECKPOINT, CODEC_SHA256, FrozenPoseCodec, geometry_from_batch
from .losses_v46 import row_losses
from .model_v46 import PoseFlowV46Model
from .rows_v46 import build_training_rows, ordered_sources
from .training_v44 import NATIVE_MODEL_FIELDS, PoseFlowV44Pipeline, PoseFlowV44PipelineConfig


class PoseFlowV46PipelineConfig(BaseConfig):
    native: dict = Field(default_factory=dict)
    rows: int = 64
    capacity: int = 50
    codec_checkpoint: str = CODEC_CHECKPOINT
    codec_sha256: str = CODEC_SHA256
    codec_width: int = 256
    latent_dim: int = 32
    latent_statistics_file: str | None = None
    pretrained_checkpoint: str | None = 'external/DiffusionRegPose/pretrain_models/edpose_r50_crowdpose.pth'
    latent_loss_weight: float = 10.
    geometry_loss_weight: float = .5
    activation_checkpointing: bool = True
    seed: int = 42
    quick_eval_enabled: bool = True
    quick_eval_images: int = 32
    lr: float = 2e-4
    lr_backbone: float = 1e-4
    weight_decay: float = 1e-4

    native_config = PoseFlowV44PipelineConfig.native_config
    build = PoseFlowV44PipelineConfig.build

    @model_validator(mode='after')
    def validate_contract(self):
        if not 0 < self.capacity < self.rows:
            raise ValueError('Rows must exceed positive sequence capacity')
        if self.latent_loss_weight < 0 or self.geometry_loss_weight < 0 or self.quick_eval_images < 1:
            raise ValueError('Invalid loss weights or evaluation count')
        return self

    def create_model(self, codec=None, native_state=None):
        config = self.native_config()
        if codec is None:
            with torch.random.fork_rng(devices=[]):
                codec = FrozenPoseCodec(FidelityPoseVAE(width=self.codec_width, latent_dim=self.latent_dim))
        native = DiffusionRegPoseModel(backbone_pretrained=False,
            separate_expansion_pose_head=config.pretrained_keep_decoder_layers is not None,
            **{name: getattr(config, name) for name in NATIVE_MODEL_FIELDS})
        if native_state is not None:
            native.load_state_dict(native_state, strict=True)
        return PoseFlowV46Model(codec=codec, native=native, rows=self.rows, capacity=self.capacity,
            hidden_dim=config.hidden_dim, num_heads=config.num_heads, feedforward_dim=config.feedforward_dim,
            activation_checkpointing=self.activation_checkpointing)

    def create_pipeline(self, *, model, max_epochs):
        model.initialization_report.update(transfer='retain only original ED-Pose backbone/projections/encoder',
            retained_encoder_source_layers=[2, 4, 6], decoder='fresh dense two-layer',
            retained_tensor_count=sum(name.startswith(('backbone.', 'input_proj.', 'transformer.')) for name in model.state_dict()))
        return PoseFlowV46Pipeline(config=self, model=model, max_epochs=max_epochs)


class PoseFlowV46Pipeline(pl.LightningModule):
    configure_optimizers = PoseFlowV44Pipeline.configure_optimizers
    on_before_optimizer_step = PoseFlowV44Pipeline.on_before_optimizer_step

    def __init__(self, *, config, model=None, max_epochs=80):
        super().__init__()
        self.config = config if isinstance(config, PoseFlowV46PipelineConfig) else PoseFlowV46PipelineConfig(**config)
        self.max_epochs = max_epochs
        self.save_hyperparameters(dict(config=self.config.model_dump(), max_epochs=max_epochs))
        self.model = self.config.create_model() if model is None else model
        self.codec_provenance = {}
        self.register_buffer('training_rng_state', torch.Generator().manual_seed(self.config.seed).get_state())

    def _losses_for_batch(self, batch, *, training=True, batch_idx=0):
        geometry, locations = geometry_from_batch(batch)
        sources = ordered_sources(self.model.codec, geometry, locations,
            batch_size=len(batch['image']), capacity=self.model.capacity)
        generator = torch.Generator()
        if training:
            generator.set_state(self.training_rng_state.cpu())
        else:
            generator.manual_seed(self.config.seed+batch_idx)
        rows = build_training_rows(sources['latents'], sources['counts'], rows=self.model.rows, generator=generator)
        if training:
            self.training_rng_state.copy_(generator.get_state().to(self.training_rng_state.device))
        context = self.model.prepare_context(batch['image'], batch.get('padding_mask'))
        output = self.model.forward_rows(context, **{name: rows[name] for name in
            ('noisy_latents', 'row_times', 'token_types', 'padding_mask')})
        losses = row_losses(output, rows, sources, geometry, self.model.codec, batch['img_hw'].flip(-1),
            latent_weight=self.config.latent_loss_weight, geometry_weight=self.config.geometry_loss_weight)
        losses['debug/people'] = sources['counts'].sum().to(losses['total'])
        return losses

    def training_step(self, batch, batch_idx):
        losses = self._losses_for_batch(batch, batch_idx=batch_idx)
        if not torch.isfinite(losses['total']):
            raise FloatingPointError('Nonfinite V46 training loss')
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
        from .quick_eval_v46 import run_v46_evaluation, tensorboard_metrics
        checkpoint_callback = self.trainer.checkpoint_callback
        root = Path(checkpoint_callback.dirpath).parent if checkpoint_callback else Path(self.trainer.default_root_dir)
        report = run_v46_evaluation(self.model, self.trainer.datamodule, root/'quick_test',
            epoch=self.current_epoch, count=self.config.quick_eval_images, seed=self.config.seed)
        self.log_dict(tensorboard_metrics(report))
        self.log('val/ordered_sequence_selection_AP', min(report['metrics']['pose_AP'], report['metrics']['bbox_AP']))

    def checkpoint_contract(self):
        return dict(version=1, architecture='V46', order='512_columns_cy_cx_wh_masked_joints_stable',
            rows=self.model.rows, capacity=self.model.capacity, state='clean_latents',
            decoder=dict(layers=2, width=self.model.hidden_dim, heads=self.model.num_heads,
                         ffn=self.model.feedforward_dim, norm='pre', dropout=0., activation='GELU'),
            frontend=self.config.native, attention='bidirectional_independent_rows_dense_image',
            noise='row_cosine_s.008_exact_endpoints', sampling='independent_whole_prefix_renoise',
            eos=dict(threshold=.5, latch=True, confidence='birth_person_probability'),
            supervision='equal_birth_refine_tasks_per_person_both_layers;per_image_balanced_EOS',
            loss_weights=[self.config.latent_loss_weight, self.config.geometry_loss_weight, 1.],
            initialization='original_EDPose_frontend_fresh_decoder', codec_sha256=self.config.codec_sha256,
            codec_dimensions=[self.config.codec_width, self.config.latent_dim], posterior_mode='mean')

    def on_save_checkpoint(self, checkpoint):
        checkpoint['v46_contract'] = self.checkpoint_contract()
        checkpoint['v46_codec_provenance'] = self.codec_provenance
        checkpoint['v46_initialization'] = self.model.initialization_report
        checkpoint['v46_global_rng'] = dict(python=random.getstate(), numpy=np.random.get_state(),
            torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])

    def on_load_checkpoint(self, checkpoint):
        if checkpoint.get('v46_contract') != self.checkpoint_contract():
            raise ValueError('Checkpoint does not satisfy the V46 contract')
        self.codec_provenance = checkpoint['v46_codec_provenance']
        self.model.initialization_report = checkpoint['v46_initialization']
        rng = checkpoint['v46_global_rng']
        random.setstate(rng['python'])
        np.random.set_state(rng['numpy'])
        torch.set_rng_state(rng['torch'].cpu())
        if rng['cuda'] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([value.cpu() for value in rng['cuda']])
