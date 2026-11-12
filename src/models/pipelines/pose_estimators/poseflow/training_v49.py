import random
from pathlib import Path

import numpy as np
import torch
from pydantic import Field

from ..diffusionregpose.model import DiffusionRegPoseModel
from .codec_v43 import geometry_from_batch
from .coordinates_v47 import ordered_coordinates
from .losses_v47 import row_losses
from .model_v49 import PoseFlowV49Model
from .rows_token_age import build_training_rows
from .training_v44 import NATIVE_MODEL_FIELDS
from .training_v48 import PoseFlowV48Pipeline, PoseFlowV48PipelineConfig


class PoseFlowV49PipelineConfig(PoseFlowV48PipelineConfig):
    rows: int = 50
    capacity: int = 49
    scale: float = Field(default=6., gt=0, allow_inf_nan=False)

    def create_model(self, native=None):
        config = self.native_config()
        if native is None:
            native = DiffusionRegPoseModel(backbone_pretrained=False,
                separate_expansion_pose_head=config.pretrained_keep_decoder_layers is not None,
                **{name: getattr(config, name) for name in NATIVE_MODEL_FIELDS})
        return PoseFlowV49Model(native=native, rows=self.rows, capacity=self.capacity, scale=self.scale,
            hidden_dim=config.hidden_dim, num_heads=config.num_heads, feedforward_dim=config.feedforward_dim,
            codec_width=self.codec_width, latent_dim=self.latent_dim,
            activation_checkpointing=self.activation_checkpointing)

    def build(self, *, max_epochs=80, load_pretrained_checkpoint=True, **kwargs):
        base = super().build(max_epochs=max_epochs, load_pretrained_checkpoint=load_pretrained_checkpoint, **kwargs)
        return PoseFlowV49Pipeline(config=self, model=base.model, max_epochs=max_epochs)


class PoseFlowV49Pipeline(PoseFlowV48Pipeline):
    def __init__(self, *, config, model=None, max_epochs=80):
        specification = config if isinstance(config, PoseFlowV49PipelineConfig) else PoseFlowV49PipelineConfig(**config)
        super().__init__(config=specification, model=model, max_epochs=max_epochs)

    def training_rows(self, batch, geometry, locations, *, generator=None):
        original = ordered_coordinates(geometry, locations, batch_size=len(batch['image']), capacity=self.model.capacity)
        rows = build_training_rows(self.model.scale*(2*original['coordinates']-1), original['counts'],
            coordinate_mask=original['coordinate_mask'], target_indices=original['target_indices'],
            row_count=self.model.rows, generator=generator)
        sources = dict(rows['sources'], coordinates=(rows['sources']['coordinates']/self.model.scale+1)/2)
        return rows, sources

    def _losses_for_batch(self, batch, *, training=True, batch_idx=0):
        geometry, locations = geometry_from_batch(batch)
        generator = None if training else torch.Generator().manual_seed(self.config.seed+batch_idx)
        rows, sources = self.training_rows(batch, geometry, locations, generator=generator)
        context = self.model.prepare_context(batch['image'], batch.get('padding_mask'))
        output = self.model.forward_rows(context, **{name: rows[name] for name in self.model.row_input_fields})
        losses = row_losses(output, rows, sources, geometry, batch['img_hw'].flip(-1),
            coordinate_weight=self.config.coordinate_loss_weight, geometry_weight=self.config.geometry_loss_weight)
        losses['debug/people'] = sources['counts'].sum().to(losses['total'])
        return losses

    def on_validation_epoch_end(self):
        if not self.config.quick_eval_enabled or self.trainer.sanity_checking:
            return
        from .quick_eval_v49 import run_v49_evaluation, tensorboard_metrics
        callback = self.trainer.checkpoint_callback
        root = Path(callback.dirpath).parent if callback else Path(self.trainer.default_root_dir)
        report = run_v49_evaluation(self.model, self.trainer.datamodule, root/'quick_test',
            epoch=self.current_epoch, count=self.config.quick_eval_images, seed=self.config.seed)
        self.log_dict(tensorboard_metrics(report))
        self.log('val/token_age_selection_AP', min(report['metrics']['pose_AP'], report['metrics']['bbox_AP']))

    def checkpoint_contract(self):
        contract = super().checkpoint_contract()
        contract.update(architecture='V49', order='random_per_image_per_forward;fixed_sources_across_rows;no_matching',
            state='x=6*(2*normalized_coordinates-1)', scale=self.model.scale,
            encoder_input='(x/scale+1)/2_for_all_valid_tokens_including_noisy_query',
            noise=dict(coefficient='sigma', birth=1., age1=.1, decay=.5, terminal_existing=0.,
                       training='independent_Gaussian_per_row_token_channel', query='pure_noise_including_EOS'),
            time_conditioning='sinusoidal_per_token_sigma*999_MLP',
            sampling='DDIM_eta0_ratio_form;one_birth_per_row;all_rows_after_EOS;no_hidden_recurrence',
            eos=dict(threshold=.5, latch=True, post_EOS_rows='remaining_to_50', confidence='birth_person_probability'),
            supervision='both_layers;birth_mean_per_person;refine_mean_rows_per_person;equal_birth_refine;balanced_EOS',
            initialization='original_EDPose_frontend;fresh_coordinate_codec_decoder_and_sigma_embedding')
        return contract

    def on_save_checkpoint(self, checkpoint):
        checkpoint['v49_contract'] = self.checkpoint_contract()
        checkpoint['v49_initialization'] = self.model.initialization_report
        checkpoint['v49_global_rng'] = dict(python=random.getstate(), numpy=np.random.get_state(),
            torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])

    def on_load_checkpoint(self, checkpoint):
        if checkpoint.get('v49_contract') != self.checkpoint_contract():
            raise ValueError('Checkpoint does not satisfy the V49 contract')
        self.model.initialization_report = checkpoint['v49_initialization']
        rng = checkpoint['v49_global_rng']
        random.setstate(rng['python'])
        np.random.set_state(rng['numpy'])
        torch.set_rng_state(rng['torch'].cpu())
        if rng['cuda'] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([value.cpu() for value in rng['cuda']])
