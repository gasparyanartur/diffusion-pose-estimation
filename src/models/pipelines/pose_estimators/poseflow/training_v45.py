import random
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pydantic import Field, model_validator

from src.configs.base_config import BaseConfig
from src.models.pipelines.pose_autoencoder.model import geometry_loss_per_person
from src.models.pipelines.pose_vae.fidelity import FidelityPoseVAE
from ..diffusionregpose.training import DiffusionRegPoseCriterion, DiffusionRegPoseMatcher
from .codec_v43 import CODEC_CHECKPOINT, CODEC_SHA256, FrozenPoseCodec
from .latent_v43 import assign_latent_sources, geometry_targets, slot_focal_loss
from .model_v45 import PoseFlowV45Model
from .training_v44 import NATIVE_MODEL_FIELDS, PoseFlowV44Pipeline, PoseFlowV44PipelineConfig


def progressive_weights(slots, *, device=None, dtype=torch.float32):
    return (2 * torch.arange(slots, 0, -1, device=device, dtype=dtype)).reciprocal()


class PoseFlowV45PipelineConfig(BaseConfig):
    native: dict = Field(default_factory=dict)
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
        native = self.native_config()
        if (native.matching_strategy != 'hungarian' or native.num_group != native.num_proposals
                or native.num_decoder_layers - native.num_box_decoder_layers != 2):
            raise ValueError('V45 requires fixed Hungarian proposals and two shared person layers')
        if self.latent_loss_weight < 0 or self.geometry_loss_weight < 0:
            raise ValueError('V45 loss weights must be nonnegative')
        if self.quick_eval_images < 1:
            raise ValueError('V45 evaluation needs at least one image')
        return self

    def create_model(self, codec=None, native_state=None):
        native = self.native_config()
        if codec is None:
            with torch.random.fork_rng(devices=[]):
                codec = FrozenPoseCodec(FidelityPoseVAE(width=self.codec_width, latent_dim=self.latent_dim))
        return PoseFlowV45Model(codec=codec, native_state=native_state, backbone_pretrained=False,
            activation_checkpointing=self.activation_checkpointing,
            separate_expansion_pose_head=native.pretrained_keep_decoder_layers is not None,
            **{name: getattr(native, name) for name in NATIVE_MODEL_FIELDS})

    def create_pipeline(self, *, model, max_epochs):
        return PoseFlowV45Pipeline(config=self, model=model, max_epochs=max_epochs)


class PoseFlowV45Pipeline(PoseFlowV44Pipeline):
    def __init__(self, *, config, model=None, max_epochs=80):
        pl.LightningModule.__init__(self)
        self.config = config if isinstance(config, PoseFlowV45PipelineConfig) else PoseFlowV45PipelineConfig(**config)
        self.max_epochs = max_epochs
        self.save_hyperparameters(dict(config=self.config.model_dump(), max_epochs=max_epochs))
        self.model = self.config.create_model() if model is None else model
        native = self.config.native_config()
        weights = native.weights.to_dataclass()
        self.frontend_criterion = DiffusionRegPoseCriterion(num_body_points=14,
            num_box_decoder_layers=native.num_box_decoder_layers, weights=weights,
            dn_label_coef=native.dn_label_coef, dn_bbox_coef=native.dn_bbox_coef,
            interm_loss_coef=native.interm_loss_coef, no_interm_box_loss=native.no_interm_box_loss)
        self.slot_matcher = DiffusionRegPoseMatcher(num_body_points=14, weights=weights)
        self.codec_provenance = {}

    def _losses_for_batch(self, batch, *, training=True, batch_idx=0):
        geometry, targets = geometry_targets(batch)
        slots = self.model.num_proposals
        if any(len(target['labels']) > slots for target in targets):
            raise ValueError('Valid person count exceeds proposal capacity')
        context = self.model.prepare_context(batch['image'], batch.get('padding_mask'), targets=targets)
        matches = self.slot_matcher(dict(pred_boxes=context['boxes'], pred_logits=context['pred_logits']),
            targets, include_keypoints=False)
        sources = assign_latent_sources(self.model.codec, geometry, targets, matches,
            slots=slots, generator=torch.Generator(), posterior_mode='mean')
        positive = sources['positive']
        image_wh = batch['img_hw'].flip(-1).to(context['features'])
        weights = progressive_weights(slots, device=image_wh.device, dtype=image_wh.dtype)
        latent_errors = torch.zeros_like(positive, dtype=image_wh.dtype)
        geometry_errors = torch.zeros_like(latent_errors)
        for stage, layer, latent in self.model.latent_predictions(context):
            latent_errors = latent_errors + torch.nn.functional.pad(
                (latent-sources['latents'][:, :stage]).square().mean(-1)*weights[:stage], (0, slots-stage))
            active = positive[:, :stage]
            if active.any():
                rows = sources['target_indices'][:, :stage][active]
                decoded = self.model.codec.decode_latents(latent[active], image_wh[:, None].expand(-1, stage, -1)[active])
                errors = geometry_loss_per_person(decoded, {name: value[rows] for name, value in geometry.items()})['loss']
                per_slot = torch.zeros_like(latent_errors[:, :stage]).masked_scatter(active, errors)
                geometry_errors = geometry_errors + torch.nn.functional.pad(per_slot*weights[:stage], (0, slots-stage))
        zero = latent_errors.sum()*0
        foreground = latent_errors[positive].mean() if positive.any() else zero
        background = latent_errors[~positive].mean() if (~positive).any() else zero
        decoded_loss = geometry_errors[positive].mean() if positive.any() else zero
        frontend = self._frontend_loss(context, targets)
        focal = slot_focal_loss(context['existence_logits'], positive)
        total = frontend + 2*focal + self.config.latent_loss_weight*(foreground+background) + self.config.geometry_loss_weight*decoded_loss
        return dict(total=total, frontend=frontend, loss_slot_focal=focal,
            latent_positive=foreground, latent_background=background, geometry=decoded_loss,
            **{'debug/positive_slots': positive.sum().float(), 'debug/stages': image_wh.new_tensor(slots)})

    def on_validation_epoch_end(self):
        if not self.config.quick_eval_enabled or self.trainer.sanity_checking:
            return
        from .quick_eval_v45 import run_v45_evaluation, tensorboard_metrics
        checkpoint_callback = self.trainer.checkpoint_callback
        root = Path(checkpoint_callback.dirpath).parent if checkpoint_callback else Path(self.trainer.default_root_dir)
        report = run_v45_evaluation(self.model, self.trainer.datamodule, root/'quick_test',
            epoch=self.current_epoch, count=self.config.quick_eval_images, seed=self.config.seed)
        self.log_dict(tensorboard_metrics(report))
        self.log('val/progressive_selection_AP', min(report['metrics']['pose_AP'], report['metrics']['bbox_AP']))

    def checkpoint_contract(self):
        return dict(version=1, architecture='V45', attention='bidirectional_prefix',
            order='native_confidence_topk', stages=self.model.num_proposals, shared_layers=2,
            state='hidden', normalization='per_person_all_active_stages_and_layers',
            initialization='fresh_EDPose', codec_sha256=self.config.codec_sha256,
            posterior_mode='mean', fixed_assignment=True)

    def on_save_checkpoint(self, checkpoint):
        checkpoint['v45_contract'] = self.checkpoint_contract()
        checkpoint['v45_codec_provenance'] = self.codec_provenance
        checkpoint['v45_global_rng'] = dict(python=random.getstate(), numpy=np.random.get_state(),
            torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])

    def on_load_checkpoint(self, checkpoint):
        if checkpoint.get('v45_contract') != self.checkpoint_contract():
            raise ValueError('Checkpoint does not satisfy the V45 contract')
        self.codec_provenance = checkpoint['v45_codec_provenance']
        rng = checkpoint['v45_global_rng']
        random.setstate(rng['python'])
        np.random.set_state(rng['numpy'])
        torch.set_rng_state(rng['torch'].cpu())
        if rng['cuda'] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([value.cpu() for value in rng['cuda']])
