from pathlib import Path

import torch
from torch.nn import functional as F

from .codec_v43 import geometry_from_batch
from .latent_v43 import geometry_targets
from .losses_v49_balanced import row_losses
from .model_v50 import PoseFlowV50Model
from .training_v44 import PoseFlowV44Pipeline
from .training_v49_balanced import PoseFlowV49BalancedPipeline, PoseFlowV49BalancedPipelineConfig
from ..diffusionregpose.training import DiffusionRegPoseCriterion, DiffusionRegPoseMatcher


class PoseFlowV50PipelineConfig(PoseFlowV49BalancedPipelineConfig):
    calibration_seed: int = 42050

    def create_model(self, native=None):
        from ..diffusionregpose.model import DiffusionRegPoseModel
        from .training_v44 import NATIVE_MODEL_FIELDS
        config = self.native_config()
        if native is None:
            native = DiffusionRegPoseModel(backbone_pretrained=False,
                separate_expansion_pose_head=config.pretrained_keep_decoder_layers is not None,
                **{name: getattr(config, name) for name in NATIVE_MODEL_FIELDS})
        return PoseFlowV50Model(native=native, rows=self.rows, capacity=self.capacity, scale=self.scale,
            hidden_dim=config.hidden_dim, num_heads=config.num_heads, feedforward_dim=config.feedforward_dim,
            codec_width=self.codec_width, latent_dim=self.latent_dim,
            activation_checkpointing=self.activation_checkpointing)

    def build(self, *, max_epochs=80, load_pretrained_checkpoint=True, **kwargs):
        base = super().build(max_epochs=max_epochs, load_pretrained_checkpoint=load_pretrained_checkpoint, **kwargs)
        base.model.initialization_report['retained_tensor_count'] = sum(
            name.startswith(('backbone.', 'input_proj.', 'transformer.', 'bbox_embed.', 'class_embed.'))
            for name in base.model.frontend.native.state_dict())
        return PoseFlowV50Pipeline(config=self, model=base.model, max_epochs=max_epochs)


class PoseFlowV50Pipeline(PoseFlowV49BalancedPipeline):
    _frontend_loss = PoseFlowV44Pipeline._frontend_loss

    def __init__(self, *, config, model=None, max_epochs=80):
        specification = config if isinstance(config, PoseFlowV50PipelineConfig) else PoseFlowV50PipelineConfig(**config)
        super().__init__(config=specification, model=model, max_epochs=max_epochs)
        native = self.config.native_config()
        weights = native.weights.to_dataclass()
        self.frontend_criterion = DiffusionRegPoseCriterion(num_body_points=14,
            num_box_decoder_layers=native.num_box_decoder_layers, weights=weights,
            dn_label_coef=native.dn_label_coef, dn_bbox_coef=native.dn_bbox_coef,
            interm_loss_coef=native.interm_loss_coef, no_interm_box_loss=native.no_interm_box_loss)
        self.slot_matcher = DiffusionRegPoseMatcher(num_body_points=14, weights=weights)

    def configure_optimizers(self):
        backbone, main = [], []
        for name, parameter in self.named_parameters():
            if parameter.requires_grad:
                (backbone if name.startswith('model.frontend.native.backbone.') else main).append(parameter)
        optimizer = torch.optim.AdamW([
            dict(params=main, lr=self.config.lr),
            dict(params=backbone, lr=self.config.lr_backbone)], weight_decay=self.config.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=(31, 61), gamma=.1)
        return dict(optimizer=optimizer, lr_scheduler=dict(scheduler=scheduler, interval='epoch'))

    def matched_rows(self, batch, context, *, generator=None):
        geometry, targets = geometry_targets(batch)
        _, locations = geometry_from_batch(batch)
        rows, sources = self.training_rows(batch, geometry, locations, generator=generator)
        matches = self.slot_matcher(dict(pred_boxes=context['boxes'], pred_logits=context['pred_logits']),
            targets, include_keypoints=False)
        candidate_for_geometry = torch.full((len(geometry['image_wh']),), -1, dtype=torch.long, device=batch['image'].device)
        for target, (candidates, people) in zip(targets, matches):
            if len(people) != len(target['labels']) or len(candidates.unique()) != len(candidates):
                raise ValueError('Selected proposal assignment must cover each person exactly once')
            candidate_for_geometry[target['geometry_indices'][people]] = candidates
        source_indices = sources['target_indices']
        identities = torch.full_like(source_indices, -1)
        valid = source_indices >= 0
        identities[valid] = candidate_for_geometry[source_indices[valid]]
        ids = F.pad(identities, (0, self.model.rows-self.model.capacity))[:, None].expand_as(rows['token_types'])
        rows['proposal_indices'] = ids.masked_fill(~rows['supervised'], -1)
        rows['selection_targets'] = rows['proposal_indices'].gather(-1, rows['predict_indices'][..., None]).squeeze(-1)
        sources['proposal_indices'] = identities
        return rows, sources, geometry, targets

    def _forward_losses(self, batch, generator, scales):
        _, targets = geometry_targets(batch)
        if any(len(target['labels']) > self.model.capacity for target in targets):
            raise ValueError('Person count exceeds V50 capacity')
        context = self.model.prepare_context(batch['image'], batch.get('padding_mask'), targets=targets)
        rows, sources, geometry, targets = self.matched_rows(batch, context, generator=generator)
        output = self.model.forward_rows(context, **{name: rows[name] for name in self.model.row_input_fields})
        losses = row_losses(output, rows, sources, geometry, batch['img_hw'].flip(-1), scales=scales,
            coordinate_weight=self.config.coordinate_loss_weight, geometry_weight=self.config.geometry_loss_weight)
        person_rows = ~rows['eos_targets']
        selection = F.cross_entropy(output['selection_logits'][person_rows], rows['selection_targets'][person_rows]) if person_rows.any() else output['selection_logits'].sum()*0
        frontend = self._frontend_loss(context, targets)
        losses.update(selection=selection, frontend=frontend)
        losses['total'] = losses['total']+selection+frontend
        losses['debug/people'] = sources['counts'].sum().to(losses['total'])
        if person_rows.any():
            losses['selection_accuracy'] = (output['selection_logits'].argmax(-1)[person_rows] == rows['selection_targets'][person_rows]).float().mean()
        return losses

    def on_validation_epoch_end(self):
        if not self.config.quick_eval_enabled or self.trainer.sanity_checking:
            return
        from .quick_eval_v50 import run_v50_evaluation, tensorboard_metrics
        callback = self.trainer.checkpoint_callback
        root = Path(callback.dirpath).parent if callback else Path(self.trainer.default_root_dir)
        report = run_v50_evaluation(self.model, self.trainer.datamodule, root/'quick_test',
            epoch=self.current_epoch, count=self.config.quick_eval_images, seed=self.config.seed)
        self.log_dict(tensorboard_metrics(report))
        self.log('val/proposal_attention_selection_AP', min(report['metrics']['pose_AP'], report['metrics']['bbox_AP']))

    def checkpoint_contract(self):
        contract = super().checkpoint_contract()
        contract.update(architecture='V50', variant='proposal_attention_gated_residual_balanced_v1',
            order='random_per_image;one_selected_slot_Hungarian_assignment;fixed_sources',
            selector='prefix_mean_shared_MLP_query;image_local_keys_plus_EOS;used_exclusion;conditional_candidate_CE_plus_marginal_balanced_EOS_BCE',
            identity='hard_gather_feature_and_reference_detached_geometry_branch;trainable_selection_keys;no_global_ID_embedding',
            residual='deattenuated_input_bypass;sigma_delta_XY/(2scale);positive_clamped_size_times_exp(clamp(sigma_delta,-10,10));exact_zero_identity;head_std1e-3',
            eos=dict(threshold=.5, latch=True, confidence='native_objectness_times_birth_nonEOS'),
            initialization='original_EDPose_native_box_frontend;fresh_dense_decoder_coordinate_codec_selector_residual_heads',
            native_losses='box_frontend_encoder_DN_once_per_batch', noise=dict(coefficient='sigma', birth=1., age1=.1, decay=.5, terminal_existing=0.))
        return contract
