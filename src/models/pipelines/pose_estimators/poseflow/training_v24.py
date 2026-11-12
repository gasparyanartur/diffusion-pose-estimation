import hashlib
import logging
from typing import ClassVar

from pydantic import Field, model_validator

from ..diffusionregpose.training import DiffusionRegPosePipelineConfig
from .model_v24 import PoseFlowV24Model
from .training import PoseFlowPipeline, PoseFlowPipelineConfig
from .training_v21 import PoseFlowV21Pipeline


class PoseFlowV24Pipeline(PoseFlowV21Pipeline):
    def training_step(self, batch, batch_idx):
        loss = PoseFlowPipeline.training_step(self, batch, batch_idx)
        if self.model.h_branch is not None:
            for index, gate in enumerate(self.model.h_branch.gates):
                self.log(f'debug/v24/h_gate_{index}', gate, batch_size=batch['image'].shape[0])
        return loss


class PoseFlowV24PipelineConfig(PoseFlowPipelineConfig):
    training_protocol: ClassVar[str] = 'matched_single_call_residual_h'
    model_class: ClassVar[type] = PoseFlowV24Model
    pipeline_class: ClassVar[type] = PoseFlowV24Pipeline
    new_model_prefixes: ClassVar[tuple[str, ...]] = ('h_branch.',)
    h_enabled: bool = False
    quick_eval_enabled: bool = True
    quick_eval_images: int = Field(default=32, ge=1)
    quick_eval_seed: int = 42
    quick_eval_steps: list[int] = Field(default_factory=lambda: [1])

    @model_validator(mode='after')
    def validate_matched_experiment(self):
        if self.flow_sampling_steps != 1 or self.quick_eval_steps != [1]:
            raise ValueError('V24 requires exactly one call')
        if self.pose_codebook_size or self.flow_renewal or self.flow_ensemble:
            raise ValueError('V24 excludes codebook, renewal and ensemble')
        if self.num_proposals != self.num_group:
            raise ValueError('V24 pose candidates and groups must agree')
        return self

    def _build_poseflow_model(self):
        fields = ('backbone_pretrained', 'timesteps', 'sampling_timesteps', 'num_queries',
                  'num_proposals', 'num_classes', 'num_feature_levels', 'num_encoder_layers',
                  'num_decoder_layers', 'hidden_dim', 'feedforward_dim', 'num_heads',
                  'num_body_points', 'num_box_decoder_layers', 'num_group', 'dn_number',
                  'dn_box_noise_scale', 'dn_label_noise_ratio', 'freeze_backbone_stem_and_layer1',
                  'freeze_backbone', 'freeze_backbone_and_transformer', 'flow_sampling_steps')
        return self.model_class(**{name:getattr(self, name) for name in fields},
            separate_expansion_pose_head=self.pretrained_keep_decoder_layers is not None,
            h_enabled=self.h_enabled, **self._extra_model_kwargs())

    def _extra_model_kwargs(self):
        return {}

    def build(self, **kwargs):
        base = DiffusionRegPosePipelineConfig.build(self, **kwargs)
        model = self._build_poseflow_model()
        result = model.load_state_dict(base.model.state_dict(), strict=False)
        expected = {name for name in model.state_dict() if name.startswith(self.new_model_prefixes)}
        if set(result.missing_keys) != expected or result.unexpected_keys:
            raise RuntimeError('V24 initialization changed shared model tensors')
        model.initialization_report = {'new_state_keys': sorted(expected),
            'shared_tensors':len(base.model.state_dict())}
        digest = hashlib.sha256()
        for name, value in sorted(model.state_dict().items()):
            if not name.startswith(self.new_model_prefixes):
                digest.update(name.encode())
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        model.initialization_report['shared_sha256'] = digest.hexdigest()
        logging.getLogger(__name__).info('V24 initialization: h_enabled=%s shared_sha256=%s',
                                        self.h_enabled, digest.hexdigest())
        fields = ('lr', 'lr_backbone', 'weight_decay', 'max_epochs', 'lr_warmup_epochs',
                  'lr_min_frac', 'scheduler_type', 'lr_decay_milestones', 'split_weight_decay',
                  'val_ap_every_n_epochs', 'diffusion_noise_seed', 'deterministic_seed')
        return self.pipeline_class(model, base.criterion,
            **{name:getattr(base, name) for name in fields},
            val_ann_file=kwargs.get('val_ann_file'), flow_padding_policy=self.flow_padding_policy,
            flow_sampling_steps=self.flow_sampling_steps, quick_eval_config=self)
