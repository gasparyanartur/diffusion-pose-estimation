from typing import ClassVar, Literal

from pydantic import Field, model_validator

from ..diffusionregpose.training import DiffusionRegPosePipelineConfig
from .model_v21 import PoseFlowV21Model
from .training import PoseFlowPipeline, PoseFlowPipelineConfig


class PoseFlowV21Pipeline(PoseFlowPipeline):
    diagnostic_namespace = 'v21'
    def __init__(self, *args, quick_eval_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.quick_eval_config = quick_eval_config
        self._quick_eval_epoch = None

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        config, trainer = self.quick_eval_config, self.trainer
        if (config is None or not config.quick_eval_enabled or trainer.sanity_checking
                or not trainer.is_global_zero or self._quick_eval_epoch == self.current_epoch):
            return
        import json
        from pathlib import Path
        from scripts.utils.quick_test_logging import comparison_tag
        from .quick_eval_v21 import run_v21_quick_test

        callback = trainer.checkpoint_callback
        run_directory = (Path(callback.dirpath).parent if callback is not None
                         else Path(trainer.default_root_dir))
        root = run_directory/'quick_test'
        directory = root/f'epoch{self.current_epoch:03d}'
        if (directory/'tensorboard_logged.json').exists():
            self._quick_eval_epoch = self.current_epoch
            return
        if (directory/'metrics.json').exists():
            report = json.loads((directory/'metrics.json').read_text())
            if report['global_step'] != self.global_step:
                raise ValueError('Existing V21 quick-test report has a different training step')
        else:
            report = run_v21_quick_test(self.model, trainer.datamodule, config, root,
                                        epoch=self.current_epoch, global_step=self.global_step)
        for condition, row in report['conditions'].items():
            for name, value in row['metrics'].items():
                if value >= 0:
                    self.log(comparison_tag(f'quick-test/{condition}/{name}'), value,
                             on_step=False, on_epoch=True, rank_zero_only=True)
        (directory/'tensorboard_logged.json').write_text(json.dumps({
            'epoch': self.current_epoch, 'global_step': self.global_step,
            'kind': 'live validation hook'})+'\n')
        self._quick_eval_epoch = self.current_epoch

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        self.log(f'debug/{self.diagnostic_namespace}/dense_gate', self.model.dense_read.gate, batch_size=batch['image'].shape[0])
        return loss


class PoseFlowV21PipelineConfig(PoseFlowPipelineConfig):
    model_class: ClassVar[type] = PoseFlowV21Model
    pipeline_class: ClassVar[type] = PoseFlowV21Pipeline
    new_state_prefixes: ClassVar[tuple[str, ...]] = ('dense_read.',)
    hypothesis_update_order: Literal['parallel', 'h_then_g'] = 'parallel'
    quick_eval_enabled: bool = True
    quick_eval_images: int = Field(default=32, ge=1)
    quick_eval_seed: int = 42
    quick_eval_steps: list[int] = Field(default_factory=lambda: [1, 2, 3])

    @model_validator(mode='after')
    def validate_fixed_hypotheses(self):
        if (not self.quick_eval_steps or min(self.quick_eval_steps) < 1
                or len(set(self.quick_eval_steps)) != len(self.quick_eval_steps)):
            raise ValueError('V21 quick-test call counts must be positive and unique')
        if not self.num_queries == self.num_proposals == self.num_group:
            raise ValueError('V21 query, proposal and group counts must be equal')
        if self.num_box_decoder_layers != 1 or self.num_decoder_layers < 2:
            raise ValueError('V21 requires one box stage followed by pose stages')
        if self.pose_codebook_size or self.flow_renewal or self.flow_ensemble:
            raise ValueError('V21 uses fixed hypotheses without codebook, renewal or ensemble')
        return self

    def _extra_model_arguments(self):
        return {}

    def _extra_pipeline_arguments(self):
        return {}

    def _build_poseflow_model(self):
        fields = ('backbone_pretrained', 'timesteps', 'sampling_timesteps', 'num_queries',
                  'num_proposals', 'num_classes', 'num_feature_levels', 'num_encoder_layers',
                  'num_decoder_layers', 'hidden_dim', 'feedforward_dim', 'num_heads',
                  'num_body_points', 'num_box_decoder_layers', 'num_group', 'dn_number',
                  'dn_box_noise_scale', 'dn_label_noise_ratio', 'freeze_backbone_stem_and_layer1',
                  'freeze_backbone', 'freeze_backbone_and_transformer', 'flow_sampling_steps',
                  'hypothesis_update_order')
        return self.model_class(**{name:getattr(self, name) for name in fields}, **self._extra_model_arguments(),
            separate_expansion_pose_head=self.pretrained_keep_decoder_layers is not None)

    def build(self, **kwargs):
        base = DiffusionRegPosePipelineConfig.build(self, **kwargs)
        model = self._build_poseflow_model()
        result = model.load_state_dict(base.model.state_dict(), strict=False)
        expected = {name for name in model.state_dict()
                    if name == 'hypothesis_slots' or name.startswith(self.new_state_prefixes)}
        if set(result.missing_keys) != expected or result.unexpected_keys:
            raise RuntimeError('V21 initialization changed the expected new-module boundary')
        model.initialization_report = {'new_state_keys':sorted(expected),
            'shared_tensors':len(base.model.state_dict()),
            'shared_elements':sum(value.numel() for value in base.model.state_dict().values())}
        fields = ('lr', 'lr_backbone', 'weight_decay', 'max_epochs', 'lr_warmup_epochs',
                  'lr_min_frac', 'scheduler_type', 'lr_decay_milestones', 'split_weight_decay',
                  'val_ap_every_n_epochs', 'diffusion_noise_seed', 'deterministic_seed')
        return self.pipeline_class(model, base.criterion, **{name:getattr(base, name) for name in fields},
            val_ann_file=kwargs.get('val_ann_file'), flow_padding_policy=self.flow_padding_policy,
            flow_sampling_steps=self.flow_sampling_steps, quick_eval_config=self,
            **self._extra_pipeline_arguments())
