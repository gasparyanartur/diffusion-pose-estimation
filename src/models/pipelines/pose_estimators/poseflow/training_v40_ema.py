from copy import deepcopy
from typing import ClassVar

import torch
from torch import nn
from torch.nn import functional
from pydantic import Field

from .latent_v35 import box_task_loss
from .model_v26 import PoseFlowV26Model
from .training_v40 import PoseFlowV40Pipeline, PoseFlowV40PipelineConfig


class EMACleanEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.num_feature_levels = model.num_feature_levels
        for name in ('backbone', 'input_proj', 'input_mlp', 'time_embedding', 'clean_layers', 'clean_norm'):
            setattr(self, name, deepcopy(getattr(model, name)))
        self.level_embed = nn.Parameter(model.level_embed.detach().clone(), requires_grad=False)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, images, input_pixels, padding_mask=None):
        encoded_image = PoseFlowV26Model.encode_image(self, images, padding_mask)
        state = self.input_mlp(input_pixels/100.)
        time_features = self.time_embedding(state.new_zeros(state.shape[0]))
        for layer in self.clean_layers:
            state = layer(state, time_features, encoded_image)
        return self.clean_norm(state)

    @torch.no_grad()
    def update_from(self, model, *, decay):
        online_parameters = dict(model.named_parameters())
        for name, parameter in self.named_parameters():
            parameter.lerp_(online_parameters[name].detach(), 1-decay)
        online_buffers = dict(model.named_buffers())
        for name, buffer in self.named_buffers():
            value = online_buffers[name].detach()
            if buffer.is_floating_point():
                buffer.lerp_(value, 1-decay)
            else:
                buffer.copy_(value)


class PoseFlowV40EMAPipeline(PoseFlowV40Pipeline):
    def __init__(self, model, *, config, **kwargs):
        super().__init__(model, config=config, **kwargs)
        self.teacher = EMACleanEncoder(model)
        self.register_buffer('ema_updates', torch.zeros((), dtype=torch.long))

    def _clean_latent_target(self, batch, targets):
        return self.teacher(batch['image'], targets['input_pixels'], batch.get('padding_mask'))

    @torch.no_grad()
    def update_ema(self):
        self.teacher.update_from(self.model, decay=self.v26_config.ema_decay)
        self.ema_updates.add_(1)

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        self.update_ema()

    @torch.no_grad()
    def _record_target_diagnostics(self, clean, latent_target, targets):
        online = clean['pred_latents'].detach()
        for label, selected in (('positive', targets['positive']), ('background', ~targets['positive'])):
            if not selected.any():
                continue
            online_codes, teacher_codes = online[selected], latent_target[selected]
            metrics = dict(mse=(online_codes-teacher_codes).square().mean(),
                cosine=functional.cosine_similarity(online_codes, teacher_codes, dim=-1).mean(),
                online_mean=online_codes.mean(), teacher_mean=teacher_codes.mean(),
                online_rms=online_codes.square().mean().sqrt(), teacher_rms=teacher_codes.square().mean().sqrt())
            for name, value in metrics.items():
                key = f'ema/online_teacher/{label}/{name}'
                self._diagnostics[key] = value
                self._diagnostic_counts[key] = int(selected.sum())
        weights = self.v26_config.weights
        teacher_loss = box_task_loss(self.model.readout(latent_target), targets,
                                    weights=(weights.cls, weights.bbox, weights.giou))
        for name, value in teacher_loss.items():
            self._diagnostics[f'ema/teacher_decode/{name}'] = value
        self._diagnostics['ema/updates'] = self.ema_updates.float().clone()
        self._diagnostics['ema/decay'] = online.new_tensor(self.v26_config.ema_decay)


class PoseFlowV40EMAPipelineConfig(PoseFlowV40PipelineConfig):
    pipeline_class: ClassVar[type] = PoseFlowV40EMAPipeline
    ema_decay: float = Field(default=.999, ge=0., lt=1., allow_inf_nan=False)
