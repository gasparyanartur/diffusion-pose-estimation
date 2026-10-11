"""V13 latent-only sampling from Gaussian or controlled corrupted states."""

import torch

from .model_v12 import PoseFlowV12Model
from .latent_v12 import ddim_update


class PoseFlowV13Model(PoseFlowV12Model):
    @torch.no_grad()
    def sample_keypoints(self, images, *, padding_mask=None, num_proposals=None,
                         generator=None, collect_step_diagnostics=False,
                         initial_latents=None, start_time=1., sampling_steps=None,
                         encoded_image=None):
        if num_proposals is not None and num_proposals != self.num_proposals:
            raise ValueError('V13 preserves its configured group count')
        steps = self.flow_sampling_steps if sampling_steps is None else sampling_steps
        if not isinstance(steps, int) or steps < 1 or not 0 < start_time <= 1:
            raise ValueError('V13 requires positive steps and a start time in (0,1]')
        memory = self.encode_image(images, padding_mask) if encoded_image is None else encoded_image
        shape = (images.shape[0], self.num_proposals, self.num_body_points+3, self.hidden_dim)
        if initial_latents is None:
            if start_time != 1:
                raise ValueError('Partial-noise sampling requires explicit initial latents')
            state = torch.randn(shape, device=images.device, dtype=images.dtype, generator=generator)
        else:
            if initial_latents.shape != shape or not torch.isfinite(initial_latents).all():
                raise ValueError('V13 initial latents have invalid shape or values')
            if initial_latents.device != images.device or initial_latents.dtype != images.dtype:
                raise ValueError('V13 initial latents must match the image device and dtype')
            state = initial_latents.clone()
        diagnostics = []
        for step in range(steps):
            times = torch.full((images.shape[0],), start_time*(1-step/steps),
                               device=images.device, dtype=images.dtype)
            next_times = torch.full_like(times, start_time*(1-(step+1)/steps))
            output = self.denoise(state, times, memory, collect_diagnostics=collect_step_diagnostics)
            next_state = ddim_update(state, output['pred_latents'], times, next_times)
            if collect_step_diagnostics:
                stages = output['aux_outputs'] + [output]
                diagnostics.append({
                    'time':times[0].item(), 'latent_state':next_state.clone(),
                    'update_rms':(next_state-state).float().square().mean().sqrt(),
                    'layers':output['latent_diagnostics'],
                    'predictions':[{name:stage[name] for name in
                                    ('pred_logits','pred_boxes','pred_keypoints')} for stage in stages],
                })
            state = next_state
        return {**output, 'latent_state':state, 'latent_step_diagnostics':diagnostics}
