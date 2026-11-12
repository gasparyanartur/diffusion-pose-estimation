import torch
from torch import nn

from .model_v40 import PoseFlowV40Model


class PoseFlowV41Model(PoseFlowV40Model):
    sampler_kind = 'coordinate box DDIM eta0'

    def __init__(self, *, coordinate_scale=6., **kwargs):
        super().__init__(**kwargs)
        self.scale = coordinate_scale
        self.clean_norm = nn.Identity()

    def boxes_to_coordinates(self, boxes):
        return self.scale*(2*boxes-1)

    def coordinates_to_boxes(self, coordinates):
        return (coordinates.clamp(-self.scale, self.scale)/self.scale+1)/2

    def readout(self, embeddings):
        output = super().readout(embeddings)
        output['pred_embeddings'] = output.pop('pred_latents')
        return output

    def encode_coordinates(self, boxes, encoded_image):
        # B = images, Q = coordinate queries, D = feature width; state <B,Q,D>.
        state = self.input_mlp(boxes)
        features = self.time_embedding(state.new_zeros(len(state)))
        for layer in self.clean_layers:
            state = layer(state, features, encoded_image)
        return self.clean_norm(state)

    def encode_clean(self, boxes, encoded_image):
        return self.readout(self.encode_coordinates(boxes, encoded_image))

    def denoise(self, noisy_coordinates, timesteps, encoded_image, *, references):
        expected = (encoded_image[0].shape[0], self.num_proposals, 4)
        if noisy_coordinates.shape != expected or timesteps.shape != (expected[0],):
            raise ValueError('Coordinate/timestep shape disagrees with query budget')
        if ((timesteps < 0) | (timesteps >= self.num_timesteps)).any():
            raise ValueError('Timestep outside diffusion schedule')
        if references['boxes'].shape != expected or references['features'].shape != (*expected[:2], self.hidden_dim):
            raise ValueError('Reference shape disagrees with query budget')
        state = self.encode_coordinates(self.coordinates_to_boxes(noisy_coordinates), encoded_image)
        condition = references['features']+self.reference_box_embedding(references['boxes'])
        features = self.time_embedding(timesteps.to(state.dtype))
        for layer in self.denoiser_layers:
            state = layer(state, features, encoded_image, reference_features=condition)
        return self.readout(self.denoiser_projection(state))

    def training_outputs(self, images, clean_boxes, *, source_coordinates, timesteps, noise,
                         references=None, padding_mask=None, context=None):
        context = self.encode_context(images, padding_mask) if context is None else context
        if references is None:
            references = self.select_references(self.propose(context))
        clean = self.encode_clean(clean_boxes, context['image'])
        noisy = self.q_sample(source_coordinates, timesteps, noise)
        predicted = self.denoise(noisy, timesteps, context['image'], references=references)
        return clean, predicted, noisy, timesteps

    @torch.no_grad()
    def sample_keypoints(self, images, *, padding_mask=None, generator=None, num_proposals=None,
                         collect_step_diagnostics=False):
        if num_proposals is not None and num_proposals != self.num_proposals:
            raise ValueError('V41 preserves its fixed coordinate query budget')
        context = self.encode_context(images, padding_mask)
        candidates = self.propose(context)
        references = self.select_references(candidates)
        state = torch.randn(len(images), self.num_proposals, 4, device=images.device,
                            dtype=images.dtype, generator=generator)
        diagnostics = []
        for time, next_time in self.ddim_timestep_pairs():
            times = torch.full((len(images),), time, device=images.device, dtype=torch.long)
            output = self.denoise(state, times, context['image'], references=references)
            clean = self.boxes_to_coordinates(output['pred_boxes'])
            noise = self.predict_noise_from_start(state, times, clean).to(state.dtype)
            if collect_step_diagnostics:
                diagnostics.append(dict(time=time, next_time=next_time,
                    coordinate_rms=state.square().mean().sqrt().item(),
                    clamp_fraction=(state.abs() > self.scale).float().mean().item()))
            state = self.ddim_step(clean, noise, time, next_time, noise=torch.zeros_like(state))
        return dict(output, coordinate_state=state, coordinate_step_diagnostics=diagnostics,
            reference_boxes=references['boxes'], proposal_boxes=candidates['pred_boxes'],
            proposal_logits=candidates['pred_logits'])
