from dataclasses import dataclass
from functools import partial

import torch
from torch import Tensor, nn

from .model import PoseFlowModel


@dataclass
class HypothesisContext:
    memory: Tensor
    position: Tensor
    padding_mask: Tensor
    hypotheses: Tensor | None = None


class ResidualHypothesisBranch(nn.Module):
    def __init__(self, width: int, heads: int, feedforward_dim: int, joints: int, stages: int):
        super().__init__()
        self.stride = joints + 1
        self.image_attention = nn.MultiheadAttention(width, heads, dropout=0., batch_first=True)
        self.write_attention = nn.MultiheadAttention(width, heads, dropout=0., batch_first=True)
        self.read_attention = nn.MultiheadAttention(width, heads, dropout=0., batch_first=True)
        self.query_norm = nn.LayerNorm(width)
        self.write_norm = nn.LayerNorm(width)
        self.output_norm = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(nn.Linear(width, feedforward_dim), nn.ReLU(),
                                         nn.Linear(feedforward_dim, width))
        self.gates = nn.Parameter(torch.zeros(stages))

    def forward(self, features: Tensor, *, layer_index: int, denoising_query_count: int = 0,
                decoder_context: HypothesisContext, **unused) -> Tensor:
        context = decoder_context
        if context.padding_mask.all(-1).any():
            raise ValueError('Residual h requires valid image tokens')
        normal = features[denoising_query_count:].transpose(0, 1)
        batch, length, width = normal.shape
        if length == 0 or length % self.stride:
            raise ValueError('Residual h requires complete person geometry groups')
        # B images, G candidates, S box/joint tokens, D channels.
        grouped = normal.reshape(batch, -1, self.stride, width)  # <B,G,S,D>
        groups = grouped.shape[1]
        if context.hypotheses is None:
            boxes = grouped[:, :, 0]
            image_message = self.image_attention(self.query_norm(boxes),
                context.memory + context.position, context.memory,
                key_padding_mask=context.padding_mask, need_weights=False)[0]
            context.hypotheses = boxes + image_message
        hypotheses = context.hypotheses.reshape(batch * groups, 1, width)
        geometry = grouped.reshape(batch * groups, self.stride, width)
        message = self.write_attention(self.query_norm(hypotheses), geometry, geometry,
                                       need_weights=False)[0]
        hypotheses = self.write_norm(hypotheses + message)
        hypotheses = self.output_norm(hypotheses + self.feedforward(hypotheses))
        context.hypotheses = hypotheses.reshape(batch, groups, width)
        local_memory = torch.cat((geometry, hypotheses), 1)
        message = self.read_attention(geometry, local_memory, local_memory,
                                      need_weights=False)[0]
        normal = (geometry + self.gates[layer_index] * message).reshape(batch, length, width)
        return torch.cat((features[:denoising_query_count], normal.transpose(0, 1)), 0)


class PoseFlowV24Model(PoseFlowModel):
    def __init__(self, *, h_enabled=False, **kwargs):
        super().__init__(**kwargs)
        self.h_enabled = bool(h_enabled)
        self.coordinate_batch_layout = 'per_image'
        self.transformer.decoder.coordinate_batch_layout = self.coordinate_batch_layout
        self.hypothesis_update_order = 'within_call_residual' if h_enabled else 'no_h'
        self.h_branch = None
        decoder = self.transformer.decoder
        pose_stages = decoder.num_layers - decoder.num_box_decoder_layers
        if self.flow_sampling_steps != 1 or pose_stages < 1:
            raise ValueError('V24 requires one call and at least one pose stage')
        if h_enabled:
            # The extra initialization must not perturb the paired control's RNG.
            with torch.random.fork_rng(devices=[]):
                self.h_branch = ResidualHypothesisBranch(decoder.d_model, decoder.num_heads,
                    self.transformer.decoder.layers[0].linear1.out_features,
                    decoder.num_body_points, pose_stages)

    def encode_features(self, inputs, timesteps=None):
        memory = super().encode_features(inputs, timesteps)
        if self.h_enabled:
            inputs.decoder_context = HypothesisContext(memory,
                inputs.position + self._level_position_offsets(inputs), inputs.padding_mask)
        return memory

    def post_image_attention_by_layer(self):
        if self.h_branch is None:
            return None
        decoder = self.transformer.decoder
        return {index: partial(self.h_branch, layer_index=index-decoder.num_box_decoder_layers)
                for index in range(decoder.num_box_decoder_layers, decoder.num_layers)}

    def sample_keypoints(self, images, *, persist_hypotheses=True, **kwargs):
        if self.flow_sampling_steps != 1:
            raise ValueError('V24 experiment permits exactly one call')
        return super().sample_keypoints(images, **kwargs)
