import torch
from torch import nn
from torch.nn import functional as F

from ..diffusionregpose.transformer import sine_embedding_for_position
from .model_v21 import PoseFlowV21Model


class TimeAffine(nn.Module):
    def __init__(self, time_width, feature_width):
        super().__init__()
        self.affine = nn.Linear(time_width, 2*feature_width)
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(self, features, time_features):
        scale, shift = self.affine(time_features).to(features.dtype).chunk(2, -1)
        return features*(1+scale[:, None])+shift[:, None]


class PoseFlowV22Model(PoseFlowV21Model):
    def __init__(self, *, time_embedding_scale=999., coordinate_batch_layout='legacy', **kwargs):
        super().__init__(**kwargs)
        if coordinate_batch_layout not in ('legacy', 'per_image'):
            raise ValueError('Unknown V22 coordinate batch layout')
        self.coordinate_batch_layout = coordinate_batch_layout
        if self.hypothesis_update_order != 'h_then_g':
            raise ValueError('V22 requires H-first decoder updates')
        if not 0 < time_embedding_scale < float('inf'):
            raise ValueError('V22 embedding scale must be finite and positive')
        self.time_embedding_scale = float(time_embedding_scale)
        width = self.transformer.decoder.d_model
        self.time_bottleneck = nn.Sequential(nn.SiLU(), nn.Linear(4*width, width))
        self.dense_query_conditioner = TimeAffine(width, width)
        self.proposal_conditioner = TimeAffine(width, width)
        self.coordinate_conditioners = nn.ModuleDict({
            name: TimeAffine(width, width) for name in ('self_q', 'self_k', 'fusion_q', 'fusion_k')
        })
        self.conditioners = nn.ModuleList(nn.ModuleDict({
            name: TimeAffine(width, width) for name in (
                'h_attention', 'g_attention', 'h_attn_norm', 'h_ffn_norm',
                'g_attn_norm', 'g_image_norm', 'g_ffn_norm')
        }) for _ in self.transformer.decoder.layers)

    def time_features(self, timesteps):
        if timesteps is None:
            raise ValueError('V22 conditioning requires flow timesteps')
        return self.time_bottleneck(self.transformer.time_mlp(timesteps*self.time_embedding_scale))

    def encode_features(self, inputs, timesteps=None):
        scaled = None if timesteps is None else timesteps*self.time_embedding_scale
        return super().encode_features(inputs, scaled)

    def prepare_hypotheses(self, hypotheses, memory, position, inputs, timesteps):
        if inputs.padding_mask.all(-1).any():
            raise ValueError('V22 dense read requires valid image tokens')
        time_features = self.time_features(timesteps)
        query = self.dense_query_conditioner(self.dense_read.query_norm(hypotheses), time_features)
        message, _ = self.dense_read.attention(query, memory+position, memory,
            key_padding_mask=inputs.padding_mask, need_weights=False)
        hypotheses = hypotheses+self.dense_read.gate*self.dense_strength(timesteps)[:, None, None]*message
        hypotheses = self.transformer.enc_output_norm(self.transformer.enc_output(hypotheses))
        return self.proposal_conditioner(hypotheses, time_features)

    def refine_diffusion_coordinates(self, diffusion_xy, reference_xy, timesteps=None):
        decoder = self.transformer.decoder
        time_features = self.time_features(timesteps)
        groups, batch_size, joints, _ = diffusion_xy.shape
        diffusion_xy = diffusion_xy.detach()
        if self.coordinate_batch_layout == 'per_image':
            diffusion_xy = diffusion_xy.permute(1, 0, 2, 3)
        diffusion_xy = diffusion_xy.reshape(batch_size, groups*joints, 2)
        def inherited_projection(module, features):
            return F.linear(features, module.weight.detach(), module.bias.detach())
        query = self.coordinate_conditioners['self_q'](inherited_projection(decoder.emb_SQ, diffusion_xy), time_features)
        key = self.coordinate_conditioners['self_k'](inherited_projection(decoder.emb_SK, diffusion_xy), time_features)
        value = inherited_projection(decoder.emb_SV, diffusion_xy)
        attention = (query*(query.shape[1]**.5)) @ key.transpose(-2, -1)
        diffusion_xy = attention.softmax(-1) @ value
        reference_xy = reference_xy.detach().permute(2, 0, 1, 3).reshape(batch_size, groups*joints, 2)
        query = self.coordinate_conditioners['fusion_q'](inherited_projection(decoder.emb_Q, diffusion_xy+reference_xy), time_features)
        key = self.coordinate_conditioners['fusion_k'](inherited_projection(decoder.emb_K, reference_xy), time_features)
        value = inherited_projection(decoder.emb_V, reference_xy)
        attention = (query*(query.shape[1]**.5)) @ key.transpose(-2, -1)
        refined = attention.softmax(-1) @ value
        return refined.view(batch_size, groups, joints, 2).permute(1, 2, 0, 3)

    def detach_decoder_references(self, references, *, layer_index, dn_count):
        decoder = self.transformer.decoder
        if layer_index != decoder.num_box_decoder_layers-1:
            return references.detach()
        # Preserve only adapter-generated joint XY; inherited geometry stays detached.
        grouped = references[:, dn_count:].reshape(references.shape[0], self.num_queries,
                                                   decoder.num_body_points+1, 4)
        joints = torch.cat((grouped[:, :, 1:, :2], grouped[:, :, 1:, 2:].detach()), -1)
        normal = torch.cat((grouped[:, :, :1].detach(), joints), 2).flatten(1, 2)
        return torch.cat((references[:, :dn_count].detach(), normal), 1)

    def refine_with_hypotheses(self, layer, geometry, references, hypotheses, inputs, memory, mask, *, timesteps=None):
        decoder = self.transformer.decoder
        index = next(index for index, candidate in enumerate(decoder.layers) if candidate is layer)
        block = self.conditioners[index]
        time_features = self.time_features(timesteps)
        ratios = torch.cat((inputs.valid_ratios, inputs.valid_ratios), -1)
        scaled = references[:, :, None]*ratios[:, None]
        positions = decoder.ref_point_head(sine_embedding_for_position(
            scaled[:, :, 0], num_pos_feats=decoder.d_model//2))
        geometry_count = geometry.shape[1]
        h_attention = block['h_attention'](hypotheses, time_features)
        g_attention = block['g_attention'](geometry, time_features)
        keys = torch.cat((g_attention+positions, h_attention), 1).transpose(0, 1)
        values = torch.cat((geometry, hypotheses), 1).transpose(0, 1)
        attended = layer.self_attn(h_attention.transpose(0, 1), keys, values,
            attn_mask=mask[:, geometry_count:, :], need_weights=False)[0].transpose(0, 1)
        hypotheses = block['h_attn_norm'](layer.norm2(hypotheses+layer.dropout2(attended)), time_features)
        update = layer.dropout3(layer.linear2(torch.relu(layer.linear1(hypotheses))))
        hypotheses = block['h_ffn_norm'](layer.norm3(hypotheses+layer.dropout4(update)), time_features)

        # Geometry reads this fully updated H; the same tensor is returned for carry.
        h_attention = block['h_attention'](hypotheses, time_features)
        keys = torch.cat((g_attention+positions, h_attention), 1).transpose(0, 1)
        values = torch.cat((geometry, hypotheses), 1).transpose(0, 1)
        attended = layer.self_attn((g_attention+positions).transpose(0, 1), keys, values,
            attn_mask=mask[:, :geometry_count, :], need_weights=False)[0].transpose(0, 1)
        geometry = block['g_attn_norm'](layer.norm2(geometry+layer.dropout2(attended)), time_features)
        crossed = layer.cross_attn(geometry+positions, scaled, memory, inputs.spatial_shapes,
                                   inputs.level_start_index, inputs.padding_mask)
        geometry = block['g_image_norm'](layer.norm1(geometry+layer.dropout1(crossed)), time_features)
        update = layer.dropout3(layer.linear2(torch.relu(layer.linear1(geometry))))
        geometry = block['g_ffn_norm'](layer.norm3(geometry+layer.dropout4(update)), time_features)
        return geometry, hypotheses
