from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from ..diffusionregpose.transformer import inverse_sigmoid, sine_embedding_for_position


class ExpandedPoseDecoder(nn.Module):
    def __init__(self, native):
        super().__init__()
        decoder = native.transformer.decoder
        first = decoder.num_box_decoder_layers
        self.layers = deepcopy(decoder.layers[first:])
        self.norm = deepcopy(decoder.norm)
        self.ref_point_head = deepcopy(decoder.ref_point_head)
        self.keypoint_embed = deepcopy(decoder.keypoint_embed)
        self.hw = deepcopy(decoder.hw)
        self.expansion_head = deepcopy(decoder.expansion_pose_embed or decoder.pose_embed[0])
        self.box_heads = deepcopy(decoder.bbox_embed[first:])
        self.pose_heads = deepcopy(decoder.pose_embed)
        self.extent_heads = deepcopy(decoder.pose_hw_embed)
        self.width, self.points = decoder.d_model, decoder.num_body_points
        self.image_projections = nn.ModuleList(layer.cross_attn.value_proj for layer in self.layers)
        for layer in self.layers:
            layer.cross_attn.value_proj = nn.Identity()

    def project_memory(self, memory, padding_mask):
        return tuple(projection(memory).masked_fill(padding_mask[..., None], 0.)
                     for projection in self.image_projections)

    def expand(self, features, boxes):
        # B images, R independent rows, one new person and K joint queries.
        points = features+self.keypoint_embed.weight
        xy = (inverse_sigmoid(boxes[..., :2])+self.expansion_head(points)[..., :2]).sigmoid()
        extent = self.hw.weight.sigmoid()*boxes[..., 2:]
        return (torch.cat((features, points), -2),
                torch.cat((boxes, torch.cat((xy, extent), -1)), -2))

    def _layer(self, layer, state, references, prefix_mask, values, shapes, starts, ratios):
        batch, rows, queries, width = state.shape
        scaled = references[..., None, :]*torch.cat((ratios, ratios), -1)[:, None, None]
        position = self.ref_point_head(sine_embedding_for_position(scaled[..., 0, :], num_pos_feats=width//2))
        sequence = state.reshape(batch*rows, queries, width).transpose(0, 1)
        query = (state+position).reshape(batch*rows, queries, width).transpose(0, 1)
        mask = F.pad(prefix_mask, (self.points+1, 0)).reshape(batch*rows, queries)
        group_mask = torch.zeros(queries, queries, dtype=torch.bool, device=state.device)
        group_mask[1:self.points+1, self.points+1:] = True
        group_mask[self.points+1:, 1:self.points+1] = True
        attended = layer.self_attn(query, query, sequence, attn_mask=group_mask if group_mask.any() else None,
                                  key_padding_mask=mask if mask.any() else None)[0]
        state = layer.norm2(sequence+layer.dropout2(attended)).transpose(0, 1).reshape_as(state)
        crossed = layer.cross_attn((state+position).reshape(batch, rows*queries, width),
            scaled.reshape(batch, rows*queries, len(shapes), 4).contiguous(), values, shapes, starts)
        state = layer.norm1(state+layer.dropout1(crossed.reshape_as(state)))
        return layer.norm3(state+layer.dropout4(layer.dropout3(layer.linear2(F.relu(layer.linear1(state))))))

    def forward(self, features, boxes, prefix, prefix_boxes, prefix_mask, image_values, shapes, starts, ratios):
        state, references = self.expand(features, boxes)
        state = torch.cat((state, prefix), -2)
        references = torch.cat((references, prefix_boxes), -2)
        predictions = []
        for index, layer in enumerate(self.layers):
            state = self._layer(layer, state, references, prefix_mask, image_values[index], shapes, starts, ratios)
            readout = self.norm(state)
            box = (self.box_heads[index](readout[..., 0, :])+inverse_sigmoid(references[..., 0, :])).sigmoid()
            point_state = readout[..., 1:self.points+1, :]
            joints = (self.pose_heads[index](point_state)+inverse_sigmoid(references[..., 1:self.points+1, :2])).sigmoid()
            predictions.append(torch.cat((joints.flatten(-2), box), -1))
            raw_points = state[..., 1:self.points+1, :]
            next_box = (self.box_heads[index](state[..., :1, :])+inverse_sigmoid(references[..., :1, :])).sigmoid()
            delta = torch.cat((self.pose_heads[index](raw_points), self.extent_heads[index](raw_points)), -1)
            next_points = (delta+inverse_sigmoid(references[..., 1:self.points+1, :])).sigmoid()
            references = torch.cat((next_box, next_points, prefix_boxes), -2).detach()
        return torch.stack(predictions)
