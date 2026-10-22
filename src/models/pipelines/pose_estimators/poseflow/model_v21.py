import torch
from torch import nn

from ..diffusionregpose.transformer import inverse_sigmoid, sine_embedding_for_position
from .model import PoseFlowModel


class HypothesisImageRead(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=0., batch_first=True)
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, hypotheses, memory, position, padding_mask, strength):
        if padding_mask.all(-1).any():
            raise ValueError('V21 dense read requires valid image tokens')
        message, _ = self.attention(self.query_norm(hypotheses), memory+position, memory,
                                    key_padding_mask=padding_mask, need_weights=False)
        return hypotheses+self.gate*strength[:, None, None]*message


class PoseFlowV21Model(PoseFlowModel):
    def __init__(self, *, hypothesis_update_order='parallel', **kwargs):
        super().__init__(**kwargs)
        if hypothesis_update_order not in ('parallel', 'h_then_g'):
            raise ValueError('Unknown V21 hypothesis update order')
        self.hypothesis_update_order = hypothesis_update_order
        if self.num_queries != self.num_proposals or self.num_queries != self.transformer.decoder.num_group:
            raise ValueError('V21 query, proposal and group counts must be equal')
        if self.transformer.decoder.num_box_decoder_layers != 1:
            raise ValueError('V21 requires one box-only decoder stage')
        width = self.transformer.decoder.d_model
        self.hypothesis_slots = nn.Parameter(torch.randn(1, self.num_queries, width)*.02)
        self.dense_read = HypothesisImageRead(width, self.transformer.decoder.num_heads)

    @staticmethod
    def dense_strength(times):
        return .1+.9*times

    def hypothesis_attention_mask(self, *, batch_size, dn_count, expanded, device, targets=None):
        decoder = self.transformer.decoder
        stride = decoder.num_body_points+1 if expanded else 1
        geometry_count = dn_count+self.num_queries*stride
        if expanded:
            base = (self.training_attention_mask(targets, batch_size, device) if dn_count
                    else self.inference_attention_mask(batch_size, device))
        else:
            base = (self.training_box_attention_mask(targets, batch_size, device) if dn_count
                    else torch.zeros(batch_size*decoder.num_heads, geometry_count, geometry_count,
                                     dtype=torch.bool, device=device))
        mask = torch.ones(batch_size*decoder.num_heads, geometry_count+self.num_queries,
                          geometry_count+self.num_queries, dtype=torch.bool, device=device)
        mask[:, :geometry_count, :geometry_count] = base
        for candidate in range(self.num_queries):
            start = dn_count+candidate*stride
            hypothesis = geometry_count+candidate
            mask[:, start:start+stride, hypothesis] = False
            mask[:, hypothesis, start:start+stride] = False
            mask[:, hypothesis, hypothesis] = False
        return mask

    def refine_with_hypotheses(self, layer, geometry, references, hypotheses, inputs, memory, mask, *, timesteps=None):
        if self.hypothesis_update_order == 'h_then_g':
            return self.refine_hypotheses_then_geometry(
                layer, geometry, references, hypotheses, inputs, memory, mask)
        decoder = self.transformer.decoder
        ratios = torch.cat((inputs.valid_ratios, inputs.valid_ratios), -1)
        scaled = references[:, :, None]*ratios[:, None]
        positions = decoder.ref_point_head(sine_embedding_for_position(
            scaled[:, :, 0], num_pos_feats=decoder.d_model//2))
        # Geometry and H occupy separate ranges; only geometry has spatial references.
        geometry_count = geometry.shape[1]
        tokens = torch.cat((geometry, hypotheses), 1).transpose(0, 1)
        position = torch.cat((positions, torch.zeros_like(hypotheses)), 1).transpose(0, 1)
        attended = layer.self_attn(tokens+position, tokens+position, tokens, attn_mask=mask,
                                    need_weights=False)[0]
        tokens = layer.norm2(tokens+layer.dropout2(attended))
        geometry, hypotheses = tokens[:geometry_count].transpose(0, 1), tokens[geometry_count:].transpose(0, 1)
        crossed = layer.cross_attn(geometry+positions, scaled, memory, inputs.spatial_shapes,
                                   inputs.level_start_index, inputs.padding_mask)
        geometry = layer.norm1(geometry+layer.dropout1(crossed))
        tokens = torch.cat((geometry, hypotheses), 1)
        update = layer.dropout3(layer.linear2(torch.relu(layer.linear1(tokens))))
        tokens = layer.norm3(tokens+layer.dropout4(update))
        return tokens[:, :geometry_count], tokens[:, geometry_count:]

    def refine_hypotheses_then_geometry(self, layer, geometry, references, hypotheses, inputs, memory, mask):
        decoder = self.transformer.decoder
        ratios = torch.cat((inputs.valid_ratios, inputs.valid_ratios), -1)
        scaled = references[:, :, None]*ratios[:, None]
        positions = decoder.ref_point_head(sine_embedding_for_position(
            scaled[:, :, 0], num_pos_feats=decoder.d_model//2))
        geometry_count = geometry.shape[1]
        tokens = torch.cat((geometry, hypotheses), 1).transpose(0, 1)
        position = torch.cat((positions, torch.zeros_like(hypotheses)), 1).transpose(0, 1)

        # Finish the H write before geometry reads the exact state returned for carry.
        attended = layer.self_attn(hypotheses.transpose(0, 1), tokens+position, tokens,
            attn_mask=mask[:, geometry_count:, :], need_weights=False)[0].transpose(0, 1)
        hypotheses = layer.norm2(hypotheses+layer.dropout2(attended))
        update = layer.dropout3(layer.linear2(torch.relu(layer.linear1(hypotheses))))
        hypotheses = layer.norm3(hypotheses+layer.dropout4(update))

        tokens = torch.cat((geometry, hypotheses), 1).transpose(0, 1)
        attended = layer.self_attn((geometry+positions).transpose(0, 1), tokens+position, tokens,
            attn_mask=mask[:, :geometry_count, :], need_weights=False)[0].transpose(0, 1)
        geometry = layer.norm2(geometry+layer.dropout2(attended))
        crossed = layer.cross_attn(geometry+positions, scaled, memory, inputs.spatial_shapes,
                                   inputs.level_start_index, inputs.padding_mask)
        geometry = layer.norm1(geometry+layer.dropout1(crossed))
        update = layer.dropout3(layer.linear2(torch.relu(layer.linear1(geometry))))
        geometry = layer.norm3(geometry+layer.dropout4(update))
        return geometry, hypotheses

    def refine_diffusion_coordinates(self, diffusion_xy, reference_xy, timesteps=None):
        return self.transformer.decoder._refine_diffusion_keypoints(diffusion_xy, reference_xy)

    def prepare_hypotheses(self, hypotheses, memory, position, inputs, timesteps):
        hypotheses = self.dense_read(hypotheses, memory, position, inputs.padding_mask, self.dense_strength(timesteps))
        return self.transformer.enc_output_norm(self.transformer.enc_output(hypotheses))

    def detach_decoder_references(self, references, *, layer_index, dn_count):
        return references.detach()

    def expand_ordered(self, features, references, diffusion_xy, *, dn_count, timesteps=None):
        decoder = self.transformer.decoder
        boxes, box_references = features[:, dn_count:], references[:, dn_count:]
        joints = decoder.num_body_points
        point_features = boxes[:, :, None]+decoder.keypoint_embed.weight[None, None, :joints]
        pose_head = decoder.expansion_pose_embed or decoder.pose_embed[0]
        point_xy = (inverse_sigmoid(box_references[:, :, None, :2])+pose_head(point_features)).sigmoid()
        point_wh = decoder.hw.weight[None, None, :joints].sigmoid()*box_references[:, :, None, 2:]
        point_xy = self.refine_diffusion_coordinates(
            diffusion_xy.reshape(features.shape[0], self.num_queries, joints, 2).permute(1, 0, 2, 3),
            point_xy.permute(1, 2, 0, 3), timesteps).permute(2, 0, 1, 3)
        expanded = torch.cat((boxes[:, :, None], point_features), 2).flatten(1, 2)
        expanded_references = torch.cat((box_references[:, :, None],
                                        torch.cat((point_xy, point_wh), -1)), 2).flatten(1, 2)
        return (torch.cat((features[:, :dn_count], expanded), 1),
                torch.cat((references[:, :dn_count], expanded_references), 1))

    def prediction(self, index, features, references, *, dn_count, expanded):
        decoder = self.transformer.decoder
        features = decoder.norm(features)
        normal, normal_references = features[:, dn_count:], references[:, dn_count:]
        if expanded:
            stride = decoder.num_body_points+1
            grouped = normal.reshape(normal.shape[0], self.num_queries, stride, -1)
            grouped_references = normal_references.reshape(normal.shape[0], self.num_queries, stride, 4)
            box_features, box_references = grouped[:, :, 0], grouped_references[:, :, 0]
            pose_head = self.pose_embed[index-decoder.num_box_decoder_layers]
            points = (pose_head(grouped[:, :, 1:])+inverse_sigmoid(grouped_references[:, :, 1:, :2])).sigmoid()
            visibility = torch.ones_like(points[..., 0]).sigmoid()
            keypoints = torch.cat((points.flatten(-2), visibility), -1)
        else:
            box_features, box_references = normal, normal_references
            keypoints = normal.new_zeros(normal.shape[0], self.num_queries, 3*decoder.num_body_points)
        output = {'pred_logits':self.class_embed[index](box_features),
                  'pred_boxes':(self.bbox_embed[index](box_features)+inverse_sigmoid(box_references)).sigmoid(),
                  'pred_keypoints':keypoints}
        denoising = None
        if dn_count:
            denoising = {'pred_logits':self.class_embed[index](features[:, :dn_count]),
                         'pred_boxes':(self.bbox_embed[index](features[:, :dn_count])
                                       +inverse_sigmoid(references[:, :dn_count])).sigmoid()}
        return output, denoising

    def update_pose_references(self, index, geometry, references, dn_count):
        decoder = self.transformer.decoder
        stride = decoder.num_body_points+1
        grouped = geometry[:, dn_count:].reshape(geometry.shape[0], self.num_queries, stride, -1)
        previous = references[:, dn_count:].reshape(geometry.shape[0], self.num_queries, stride, 4)
        boxes = (self.bbox_embed[index](grouped[:, :, 0])+inverse_sigmoid(previous[:, :, 0])).sigmoid()
        pose_index = index-decoder.num_box_decoder_layers
        joints = torch.cat((self.pose_embed[pose_index](grouped[:, :, 1:]),
                            self.pose_hw_embed[pose_index](grouped[:, :, 1:])), -1)
        joints = (joints+inverse_sigmoid(previous[:, :, 1:])).sigmoid()
        normal = torch.cat((boxes[:, :, None], joints), 2).flatten(1, 2)
        denoising = (self.bbox_embed[index](geometry[:, :dn_count])
                     +inverse_sigmoid(references[:, :dn_count])).sigmoid()
        return torch.cat((denoising, normal), 1)

    def forward(self, images, diffusion_keypoints, timesteps, targets=None, padding_mask=None,
                *, hypothesis_memory=None, collect_codebook_diagnostics=False,
                collect_codebook_attention=False):
        if padding_mask is None:
            padding_mask = torch.zeros(images.shape[0], *images.shape[-2:], device=images.device, dtype=torch.bool)
        if (timesteps.shape != images.shape[:1] or not torch.isfinite(timesteps).all()
                or ((timesteps < 0) | (timesteps > 1)).any()):
            raise ValueError('V21 flow times must be finite and in [0,1]')
        expected = (images.shape[0], self.num_queries, self.transformer.decoder.d_model)
        hypotheses = self.hypothesis_slots.expand(expected) if hypothesis_memory is None else hypothesis_memory
        if (hypotheses.shape != expected or hypotheses.device != images.device
                or hypotheses.dtype != images.dtype or not torch.isfinite(hypotheses).all()):
            raise ValueError('V21 hypothesis memory must match batch, candidates, device and dtype')
        if diffusion_keypoints.shape != (images.shape[0], self.num_queries, 2*self.transformer.decoder.num_body_points):
            raise ValueError('V21 joint flow state shape disagrees with candidate count')
        if padding_mask.flatten(1).all(-1).any():
            raise ValueError('V21 requires valid image pixels')
        if not torch.isfinite(diffusion_keypoints).all():
            raise ValueError('V21 joint flow state must be finite')
        inputs = self.prepare_transformer_inputs(images, padding_mask)
        memory = self.encode_features(inputs, timesteps)
        position = inputs.position+self._level_position_offsets(inputs)
        hypotheses = self.prepare_hypotheses(hypotheses, memory, position, inputs, timesteps)
        initial_logits = self.transformer.enc_out_class_embed(hypotheses)
        initial_box_logits = self.transformer.enc_out_bbox_embed(hypotheses)
        geometry, references = hypotheses.detach(), initial_box_logits.detach().sigmoid()
        dn_count, dn_state = 0, None
        if self.training and targets is not None and self.transformer.decoder.num_dn:
            dn_features, dn_references, dn_state = self._prepare_denoising_queries(targets, images.device)
            dn_count = dn_features.shape[1]
            geometry = torch.cat((dn_features, geometry), 1)
            references = torch.cat((dn_references.sigmoid(), references), 1)
        predictions, denoising = [], []
        decoder = self.transformer.decoder
        for index, layer in enumerate(decoder.layers):
            expanded = index >= decoder.num_box_decoder_layers
            mask = self.hypothesis_attention_mask(batch_size=images.shape[0], dn_count=dn_count,
                expanded=expanded, device=images.device, targets=targets)
            geometry, hypotheses = self.refine_with_hypotheses(
                layer, geometry, references, hypotheses, inputs, memory, mask, timesteps=timesteps)
            prediction, dn_prediction = self.prediction(index, geometry, references,
                                                         dn_count=dn_count, expanded=expanded)
            predictions.append(prediction)
            if dn_prediction is not None:
                denoising.append(dn_prediction)
            if not expanded:
                references = (self.bbox_embed[index](geometry)+inverse_sigmoid(references)).sigmoid()
                geometry, references = self.expand_ordered(geometry, references,
                    (diffusion_keypoints/self.scale+1)/2, dn_count=dn_count, timesteps=timesteps)
            else:
                references = self.update_pose_references(index, geometry, references, dn_count)
            references = self.detach_decoder_references(references, layer_index=index, dn_count=dn_count)
        result = {**predictions[-1], 'aux_outputs':predictions[:-1], 'hypothesis_memory':hypotheses}
        if self.training:
            result['interm_outputs'] = {'pred_logits':initial_logits, 'pred_boxes':initial_box_logits.sigmoid(),
                'pred_keypoints':initial_logits.new_zeros(images.shape[0], self.num_queries, 3*decoder.num_body_points)}
        if dn_state is not None:
            result.update(dn_class_input=dn_state['labels'], dn_bbox_input=dn_state['boxes'],
                          dn_class_pred=denoising[-1]['pred_logits'], dn_bbox_pred=denoising[-1]['pred_boxes'],
                          dn_aux_outputs=denoising[:-1])
        return result

    @torch.no_grad()
    def sample_keypoints(self, images, *, padding_mask=None, num_proposals=None, generator=None,
                        persist_hypotheses=True):
        if not isinstance(self.flow_sampling_steps, int) or self.flow_sampling_steps < 1:
            raise ValueError('V21 sampling requires a positive integer call count')
        if num_proposals is not None and num_proposals != self.num_proposals:
            raise ValueError('V21 sampling requires the configured fixed candidate count')
        source = torch.randn(images.shape[0], self.num_proposals, 2*self.transformer.decoder.num_body_points,
                             device=images.device, dtype=images.dtype, generator=generator)
        state, memory = source.clone(), None
        for index in range(self.flow_sampling_steps):
            times = images.new_full(images.shape[:1], index/self.flow_sampling_steps)
            output = self(images, state, times, padding_mask=padding_mask, hypothesis_memory=memory)
            endpoint = (output['pred_keypoints'][..., :source.shape[-1]]*2-1)*self.scale
            state = state+(endpoint-source)/self.flow_sampling_steps
            memory = output['hypothesis_memory'] if persist_hypotheses else None
        return {**output, 'flow_keypoints':state}
