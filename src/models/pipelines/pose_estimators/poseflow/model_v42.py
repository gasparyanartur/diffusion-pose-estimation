from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F

from ..diffusionregpose.model import DiffusionRegPoseModel
from ..diffusionregpose.transformer import (
    MLP, MultiScaleDeformableAttention, inverse_sigmoid, sine_embedding_for_position,
)


class PoseFlowV42Model(DiffusionRegPoseModel):
    def __init__(self, *, native_state=None, pose_readout_mode='absolute',
                 confidence_mode='pose_conditioned', **kwargs):
        if pose_readout_mode not in ('absolute', 'box_center_residual'):
            raise ValueError('Unknown V42 pose readout mode')
        if confidence_mode not in ('pose_conditioned', 'image_only'):
            raise ValueError('Unknown V42 confidence mode')
        super().__init__(**kwargs)
        self.pose_readout_mode = pose_readout_mode
        self.confidence_mode = confidence_mode
        if native_state is not None:
            self.load_state_dict(native_state, strict=True)
        decoder = self.transformer.decoder
        if not 1 <= decoder.num_box_decoder_layers < decoder.num_layers:
            raise ValueError('V42 requires box selection followed by pose refinement')
        if self.num_proposals != decoder.num_group or self.num_proposals > self.num_queries:
            raise ValueError('V42 requires proposals == groups <= queries')
        width = decoder.d_model
        points = decoder.num_body_points
        self.pose_encoder = MLP(2 * points, width, width, 2)
        self.pose_readers = nn.ModuleList([
            MLP(width, width, 2 * points, 3)
            for _ in range(decoder.num_layers - decoder.num_box_decoder_layers)
        ])
        self.person_attention = nn.ModuleList()
        for layer in decoder.layers[decoder.num_box_decoder_layers:]:
            original = layer.cross_attn
            attention = MultiScaleDeformableAttention(
                width, original.num_levels, original.num_heads,
                original.num_points * (points + 1),
            )
            attention.value_proj.load_state_dict(original.value_proj.state_dict())
            attention.output_proj.load_state_dict(original.output_proj.state_dict())
            self.person_attention.append(attention)
        # These native modules implement joint expansion and are absent in V42.
        for name in ('pose_embed', 'pose_hw_embed', 'expansion_pose_embed'):
            delattr(self, name)
            delattr(decoder, name)
        for name in ('hw', 'keypoint_embed', 'emb_Q', 'emb_K', 'emb_V', 'emb_SQ', 'emb_SK', 'emb_SV'):
            delattr(decoder, name)
        if confidence_mode == 'image_only':
            self.image_confidence_head = deepcopy(self.class_embed[decoder.num_box_decoder_layers - 1])

    def read_pose(self, features, reference_boxes, *, layer_index):
        coordinate_logits = self.pose_readers[layer_index](features)
        if self.pose_readout_mode == 'box_center_residual':
            center_logits = inverse_sigmoid(reference_boxes[..., :2])
            coordinate_logits = coordinate_logits + center_logits.repeat(1, 1, self.transformer.decoder.num_body_points)
        return coordinate_logits.sigmoid()

    def _person_layer(self, layer, attention, features, position, references, memory, inputs, mask, dn_count):
        query = features + position
        attended = layer.self_attn(query, query, features, attn_mask=mask)[0]
        features = layer.norm2(features + layer.dropout2(attended))
        query = (features + position).transpose(0, 1)
        references = references.transpose(0, 1).contiguous()
        arguments = (memory, inputs.spatial_shapes, inputs.level_start_index, inputs.padding_mask)
        person_reads = attention(query[:, dn_count:], references[:, dn_count:], *arguments)
        if dn_count:
            dn_reads = layer.cross_attn(query[:, :dn_count], references[:, :dn_count], *arguments)
            person_reads = torch.cat((dn_reads, person_reads), dim=1)
        features = layer.norm1(features + layer.dropout1(person_reads.transpose(0, 1)))
        return layer.norm3(features + layer.dropout4(layer.dropout3(layer.linear2(F.relu(layer.linear1(features))))))

    def forward(self, images, diffusion_keypoints, timesteps, targets=None, padding_mask=None, *, pose_input_builder=None):
        decoder = self.transformer.decoder
        expected = (images.shape[0], self.num_proposals, decoder.num_body_points * 2)
        if tuple(diffusion_keypoints.shape) != expected:
            raise ValueError(f'V42 noisy pose shape must be {expected}')
        if padding_mask is None:
            padding_mask = torch.zeros(images.shape[0], *images.shape[2:], dtype=torch.bool, device=images.device)
        inputs = self.prepare_transformer_inputs(images, padding_mask)
        memory = self.encode_features(inputs, timesteps)
        features, reference_logits, proposal_logits, proposal_boxes = self.select_two_stage_queries(memory, inputs)
        memory = self.decoder_feature_memory(inputs, memory)
        dn_state = None
        dn_count = 0
        mask = None
        if self.training and targets is not None:
            dn_features, dn_references, dn_state = self._prepare_denoising_queries(targets, images.device)
            dn_count = decoder.num_dn
            features = torch.cat((dn_features, features), dim=1)
            reference_logits = torch.cat((dn_references, reference_logits), dim=1)
            mask = self.training_box_attention_mask(targets, images.shape[0], images.device)
        # L = queries, B = images, D = person feature width.
        features = features.transpose(0, 1)  # <L, B, D>
        references = reference_logits.sigmoid().transpose(0, 1)
        predictions, dn_predictions = [], []
        final_pose_features = None
        pose_target_matches = None
        image_confidence = None
        for index, layer in enumerate(decoder.layers):
            scaled_references = references[:, :, None] * torch.cat((inputs.valid_ratios, inputs.valid_ratios), -1)[None]
            position = decoder.ref_point_head(sine_embedding_for_position(
                scaled_references[:, :, 0], num_pos_feats=decoder.d_model // 2,
            ))
            if index < decoder.num_box_decoder_layers:
                features = layer(features, tgt_query_pos=position,
                    tgt_reference_points=scaled_references, memory=memory,
                    memory_level_start_index=inputs.level_start_index,
                    memory_spatial_shapes=inputs.spatial_shapes,
                    memory_key_padding_mask=inputs.padding_mask, self_attn_mask=mask)
            else:
                features = self._person_layer(layer, self.person_attention[index-decoder.num_box_decoder_layers],
                    features, position, scaled_references, memory, inputs, mask, dn_count)
            normalized = decoder.norm(features).transpose(0, 1)
            input_boxes = references.transpose(0, 1)
            all_boxes = (self.bbox_embed[index](normalized) + inverse_sigmoid(input_boxes)).sigmoid()
            all_logits = self.class_embed[index](normalized)
            normal_features = normalized[:, dn_count:]
            keypoints = normal_features.new_zeros(*normal_features.shape[:2], decoder.num_body_points * 3)
            if index >= decoder.num_box_decoder_layers:
                xy = self.read_pose(normal_features, input_boxes[:, dn_count:],
                    layer_index=index-decoder.num_box_decoder_layers)
                visibility = xy.new_ones(*xy.shape[:2], decoder.num_body_points).sigmoid()
                keypoints = torch.cat((xy, visibility), dim=-1)
                final_pose_features = normal_features
            normal_logits = all_logits[:, dn_count:] if image_confidence is None else image_confidence
            predictions.append(dict(pred_logits=normal_logits, pred_boxes=all_boxes[:, dn_count:], pred_keypoints=keypoints))
            if dn_count:
                dn_predictions.append(dict(pred_logits=all_logits[:, :dn_count], pred_boxes=all_boxes[:, :dn_count]))
            # Native reference updates use unnormalized features and stop their gradient.
            references = (self.bbox_embed[index](features) + inverse_sigmoid(references)).sigmoid().detach()
            if index == decoder.num_box_decoder_layers - 1:
                selection_logits = self.class_embed[index](features[dn_count:])
                scores = selection_logits.max(-1).values
                selected = scores.topk(self.num_proposals, dim=0).indices
                person_features = torch.gather(features[dn_count:], 0, selected[..., None].expand(-1, -1, decoder.d_model))
                person_boxes = torch.gather(references[dn_count:], 0, selected[..., None].expand(-1, -1, 4))
                if self.confidence_mode == 'image_only':
                    image_features = normalized[:, dn_count:].gather(1,
                        selected.transpose(0, 1)[..., None].expand(-1, -1, decoder.d_model))
                    image_confidence = self.image_confidence_head(image_features)
                if pose_input_builder is not None:
                    selected_logits = torch.gather(selection_logits, 0,
                        selected[..., None].expand(-1, -1, self.num_classes))
                    diffusion_keypoints, pose_target_matches = pose_input_builder(
                        selected_logits.transpose(0, 1), person_boxes.transpose(0, 1), timesteps, targets,
                    )
                    if tuple(diffusion_keypoints.shape) != expected:
                        raise ValueError(f'V42 pose-input builder must return coordinates shaped {expected}')
                noisy_xy = ((diffusion_keypoints / self.scale) + 1) / 2
                person_features = person_features + self.pose_encoder(noisy_xy).transpose(0, 1)
                features = torch.cat((features[:dn_count], person_features), dim=0)
                references = torch.cat((references[:dn_count], person_boxes), dim=0)
                if mask is not None:
                    mask = mask[:, :dn_count+self.num_proposals, :dn_count+self.num_proposals]
        result = {**predictions[-1], 'aux_outputs': predictions[:-1], 'pose_features': final_pose_features}
        if getattr(self, 'return_pose_references', False):
            result['pose_reference_boxes'] = input_boxes[:, dn_count:].detach()
        if pose_target_matches is not None:
            result['pose_target_matches'] = pose_target_matches
        if self.training:
            result['interm_outputs'] = dict(pred_logits=proposal_logits, pred_boxes=proposal_boxes,
                pred_keypoints=proposal_boxes.new_zeros(*proposal_boxes.shape[:2], decoder.num_body_points * 3))
        if dn_state is not None:
            result.update(dn_class_input=dn_state['labels'], dn_bbox_input=dn_state['boxes'],
                dn_class_pred=dn_predictions[-1]['pred_logits'], dn_bbox_pred=dn_predictions[-1]['pred_boxes'],
                dn_aux_outputs=dn_predictions[:-1])
        return result
