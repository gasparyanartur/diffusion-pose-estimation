import torch
from torch import nn

from .expanded_pose_v53 import ExpandedPoseDecoder
from .model_v50 import ProposalFrontend
from .proposal_context_v55 import ProposalContext


class FullDepthBirthDecoder(nn.Module):
    """Takes ownership of a complete native model; decodes supplied proposal IDs.

    Prefixes are encoded features [B, R, Q, D], not target coordinates. This
    adapter has no selector, EOS policy, sampling loop or training recipe.
    """

    def __init__(self, *, native):
        super().__init__()
        decoder = native.transformer.decoder
        if (len(native.transformer.encoder.layers) != 6 or len(decoder.layers) != 6
                or decoder.num_box_decoder_layers != 2 or decoder.num_body_points != 14):
            raise ValueError('Expected full-depth native six-encoder/two-box/four-pose model')
        self.pose_decoder = ExpandedPoseDecoder(native)
        if decoder.expansion_pose_embed is None:
            self.pose_decoder.expansion_head = self.pose_decoder.pose_heads[0]
        self.proposal_context = ProposalContext(decoder.d_model, decoder.num_heads)
        self.frontend = ProposalFrontend(native)

    def prepare_context(self, images, padding_mask=None):
        context = self.frontend.prepare_context(images, padding_mask)
        context['image_values'] = self.pose_decoder.project_memory(
            context['memory'], context['inputs'].padding_mask)
        return context

    def _selected_features(self, context, indices, valid, prefix, prefix_valid):
        batch = torch.arange(len(indices), device=indices.device)[:, None]
        safe_indices = indices.masked_fill(~valid, 0)
        features = context['features'][batch, safe_indices]
        boxes = context['boxes'][batch, safe_indices]
        if prefix is not None:
            batch_size, rows, width = features.shape
            flat_features = features.reshape(batch_size*rows, 1, width)
            flat_prefix = prefix.reshape(batch_size*rows, 1, prefix.shape[2], width)
            flat_valid = (prefix_valid & valid[..., None]).reshape(batch_size*rows, 1, -1)
            corrected = self.proposal_context(flat_features, flat_prefix, flat_valid)[:, 0, 0]
            correction = corrected.reshape_as(features)-features.detach()
            features = features+correction
        return features, boxes, batch, safe_indices

    def _validate_inputs(self, context, indices, valid_mask, prefix, prefix_valid):
        if (indices.ndim != 2 or indices.dtype != torch.long
                or indices.shape[0] != len(context['features'])
                or indices.device != context['features'].device):
            raise ValueError('Expected proposal indices [B,R] on context device')
        valid = torch.ones_like(indices, dtype=torch.bool) if valid_mask is None else valid_mask
        if (valid.shape != indices.shape or valid.dtype != torch.bool or valid.device != indices.device
                or ((indices[valid] < 0) | (indices[valid] >= context['features'].shape[1])).any()):
            raise ValueError('Invalid proposal indices or valid_mask')
        if prefix is None:
            if prefix_valid is not None:
                raise ValueError('prefix_valid requires prefix features')
        elif (prefix.ndim != 4 or prefix.shape[:2] != indices.shape
                or prefix.shape[-1] != self.pose_decoder.width
                or prefix.device != indices.device or prefix.dtype != context['features'].dtype
                or prefix_valid is None or prefix_valid.shape != prefix.shape[:-1]
                or prefix_valid.dtype != torch.bool or prefix_valid.device != indices.device
                or not torch.isfinite(prefix[prefix_valid & valid[..., None]]).all()):
            raise ValueError('Invalid prefix features or prefix_valid mask')
        return valid

    def decode_proposals(self, context, proposal_indices, *, valid_mask=None,
                         prefix=None, prefix_valid=None, collect_trace=False):
        # B images, R independent supplied proposals, each expanded to 15 queries.
        valid = self._validate_inputs(context, proposal_indices, valid_mask, prefix, prefix_valid)
        batch_size, rows = proposal_indices.shape
        if rows:
            features, boxes, batch, safe = self._selected_features(
                context, proposal_indices, valid, prefix, prefix_valid)
            empty = features.new_zeros(batch_size, rows, 0, self.pose_decoder.width)
            empty_boxes = boxes.new_zeros(batch_size, rows, 0, 4)
            empty_mask = torch.zeros(batch_size, rows, 0, dtype=torch.bool, device=features.device)
            inputs = context['inputs']
            layers = self.pose_decoder(features[:, :, None], boxes[:, :, None], empty, empty_boxes,
                empty_mask, context['image_values'], inputs.spatial_shapes,
                inputs.level_start_index, inputs.valid_ratios)
            layers = layers.masked_fill(~valid[None, ..., None], 0.)
            logits = context['pred_logits'][batch, safe].masked_fill(~valid[..., None], -30.)
            query_indices = context['indices'][batch, safe].masked_fill(~valid, -1)
        else:
            layers = context['features'].new_zeros(4, batch_size, 0, 32)
            logits = context['pred_logits'][:, :0]
            query_indices = context['indices'][:, :0]
        coordinates = layers[-1]
        scores = logits[..., 1].sigmoid().masked_fill(~valid, 0.)
        existence = torch.logit(scores.clamp(1e-8, 1-1e-7)).masked_fill(~valid, -30.)
        output = dict(pred_coordinates=coordinates, pred_boxes=coordinates[..., 28:],
            pred_keypoints=torch.cat((coordinates[..., :28], valid[..., None].expand(-1, -1, 14)), -1),
            pred_logits=torch.stack((torch.full_like(existence, -30.), existence), -1), scores=scores,
            proposal_indices=proposal_indices.masked_fill(~valid, -1), query_indices=query_indices,
            valid_mask=valid, person_counts=valid.sum(-1))
        if collect_trace:
            output['trace'] = [dict(layer=index+1, proposal_indices=output['proposal_indices'],
                valid_mask=valid, pred_coordinates=coordinates) for index, coordinates in enumerate(layers)]
        return output
