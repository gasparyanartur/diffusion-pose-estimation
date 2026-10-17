import math

import torch
from torch import nn

from .model_v14 import PoseFlowV14Model, V14AttentionUpdate, V14EvidenceExtractor


class V15HypothesisUpdate(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.query_norm = nn.Identity()
        self.evidence_norm = nn.LayerNorm(width)
        self.retention_logit = nn.Parameter(torch.tensor(math.log(9.)))

    def forward(self, memory, global_tokens, *, collect_weights=False):
        evidence, weights = self.attention(self.query_norm(memory), global_tokens, global_tokens,
            need_weights=collect_weights, average_attn_weights=False)
        evidence = self.evidence_norm(evidence)
        retention = self.retention_logit.sigmoid()
        return retention * memory + (1-retention) * evidence, weights


class V15EvidenceUpdate(V14EvidenceExtractor):
    def __init__(self, width, heads, levels, points, joints, feedforward, dropout):
        super().__init__(width, heads, levels, points, joints)
        self.ffn = nn.Sequential(nn.Linear(width, feedforward), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(feedforward, width))
        self.ffn_norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)
        self.image_message_transform = nn.Identity()

    def forward(self, representations, geometry, image):
        batch, groups, roles, width = representations.shape
        boxes = geometry['pred_boxes']
        joints = geometry['pred_keypoints'][..., :2*(roles-1)].reshape(batch, groups, roles-1, 2)
        position = torch.cat((self.box_position(boxes)[:, :, None], self.joint_position(joints)), 2)
        queries = representations + position + self.roles.weight[None, None]
        ratios = image.valid_ratios[:, None]
        box_references = boxes[:, :, None] * torch.cat((ratios, ratios), -1)
        joint_references = joints.flatten(1, 2)[:, :, None] * ratios
        arguments = (image.memory, image.spatial_shapes, image.level_start_index, image.padding_mask)
        box_message = self.image_attention(queries[:, :, 0], box_references, *arguments)
        joint_message = self.image_attention(queries[:, :, 1:].flatten(1, 2), joint_references, *arguments)
        message = torch.cat((box_message[:, :, None], joint_message.reshape(batch, groups, roles-1, width)), 2)
        message = self.image_message_transform(message)
        updated = self.norm(representations + self.dropout(message))
        return self.ffn_norm(updated + self.dropout(self.ffn(updated)))


class PoseFlowV15DecoderLayer(nn.Module):
    def __init__(self, width, heads, feedforward, dropout):
        super().__init__()
        self.state_update = V14AttentionUpdate(width, heads, feedforward, dropout)
        self.aggregate = V14AttentionUpdate(width, heads, feedforward, dropout)

    def read_hypotheses(self, representations, memory):
        updated, _ = self.state_update(representations.flatten(1, 2), memory)
        return updated.reshape_as(representations)

    def coordinate_person(self, representations):
        batch, groups, roles, width = representations.shape
        grouped = representations.reshape(batch*groups, roles, width)
        updated, _ = self.aggregate(grouped, grouped)
        return updated.reshape_as(representations)


class PoseFlowV15Model(PoseFlowV14Model):
    def __init__(self, image_model, *, sampling_steps=3):
        super().__init__(image_model, sampling_steps=sampling_steps)
        original = image_model.transformer.decoder.layers[0]
        width, heads = self.hidden_dim, original.self_attn.num_heads
        feedforward, dropout = original.linear1.out_features, original.dropout1.p
        self.evidence_extractor = V15EvidenceUpdate(width, heads, self.num_feature_levels,
            original.cross_attn.num_points, self.num_body_points, feedforward, dropout)
        self.hypothesis_update = V15HypothesisUpdate(width, heads, dropout)
        self.layers = nn.ModuleList([PoseFlowV15DecoderLayer(width, heads, feedforward, dropout)
                                    for _ in image_model.transformer.decoder.layers])
        self.initialization_report['decoder_transferred'] = []
        self.initialization_report['memory_retention_initial'] = .9
        state = self.state_dict()
        total_elements = sum(value.numel() for value in state.values())
        self.initialization_report.update(total_state_tensors=len(state),
            total_state_elements=total_elements,
            retained_image_element_fraction=self.initialization_report['retained_image_elements']/total_elements)

    def denoise(self, noisy_latents, timesteps, encoded_image, *, hypothesis_memory=None,
                collect_diagnostics=False):
        expected = (noisy_latents.shape[0], self.num_proposals, self.num_state_tokens, self.hidden_dim)
        if noisy_latents.shape != expected or timesteps.shape != (expected[0],):
            raise ValueError('V15 latent/timestep shapes disagree with configuration')
        if not torch.isfinite(noisy_latents).all() or not torch.isfinite(timesteps).all() or ((timesteps < 0) | (timesteps > 1)).any():
            raise ValueError('V15 requires finite latents and times in [0,1]')
        if hypothesis_memory is None:
            hypothesis_memory = self.hypothesis_slots.expand(expected[0], -1, -1)
        if (hypothesis_memory.shape != (expected[0], self.num_hypotheses, self.hidden_dim)
                or hypothesis_memory.device != noisy_latents.device or hypothesis_memory.dtype != noisy_latents.dtype
                or not torch.isfinite(hypothesis_memory).all()):
            raise ValueError('V15 hypothesis memory shape/device/dtype/values disagree with state')
        time_features = self.time_embedding(timesteps)[:, None, None]
        state = self.input_projection(noisy_latents) + self.state_roles.weight[None, None] + time_features
        global_tokens = self.global_token.expand(expected[0], self.num_proposals, -1)
        representations = torch.cat((state, global_tokens[:, :, None]), 2)
        predictions, diagnostics = [], []
        for index, layer in enumerate(self.layers):
            previous_global, previous_memory = representations[:, :, -1], hypothesis_memory
            representations = layer.read_hypotheses(representations, hypothesis_memory)
            sampling_geometry = self.codec.decode(self.latent_head(representations[:, :, :-1]))
            local = self.evidence_extractor(representations[:, :, :-1], sampling_geometry, encoded_image)
            representations = torch.cat((local, representations[:, :, -1:]), 2)
            representations = layer.coordinate_person(representations)
            global_tokens = representations[:, :, -1]
            hypothesis_memory, attention = self.hypothesis_update(hypothesis_memory, global_tokens,
                collect_weights=collect_diagnostics)
            clean = self.latent_head(representations[:, :, :-1])
            geometry = self.codec.decode(clean)
            predictions.append({'pred_latents': clean, **geometry})
            if collect_diagnostics:
                probabilities = attention.detach().float().clamp_min(0)
                probabilities = probabilities / probabilities.sum(-1, keepdim=True).clamp_min(1e-8)
                averaged = probabilities.mean(1)
                row = self._iteration_diagnostics(index, clean, global_tokens, previous_global,
                    hypothesis_memory, previous_memory, averaged, geometry)
                row['hypothesis_attention_entropy_head_averaged'] = row.pop('hypothesis_attention_entropy')
                row['hypothesis_attention_entropy_per_head'] = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(-1).mean()
                row['memory_retention'] = self.hypothesis_update.retention_logit.sigmoid().detach()
                row['joint_candidate_std'] = geometry['pred_keypoints'][..., :2*self.num_body_points].detach().float().std(1, unbiased=False).mean()
                row['box_candidate_std'] = geometry['pred_boxes'].detach().float().std(1, unbiased=False).mean()
                diagnostics.append(row)
        return {**predictions[-1], 'aux_outputs': predictions[:-1],
                'hypothesis_memory': hypothesis_memory, 'latent_diagnostics': diagnostics}
