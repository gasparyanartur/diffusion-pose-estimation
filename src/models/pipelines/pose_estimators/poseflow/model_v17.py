import torch
from torch import nn

from .model_v14 import PoseFlowV14Model, V14AttentionUpdate, tensor_rms
from .model_v15 import V15EvidenceUpdate


class V17AttentionUpdate(V14AttentionUpdate):
    def forward(self, queries, context, *, collect_weights=False):
        update, weights = self.attention(queries, context, context,
            need_weights=collect_weights, average_attn_weights=False)
        queries = self.attention_norm(queries + self.dropout(update))
        return self.ffn_norm(queries + self.dropout(self.ffn(queries))), weights


def attention_entropy(probabilities):
    probabilities = probabilities.detach().float().clamp_min(0)
    probabilities = probabilities / probabilities.sum(-1, keepdim=True).clamp_min(1e-8)
    return -(probabilities * probabilities.clamp_min(1e-8).log()).sum(-1).mean()


class PoseFlowV17Model(PoseFlowV14Model):
    def __init__(self, image_model, *, sampling_steps=3):
        super().__init__(image_model, sampling_steps=sampling_steps)
        original = image_model.transformer.decoder.layers[0]
        width, heads = self.hidden_dim, original.self_attn.num_heads
        feedforward, dropout = original.linear1.out_features, original.dropout1.p
        self.num_hypotheses = self.num_proposals
        self.hypothesis_slots = nn.Parameter(torch.randn(1, self.num_hypotheses, width) * .02)
        self.evidence_extractor = V15EvidenceUpdate(width, heads, self.num_feature_levels,
            original.cross_attn.num_points, self.num_body_points, feedforward, dropout)
        del self.hypothesis_update
        self.hypothesis_attention = V17AttentionUpdate(width, heads, feedforward, dropout)
        self.layers = nn.ModuleList([
            V17AttentionUpdate(width, heads, feedforward, dropout)
            for _ in image_model.transformer.decoder.layers
        ])
        state = self.state_dict()
        total = sum(value.numel() for value in state.values())
        fresh = self.initialization_report['fresh_components']
        self.initialization_report.update(
            fresh_components=[name for name in fresh if name != 'hypothesis_update'] + ['hypothesis_attention'],
            total_state_tensors=len(state), total_state_elements=total,
            retained_image_element_fraction=self.initialization_report['retained_image_elements']/total)

    def refine_before_image(self, index, representations):
        return representations

    def denoise(self, noisy_latents, timesteps, encoded_image, *, hypothesis_memory=None,
                collect_diagnostics=False):
        batch = noisy_latents.shape[0]
        expected = (batch, self.num_proposals, self.num_state_tokens, self.hidden_dim)
        if noisy_latents.shape != expected or timesteps.shape != (batch,):
            raise ValueError('V17 latent/timestep shapes disagree with configuration')
        if (not torch.isfinite(noisy_latents).all() or not torch.isfinite(timesteps).all()
                or ((timesteps < 0) | (timesteps > 1)).any()):
            raise ValueError('V17 requires finite latents and times in [0,1]')
        if hypothesis_memory is None:
            hypothesis_memory = self.hypothesis_slots.expand(batch, -1, -1)
        if (hypothesis_memory.shape != (batch, self.num_proposals, self.hidden_dim)
                or hypothesis_memory.device != noisy_latents.device
                or hypothesis_memory.dtype != noisy_latents.dtype
                or not torch.isfinite(hypothesis_memory).all()):
            raise ValueError('V17 memory must match candidate shape, device and dtype')
        state = (self.input_projection(noisy_latents) + self.state_roles.weight[None, None]
                 + self.time_embedding(timesteps)[:, None, None])
        global_tokens = self.global_token.expand(batch, self.num_proposals, -1)
        representations = torch.cat((state, global_tokens[:, :, None]), 2)
        predictions, diagnostics = [], []
        for index, layer in enumerate(self.layers):
            previous_global, previous_memory = representations[:, :, -1], hypothesis_memory
            hypothesis_memory, bank_attention = self.hypothesis_attention(
                hypothesis_memory, hypothesis_memory, collect_weights=collect_diagnostics)
            pre_image_representations = representations
            refined_representations = self.refine_before_image(index, representations)
            representations = refined_representations
            sampling_geometry = self.codec.decode(self.latent_head(representations[:, :, :self.num_state_tokens]))
            local = self.evidence_extractor(representations[:, :, :self.num_state_tokens],
                                            sampling_geometry, encoded_image)
            tokens = torch.cat((local, representations[:, :, -1:], hypothesis_memory[:, :, None]), 2)
            # Each candidate is an independent attention group: box, joints, global, hypothesis.
            grouped = tokens.flatten(0, 1)
            updated, local_attention = layer(grouped, grouped, collect_weights=collect_diagnostics)
            updated = updated.reshape_as(tokens)
            representations, hypothesis_memory = updated[:, :, :-1], updated[:, :, -1]
            global_tokens = representations[:, :, -1]
            clean = self.latent_head(representations[:, :, :self.num_state_tokens])
            geometry = self.codec.decode(clean)
            predictions.append({'pred_latents': clean, **geometry})
            if collect_diagnostics:
                row = self._iteration_diagnostics(index, clean, global_tokens, previous_global,
                    hypothesis_memory, previous_memory, bank_attention.mean(1), geometry)
                row['pre_image_input_rms'] = tensor_rms(pre_image_representations)
                row['pre_image_output_rms'] = tensor_rms(refined_representations)
                row['pre_image_update_rms'] = tensor_rms(refined_representations-pre_image_representations)
                row['sampling_box_size_mean'] = sampling_geometry['pred_boxes'][..., 2:].detach().float().mean()
                row['sampling_joint_candidate_std'] = sampling_geometry['pred_keypoints'][..., :2*self.num_body_points].detach().float().std(1, unbiased=False).mean()
                row['hypothesis_self_attention_entropy_head_averaged'] = row.pop('hypothesis_attention_entropy')
                row['hypothesis_self_attention_entropy_per_head'] = attention_entropy(bank_attention)
                row['local_attention_entropy_per_head'] = attention_entropy(local_attention)
                row['local_hypothesis_attention_entropy_per_head'] = attention_entropy(local_attention[:, :, -1])
                row['local_hypothesis_read_mass'] = local_attention[:, :, :self.num_state_tokens, -1].detach().mean()
                row['local_hypothesis_write_mass'] = local_attention[:, :, -1, :self.num_state_tokens].detach().sum(-1).mean()
                row['joint_candidate_std'] = geometry['pred_keypoints'][..., :2*self.num_body_points].detach().float().std(1, unbiased=False).mean()
                row['box_candidate_std'] = geometry['pred_boxes'].detach().float().std(1, unbiased=False).mean()
                diagnostics.append(row)
        return {**predictions[-1], 'aux_outputs': predictions[:-1],
                'hypothesis_memory': hypothesis_memory, 'latent_diagnostics': diagnostics}
