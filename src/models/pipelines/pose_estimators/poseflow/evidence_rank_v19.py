import torch
from torch import nn

from .model_v19 import PoseFlowV19Model


def corrupt_evidence(evidence, noise_levels, *, noise=None):
    scale = evidence.detach().float().square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
    scale = scale.to(evidence.dtype)
    noise = torch.randn_like(evidence) if noise is None else noise
    return [evidence if sigma == 0 else evidence + sigma*scale*noise for sigma in noise_levels]


def evidence_ranking_loss(scores, positive, noise_levels, *, margin=1.):
    # Scores: B images, N candidates, three corruption levels.
    denominator = positive.sum().clamp_min(1)
    metrics, losses = {}, []
    for first, second in [(0, 1), (0, 2), (1, 2)]:
        gap = scores[..., first]-scores[..., second]
        required = margin*(noise_levels[second]**2-noise_levels[first]**2)/noise_levels[-1]**2
        losses.append(((required-gap).relu()*positive).sum()/denominator)
        prefix = f'pair{first}{second}'
        metrics[f'{prefix}/score_gap'] = (gap.detach()*positive).sum()/denominator
        metrics[f'{prefix}/violation_fraction'] = ((gap.detach() < required)*positive).sum()/denominator
    for level in range(3):
        metrics[f'level{level}/score'] = (scores[..., level].detach()*positive).sum()/denominator
    return torch.stack(losses).mean(), metrics


class PoseFlowV19EvidenceRankModel(PoseFlowV19Model):
    def __init__(self, image_model, *, sampling_steps=3):
        super().__init__(image_model, sampling_steps=sampling_steps)
        self.detach_decoder_geometry = True
        self.evidence_confidence = nn.Linear(self.hidden_dim, 1, bias=False)
        self.evidence_noise_levels = [0., .25, .5]
        self.evidence_rank_enabled = True
        self.initialization_report['fresh_components'].append('evidence_confidence')
        state = self.state_dict()
        total = sum(value.numel() for value in state.values())
        self.initialization_report.update(total_state_tensors=len(state), total_state_elements=total,
            copied_source_fraction=self.initialization_report['copied_source_elements']/total)
        for layer in self.layers:
            for module in layer.block.modules():
                if ((isinstance(module, nn.Dropout) and module.p > 0)
                        or (isinstance(module, nn.MultiheadAttention) and module.dropout > 0)):
                    raise ValueError('Evidence-ranking branches require zero decoder dropout')

    def _decoder_update(self, layer, features, memory, references, positions, image, collect_weights):
        if not (self.training and torch.is_grad_enabled() and self.evidence_rank_enabled):
            return super()._decoder_update(layer, features, memory, references, positions, image, collect_weights)
        evidence = layer.read_image(features, references, positions, image)
        messages = corrupt_evidence(evidence, self.evidence_noise_levels)
        clean_features, clean_memory, weights = layer.update_from_evidence(features, memory, positions,
            messages[0], collect_weights=collect_weights)
        scores = [self.evidence_confidence(clean_memory).squeeze(-1)]
        memory_rms = [clean_memory.detach().float().square().mean(-1).sqrt()]
        for message in messages[1:]:
            _, updated_memory, _ = layer.update_from_evidence(features, memory, positions, message)
            scores.append(self.evidence_confidence(updated_memory).squeeze(-1))
            memory_rms.append(updated_memory.detach().float().square().mean(-1).sqrt())
        statistics = {f'level{index}/hypothesis_rms': value for index, value in enumerate(memory_rms)}
        statistics['evidence_rms'] = evidence.detach().float().square().mean((-1, -2)).sqrt()
        return clean_features, clean_memory, weights, {
            'evidence_rank_scores': torch.stack(scores, -1), 'evidence_rank_statistics': statistics}
