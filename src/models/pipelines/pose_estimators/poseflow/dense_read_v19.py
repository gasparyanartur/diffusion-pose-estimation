import torch
from torch import nn

from .model_v14 import tensor_rms
from .model_v19 import PoseFlowV19Model


class DenseHypothesisRead(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=0., batch_first=True)
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, hypotheses, timesteps, image, *, collect_diagnostics=False):
        if image.position is None or image.position.shape != image.memory.shape:
            raise ValueError('Dense hypothesis reads require matching image position features')
        if image.padding_mask.all(-1).any():
            raise ValueError('Dense hypothesis reads require a valid image token for every image')
        if (timesteps.shape != hypotheses.shape[:1] or not torch.isfinite(timesteps).all()
                or ((timesteps < 0) | (timesteps > 1)).any()):
            raise ValueError('Dense hypothesis read times must match batch and lie in [0,1]')
        # B = images, N = hypotheses, L = spatial tokens, D = width.
        message, weights = self.attention(
            self.query_norm(hypotheses), image.memory+image.position, image.memory,
            key_padding_mask=image.padding_mask, need_weights=collect_diagnostics,
            average_attn_weights=False)
        strength = .1+.9*(1-timesteps)
        update = (self.gate*strength)[:, None, None]*message  # <B,N,D>
        updated = hypotheses+update
        diagnostics = {}
        if collect_diagnostics:
            diagnostics = {
                'dense_read/gate': self.gate.detach(),
                'dense_read/time_strength_mean': strength.detach().mean(),
                'dense_read/time_strength_min': strength.detach().min(),
                'dense_read/time_strength_max': strength.detach().max(),
                'dense_read/message_rms': tensor_rms(message),
                'dense_read/update_rms': tensor_rms(update),
                'dense_read/incoming_memory_rms': tensor_rms(hypotheses),
                'dense_read/outgoing_memory_rms': tensor_rms(updated),
            }
            probability = weights.detach().float().clamp_min(0)
            entropy = -(probability*probability.clamp_min(1e-8).log()).sum(-1)
            for head, value in enumerate(entropy.mean((0, 2))):
                diagnostics[f'dense_read/attention_entropy_head{head}'] = value
        return updated, diagnostics


class PoseFlowV19DenseReadModel(PoseFlowV19Model):
    def __init__(self, image_model, *, sampling_steps=3):
        super().__init__(image_model, sampling_steps=sampling_steps)
        heads = image_model.transformer.decoder.layers[0].self_attn.num_heads
        self.dense_read = DenseHypothesisRead(self.hidden_dim, heads)
        self.detach_decoder_geometry = True
        total = sum(value.numel() for value in self.state_dict().values())
        self.initialization_report.update(
            fresh_components=self.initialization_report['fresh_components']+['dense_read'],
            total_state_tensors=len(self.state_dict()), total_state_elements=total,
            copied_source_fraction=self.initialization_report['copied_source_elements']/total)

    def _read_hypothesis_image(self, memory, timesteps, image, collect_diagnostics):
        return self.dense_read(memory, timesteps, image, collect_diagnostics=collect_diagnostics)

    def load_baseline_state_dict(self, state):
        current = self.state_dict()
        new_keys = {name for name in current if name.startswith('dense_read.')}
        baseline_keys = current.keys()-new_keys
        if state.keys() != baseline_keys:
            raise ValueError('Expected exactly the V19 baseline model state, without ranking/dense keys')
        if any(state[name].shape != current[name].shape for name in baseline_keys):
            raise ValueError('V19 baseline tensor shapes do not match the dense-read model')
        result = self.load_state_dict(state, strict=False)
        if set(result.missing_keys) != new_keys or result.unexpected_keys:
            raise RuntimeError('V19 baseline import changed the expected new-module boundary')
        loaded = sum(current[name].numel() for name in baseline_keys)
        report = {'loaded_tensors': len(baseline_keys), 'loaded_elements': loaded,
                  'loaded_state_fraction': loaded/sum(value.numel() for value in current.values()),
                  'new_keys': sorted(new_keys)}
        self.initialization_report['baseline_transfer'] = report
        return report
