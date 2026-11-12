from types import SimpleNamespace

import pytest
import torch
from torch import nn

from scripts.investigation.diffusionregpose_inspection.state_sensitivity import (
    mask_decoder_routes, replace_joint_rows,
)


def test_replace_joint_rows_preserves_boxes_and_broadcasts_bias():
    values = torch.arange(24.).reshape(1, 6, 4)
    bias = torch.tensor([1., 2., 3., 4.])
    changed = replace_joint_rows(values, stride=3, axis=1, replacement=bias)
    torch.testing.assert_close(changed[:, [0, 3]], values[:, [0, 3]])
    torch.testing.assert_close(changed[:, [1, 2, 4, 5]], bias.expand(1, 4, 4))
    torch.testing.assert_close(values, torch.arange(24.).reshape(1, 6, 4))


class FakeCrossAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_proj = nn.Linear(4, 4)

    def forward(self, value):
        return self.output_proj(value)


def test_route_masks_keep_box_updates_and_remove_hooks_on_exception():
    layers = nn.ModuleList([nn.Module(), nn.Module(), nn.Module()])
    for layer in layers:
        layer.cross_attn = FakeCrossAttention()
        layer.self_attn = nn.MultiheadAttention(4, 1)
        layer.linear2 = nn.Linear(4, 4)
    model = nn.Module().eval()
    model.transformer = SimpleNamespace(decoder=SimpleNamespace(
        layers=layers, num_box_decoder_layers=1, num_body_points=2))
    batch_first = torch.randn(1, 6, 4)
    sequence_first = batch_first.transpose(0, 1)
    cross_before = layers[1].cross_attn(batch_first)
    attention_before = layers[1].self_attn(sequence_first, sequence_first, sequence_first)
    feedforward_before = layers[1].linear2(sequence_first)
    with pytest.raises(RuntimeError, match='intentional'):
        with mask_decoder_routes(model, first_image=True, first_updates=True):
            crossed = layers[1].cross_attn(batch_first)
            attended, weights = layers[1].self_attn(sequence_first, sequence_first, sequence_first)
            feedforward = layers[1].linear2(sequence_first)
            torch.testing.assert_close(crossed[:, [0, 3]], cross_before[:, [0, 3]])
            torch.testing.assert_close(crossed[:, [1, 2, 4, 5]],
                                       layers[1].cross_attn.output_proj.bias.expand(1, 4, 4))
            torch.testing.assert_close(attended[[0, 3]], attention_before[0][[0, 3]])
            torch.testing.assert_close(weights, attention_before[1])
            torch.testing.assert_close(feedforward[[0, 3]], feedforward_before[[0, 3]])
            assert attended[[1, 2, 4, 5]].count_nonzero() == 0
            assert feedforward[[1, 2, 4, 5]].count_nonzero() == 0
            raise RuntimeError('intentional')
    for layer in layers:
        for module in layer.modules():
            assert not module._forward_hooks
    torch.testing.assert_close(layers[1].cross_attn(batch_first), cross_before)
