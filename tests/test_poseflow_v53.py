from copy import deepcopy

import pytest
import torch

from src.models.pipelines.pose_estimators.diffusionregpose.model import DiffusionRegPoseModel
from src.models.pipelines.pose_estimators.diffusionregpose.transformer import inverse_sigmoid, sine_embedding_for_position
from src.models.pipelines.pose_estimators.poseflow.expanded_pose_v53 import ExpandedPoseDecoder
from tests.test_poseflow_v44 import arguments


def native_pose(native, feature, box, memory, shapes, starts, ratios):
    decoder = deepcopy(native.transformer.decoder).eval()
    decoder.num_group, decoder.num_dn = 1, 0
    state, references, _ = decoder._expand_keypoint_queries(feature.transpose(0, 1), box.transpose(0, 1),
        torch.zeros(len(feature), 1, 28), state_keypoint_gate=torch.zeros(len(feature)))
    result = []
    for index in range(decoder.num_box_decoder_layers, decoder.num_layers):
        scaled = references[:, :, None]*torch.cat((ratios, ratios), -1)[None]
        position = decoder.ref_point_head(sine_embedding_for_position(scaled[:, :, 0], num_pos_feats=16))
        state = decoder.layers[index](state, tgt_query_pos=position, tgt_reference_points=scaled,
            memory=memory, memory_level_start_index=starts, memory_spatial_shapes=shapes)
        readout = decoder.norm(state).transpose(0, 1)
        boxes = (decoder.bbox_embed[index](readout[:, 0])+inverse_sigmoid(references[0])).sigmoid()
        joints = (decoder.pose_embed[index-decoder.num_box_decoder_layers](readout[:, 1:])+
                  inverse_sigmoid(references[1:, :, :2].transpose(0, 1))).sigmoid()
        result.append(torch.cat((joints.flatten(-2), boxes), -1))
        state, references = decoder._update_expanded_queries(index, state, references)
        references = references.detach()
    return torch.stack(result)


def test_expansion_cached_decoder_matches_native_with_no_prefix_batch_two():
    torch.manual_seed(53)
    native = DiffusionRegPoseModel(**arguments()).eval()
    adapter = ExpandedPoseDecoder(native).eval()
    memory, feature = torch.randn(2, 7, 32), torch.randn(2, 1, 32)
    box = torch.rand(2, 1, 4)*.6+.2
    shapes, starts, ratios = torch.tensor([[2, 2], [1, 2], [1, 1]]), torch.tensor([0, 4, 6]), torch.ones(2, 3, 2)
    expected = native_pose(native, feature, box, memory, shapes, starts, ratios)
    projected = adapter.project_memory(memory, torch.zeros(2, 7, dtype=torch.bool))
    actual = adapter(feature[:, None], box[:, None], torch.zeros(2, 1, 0, 32),
        torch.zeros(2, 1, 0, 4), torch.zeros(2, 1, 0, dtype=torch.bool), projected, shapes, starts, ratios)
    torch.testing.assert_close(actual[:, :, 0], expected, atol=0, rtol=0)
    assert torch.equal(adapter.keypoint_embed.weight, native.transformer.decoder.keypoint_embed.weight)


def test_rows_random_order_clean_prefix_and_empty_capacity():
    from src.models.pipelines.pose_estimators.poseflow.rows_v53 import training_rows
    clean = torch.rand(2, 6, 32)
    sources = dict(coordinates=clean, counts=torch.tensor([3, 0]), coordinate_mask=torch.ones_like(clean, dtype=torch.bool),
                   target_indices=torch.arange(6).repeat(2, 1))
    rows, shuffled = training_rows(sources, generator=torch.Generator().manual_seed(53))
    assert rows['birth_rows'].sum().item() == 3
    assert rows['eos_targets'].sum().item() == 2
    assert rows['row_valid'].sum().item() == 5
    assert rows['prefix_coordinates'].shape == (2, 4, 4, 32)
    assert not rows['prefix_coordinates'][rows['token_types'] != 1].any()
    torch.testing.assert_close(rows['prefix_coordinates'][0, 2, :2], shuffled['coordinates'][0, :2])
    with pytest.raises(ValueError, match='capacity'):
        training_rows(dict(sources, counts=torch.tensor([7, 0])))


def model():
    from src.models.pipelines.pose_estimators.poseflow.model_v53 import PoseFlowV53Model
    return PoseFlowV53Model(native=DiffusionRegPoseModel(**arguments()), capacity=6).eval()


def test_model_parallel_rows_masks_gradients_and_fixed_births():
    from tests.test_poseflow_v52 import context
    from src.models.pipelines.pose_estimators.poseflow.rows_v48 import training_rows
    network = model().train()
    # The random native fixture has zero final heads; pretrained heads are nonzero.
    for head in (*network.pose_decoder.pose_heads, *network.pose_decoder.box_heads):
        torch.nn.init.normal_(head.layers[-1].weight, std=.01)
    cached = context(SimpleCache(network))
    rows = training_rows(torch.rand(2, 6, 32), torch.tensor([2, 1]))
    rows['proposal_indices'] = torch.arange(3).expand(2, 3, 3).clone()
    values = {name: rows[name] for name in network.row_input_fields}
    output = network.forward_rows(cached, **values)
    assert output['pred_coordinates'].shape == (2, 2, 3, 32)
    for row in range(3):
        independent = network.forward_rows(cached, **{name: value[:, row:row+1] for name, value in values.items()})
        torch.testing.assert_close(output['pred_coordinates'][:, :, row:row+1], independent['pred_coordinates'], atol=2e-6, rtol=2e-5)
    changed = deepcopy(values)
    changed['prefix_coordinates'][rows['token_types'] != 1] = float('nan')
    torch.testing.assert_close(output['pred_coordinates'], network.forward_rows(cached, **changed)['pred_coordinates'], atol=0, rtol=0)
    output['pred_coordinates'].square().sum().backward()
    assert network.pose_decoder.keypoint_embed.weight.grad.abs().sum() > 0
    assert network.pose_decoder.layers[0].cross_attn.sampling_offsets.weight.grad.abs().sum() > 0
    assert network.coordinate_input[0].weight.grad.abs().sum() > 0
    network.eval()
    network.selector.eos_bias.data.fill_(-1e6)
    generated = network.predict(cached, torch.tensor([[80., 64.], [64., 80.]]), collect_trace=True)
    assert generated['person_counts'].tolist() == [6, 6]
    assert generated['generation_steps'].tolist() == [6, 6]
    for step in generated['trace']:
        if step['layer'] == 2:
            born = step['image_indices'][step['birth']]
            rank = step['step']-1
            assert torch.equal(generated['pred_coordinates'][born, rank], step['pred_coordinates'][step['birth']])
    network.selector.eos_bias.data.fill_(1e6)
    empty = network.predict(cached, torch.tensor([[80., 64.], [64., 80.]]))
    assert empty['person_counts'].tolist() == [0, 0] and empty['generation_steps'].tolist() == [1, 1]


class SimpleCache:
    def __init__(self, network):
        self.network = network

    def _project_image_context(self, memory, position, padding_mask):
        return dict(memory=memory, image_values=self.network.pose_decoder.project_memory(memory, padding_mask))


def test_independent_row_gradient_parity_and_per_image_stopping(monkeypatch):
    from tests.test_poseflow_v52 import context
    from src.models.pipelines.pose_estimators.poseflow.rows_v48 import training_rows
    network = model()
    for head in (*network.pose_decoder.pose_heads, *network.pose_decoder.box_heads):
        torch.nn.init.normal_(head.layers[-1].weight, std=.01)
    independent = deepcopy(network)
    torch.manual_seed(531)
    cached = context(SimpleCache(network))
    torch.manual_seed(531)
    other_cached = context(SimpleCache(independent))
    rows = training_rows(torch.rand(2, 6, 32), torch.tensor([2, 1]))
    rows['proposal_indices'] = torch.arange(3).expand(2, 3, 3).clone()
    values = {name: rows[name] for name in network.row_input_fields}
    batched = network.forward_rows(cached, **values)['pred_coordinates']
    batched.square().sum().backward()
    outputs = [independent.forward_rows(other_cached, **{name: value[:, row:row+1] for name, value in values.items()})['pred_coordinates'] for row in range(3)]
    sum(value.square().sum() for value in outputs).backward()
    for name, parameter in network.named_parameters():
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, dict(independent.named_parameters())[name].grad, atol=3e-5, rtol=3e-4)
    with torch.no_grad():
        cached['features'][:, 0, 0] = torch.tensor([-7., 7.])
    select = network.selector.forward
    def scheduled(prefix, candidates, used):
        scores, _ = select(prefix, candidates, used)
        limit = torch.where(candidates[:, :1, 0] < 0, 1, 3)
        eos = torch.where(used.sum(-1) >= limit, 20., -20.)
        return scores, eos
    monkeypatch.setattr(network.selector, 'forward', scheduled)
    sizes = torch.tensor([[80., 64.], [64., 80.]])
    together = network.predict(cached, sizes)
    assert together['person_counts'].tolist() == [1, 3]
    assert together['generation_steps'].tolist() == [2, 4]
    from src.models.pipelines.pose_estimators.poseflow.model_v53 import select_context
    for image in range(2):
        alone = network.predict(select_context(cached, torch.tensor([image])), sizes[image:image+1])
        torch.testing.assert_close(together['pred_coordinates'][image:image+1], alone['pred_coordinates'], atol=3e-6, rtol=3e-5)
