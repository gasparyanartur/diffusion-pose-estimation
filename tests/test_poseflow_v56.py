from copy import deepcopy
import json

import pytest
import pytorch_lightning as pl
import torch

from src.models.pipelines.pose_estimators.poseflow.model_v56 import PoseFlowV56Model
from src.models.pipelines.pose_estimators.poseflow.training_v56 import PoseFlowV56Pipeline, PoseFlowV56PipelineConfig
from tests.test_full_depth_birth import native_model, cached_context
from tests.test_poseflow_v43_training import batch
from tests.test_poseflow_v44 import arguments
from tests.test_poseflow_v55 import opened


def pipeline():
    config = PoseFlowV56PipelineConfig(native=dict(arguments(), num_encoder_layers=6,
        num_decoder_layers=6, num_box_decoder_layers=2), capacity=6, rows=7,
        pretrained_checkpoint=None, quick_eval_enabled=False, calibration_batches=2, residual_bank_capacity=8)
    module = PoseFlowV56Pipeline(config=config, model=PoseFlowV56Model(native=native_model(), capacity=6))
    module.set_residual_bank(torch.full((4, 32), .1), torch.ones(4, 14, dtype=torch.bool), dict(split='train', image_ids=[1, 2]))
    module.set_calibration([[1., 2.]]*2, dict(source='unit'))
    return module


def test_initial_birth_matches_preserved_decoder_and_no_future_information():
    from src.models.pipelines.pose_estimators.poseflow.full_depth_birth import FullDepthBirthDecoder
    from src.models.pipelines.pose_estimators.poseflow.rows_v48 import training_rows
    native = native_model()
    reference = FullDepthBirthDecoder(native=deepcopy(native)).eval()
    model = PoseFlowV56Model(native=native, capacity=6).eval()
    context = cached_context(reference)
    context['existence_logits'] = context['pred_logits'][..., 1]
    rows = training_rows(torch.rand(2, 6, 32), torch.tensor([2, 1]))
    rows['proposal_indices'] = torch.tensor([0, 1, 2])[None, None].expand(2, 3, 3).clone()
    output = model.forward_rows(context, **{name: rows[name] for name in model.row_input_fields})
    expected = reference.decode_proposals(context, output['selected_proposals'], collect_trace=True)
    torch.testing.assert_close(output['pred_coordinates'], torch.stack([item['pred_coordinates'] for item in expected['trace']]), atol=0, rtol=0)
    changed = deepcopy(rows)
    changed['prefix_coordinates'][changed['token_types'] != 1] = float('nan')
    torch.testing.assert_close(output['pred_coordinates'], model.forward_rows(context,
        **{name: changed[name] for name in model.row_input_fields})['pred_coordinates'], atol=0, rtol=0)
    assert model.pose_decoder.expansion_head is model.pose_decoder.pose_heads[0]


def test_one_assignment_four_layer_losses_and_live_prefix_encoder_gradients(monkeypatch):
    module = pipeline().train()
    opened(module.model.proposal_context)
    calls = []
    prepare, match = module.model.prepare_context, module.slot_matcher.forward
    def context(*args, **kwargs):
        calls.append('context')
        return prepare(*args, **kwargs)
    def assignment(*args, **kwargs):
        calls.append('match')
        return match(*args, **kwargs)
    monkeypatch.setattr(module.model, 'prepare_context', context)
    monkeypatch.setattr(module.slot_matcher, 'forward', assignment)
    losses = module._losses_for_batch(batch((2, 1)))
    assert calls == ['context', 'match']
    for task in ('pose_birth', 'box_birth', 'selection', 'eos'):
        parameters = (module.model.coordinate_input[0].weight,
                      module.model.proposal_context.attention.out_proj.weight)
        gradients = torch.autograd.grad(losses[task], parameters, retain_graph=True, allow_unused=True)
        assert all(value is not None and value.abs().sum() > 0 for value in gradients), task
    losses['total'].backward()
    assert all(head.layers[-1].weight.grad.abs().sum() > 0 for head in module.model.pose_decoder.pose_heads)
    groups = module.gradient_clipping_families()
    ids = [id(parameter) for group in groups.values() for parameter in group]
    assert len(ids) == len(set(ids)) == sum(parameter.requires_grad for parameter in module.model.parameters())


def test_checkpoint_gradients_empty_capacity_sampling_and_asset_free_restore(tmp_path):
    module = pipeline().train()
    opened(module.model.proposal_context)
    direct = deepcopy(module)
    direct.model.activation_checkpointing = False
    sample = batch((2, 0))
    for instance in (module, direct):
        torch.manual_seed(56)
        instance._losses_for_batch(sample)['total'].backward()
    for name, parameter in module.named_parameters():
        other = dict(direct.named_parameters())[name]
        assert (parameter.grad is None) == (other.grad is None)
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, other.grad, atol=3e-5, rtol=3e-4)
    for counts in ((0,), (1,)):
        missing = batch(counts)
        missing['gt_labeled'].zero_()
        assert module._losses_for_batch(missing)['pose_birth'] == 0
    assert json.loads(json.dumps(module.checkpoint_contract())) == module.checkpoint_contract()
    record = dict(state_dict=deepcopy(module.state_dict()), hyper_parameters=module.hparams,
        **{'pytorch-lightning_version': pl.__version__})
    module.on_save_checkpoint(record)
    expected = module._losses_for_batch(sample)['total']
    path = tmp_path/'v56.ckpt'
    torch.save(record, path)
    restored = PoseFlowV56Pipeline.load_from_checkpoint(path, weights_only=False).train()
    torch.testing.assert_close(restored._losses_for_batch(sample)['total'], expected, atol=0, rtol=0)
    for key in ('v56_contract', 'v56_prefix_bank'):
        broken = deepcopy(record)
        broken.pop(key)
        with pytest.raises(ValueError):
            restored.on_load_checkpoint(broken)
    restored.eval()
    context = restored.model.prepare_context(sample['image'])
    restored.model.selector.eos_bias.data.fill_(-1e6)
    output = restored.model.predict(context, sample['img_hw'].flip(-1), collect_trace=True)
    assert output['person_counts'].tolist() == [6, 6]
    assert len(output['trace']) == 24
    restored.model.selector.eos_bias.data.fill_(1e6)
    assert restored.model.predict(context, sample['img_hw'].flip(-1))['person_counts'].tolist() == [0, 0]
    overflow = batch((2,))
    for name in ('gt_boxes_pixel', 'gt_keypoints', 'gt_labeled', 'gt_box_labeled', 'person_mask'):
        overflow[name] = overflow[name].repeat_interleave(4, dim=1)[:, :7]
    with pytest.raises(ValueError, match='capacity'):
        module._losses_for_batch(overflow)
