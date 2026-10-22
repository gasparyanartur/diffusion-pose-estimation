import pytest
import torch

from src.models.pipelines.pose_estimators.poseflow.training_v23 import PoseFlowV23PipelineConfig, sample_two_step_schedule
from tests.test_poseflow import _batch
from tests.test_poseflow_v22 import configuration as v22_configuration


def configuration(**overrides):
    return PoseFlowV23PipelineConfig(**(v22_configuration(coordinate_batch_layout='per_image').model_dump() | overrides))


def test_schedule_uniform_supervised_time_and_balanced_branches():
    generator = torch.Generator().manual_seed(42)
    chosen, branches = [], []
    for _ in range(2000):
        first, second, use_second = sample_two_step_schedule(8, generator)
        assert ((0 <= first) & (first <= second) & (second < 1)).all()
        chosen.append(second if use_second else first); branches.append(use_second)
    values = torch.cat(chosen)
    assert .46 < sum(branches)/len(branches) < .54
    assert abs(float(values.mean())-.5) < .015
    assert abs(float((values < .25).float().mean())-.25) < .02


@pytest.mark.parametrize('use_second', [False, True])
def test_actual_loss_uses_only_selected_step_and_detaches_warmup(monkeypatch, use_second):
    from src.models.pipelines.pose_estimators.poseflow import training_v23
    pipeline = configuration(dn_number=2).build(max_epochs=2).train()
    monkeypatch.setattr(training_v23, 'sample_two_step_schedule', lambda count,generator: (
        torch.full((count,), .2), torch.full((count,), .7), use_second))
    sources = []
    original_builder = training_v23.build_poseflow_targets
    def build(*args, **kwargs):
        result = original_builder(*args, **kwargs)
        sources.append(result[1])
        return result
    monkeypatch.setattr(training_v23, 'build_poseflow_targets', build)
    criterion_calls = []
    criterion_hook = pipeline.criterion.register_forward_hook(lambda module,args,output:criterion_calls.append(output))
    calls = []
    def capture(module, args, kwargs, output):
        calls.append({'args': args, 'kwargs': kwargs, 'output': output, 'grad': torch.is_grad_enabled(), 'training': module.training})
    hook = pipeline.model.register_forward_hook(capture, with_kwargs=True)
    loss = pipeline._losses_for_batch(_batch() | {'image': torch.randn(1,3,64,64)})['total']
    hook.remove(); criterion_hook.remove()
    assert len(criterion_calls) == 1
    assert len(calls) == (2 if use_second else 1)
    assert calls[-1]['grad'] and calls[-1]['training']
    assert len(calls[-1]['args']) >= 4 and calls[-1]['args'][3] is not None
    assert torch.all(calls[-1]['args'][2] == (.7 if use_second else .2))
    if use_second:
        warm, supervised = calls
        assert not warm['grad'] and not warm['training']
        assert warm['kwargs'].get('targets') is None and len(warm['args']) == 3
        memory = supervised['kwargs']['hypothesis_memory']
        assert torch.equal(memory, warm['output']['hypothesis_memory']) and not memory.requires_grad
        state = supervised['args'][1]
        endpoint = (warm['output']['pred_keypoints'][...,:4]*2-1)*pipeline.model.scale
        expected = warm['args'][1]+.5*(endpoint-sources[0])
        torch.testing.assert_close(state, expected, atol=0, rtol=0)
        assert not state.requires_grad
    loss.backward()
    if use_second:
        assert pipeline.model.hypothesis_slots.grad is None
    else:
        assert pipeline.model.hypothesis_slots.grad is not None
    assert pipeline.model.training
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in pipeline.model.parameters())


def test_schedule_rng_strict_restore_continues():
    config = configuration()
    pipeline = config.build(max_epochs=2)
    pipeline._sample_schedule(3, torch.device('cpu'), torch.float32)
    saved = {key:value.clone() for key,value in pipeline.state_dict().items()}
    expected = pipeline._sample_schedule(3, torch.device('cpu'), torch.float32)
    restored = config.build(max_epochs=2)
    restored.load_state_dict(saved, strict=True)
    actual = restored._sample_schedule(3, torch.device('cpu'), torch.float32)
    for first, second in zip(expected, actual):
        assert torch.equal(first, second) if torch.is_tensor(first) else first == second


def test_target_builder_default_parity_explicit_times_and_validation():
    from src.models.pipelines.pose_estimators.poseflow.training import build_poseflow_targets
    config = configuration()
    model = config._build_poseflow_model()
    arguments = dict(num_proposals=3, num_body_points=2, padding_policy=config.flow_padding_policy)
    torch.manual_seed(9)
    expected = build_poseflow_targets(_batch(), model, **arguments)
    torch.manual_seed(9)
    actual = build_poseflow_targets(_batch(), model, **arguments, flow_times=expected[2])
    for first, second in zip(expected[:3], actual[:3]):
        torch.testing.assert_close(first, second, atol=0, rtol=0)
    for invalid in (torch.tensor([float('nan')]), torch.tensor([1.1]), torch.tensor([.2,.3])):
        with pytest.raises(ValueError, match='Explicit flow times'):
            build_poseflow_targets(_batch(), model, **arguments, flow_times=invalid)


def test_validation_keeps_single_call_and_does_not_advance_schedule():
    pipeline = configuration().build(max_epochs=2).eval()
    before = pipeline.two_step_rng_state.clone()
    calls = []
    hook = pipeline.model.register_forward_hook(lambda module,args,output:calls.append(output))
    loss = pipeline._losses_for_batch(_batch() | {'image':torch.randn(1,3,64,64)})['total']
    hook.remove()
    assert loss.isfinite() and len(calls) == 1
    assert torch.equal(before, pipeline.two_step_rng_state)
