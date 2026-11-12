from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from src.models.pipelines.pose_estimators.poseflow.dino_box_teacher import (
    BoxFeatureTargets, FrozenDinoBoxTeacher, build_box_targets, pool_dino_boxes,
)
from src.models.pipelines.pose_estimators.poseflow.model_v25 import HFeatureGridDecoder
from src.models.pipelines.pose_estimators.poseflow.training_v25 import (
    PoseFlowV25PipelineConfig, matched_grid_loss,
)
from src.models.pipelines.pose_estimators.poseflow.training import build_poseflow_targets
from tests.test_poseflow import _batch
from tests.test_poseflow_v24 import configuration as v24_configuration


@pytest.fixture(autouse=True)
def preserve_rng():
    with torch.random.fork_rng(devices=[]):
        yield


def configuration(**overrides):
    values = v24_configuration(h_enabled=True).model_dump()
    return PoseFlowV25PipelineConfig(**(values | {'h_dino_grid_size': 2} | overrides))


class FakeDino(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))
        self.inputs = []

    def forward(self, image):
        self.inputs.append(image.detach().clone())
        patches = torch.nn.functional.avg_pool2d(image, 16).flatten(2).transpose(1, 2)
        patches = patches.repeat(1, 1, 128) * self.scale
        prefix = patches.new_zeros(1, 5, 384)
        return SimpleNamespace(last_hidden_state=torch.cat((prefix, patches), dim=1))


class FakeTeacher:
    def __init__(self, grid_size=2):
        self.network = FakeDino().eval().requires_grad_(False)
        self.grid_size = grid_size
        self.calls = 0

    def targets(self, images, image_hw, boxes):
        self.calls += 1
        return build_box_targets(self.network, images, image_hw, boxes,
                                 grid_size=self.grid_size, skip_tokens=5)


def batch():
    return _batch() | {'image': torch.randn(1, 3, 64, 64),
                       'img_hw': torch.tensor([[64, 64]])}


def test_grid_decoder_uses_h_and_fixed_positions_and_has_gradients():
    decoder = HFeatureGridDecoder(32, grid_size=7, feature_dim=8)
    h = torch.randn(2, 3, 32, requires_grad=True)
    output = decoder(h)
    assert output.shape == (2, 3, 49, 8)
    assert not torch.equal(output[..., 0, :], output[..., 1, :])
    output.square().mean().backward()
    assert h.grad.abs().sum() > 0
    assert all(p.grad is not None for p in decoder.parameters())


def test_roi_alignment_uses_image_size_and_detaches_boxes():
    y, x = torch.meshgrid(torch.arange(4) + .5, torch.arange(4) + .5, indexing='ij')
    feature = torch.stack((x, y))
    boxes = torch.tensor([[.5, .5, .5, .5], [.5, .5, 0., .5],
                          [float('nan'), .5, .5, .5]], requires_grad=True)
    result = pool_dino_boxes(feature, boxes, (64, 64), grid_size=2)
    assert result.valid.tolist() == [True, False, False]
    expected = torch.tensor([[1.5, 1.5], [2.5, 1.5], [1.5, 2.5], [2.5, 2.5]])
    torch.testing.assert_close(result.features[0], expected)
    assert not result.features.requires_grad
    assert torch.count_nonzero(result.features[1:]) == 0
    empty = pool_dino_boxes(feature, boxes[:0], (64, 64), grid_size=2)
    assert empty.features.shape == (0, 4, 2)


def test_teacher_unpads_each_image_and_preserves_normalized_pixels():
    network = FakeDino().requires_grad_(False).eval()
    images = torch.randn(2, 3, 64, 80, requires_grad=True)
    image_hw = torch.tensor([[49, 65], [64, 48]])
    boxes = [torch.tensor([[.5, .5, 1., 1.]]), torch.tensor([[.5, .5, 1., 1.]])]
    targets = build_box_targets(network, images, image_hw, boxes, grid_size=2, skip_tokens=5)
    assert network.inputs[0].shape == (1, 3, 64, 80)
    assert network.inputs[1].shape == (1, 3, 64, 48)
    assert torch.equal(network.inputs[0][0, :, :49, :65], images[0, :, :49, :65])
    assert torch.count_nonzero(network.inputs[0][0, :, 49:]) == 0
    assert all(t.features.shape == (1, 4, 384) and not t.features.requires_grad for t in targets)
    separate = build_box_targets(network, images[1:2], image_hw[1:2], boxes[1:], grid_size=2, skip_tokens=5)
    torch.testing.assert_close(targets[1].features, separate[0].features, rtol=0, atol=0)


def test_matched_grid_loss_respects_permutation_repeated_matches_and_gradient_boundary():
    features = torch.tensor([[[1., 0., 0.]], [[0., 1., 0.]]], requires_grad=True)
    target = BoxFeatureTargets(features, torch.ones(2, dtype=torch.bool))
    prediction = features.detach()[[1, 0, 1, 0]][None].clone().requires_grad_()
    matches = [(torch.tensor([0, 1, 2]), torch.tensor([1, 0, 1]))]
    result = matched_grid_loss(prediction, [target], matches)
    torch.testing.assert_close(result['loss_h_dino'], torch.tensor(0.))
    assert result['h_dino_positive_assignments'] == 3
    with torch.no_grad():
        prediction[0, 1, 0, 1] = .3
    result = matched_grid_loss(prediction, [target], matches)
    result['loss_h_dino'].backward()
    assert prediction.grad.abs().sum() > 0 and features.grad is None
    assert torch.count_nonzero(prediction.grad[0, 3]) == 0
    assert result['h_dino_correct_minus_wrong_cosine'] > .8
    invalid = BoxFeatureTargets(features.detach(), torch.zeros(2, dtype=torch.bool))
    result = matched_grid_loss(prediction, [invalid], matches)
    assert result['loss_h_dino'] == 0 and result['h_dino_skipped_assignments'] == 3
    empty = [(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))]
    assert matched_grid_loss(prediction, [target], empty)['loss_h_dino'] == 0


def test_precomputed_final_pose_assignments_are_reused():
    pipeline = configuration().build(max_epochs=2).eval()
    data = batch()
    state, _, times, targets = build_poseflow_targets(data, pipeline.model,
        num_proposals=3, num_body_points=2, padding_policy='source_prior', noise_seed=42)
    output = pipeline.model(data['image'], state, times)
    output = {key: output[key] for key in ('pred_logits', 'pred_boxes', 'pred_keypoints')}
    matches = pipeline.criterion.matcher(output, targets)
    expected = pipeline.criterion(output, targets)
    class ForbiddenMatcher(torch.nn.Module):
        def forward(self, *args, **kwargs):
            raise AssertionError('Final matching must not run again')
    original = pipeline.criterion.matcher
    original.forward = ForbiddenMatcher().forward
    actual = pipeline.criterion(output, targets, final_matches=matches)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


def test_zero_weight_exact_v24_pose_loss_rng_and_shared_gradients():
    torch.manual_seed(42)
    base = v24_configuration(h_enabled=True).build(max_epochs=2).train()
    torch.manual_seed(42)
    variant = configuration(h_dino_weight=0).build(max_epochs=2).train()
    with torch.no_grad():
        base.model.h_branch.gates.fill_(.1)
        variant.model.h_branch.gates.fill_(.1)
    for name, value in base.model.state_dict().items():
        assert torch.equal(value, variant.model.state_dict()[name]), name
    data = batch()
    torch.manual_seed(91)
    first = base._losses_for_batch(data)['total']
    expected_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(91)
    second = variant._losses_for_batch(data)['total']
    assert torch.equal(expected_rng, torch.random.get_rng_state())
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    first.backward(); second.backward()
    for name, parameter in base.model.named_parameters():
        other = dict(variant.model.named_parameters())[name]
        if parameter.grad is None:
            assert other.grad is None, name
        else:
            torch.testing.assert_close(parameter.grad, other.grad, rtol=0, atol=0)
    assert variant._dino_teacher is None
    assert all(p.grad is None for p in variant.model.h_decoder.parameters())


def test_auxiliary_gradients_teacher_ownership_and_inference_skip():
    pipeline = configuration().build(max_epochs=2).train()
    teacher = FakeTeacher()
    pipeline._dino_teacher = teacher
    losses = pipeline._losses_for_batch(batch())
    losses['loss_h_dino'].backward()
    assert teacher.calls == 1 and not teacher.network.training
    for module in (pipeline.model.h_decoder, pipeline.model.h_branch.image_attention,
                   pipeline.model.h_branch.write_attention):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and sum(g.abs().sum() for g in grads) > 0
    assert all(p.grad is None for p in teacher.network.parameters())
    assert not any('teacher' in name for name in pipeline.state_dict())
    assert all(p is not teacher.network.scale for p in pipeline.parameters())
    def forbidden(*args):
        raise AssertionError('Auxiliary decoder must not execute during pose sampling')
    pipeline.model.h_decoder.forward = forbidden
    pipeline.eval().model.sample_keypoints(torch.randn(1, 3, 64, 64),
                                          generator=torch.Generator().manual_seed(42))
    assert teacher.calls == 1


def test_strict_restore_and_explicit_baseline_import(tmp_path):
    from scripts.evaluation.evaluate_poseflow import load_poseflow_model
    config = configuration(h_dino_weight=0)
    pipeline = config.build(max_epochs=2).eval()
    baseline = v24_configuration(h_enabled=True).build(max_epochs=2)
    report = pipeline.model.load_v24_state(baseline.model.state_dict())
    assert report['new_state_keys'] and all(k.startswith('h_decoder.') for k in report['new_state_keys'])
    with pytest.raises(RuntimeError):
        pipeline.model.load_v24_state({k:v for k,v in baseline.model.state_dict().items() if k != 'h_branch.gates'})
    torch.save({'state_dict': pipeline.state_dict()}, tmp_path/'model.ckpt')
    OmegaConf.save(OmegaConf.create({'pipeline': config.model_dump() | {
        '_target_':'src.models.pipelines.pose_estimators.poseflow.PoseFlowV25PipelineConfig'}}), tmp_path/'config.yaml')
    restored, _ = load_poseflow_model(tmp_path/'model.ckpt', tmp_path/'config.yaml')
    restored.eval()
    image = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        expected = pipeline.model.sample_keypoints(image, generator=torch.Generator().manual_seed(42))
        actual = restored.sample_keypoints(image, generator=torch.Generator().manual_seed(42))
    for key in ('pred_logits', 'pred_boxes', 'pred_keypoints'):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    assert not any('teacher' in k for k in restored.state_dict())


def test_variant_recipe_and_configuration_validation():
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=str(Path('conf').resolve()), version_base=None):
        config = compose(config_name='benchmark_poseflow_v25_h_dino')
    parsed = PoseFlowV25PipelineConfig.from_hydra_config(config.pipeline)
    assert parsed.h_enabled and parsed.h_dino_grid_size == 7 and parsed.h_dino_weight == 1
    assert parsed.dino_teacher_sha256 == '4610ad75edef83e75afdebf162d148dc628045ea6cbb83d67d4708c709c4f91d'
    with pytest.raises(ValueError):configuration(h_enabled=False)
    with pytest.raises(ValueError):configuration(h_dino_weight=-1)
    with pytest.raises(ValueError):configuration(dino_teacher_sha256='bad')


def test_empty_images_skip_teacher_and_empty_assignments_stay_differentiable():
    network = FakeDino()
    empty = [torch.empty(0, 4)]
    result = build_box_targets(network, torch.randn(1, 3, 64, 64),
                              torch.tensor([[64, 64]]), empty, grid_size=2)
    assert not network.inputs and result[0].features.shape == (0, 4, 384)
    pipeline = configuration().build(max_epochs=2).train()
    data = batch();data['gt_labeled'].zero_()
    losses = pipeline._losses_for_batch(data)
    assert pipeline._dino_teacher is None
    assert losses['loss_h_dino'] == 0 and losses['h_dino_positive_assignments'] == 0
    losses['total'].backward()
    assert all(torch.isfinite(value) for value in losses.values())


def test_distributed_normalization_handles_one_positive_across_two_ranks(monkeypatch):
    monkeypatch.setattr(torch.distributed, 'is_available', lambda: True)
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda: 2)
    # One local match and zero matches on the other rank: averaged gradients must
    # still equal the gradient of a single global matched candidate.
    monkeypatch.setattr(torch.distributed, 'all_reduce', lambda value: None)
    prediction = torch.tensor([[[[0., 1.]]]], requires_grad=True)
    target = BoxFeatureTargets(torch.tensor([[[1., 0.]]]), torch.tensor([True]))
    pairs = [(torch.tensor([0]), torch.tensor([0]))]
    loss = matched_grid_loss(prediction, [target], pairs)['loss_h_dino']
    torch.testing.assert_close(loss, torch.tensor(2.))
    loss.backward()
    torch.testing.assert_close(prediction.grad[0, 0, 0] / 2, torch.tensor([-1., 0.]))
