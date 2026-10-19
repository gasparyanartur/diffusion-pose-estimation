from pathlib import Path

import pytest
import torch

from src.models.pipelines.pose_estimators.poseflow.guidance_v20 import (
    project_pose3d, projected_pose_loss, score_distillation_loss, gfpose_marginal,
)
from src.models.pipelines.pose_estimators.poseflow.model_v20 import PoseFlowV20Model
from src.models.pipelines.pose_estimators.poseflow.training_v20 import PoseFlowV20PipelineConfig
from tests.test_poseflow import _batch
from tests.test_poseflow_v19 import configuration, source_model


class GaussianTeacher:
    def __init__(self):
        self.calls = 0

    def score(self, poses, times):
        self.calls += 1
        assert not torch.is_grad_enabled()
        return -poses


def v20_configuration(**overrides):
    values = configuration(detach_decoder_geometry=True).model_dump()
    return PoseFlowV20PipelineConfig(**(values | overrides))


def test_teacher_frame_projection_joint_mapping_and_missing_labels():
    pose = torch.arange(51.).reshape(1, 17, 3).requires_grad_()
    projected = project_pose3d(pose, joints=14)
    indices = [11, 14, 12, 15, 13, 16, 4, 1, 5, 2, 6, 3]
    torch.testing.assert_close(projected, (pose[:, indices, :2]/30+1)/2)
    target = projected.detach()+.1
    labeled = torch.zeros(1, 14, dtype=torch.bool)
    labeled[:, 0] = True
    target = torch.cat((target, torch.zeros(1, 2, 2)), 1)
    loss = projected_pose_loss(pose, target, labeled)
    torch.testing.assert_close(loss, torch.tensor(.2))
    loss.backward()
    assert pose.grad[:, 11, :2].abs().sum() > 0
    assert pose.grad[:, 14].abs().sum() == 0
    assert pose.grad[..., 2].abs().sum() == 0
    assert projected_pose_loss(pose, target, torch.zeros_like(labeled)) == 0


def test_distillation_sign_schedule_and_detached_teacher():
    pose = torch.randn(2, 17, 3, requires_grad=True)
    times = torch.tensor([.1, .6])
    noise = torch.randn_like(pose)
    alpha, sigma = gfpose_marginal(times)
    integral = .1*times+9.95*times.square()
    torch.testing.assert_close(alpha, torch.exp(-integral/2))
    torch.testing.assert_close(sigma, 1-torch.exp(-integral))
    teacher = GaussianTeacher()
    loss, metrics = score_distillation_loss(pose, teacher, times, noise)
    loss.backward()
    expected = (sigma[:, None, None]*(alpha[:, None, None]*pose.detach()
                +sigma[:, None, None]*noise)-noise)/pose.numel()
    torch.testing.assert_close(pose.grad, expected)
    assert teacher.calls == 1 and metrics['prior/force_rms'].isfinite()
    with pytest.raises(ValueError):
        gfpose_marginal(torch.tensor([0.]))
    with pytest.raises(ValueError):
        gfpose_marginal(torch.tensor([float('nan')]))


def test_dense_baseline_parity_and_new_head_gradients():
    from src.models.pipelines.pose_estimators.poseflow.dense_read_v19 import PoseFlowV19DenseReadModel
    torch.manual_seed(42)
    baseline = PoseFlowV19DenseReadModel(source_model()).eval()
    baseline.dense_read.gate.data.fill_(.2)
    model = PoseFlowV20Model(source_model()).eval()
    report = model.load_dense_state_dict(baseline.state_dict())
    assert report['new_keys'] == ['pose3d_head.bias', 'pose3d_head.weight']
    image, state = torch.randn(1, 3, 64, 64), torch.randn(1, 3, 8)
    for steps in [1, 3]:
        with torch.no_grad():
            expected = baseline.sample_keypoints(image, initial_latents=state, sampling_steps=steps)
            actual = model.sample_keypoints(image, initial_latents=state, sampling_steps=steps)
        for key in ('pred_coordinates', 'pred_logits', 'hypothesis_memory'):
            torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
    output = model.denoise(state, torch.ones(1), model.encode_image(image))
    pose = model.read_pose3d(output['hypothesis_memory'])
    loss, _ = score_distillation_loss(pose.flatten(0, 1), GaussianTeacher(),
        torch.full((3,), .3), torch.randn(3, 17, 3))
    gradients = torch.autograd.grad(loss, [model.pose3d_head.weight, model.hypothesis_slots,
        model.hypothesis_attention.attention.in_proj_weight, model.dense_read.attention.in_proj_weight])
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients)
    with pytest.raises(ValueError):
        model.load_dense_state_dict(model.state_dict())


def test_final_call_auxiliary_same_assignment_and_student_gradients(monkeypatch):
    pipeline = v20_configuration(rollout_probability=1., rollout_call_counts=[3]).build(max_epochs=2).train()
    teacher = GaussianTeacher()
    pipeline._teacher = teacher
    batch = _batch() | {'image': torch.randn(1, 3, 64, 64)}
    outputs = []
    original = pipeline.model.denoise
    def capture(*args, **kwargs):
        output = original(*args, **kwargs)
        outputs.append(output)
        return output
    monkeypatch.setattr(pipeline.model, 'denoise', capture)
    losses = pipeline._losses_for_batch(batch)
    assert len(outputs) == 3 and teacher.calls == 1
    torch.testing.assert_close(losses['total'], losses['loss_prediction']+losses['loss_pose3d_weighted'])
    gradients = torch.autograd.grad(losses['loss_pose3d_weighted'],
        [outputs[0]['hypothesis_memory'], pipeline.model.pose3d_head.weight,
         pipeline.model.hypothesis_slots, pipeline.model.dense_read.gate], retain_graph=True)
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients)
    task_gradient = torch.autograd.grad(losses['loss_prediction'],
        pipeline.model.layers[-1].block.self_attn.in_proj_weight, retain_graph=True)[0]
    assert task_gradient.abs().sum() > 0
    losses['total'].backward()
    empty = batch | {'person_mask': torch.zeros(1, 2, dtype=torch.bool)}
    previous = teacher.calls
    losses = pipeline._losses_for_batch(empty)
    assert losses['loss_pose3d_prior'] == losses['loss_pose3d_projection'] == 0
    assert teacher.calls == previous and losses['total'].isfinite()


def test_zero_auxiliary_parity_rng_and_checkpoint_without_teacher(tmp_path):
    from tests.test_poseflow_v19_dense_read_integration import dense_configuration
    from scripts.evaluation.evaluate_poseflow import load_poseflow_model
    from omegaconf import OmegaConf
    baseline = dense_configuration(rollout_probability=0.).build(max_epochs=2).train()
    pipeline = v20_configuration(rollout_probability=0., pose3d_prior_weight=0.,
        pose3d_projection_weight=0., gfpose_checkpoint='missing.pth').build(max_epochs=2).train()
    pipeline.model.load_dense_state_dict(baseline.model.state_dict())
    batch = _batch() | {'image': torch.randn(1, 3, 64, 64)}
    torch.manual_seed(17)
    expected = baseline._losses_for_batch(batch)
    expected_rng = torch.random.get_rng_state()
    torch.manual_seed(17)
    actual = pipeline._losses_for_batch(batch)
    assert actual.keys() == expected.keys()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)
    assert torch.equal(torch.random.get_rng_state(), expected_rng)
    state = pipeline.state_dict()
    assert not any('teacher' in key for key in state)
    assert pipeline._teacher is None
    torch.save({'state_dict': state}, tmp_path/'v20.ckpt')
    config_path = tmp_path/'config.yaml'
    OmegaConf.save(OmegaConf.create({'pipeline': pipeline.v12_config.model_dump() | {
        '_target_': 'src.models.pipelines.pose_estimators.poseflow.PoseFlowV20PipelineConfig'}}), config_path)
    restored, _ = load_poseflow_model(tmp_path/'v20.ckpt', config_path)
    assert isinstance(restored, PoseFlowV20Model)
    state.pop('model.pose3d_head.weight')
    torch.save({'state_dict': state}, tmp_path/'broken.ckpt')
    with pytest.raises(RuntimeError, match='Missing key'):
        load_poseflow_model(tmp_path/'broken.ckpt', config_path)


def test_guidance_noise_is_independent_and_restorable():
    pipeline = v20_configuration().build(max_epochs=2)
    poses = torch.zeros(3, 17, 3)
    torch.manual_seed(13)
    expected_rng = torch.random.get_rng_state()
    first = pipeline._guidance_noise(poses)
    assert torch.equal(torch.random.get_rng_state(), expected_rng)
    saved = {key: value.clone() for key, value in pipeline.state_dict().items()}
    second = pipeline._guidance_noise(poses)
    assert not torch.equal(first[1], second[1])
    pipeline.load_state_dict(saved, strict=True)
    repeated = pipeline._guidance_noise(poses)
    assert all(torch.equal(a, b) for a, b in zip(second, repeated))


def test_v20_configuration_inherits_dense_recipe():
    from hydra import compose, initialize_config_dir
    from src.models.pipelines.pose_estimators.poseflow import PoseFlowPipelineConfig
    with initialize_config_dir(config_dir=str(Path('conf').resolve()), version_base=None):
        resolved = compose(config_name='benchmark_poseflow_v20')
    config = PoseFlowPipelineConfig.from_hydra_config(resolved.pipeline)
    assert isinstance(config, PoseFlowV20PipelineConfig)
    assert config.detach_decoder_geometry and config.rollout_probability == .5
    assert resolved.training.max_epochs == 80 and resolved.training.grad_clip == 2
    assert resolved.run_name == 'poseflow_v20_training'
    assert config.pose3d_prior_weight == 1. and config.pose3d_projection_weight == 1.
    with pytest.raises(ValueError):
        v20_configuration(pose3d_noise_min=.7, pose3d_noise_max=.6)


def test_frozen_teacher_ema_and_hash_contract(tmp_path):
    from src.models.pipelines.pose_estimators.poseflow.gfpose.network import GFPoseNetwork
    from src.models.pipelines.pose_estimators.poseflow.gfpose.teacher import FrozenGFPose
    network = GFPoseNetwork()
    shadows = [torch.zeros_like(parameter) for parameter in network.parameters()]
    checkpoint = tmp_path/'teacher.pth'
    torch.save({'model_state_dict': network.state_dict(),
                'ema': {'shadow_params': shadows}}, checkpoint)
    with pytest.raises(ValueError, match='SHA256'):
        FrozenGFPose(checkpoint, '0'*64)
    torch.manual_seed(31)
    expected_rng = torch.random.get_rng_state()
    teacher = FrozenGFPose(checkpoint, FrozenGFPose._digest(checkpoint))
    assert torch.equal(torch.random.get_rng_state(), expected_rng)
    assert not teacher.network.training
    assert all(not p.requires_grad and torch.count_nonzero(p) == 0
               for p in teacher.network.parameters())
    score = teacher.score(torch.randn(2, 17, 3), torch.tensor([.1, .6]))
    assert score.dtype == torch.float32 and not score.requires_grad
    assert score.abs().sum() == 0
    with pytest.raises(FileNotFoundError):
        FrozenGFPose(tmp_path/'missing', '0'*64)
