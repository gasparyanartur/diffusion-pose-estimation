import pytest
import torch
from tests.test_poseflow import _batch
from tests.test_poseflow_v13 import _config
from src.models.pipelines.pose_estimators.poseflow.latent_v12 import build_v12_targets
from src.models.pipelines.pose_estimators.poseflow.losses_v13 import matched_layer_losses
from src.models.pipelines.pose_estimators.diffusionregpose.training import DiffusionRegPoseOTAMatcher


def test_ota_duplicate_targets_receive_identical_latent_and_geometry_supervision():
    pipeline = _config(matching_strategy='ota').build(max_epochs=2)
    assert isinstance(pipeline.matcher, DiffusionRegPoseOTAMatcher)
    batch = _batch() | {'person_mask': torch.tensor([[True, False]])}
    targets = build_v12_targets(batch, groups=3)
    clean = pipeline.model.codec.encode(targets)
    slot = targets['real'][0].nonzero().item()
    output = {'pred_latents': clean[:,slot:slot+1].detach().expand(-1,3,-1,-1).clone(),
              'pred_logits': torch.full((1,3,1), 8., requires_grad=True),
              'pred_boxes': targets['boxes'][:,slot:slot+1].expand(-1,3,-1).clone(),
              'pred_keypoints': torch.cat((targets['joints'][:,slot:slot+1].flatten(-2),
                  targets['labeled'][:,slot:slot+1].float()), -1).expand(-1,3,-1).clone()}
    values = matched_layer_losses(output, targets, clean, pipeline.model.codec,
                                  pipeline.matcher, pipeline.v12_config.weights.to_dataclass())
    assert values['loss_latent'].abs() < 1e-6 and values['loss_keypoints'] == 0
    assert values['positive_assignments'] == 3 and values['target_people'] == 1
    assert values['assignments_per_person'] == 3
    probability = output['pred_logits'].sigmoid()
    expected = (.25 * torch.nn.functional.softplus(-output['pred_logits']) * (1-probability)**2).mean()
    torch.testing.assert_close(values['loss_ce'], expected)
    values['loss_ce'].backward()
    assert (output['pred_logits'].grad < 0).all()


def test_ota_empty_targets_and_experiment_config():
    pipeline = _config(matching_strategy='ota', ota_candidate_topk=3).build(max_epochs=2)
    batch = _batch() | {'person_mask': torch.zeros(1,2,dtype=torch.bool),
                       'image': torch.randn(1,3,64,64)}
    losses = pipeline._losses_for_batch(batch)
    assert all(torch.isfinite(value) for value in losses.values())
    assert losses['loss_bbox'] == losses['loss_keypoints'] == 0
    from pathlib import Path
    from hydra import compose, initialize_config_dir
    from src.models.pipelines.pose_estimators.poseflow import PoseFlowPipelineConfig
    with initialize_config_dir(config_dir=str(Path('conf').resolve()),version_base=None):
        config = compose(config_name='benchmark_poseflow_v13_ota_reconstruction_decay')
    parsed = PoseFlowPipelineConfig.from_hydra_config(config.pipeline)
    assert parsed.matching_strategy == 'ota' and parsed.ota_candidate_topk == 5
    assert parsed.reconstruction_weight_end == .01 and config.training.grad_clip == 2
    assert config.training.resume is None and config.training.max_epochs == 80
