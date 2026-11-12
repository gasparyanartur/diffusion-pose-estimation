import torch

from src.models.pipelines.pose_autoencoder.model import geometry_loss
from tests.test_pose_autoencoder import identified_geometry
from tests.test_poseflow_v43_codec import small_codec


def test_tiny_valid_boxes_outside_joints_and_missing_nan_targets_remain_finite():
    codec = small_codec()
    target = identified_geometry(2)
    target['boxes_cxcywh_px'][0, 2:] = torch.tensor([.01, .02])
    target['keypoints_px'][1, 0] = torch.tensor([-1000., 3000.])
    target['joint_mask'][:, 2] = False
    target['keypoints_px'][:, 2] = float('nan')
    mean, _ = codec.encode_posterior(target)
    latent = codec.standardize(mean).detach().requires_grad_()
    prediction = codec.decode_latents(latent, target['image_wh'])
    assert (prediction['boxes_cxcywh_px'][:, 2:] > 0).all()
    loss = geometry_loss(prediction, target)['loss']
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(latent.grad).all()
    assert latent.grad.abs().sum() > 0 and all(parameter.grad is None for parameter in codec.parameters())
