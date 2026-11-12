import json

import pytest
import torch

from src.data.pose_geometry import normalize_geometry, restore_geometry
from src.models.pipelines.pose_autoencoder.model import PoseBoxAutoencoder, geometry_loss, joint_exponents
from src.models.pipelines.pose_autoencoder.evaluation import CodecAPEvaluator, perturb_latents
from tests.test_pose_vae import geometry


def identified_geometry(count=3):
    batch = geometry(count)
    batch['annotation_id'] = torch.arange(1, count+1)
    batch['image_id'] = torch.arange(1, count+1)
    return batch


def annotation_file(directory, batch):
    annotations, images = [], []
    for index in range(len(batch['image_wh'])):
        box = batch['boxes_cxcywh_px'][index]
        points = torch.cat((batch['keypoints_px'][index], batch['joint_mask'][index, :, None].float()*2), -1)
        annotations.append(dict(id=int(batch['annotation_id'][index]), image_id=int(batch['image_id'][index]),
                                category_id=1, iscrowd=0, num_keypoints=int(batch['joint_mask'][index].sum()),
                                bbox=torch.cat((box[:2]-box[2:]/2, box[2:])).tolist(),
                                keypoints=points.flatten().tolist()))
        images.append(dict(id=int(batch['image_id'][index]), width=640, height=360, crowdIndex=.1))
    path = directory/'annotations.json'
    path.write_text(json.dumps(dict(images=images, annotations=annotations, categories=[dict(id=1, name='person')])) )
    return path


def test_codec_is_deterministic_and_decoder_has_no_coordinate_bypass():
    model = PoseBoxAutoencoder()
    batch = geometry(1)
    latent = model.encode(**batch)
    assert latent.shape == (1, 32)
    torch.testing.assert_close(latent, model.encode(**batch), rtol=0, atol=0)
    assert not any('logvar' in name for name, _ in model.named_parameters())
    prediction = model.decode(latent, batch['image_wh'])
    assert prediction['keypoints_px'].shape == (1, 14, 2)
    assert (prediction['boxes_cxcywh_px'][:, 2:] > 0).all()
    geometry_loss(prediction, batch)['loss'].backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())
    changed = dict(batch, keypoints_px=batch['keypoints_px'].clone())
    changed['keypoints_px'][:, 2] = float('nan')
    torch.testing.assert_close(model.encode(**changed), latent, rtol=0, atol=0)


def test_oks_weighted_loss_uses_area_sigmas_and_missing_mask():
    batch = geometry(1)
    prediction = {name: value.clone() for name, value in batch.items()}
    assert geometry_loss(prediction, batch)['loss'] == 0
    prediction['keypoints_px'][:, 0, 0] += 10
    expected = 100/(8*.079**2*.53*100*160)
    torch.testing.assert_close(joint_exponents(prediction, batch)[0, 0], torch.tensor(expected))
    prediction['keypoints_px'][:, 2] = float('nan')
    assert torch.isfinite(geometry_loss(prediction, batch)['loss'])
    prediction['boxes_cxcywh_px'][:, 0] += 10
    torch.testing.assert_close(geometry_loss(prediction, batch)['box_center'], torch.tensor(.005))


def test_native_ap_ceilings_order_and_paired_errors(tmp_path):
    batch = identified_geometry()
    evaluator = CodecAPEvaluator(annotation_file(tmp_path, batch))
    relative, placement = normalize_geometry(**{key:batch[key] for key in
                                                ('keypoints_px','boxes_cxcywh_px','joint_mask','image_wh')})
    for prediction in (batch, restore_geometry(relative, placement, batch['image_wh'])):
        metrics = evaluator.evaluate(prediction, batch)
        for name in ('pose_AP', 'bbox_AP', 'pose_AP75', 'pose_AP95', 'bbox_AP95'):
            assert metrics[name] == pytest.approx(100)
        assert metrics['paired_OKS_mean'] == pytest.approx(1)
        assert metrics['joint_error_p95'] < 1e-6
    changed = dict(batch, keypoints_px=batch['keypoints_px']+20)
    assert evaluator.evaluate(changed, batch)['pose_AP'] < 100
    with pytest.raises(ValueError, match='order'):
        evaluator.evaluate(batch, dict(batch, annotation_id=batch['annotation_id'].flip(0)))


def test_latent_noise_reproducible_scaled_and_zero_is_exact():
    latent = torch.arange(96).reshape(3, 32).float()
    std = torch.linspace(.1, 3, 32)
    first = perturb_latents(latent, std, .05, seed=1042)
    second = perturb_latents(latent, std, .05, seed=1042)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(perturb_latents(latent, std, 0, seed=1042), latent, rtol=0, atol=0)
    torch.testing.assert_close((first-latent), .5*(perturb_latents(latent, std, .1, seed=1042)-latent),
                               atol=4e-6, rtol=1e-4)
