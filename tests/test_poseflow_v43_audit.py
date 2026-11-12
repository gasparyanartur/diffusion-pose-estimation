from copy import deepcopy
import json
from types import SimpleNamespace

import torch
from torchvision.io import write_png

from src.models.pipelines.pose_estimators.poseflow.audit_v43 import AugmentedGeometryDataset, compute_codec_audit
from src.models.pipelines.pose_autoencoder.evaluation import export_reconstructions
from tests.test_poseflow_v43_codec import small_codec
from tests.test_pose_autoencoder import annotation_file, identified_geometry


def test_actual_transform_audit_preserves_annotation_identity_for_exports(tmp_path):
    batch = identified_geometry(1)
    path = annotation_file(tmp_path, batch)
    record = json.loads(path.read_text())
    record['images'][0]['file_name'] = 'image.png'
    path.write_text(json.dumps(record))
    write_png(torch.zeros(3, 360, 640, dtype=torch.uint8), str(tmp_path/'image.png'))
    config = SimpleNamespace(train_resize_scales=(128,), train_max_size=200,
        train_crop_resize_scales=(64,), train_crop_size=(60, 100), train_random_flip=True,
        train_random_crop=True, eval_resize_scale=128, eval_max_size=200)
    sample = AugmentedGeometryDataset(path, tmp_path, config)[0]
    for target in sample.values():
        assert target['annotation_id'].tolist() == [1]
        export_reconstructions(tmp_path/'example.png', target, target)
    assert (tmp_path/'example.png').exists()


def test_audit_uses_augmented_posterior_moments_and_preserves_rng_and_weights():
    codec = small_codec()
    saved = deepcopy(codec.model.state_dict())
    original = identified_geometry(3)
    augmented = {name: value.clone() for name, value in original.items()}
    augmented['boxes_cxcywh_px'][:, :2] *= .7
    augmented['keypoints_px'] *= .7
    batches = [dict(uncropped=original, augmented=augmented)]
    rng = torch.get_rng_state().clone()
    result, statistics, examples = compute_codec_audit(codec, batches)
    mean, logvar = codec.encode_posterior(augmented)
    torch.testing.assert_close(statistics['mean'], mean.double().mean(0))
    torch.testing.assert_close(statistics['std'], (mean.double().var(0, unbiased=False)+logvar.double().exp().mean(0)).sqrt())
    assert torch.equal(rng, torch.get_rng_state())
    assert result['augmented']['people'] == result['uncropped']['people'] == 3
    assert result['augmented']['normalization_max_error'] < 1e-4
    assert len(result['augmented']['transformed_target_sha256']) == 64
    assert examples['augmented']['target']['keypoints_px'].shape == (3, 14, 2)
    assert result['augmented']['mean']['joint_error_p95'] > 0
    for name, value in saved.items():
        torch.testing.assert_close(codec.model.state_dict()[name], value, rtol=0, atol=0)
