import hashlib
from copy import deepcopy

import pytest
import torch

from src.models.pipelines.pose_vae.fidelity import FidelityPoseVAE
from src.models.pipelines.pose_estimators.poseflow.codec_v43 import (
    FrozenPoseCodec, PosteriorMoments, geometry_from_batch, load_frozen_codec,
)
from tests.test_pose_autoencoder import identified_geometry


def small_codec():
    model = FidelityPoseVAE(width=32, latent_dim=8)
    return FrozenPoseCodec(model, torch.arange(8).float(), torch.linspace(.2, 2., 8))


def test_posterior_moments_include_variance_and_are_batch_independent():
    mean = torch.tensor([[1., 3.], [5., 7.], [2., 4.]])
    logvar = torch.tensor([[.1, .2], [.3, .4], [.5, .6]]).log()
    moments = PosteriorMoments(2)
    moments.update(mean[:1], logvar[:1])
    moments.update(mean[1:], logvar[1:])
    center, scale = moments.finalize()
    torch.testing.assert_close(center, mean.double().mean(0))
    torch.testing.assert_close(scale, (mean.double().var(0, unbiased=False)+logvar.double().exp().mean(0)).sqrt())
    with pytest.raises(ValueError, match='empty'):
        PosteriorMoments(2).finalize()
    with pytest.raises(ValueError, match='finite'):
        moments.update(mean * float('nan'), logvar)


def test_affine_sampling_decoder_parity_and_frozen_input_gradients():
    codec = small_codec().train()
    batch = identified_geometry(1)
    saved = deepcopy(codec.state_dict())
    assert not codec.training and not codec.model.training and not any(p.requires_grad for p in codec.parameters())
    mean, logvar = codec.encode_posterior(batch)
    generator = torch.Generator().manual_seed(42)
    standardized = codec.sample_latents(batch, generator=generator)
    expected = mean+(.5*logvar).exp()*torch.randn(mean.shape, generator=torch.Generator().manual_seed(42))
    torch.testing.assert_close(codec.unstandardize(standardized), expected)
    expected_geometry = codec.model.decode(expected, batch['image_wh'])
    predicted = codec.decode_latents(standardized, batch['image_wh'])
    for name in predicted:
        torch.testing.assert_close(predicted[name], expected_geometry[name], atol=1e-4, rtol=1e-5)
    latent = standardized.detach().requires_grad_()
    codec.decode_latents(latent, batch['image_wh'])['keypoints_px'].square().mean().backward()
    assert torch.isfinite(latent.grad).all() and latent.grad.abs().sum() > 0
    assert all(p.grad is None for p in codec.parameters())
    for name, value in codec.state_dict().items():
        torch.testing.assert_close(value, saved[name], rtol=0, atol=0)
    restored = small_codec()
    restored.load_state_dict(saved, strict=True)
    torch.testing.assert_close(restored.sample_latents(batch, generator=torch.Generator().manual_seed(42)), standardized)


def test_statistics_must_be_ready_finite_positive_and_correct_dimension():
    codec = FrozenPoseCodec(FidelityPoseVAE(width=32, latent_dim=8))
    with pytest.raises(RuntimeError, match='statistics'):
        codec.sample_latents(identified_geometry(1), generator=torch.Generator())
    for mean, std in ((torch.zeros(7), torch.ones(7)), (torch.zeros(8), torch.zeros(8)),
                      (torch.full((8,), float('nan')), torch.ones(8))):
        with pytest.raises(ValueError):
            codec.set_statistics(mean, std)


def test_batch_conversion_uses_transformed_unpadded_frame_and_filters_padding():
    points = torch.zeros(2, 3, 14, 2)
    points[0, 0, 0] = torch.tensor([-1.5, 1.5])
    points[0, 0, 1] = float('nan')
    labels = torch.zeros(2, 3, 14, dtype=torch.bool)
    labels[:, 0, 0] = True
    data = dict(gt_keypoints=points, gt_labeled=labels,
        gt_boxes_pixel=torch.tensor([[[10., 20., 30., 50.], [0., 0., 0., 0.], [1., 2., -1., 3.]]]*2),
        person_mask=torch.tensor([[1, 0, 1]]*2), gt_box_labeled=torch.ones(2, 3),
        img_hw=torch.tensor([[100., 200.], [200., 50.]]), original_img_hw=torch.ones(2, 2)*900,
        image_id=torch.tensor([11, 22]))
    geometry, locations = geometry_from_batch(data)
    assert locations.tolist() == [[0, 0], [1, 0]]
    torch.testing.assert_close(geometry['image_wh'], torch.tensor([[200., 100.], [50., 200.]]))
    torch.testing.assert_close(geometry['boxes_cxcywh_px'][0], torch.tensor([25., 45., 30., 50.]))
    torch.testing.assert_close(geometry['keypoints_px'][0, 0], torch.tensor([-50., 125.]))
    assert not geometry['joint_mask'][:, 1:].any()
    assert torch.isfinite(small_codec().encode_posterior(geometry)[0]).all()
    data['gt_labeled'].zero_()
    empty, locations = geometry_from_batch(data)
    assert empty['keypoints_px'].shape == (0, 14, 2) and locations.shape == (0, 2)


def test_strict_asset_loading_checks_hash_shapes_and_preserves_rng(tmp_path):
    model = FidelityPoseVAE(width=32, latent_dim=8)
    path = tmp_path/'codec.ckpt'
    torch.save(dict(state_dict={'model.'+name: value for name, value in model.state_dict().items()},
                    hyper_parameters=dict(width=32, latent_dim=8)), path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    rng = torch.get_rng_state().clone()
    loaded = load_frozen_codec(path, expected_sha256=digest, width=32, latent_dim=8)
    assert torch.equal(rng, torch.get_rng_state())
    for name, value in model.state_dict().items():
        torch.testing.assert_close(loaded.model.state_dict()[name], value, rtol=0, atol=0)
    with pytest.raises(ValueError, match='hash'):
        load_frozen_codec(path, expected_sha256='0'*64, width=32, latent_dim=8)
    with pytest.raises(ValueError, match='dimension'):
        load_frozen_codec(path, expected_sha256=digest)
