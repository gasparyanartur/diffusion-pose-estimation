import pytest
import torch

from tests.test_poseflow_v22 import configuration


@pytest.mark.parametrize('batch_size', [2, 4])
def test_coordinate_attention_preserves_image_and_timestep_isolation(batch_size):
    torch.manual_seed(42)
    model = configuration(coordinate_batch_layout='per_image')._build_poseflow_model().eval()
    groups, joints = 5, 3
    noisy = torch.rand(groups, batch_size, joints, 2)
    reference = torch.rand(groups, joints, batch_size, 2)
    times = torch.linspace(.1, .9, batch_size)
    with torch.no_grad():
        for module in model.coordinate_conditioners.values():
            module.affine.weight.normal_(std=.02)
        actual = model.refine_diffusion_coordinates(noisy, reference, times)
        individual = torch.cat([
            model.refine_diffusion_coordinates(noisy[:, i:i+1], reference[:, :, i:i+1], times[i:i+1])
            for i in range(batch_size)], dim=2)
        torch.testing.assert_close(actual, individual, atol=2e-7, rtol=1e-6)
        changed_noisy, changed_reference, changed_times = noisy.clone(), reference.clone(), times.clone()
        changed_noisy[:, 1:] = 1-changed_noisy[:, 1:]
        changed_reference[:, :, 1:] = 1-changed_reference[:, :, 1:]
        changed_times[1:] = 1-changed_times[1:]
        changed = model.refine_diffusion_coordinates(changed_noisy, changed_reference, changed_times)
        torch.testing.assert_close(changed[:, :, 0], actual[:, :, 0], atol=0, rtol=0)


def test_single_image_parity_and_historical_default():
    torch.manual_seed(7)
    legacy = configuration()._build_poseflow_model().eval()
    assert legacy.coordinate_batch_layout == 'legacy'
    fixed = configuration(coordinate_batch_layout='per_image')._build_poseflow_model().eval()
    fixed.load_state_dict(legacy.state_dict(), strict=True)
    noisy, reference = torch.rand(5, 1, 3, 2), torch.rand(5, 3, 1, 2)
    times = torch.tensor([.3])
    expected = legacy.refine_diffusion_coordinates(noisy, reference, times)
    actual = fixed.refine_diffusion_coordinates(noisy, reference, times)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    image, state = torch.randn(1, 3, 64, 64), torch.randn(1, 3, 4)
    legacy.flow_sampling_steps = fixed.flow_sampling_steps = 3
    expected = legacy.sample_keypoints(image, generator=torch.Generator().manual_seed(42))
    actual = fixed.sample_keypoints(image, generator=torch.Generator().manual_seed(42))
    for name in ('pred_logits', 'pred_boxes', 'pred_keypoints', 'hypothesis_memory', 'flow_keypoints'):
        torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)
    with pytest.raises(ValueError):
        configuration(coordinate_batch_layout='unknown')
