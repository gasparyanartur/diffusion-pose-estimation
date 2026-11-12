from copy import deepcopy

import pytest
import torch

from src.models.pipelines.pose_estimators.poseflow.sampling_v46 import ddim_transition
from src.models.pipelines.pose_estimators.poseflow.rows_v46 import row_alpha_bar
from tests.test_poseflow_v46 import model, context


def test_ddim_matches_independent_arithmetic_and_oracle_entire_trajectory():
    clean, noise = torch.randn(2, 6, 8), torch.randn(2, 6, 8)
    times = torch.linspace(1., 0., 64)
    state = noise.clone()
    for current, following in zip(times[:-1], times[1:]):
        updated = ddim_transition(state, clean, current, following)
        alpha = row_alpha_bar(following)
        expected = alpha.sqrt()*clean+(1-alpha).sqrt()*noise
        torch.testing.assert_close(updated, expected, atol=3e-6, rtol=3e-5)
        state = updated
    torch.testing.assert_close(state, clean, rtol=0, atol=0)
    # Non-oracle estimate: compare against independent FP64 residual arithmetic.
    state, prediction = torch.randn(2, 6, 8), torch.randn(2, 6, 8)
    current, following = torch.tensor(.7), torch.tensor(.4)
    a, b = row_alpha_bar(torch.stack((current, following)).double())
    expected = b.sqrt()*prediction.double()+(1-b).sqrt()*(state.double()-a.sqrt()*prediction.double())/(1-a).sqrt()
    torch.testing.assert_close(ddim_transition(state, prediction, current, following).double(), expected, atol=1e-6, rtol=1e-5)


def test_ddim_terminal_and_invalid_time_rejection():
    state, prediction = torch.randn(1, 6, 8), torch.randn(1, 6, 8)
    torch.testing.assert_close(ddim_transition(state, prediction, 0., 0.), prediction, rtol=0, atol=0)
    for current, following in ((.2, .3), (-.1, 0.), (1.1, .5)):
        with pytest.raises(ValueError, match='time'):
            ddim_transition(state, prediction, current, following)


def test_sampler_default_exact_outputs_and_rng_and_ddim_determinism():
    network = model()
    for head in network.eos_heads:
        head.weight.data.zero_()
        head.bias.data.fill_(-4.)
    image, sizes = context(), torch.tensor([[100., 180.]])
    outputs, rng = [], []
    for kwargs in ({}, {'sampler': 'renoise'}, {'sampler': 'ddim'}, {'sampler': 'ddim'}):
        generator = torch.Generator().manual_seed(18)
        outputs.append(network.predict(image, sizes, generator=generator, **kwargs))
        rng.append(generator.get_state())
    for key in ('pred_latents', 'pred_logits', 'person_counts', 'stop_reason'):
        torch.testing.assert_close(outputs[0][key], outputs[1][key], atol=0, rtol=0)
        torch.testing.assert_close(outputs[2][key], outputs[3][key], atol=0, rtol=0)
    assert torch.equal(rng[0], rng[1]) and torch.equal(rng[0], rng[2])
    assert not torch.allclose(outputs[0]['pred_latents'], outputs[2]['pred_latents'])
    with pytest.raises(ValueError, match='sampler'):
        network.predict(image, sizes, sampler='ddpm')


def test_newborn_uses_noise_but_existing_person_uses_residual_after_eos(monkeypatch):
    network = model()
    seen = []
    def forward(context, noisy_latents, row_times, token_types, padding_mask):
        row = len(seen)
        seen.append(noisy_latents.clone())
        prediction = noisy_latents.new_full((*token_types.shape, 8), row+10.)
        eos = noisy_latents.new_full(token_types.shape, -10. if row < 2 else 10.)
        return dict(pred_latents=torch.stack((prediction, prediction)), eos_logits=torch.stack((eos, eos)))
    monkeypatch.setattr(network, 'forward_rows', forward)
    output = network.predict(context(), torch.tensor([[100., 180.]]),
        generator=torch.Generator().manual_seed(13), sampler='ddim')
    generator = torch.Generator().manual_seed(13)
    noise1, noise2 = [torch.randn(1, 6, 8, generator=generator) for _ in range(2)]
    times = torch.linspace(1., 0., 8)
    a1, a2 = row_alpha_bar(times[1:3])
    torch.testing.assert_close(seen[1][0, 0, 0], a1.sqrt()*10+(1-a1).sqrt()*noise1[0, 0])
    expected = ddim_transition(seen[1][0, 0, 0], torch.full((8,), 11.), times[1], times[2])
    torch.testing.assert_close(seen[2][0, 0, 0], expected)
    torch.testing.assert_close(seen[2][0, 0, 1], a2.sqrt()*11+(1-a2).sqrt()*noise2[0, 1])
    assert output['person_counts'].tolist() == [2] and output['stop_reason'].tolist() == [1]
    assert output['eos_row'].tolist() == [3]
    assert torch.all(output['pred_latents'][0, :2] == 17)


def test_per_image_ddim_rng_batch_invariance_and_unchanged_state():
    network = model()
    for head in network.eos_heads:
        head.weight.data.zero_()
        head.bias.data.fill_(-4.)
    saved = deepcopy(network.state_dict())
    image, sizes = context(2), torch.tensor([[100., 180.], [170., 90.]])
    output = network.predict(image, sizes, generator=[torch.Generator().manual_seed(s) for s in (14, 21)], sampler='ddim')
    for index, seed in enumerate((14, 21)):
        single = network.predict({k: v[index:index+1] for k, v in image.items()}, sizes[index:index+1],
                                 generator=torch.Generator().manual_seed(seed), sampler='ddim')
        torch.testing.assert_close(output['pred_latents'][index], single['pred_latents'][0], atol=1e-5, rtol=1e-4)
    for key, value in saved.items():
        assert torch.equal(value, network.state_dict()[key])
