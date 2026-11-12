from copy import deepcopy

import pytest
import pytorch_lightning as pl
import torch

from src.models.pipelines.pose_estimators.poseflow.rows_v46 import PERSON, row_alpha_bar
from src.models.pipelines.pose_estimators.poseflow.training_v46 import PoseFlowV46Pipeline, PoseFlowV46PipelineConfig
from tests.test_poseflow_v46 import model, context
from tests.test_poseflow_v44 import arguments
from tests.test_poseflow_v43_training import batch


def pipeline():
    config = PoseFlowV46PipelineConfig(native=arguments(), rows=8, capacity=6, codec_width=32, latent_dim=8,
        pretrained_checkpoint=None, quick_eval_enabled=False)
    return PoseFlowV46Pipeline(config=config, model=model(), max_epochs=80)


@pytest.mark.parametrize('eos_at,expected,reason', [(0, 0, 1), (2, 2, 1), (99, 6, 2)])
def test_sampling_eos_latches_capacity_identity_and_whole_prefix_noise(monkeypatch, eos_at, expected, reason):
    network = model()
    seen = []
    def forward(image, noisy_latents, row_times, token_types, padding_mask):
        row = len(seen)
        seen.append((noisy_latents.clone(), token_types.clone(), row_times.clone()))
        shape = (*token_types.shape, 8)
        latent = torch.arange(token_types.shape[-1], dtype=torch.float32)[None, None, :, None].expand(shape)+(row+1)*10
        eos = torch.full(token_types.shape, -10.)
        if row == eos_at:
            eos.fill_(10.)
        return dict(pred_latents=torch.stack([latent-1, latent]), eos_logits=torch.stack([eos, eos]))
    monkeypatch.setattr(network, 'forward_rows', forward)
    monkeypatch.setattr(network.codec, 'encode_posterior', lambda *_: pytest.fail('Sampling accessed GT'))
    output = network.predict(context(), torch.tensor([[160., 96.]]), generator=torch.Generator().manual_seed(42), collect_trace=True)
    assert output['person_counts'].tolist() == [expected]
    assert output['stop_reason'].tolist() == [reason]
    assert len(seen) == 8 and len(output['trace']) == 16
    assert output['sequence_indices'][0, :expected].tolist() == list(range(expected))
    if expected:
        torch.testing.assert_close(output['pred_latents'][0, :expected, 0], 80+torch.arange(expected).float())
        assert output['scores'][0, :expected].min() > .99
        # Row2 re-noises the clean birth10, not the transformer's hidden state or a DDIM state.
        generator = torch.Generator().manual_seed(42)
        noise = torch.randn(1, 6, 8, generator=generator)
        alpha = row_alpha_bar(torch.tensor([6/7]))
        torch.testing.assert_close(seen[1][0][0, 0, 0], alpha.sqrt()*10+(1-alpha).sqrt()*noise[0, 0])
    assert not output['valid_mask'][0, expected:].any()
    again = network.predict(context(), torch.tensor([[160., 96.]]), generator=torch.Generator().manual_seed(42))
    assert 'trace' not in again


def test_pipeline_one_context_rng_frozen_codec_and_strict_self_contained_restore(monkeypatch, tmp_path):
    instance = pipeline().eval()
    sample = batch((2, 1))
    original = instance.model.prepare_context
    calls = []
    def prepare(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(instance.model, 'prepare_context', prepare)
    assert torch.isfinite(instance._losses_for_batch(sample)['total']) and len(calls) == 1
    state = deepcopy(instance.model.codec.state_dict())
    saved = dict(state_dict=instance.state_dict(), hyper_parameters=instance.hparams,
                 **{'pytorch-lightning_version': pl.__version__})
    instance.on_save_checkpoint(saved)
    path = tmp_path/'v46.ckpt'
    torch.save(saved, path)
    # Loading can create architecture but must never read external initialization assets.
    monkeypatch.setattr(PoseFlowV46PipelineConfig, 'build', lambda *_, **__: pytest.fail('Restored through initializers'))
    restored = PoseFlowV46Pipeline.load_from_checkpoint(path, weights_only=False).eval()
    for key, value in state.items():
        torch.testing.assert_close(value, restored.model.codec.state_dict()[key], rtol=0, atol=0)
    first = instance._losses_for_batch(sample)['total']
    second = restored._losses_for_batch(sample)['total']
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    invalid = deepcopy(saved)
    invalid['v46_contract']['order'] = 'random'
    with pytest.raises(ValueError, match='contract'):
        restored.on_load_checkpoint(invalid)


def test_deterministic_sampling_per_image_and_validity_mask(monkeypatch):
    network = model()
    for head in network.eos_heads:
        head.weight.data.zero_()
        head.bias.data.fill_(-4.)
    image = context(2)
    sizes = torch.tensor([[100., 180.], [170., 90.]])
    generators = [torch.Generator().manual_seed(value) for value in (14, 21)]
    output = network.predict(image, sizes, generator=generators)
    for index, seed in enumerate((14, 21)):
        one = network.predict({key: value[index:index+1] for key, value in image.items()}, sizes[index:index+1],
                              generator=torch.Generator().manual_seed(seed))
        torch.testing.assert_close(output['pred_latents'][index], one['pred_latents'][0], atol=2e-6, rtol=2e-5)
