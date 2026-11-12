from copy import deepcopy

import pytest
import pytorch_lightning as pl
import torch

from src.models.pipelines.pose_estimators.poseflow.rows_v46 import PERSON, row_alpha_bar
from src.models.pipelines.pose_estimators.poseflow.sampling_v46 import ddim_transition
from src.models.pipelines.pose_estimators.poseflow.training_v47 import PoseFlowV47Pipeline, PoseFlowV47PipelineConfig
from tests.test_poseflow_v44 import arguments
from tests.test_poseflow_v43_training import batch
from tests.test_poseflow_v47 import context, model


@pytest.mark.parametrize('eos_at,count,reason', [(0, 0, 1), (2, 2, 1), (99, 6, 2)])
def test_coordinate_ddim_newborns_eos_capacity_and_traces(monkeypatch, eos_at, count, reason):
    network, seen = model(), []
    def forward(image, noisy_coordinates, row_times, token_types, padding_mask):
        row = len(seen)
        seen.append(noisy_coordinates.clone())
        coordinates = noisy_coordinates.new_full((*token_types.shape, 32), .1*(row+1))
        eos = noisy_coordinates.new_full(token_types.shape, 20. if row == eos_at else -20.)
        return dict(pred_coordinates=torch.stack([coordinates/2, coordinates]), eos_logits=torch.stack([eos, eos]))
    monkeypatch.setattr(network, 'forward_rows', forward)
    output = network.predict(context(network), torch.tensor([[200., 100.]]),
                             generator=torch.Generator().manual_seed(7), collect_trace=True)
    assert output['person_counts'].tolist() == [count] and output['stop_reason'].tolist() == [reason]
    assert len(seen) == 8 and len(output['trace']) == 16
    assert not output['valid_mask'][0, count:].any()
    if count:
        assert output['eos_row'].tolist() == ([3] if eos_at == 2 else [0])
        torch.testing.assert_close(output['pred_coordinates'][0, :count], torch.full((count, 32), .8))
        noise = torch.randn(1, 6, 32, generator=torch.Generator().manual_seed(7))
        alpha = row_alpha_bar(torch.tensor(6/7))
        torch.testing.assert_close(seen[1][0, 0, 0], alpha.sqrt()*6*(.2-1)+(1-alpha).sqrt()*noise[0, 0])
        expected = ddim_transition(seen[1][:, 0, :1], torch.full((1, 1, 32), 6*(.4-1)), 6/7, 5/7)
        torch.testing.assert_close(seen[2][:, 0, :1], expected)
    assert 'proposal_indices' not in output and 'reference_indices' not in output


def pipeline():
    config = PoseFlowV47PipelineConfig(native=arguments(), rows=8, capacity=6, codec_width=32,
        latent_dim=8, pretrained_checkpoint=None, quick_eval_enabled=False)
    return PoseFlowV47Pipeline(config=config, model=model(), max_epochs=80)


def test_pipeline_single_context_private_rng_and_asset_free_restore(monkeypatch, tmp_path):
    instance = pipeline().eval()
    calls = []
    original = instance.model.prepare_context
    def prepare(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(instance.model, 'prepare_context', prepare)
    assert torch.isfinite(instance._losses_for_batch(batch())['total']) and len(calls) == 1
    saved = dict(state_dict=deepcopy(instance.state_dict()), hyper_parameters=instance.hparams,
                 **{'pytorch-lightning_version': pl.__version__})
    instance.on_save_checkpoint(saved)
    path = tmp_path/'v47.ckpt'
    torch.save(saved, path)
    monkeypatch.setattr(PoseFlowV47PipelineConfig, 'build', lambda *_, **__: pytest.fail('External initialization'))
    restored = PoseFlowV47Pipeline.load_from_checkpoint(path, weights_only=False).eval()
    sample = batch()
    torch.testing.assert_close(instance._losses_for_batch(sample)['total'], restored._losses_for_batch(sample)['total'], rtol=0, atol=0)
    invalid = deepcopy(saved)
    invalid['v47_contract']['scale'] = 1.
    with pytest.raises(ValueError, match='contract'):
        restored.on_load_checkpoint(invalid)


def test_sampling_seed_batch_independence_and_cached_keys():
    network = model()
    for head in network.eos_heads:
        head.weight.data.zero_()
        head.bias.data.fill_(-5.)
    image = context(network, 2)
    sizes = torch.tensor([[100., 200.], [170., 90.]])
    output = network.predict(image, sizes, generator=[torch.Generator().manual_seed(seed) for seed in (3, 7)])
    for index, seed in enumerate((3, 7)):
        one = {key: tuple(item[index:index+1] for item in value) if isinstance(value, tuple)
               else value[index:index+1] for key, value in image.items()}
        prediction = network.predict(one, sizes[index:index+1], generator=torch.Generator().manual_seed(seed))
        torch.testing.assert_close(output['pred_coordinates'][index], prediction['pred_coordinates'][0], atol=2e-6, rtol=2e-5)
        assert 'trace' not in prediction
