from copy import deepcopy

import pytest
import torch

from src.models.pipelines.pose_estimators.poseflow.codec_v43 import geometry_from_batch
from src.models.pipelines.pose_estimators.poseflow.coordinates_v47 import ordered_coordinates
from src.models.pipelines.pose_estimators.poseflow.losses_v47 import row_losses
from src.models.pipelines.pose_estimators.poseflow.rows_token_age import build_training_rows
from src.models.pipelines.pose_estimators.poseflow.training_v49 import PoseFlowV49Pipeline, PoseFlowV49PipelineConfig
from tests.test_poseflow_v43_training import batch
from tests.test_poseflow_v44 import arguments
from tests.test_poseflow_v49 import model


def pipeline():
    config = PoseFlowV49PipelineConfig(native=arguments(), rows=7, capacity=6, codec_width=32,
        latent_dim=8, pretrained_checkpoint=None, quick_eval_enabled=False)
    return PoseFlowV49Pipeline(config=config, model=model(), max_epochs=80)


@pytest.mark.parametrize('counts', [(2, 1), (0,)])
def test_pipeline_encodes_image_once_has_finite_all_active_losses_and_codec_gradients(counts, monkeypatch):
    module = pipeline().train()
    calls = []
    original = module.model.prepare_context
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(module.model, 'prepare_context', counted)
    losses = module._losses_for_batch(batch(counts))
    assert len(calls) == 1
    torch.testing.assert_close(losses['total'], 5*(losses['coordinate_birth']+losses['coordinate_refine'])+
        .25*(losses['geometry_birth']+losses['geometry_refine'])+losses['eos'])
    losses['total'].backward()
    assert torch.isfinite(losses['total'])
    for head in module.model.eos_heads:
        assert head.weight.grad.abs().sum() > 0
    if sum(counts):
        assert module.model.person_codec.encoder[0].weight.grad.abs().sum() > 0
        for head in module.model.latent_heads:
            assert head.weight.grad.abs().sum() > 0
    else:
        assert losses['coordinate_birth'] == losses['coordinate_refine'] == 0


def test_source_loss_reduction_weights_each_person_equally_and_ignores_missing_joints():
    sample = batch((2, 1))
    geometry, locations = geometry_from_batch(sample)
    sources = ordered_coordinates(geometry, locations, batch_size=2, capacity=6)
    sources['coordinate_mask'][0, 0, :6] = False
    rows = build_training_rows(sources['coordinates'], sources['counts'], row_count=7,
        coordinate_mask=sources['coordinate_mask'], target_indices=sources['target_indices'],
        generator=torch.Generator().manual_seed(1))
    targets = torch.nn.functional.pad(rows['sources']['coordinates'], (0,0,0,1))
    prediction = targets[None,:,None].expand(2,2,7,7,32).clone()
    for image, count in enumerate((2,1)):
        for rank in range(count):
            prediction[:,image,:,rank] += 1+rank
    missing = ~torch.nn.functional.pad(rows['sources']['coordinate_mask'], (0,0,0,1))
    prediction = prediction.masked_fill(missing[None,:,None], 10000.).requires_grad_()
    losses = row_losses(dict(pred_coordinates=prediction,eos_logits=torch.zeros(2,2,7,7)), rows,
        rows['sources'], geometry, sample['img_hw'].flip(-1))
    torch.testing.assert_close(losses['coordinate_birth'], torch.tensor(2.))
    torch.testing.assert_close(losses['coordinate_refine'], torch.tensor(2.))
    (losses['coordinate_birth']+losses['coordinate_refine']).backward()
    assert prediction.grad.masked_select(missing[None,:,None]).eq(0).all()


def test_strict_restore_without_initializer_and_global_rng(tmp_path):
    import pytorch_lightning as pl
    module = pipeline()
    record = dict(state_dict=module.state_dict(),hyper_parameters=module.hparams,
                  **{'pytorch-lightning_version':pl.__version__})
    module.on_save_checkpoint(record)
    expected_rng = torch.get_rng_state().clone()
    filename = tmp_path/'state.ckpt'
    torch.save(record, filename)
    restored = PoseFlowV49Pipeline.load_from_checkpoint(filename, weights_only=False)
    assert torch.equal(expected_rng, torch.get_rng_state())
    for key, value in module.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])
    broken = deepcopy(record)
    broken['v49_contract']['noise'] = 'wrong'
    with pytest.raises(ValueError,match='V49'):
        module.on_load_checkpoint(broken)
