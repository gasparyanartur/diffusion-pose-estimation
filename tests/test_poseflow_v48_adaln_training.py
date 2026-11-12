from copy import deepcopy
from pathlib import Path

from hydra import compose, initialize_config_dir
import pytest
import pytorch_lightning as pl
import torch

from src.models.pipelines.pose_estimators.poseflow.training_v48_adaln import PoseFlowV48AdaLNPipeline, PoseFlowV48AdaLNPipelineConfig
from src.models.pipelines.pose_estimators.poseflow.coordinates_v47 import ordered_coordinates
from src.models.pipelines.pose_estimators.poseflow.codec_v43 import geometry_from_batch
from src.models.pipelines.pose_estimators.poseflow.rows_v48 import training_rows
from src.models.pipelines.pose_estimators.poseflow.losses_v48 import row_losses
from tests.test_poseflow_v43_training import batch
from tests.test_poseflow_v44 import arguments
from tests.test_poseflow_v48 import context
from tests.test_poseflow_v48_adaln import pair


def test_modulation_and_embedding_gradients_checkpoint_parity_and_masking():
    _, checkpointed = pair()
    plain = deepcopy(checkpointed)
    plain.activation_checkpointing = False
    sample = batch((2, 0))
    geometry, locations = geometry_from_batch(sample)
    sources = ordered_coordinates(geometry, locations, batch_size=2, capacity=6)
    rows = training_rows(sources['coordinates'], sources['counts'])
    raw = dict(memory=torch.randn(2, 7, 32), position=torch.randn(2, 7, 32), padding_mask=torch.zeros(2, 7, dtype=torch.bool))
    for network in (checkpointed, plain):
        network.train()
        optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            output = network.forward_rows(network._project_image_context(**raw), **{key: rows[key] for key in network.row_input_fields})
            loss = row_losses(output, rows, sources, geometry, sample['img_hw'].flip(-1))['total']
            loss.backward()
            assert all(layer.image_modulation[-1].weight.grad.abs().sum() > 0 for layer in network.layers)
            if step:
                assert network.generation_step_embedding.weight.grad.abs().sum() > 0
            optimizer.step()
    for (name, value), (_, other) in zip(checkpointed.named_parameters(), plain.named_parameters()):
        torch.testing.assert_close(value, other, atol=2e-6, rtol=2e-5, msg=name)


def test_strict_variant_restore_without_initializer_assets(monkeypatch, tmp_path):
    config = PoseFlowV48AdaLNPipelineConfig(native=arguments(), rows=7, capacity=6, codec_width=32,
        latent_dim=8, pretrained_checkpoint=None, quick_eval_enabled=False)
    _, network = pair()
    module = PoseFlowV48AdaLNPipeline(config=config, model=network).eval()
    saved = dict(state_dict=deepcopy(module.state_dict()), hyper_parameters=module.hparams,
                 **{'pytorch-lightning_version': pl.__version__})
    module.on_save_checkpoint(saved)
    assert saved['v48_contract']['architecture'] == 'V48'
    assert saved['v48_contract']['variant'] == 'image_cross_attention_generation_step_adaln'
    path = tmp_path/'v48_adaln.ckpt'; torch.save(saved, path)
    monkeypatch.setattr(PoseFlowV48AdaLNPipelineConfig, 'build', lambda *_, **__: pytest.fail('External initializer'))
    restored = PoseFlowV48AdaLNPipeline.load_from_checkpoint(path, weights_only=False).eval()
    sample = batch()
    torch.testing.assert_close(module._losses_for_batch(sample)['total'], restored._losses_for_batch(sample)['total'], atol=0, rtol=0)
    invalid = deepcopy(saved);invalid['v48_contract'].pop('variant')
    with pytest.raises(ValueError, match='contract'):
        restored.on_load_checkpoint(invalid)


def test_variant_preset_preserves_v48_recipe():
    with initialize_config_dir(version_base='1.3', config_dir=str(Path('conf').resolve())):
        config = compose(config_name='benchmark_poseflow_v48_adaln')
    spec = PoseFlowV48AdaLNPipelineConfig.from_hydra_config(config.pipeline)
    assert spec.capacity == 50 and spec.rows == 51
    assert config.training.max_epochs == 80 and config.training.precision == 32
    assert config.data.test_ann == config.data.val_ann


def test_variant_launcher_requires_matching_conditioning_verification():
    from scripts.training.train_poseflow_v48_adaln import validate_verification
    with pytest.raises(ValueError, match='verification'):
        validate_verification(None, None, 'unused')
