from pathlib import Path
import random
import hashlib
from types import SimpleNamespace
import pytest

from hydra import compose, initialize_config_dir
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from src.models.pipelines.pose_estimators.poseflow.quick_eval_v43 import evaluate_v43_grid
from src.models.pipelines.pose_estimators.poseflow.training_v43 import PoseFlowV43Pipeline, PoseFlowV43PipelineConfig
from src.models.pipelines.pose_estimators.poseflow.training_v43 import validate_statistics_data
from tests.test_poseflow_v43 import model
from tests.test_poseflow_v43_training import batch, pipeline


@pytest.mark.parametrize('field', ['training', 'top_level'])
def test_training_entrypoint_rejects_generic_skip_before_initialization(tmp_path, monkeypatch, field):
    from scripts.training.train_poseflow_v43 import main
    from hydra.core.hydra_config import HydraConfig
    with initialize_config_dir(config_dir=str(Path('conf').resolve()), version_base=None):
        config = compose(config_name='benchmark_poseflow_v43_frozen_vae')
    if field == 'training':
        config.training.skip_training = True
    else:
        config.skip_training = True
    monkeypatch.setattr(HydraConfig, 'get', lambda: SimpleNamespace(runtime=SimpleNamespace(output_dir=str(tmp_path))))
    def forbidden(*args, **kwargs):
        pytest.fail('Evaluation-only request reached training initialization')
    monkeypatch.setattr(PoseFlowV43PipelineConfig, 'build', forbidden)
    with pytest.raises(ValueError, match='evaluation entrypoint'):
        main.__wrapped__(config)


def test_training_rejects_statistics_from_different_annotations_or_augmentation(tmp_path):
    path = tmp_path/'annotations.json'
    path.write_text('{}')
    specification = dict(external_train_transforms=True, train_random_flip=True, train_random_crop=True,
        train_resize_scales=(256,), train_max_size=427, train_crop_resize_scales=(128,160,192),
        train_crop_size=(123,192))
    data = SimpleNamespace(root=tmp_path, train_ann='annotations.json', **specification)
    provenance = dict(annotation_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), data_config=specification)
    validate_statistics_data(provenance, data)
    data.train_random_crop = False
    with pytest.raises(ValueError, match='augmentation'):
        validate_statistics_data(provenance, data)
    path.write_text('{"changed":true}')
    with pytest.raises(ValueError, match='annotation'):
        validate_statistics_data(provenance, data)


def test_recipe_keeps_historical_data_but_embeds_frozen_codec_and_independent_last():
    with initialize_config_dir(config_dir=str(Path('conf').resolve()), version_base=None):
        config = compose(config_name='benchmark_poseflow_v43_frozen_vae')
    parsed = PoseFlowV43PipelineConfig.from_hydra_config(config.pipeline)
    assert parsed.latent_dim == 32 and parsed.hidden_dim == 256
    assert parsed.num_proposals == 50 and parsed.reference_candidates == 200
    assert config.data.train_ann.endswith('crowdpose_trainval.json') and config.data.train_random_crop
    assert config.training.max_epochs == 80 and config.training.accumulate_grad_batches == 2
    assert not config.training.test_after_fit and config.training.checkpoint.save_last_every_epoch
    assert config.training.checkpoint.save_top_k == 80
    assert config.data.batch_size == 4 and parsed.latent_statistics_file


def test_evaluator_threads_metadata_restores_rng_and_reports_actual_scope(tmp_path):
    network = model()
    data = batch((1,))
    data['original_img_hw'] = torch.tensor([[360., 640.]])
    state = torch.get_rng_state().clone()
    original_steps = network.flow_sampling_steps
    report = evaluate_v43_grid(network, [data], tmp_path, 'unused.json', [0],
        metric_evaluator=lambda records, *args: dict(pose_AP=float(len(records)), bbox_AP=0.))
    assert report['conditioning'] == 'standardized latent Gaussian; no GT geometry'
    assert len(report['conditions']) == 3 and report['images'] == 1
    assert network.flow_sampling_steps == original_steps and torch.equal(state, torch.get_rng_state())
    assert report['raw_confidence_max_difference'] == 0
    assert (tmp_path/'metrics.json').exists()


def test_native_lightning_checkpoint_load_needs_no_initialization_assets(tmp_path):
    module = pipeline()
    module.config.codec_checkpoint = 'unavailable-codec.ckpt'
    module.config.pretrained_checkpoint = 'unavailable-edpose.pth'
    module.config.latent_statistics_file = 'unavailable-statistics.pt'
    module.hparams['config'] = module.config.model_dump()
    data = batch((1, 1))
    samples = [{name: value[index] for name, value in data.items()} for index in range(2)]
    trainer = pl.Trainer(accelerator='cpu', devices=1, max_epochs=1, logger=False,
        enable_checkpointing=False, enable_progress_bar=False, limit_val_batches=0, gradient_clip_val=.1)
    trainer.fit(module, train_dataloaders=DataLoader(samples, batch_size=2))
    path = tmp_path/'native.ckpt'
    trainer.save_checkpoint(path)
    expected_rng = torch.get_rng_state().clone()
    expected_python = random.getstate()
    expected_numpy = np.random.get_state()
    torch.rand(4)
    random.random()
    np.random.rand()
    restored = PoseFlowV43Pipeline.load_from_checkpoint(path, weights_only=False)
    assert torch.equal(expected_rng, torch.get_rng_state())
    assert random.getstate() == expected_python
    assert np.array_equal(np.random.get_state()[1], expected_numpy[1])
    for name, value in module.state_dict().items():
        torch.testing.assert_close(value.cpu(), restored.state_dict()[name], atol=0, rtol=0)
    assert not restored.model.codec.model.training
    assert restored.model.codec.statistics_ready
