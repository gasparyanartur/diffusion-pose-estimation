from pathlib import Path

import pytorch_lightning as pl
import pytest
import torch
from hydra import compose, initialize_config_dir
from torch.utils.data import DataLoader

from src.models.pipelines.pose_autoencoder.training import PoseAutoencoderPipeline, PoseAutoencoderConfig
from src.models.pipelines.pose_autoencoder.experiment import clean_gate, robust_gain, latent_statistics
from tests.test_pose_autoencoder import identified_geometry


def test_clean_and_robust_recipe():
    with initialize_config_dir(config_dir=str(Path('conf').resolve()), version_base=None):
        clean = compose(config_name='benchmark_pose_autoencoder')
        robust = compose(config_name='benchmark_pose_autoencoder_robust')
    assert clean.training.max_epochs == 100 and clean.pipeline.lr == .001
    assert robust.training.max_epochs == 50 and robust.pipeline.lr == .0001
    assert robust.pipeline.min_lr == .000001 and robust.pipeline.phase == 'robust'
    assert clean.data.batch_size == robust.data.batch_size == 512
    assert not clean.training.test_after_fit and not robust.training.test_after_fit
    assert clean.training.checkpoint.monitor == 'val/selection_score'
    assert PoseAutoencoderConfig.from_hydra_config(clean.pipeline).build().model.latent_dim == 32


@pytest.mark.parametrize('phase', ['clean', 'robust'])
@pytest.mark.parametrize('accelerator', ['cpu', 'gpu'])
def test_native_restore_and_exact_next_update_with_frozen_encoder(tmp_path, phase, accelerator):
    if accelerator == 'gpu' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    batch = identified_geometry(8)
    samples = [{key:value[index] for key,value in batch.items()} for index in range(8)]
    module = PoseAutoencoderPipeline(width=32, latent_dim=8, phase=phase, max_epochs=100)
    if phase == 'robust':
        module.set_latent_statistics(torch.zeros(8), torch.linspace(.1, 2, 8))
    encoder_before = {key:value.clone() for key,value in module.model.state_dict().items()
                      if key.startswith(('encoder.', 'latent_head.'))}
    trainer = pl.Trainer(accelerator=accelerator, devices=1, max_epochs=1, logger=False,
                         enable_checkpointing=False, enable_progress_bar=False, limit_val_batches=0,
                         gradient_clip_val=1.)
    trainer.fit(module, train_dataloaders=DataLoader(samples, batch_size=4))
    path = tmp_path/'native.ckpt'
    trainer.save_checkpoint(path)
    restored = PoseAutoencoderPipeline.load_from_checkpoint(path, weights_only=False)
    torch.testing.assert_close(module.training_generator.get_state(), restored.training_generator.get_state())
    configuration = restored.configure_optimizers()
    checkpoint = torch.load(path, weights_only=False)
    configuration['optimizer'].load_state_dict(checkpoint['optimizer_states'][0])
    configuration['lr_scheduler']['scheduler'].load_state_dict(checkpoint['lr_schedulers'][0])
    assert configuration['lr_scheduler']['scheduler'].state_dict() == trainer.lr_scheduler_configs[0].scheduler.state_dict()
    for model, optimizer in ((module, trainer.optimizers[0]), (restored, configuration['optimizer'])):
        optimizer.zero_grad()
        loss, _ = model.training_objective(identified_geometry(4))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
    for key,value in module.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
    if phase == 'robust':
        for key,value in encoder_before.items():
            torch.testing.assert_close(value, module.model.state_dict()[key], rtol=0, atol=0)
        assert not module.model.encoder.training


def test_uncalibrated_robust_training_is_rejected_and_statistics_use_supplied_loader():
    module = PoseAutoencoderPipeline(width=32, latent_dim=8, phase='robust')
    with pytest.raises(ValueError, match='statistics'):
        module.training_objective(identified_geometry())
    batch = identified_geometry(4)
    samples = [{key:value[index] for key,value in batch.items()} for index in range(4)]
    mean, std = latent_statistics(module.model, DataLoader(samples, batch_size=2), 'cpu')
    expected = module.model.encode(**{key:batch[key] for key in
                                     ('keypoints_px','boxes_cxcywh_px','joint_mask','image_wh')})
    torch.testing.assert_close(mean, expected.mean(0))
    torch.testing.assert_close(std, expected.std(0, unbiased=False), atol=1e-6, rtol=0)


def test_gates_require_both_apis_and_robust_improvement():
    assert clean_gate(dict(pose_AP=97.5, bbox_AP=99))
    assert not clean_gate(dict(pose_AP=99, bbox_AP=97.49))
    control = [dict(pose_AP=95., bbox_AP=96.)]*3
    assert robust_gain(control, [dict(pose_AP=96., bbox_AP=97.)]*3)
    assert not robust_gain(control, [dict(pose_AP=96., bbox_AP=95.)]*3)
