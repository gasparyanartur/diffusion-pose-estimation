import pytorch_lightning as pl
import pytest
import torch
from torch.utils.data import DataLoader
from hydra import compose, initialize_config_dir
from pathlib import Path

from src.models.pipelines.pose_autoencoder.model import PoseBoxAutoencoder, geometry_loss
from src.models.pipelines.pose_vae.fidelity import (
    FidelityPoseVAE, FidelityVAEPipeline, calibrated_kl_weight, mask_joints,
)
from src.models.pipelines.pose_vae.model import prior_kl
from tests.test_pose_autoencoder import identified_geometry


def test_ae_transfer_is_exact_and_variance_is_calibrated():
    torch.manual_seed(42)
    autoencoder = PoseBoxAutoencoder(width=32, latent_dim=8)
    model = FidelityPoseVAE(width=32, latent_dim=8)
    std = torch.linspace(.1, 2, 8)
    model.initialize_from_autoencoder(autoencoder, std)
    batch = identified_geometry(1)
    arguments = {key: batch[key] for key in ('keypoints_px', 'boxes_cxcywh_px', 'joint_mask', 'image_wh')}
    expected = autoencoder.encode(**arguments)
    mean, logvar = model.encode(**arguments)
    torch.testing.assert_close(mean, expected, rtol=0, atol=0)
    torch.testing.assert_close(logvar.exp().sqrt(), (.05*std)[None])
    for key, value in model.decode(mean, batch['image_wh']).items():
        torch.testing.assert_close(value, autoencoder.decode(expected, batch['image_wh'])[key], rtol=0, atol=0)


def test_masking_keeps_clean_targets_and_natural_missing_labels():
    batch = identified_geometry(8)
    batch['joint_mask'][:, 0] = False
    batch['keypoints_px'][:, 0] = float('nan')
    saved = {key: value.clone() for key, value in batch.items()}
    first = mask_joints(batch, 1., torch.Generator().manual_seed(42))
    second = mask_joints(batch, 1., torch.Generator().manual_seed(42))
    assert first['joint_mask'].any(-1).all()
    assert not first['joint_mask'][:, 0].any()
    assert not (first['joint_mask'] & ~batch['joint_mask']).any()
    assert (~first['joint_mask'] & batch['joint_mask']).any()
    torch.testing.assert_close(first['joint_mask'], second['joint_mask'])
    for key in batch:
        torch.testing.assert_close(batch[key], saved[key], equal_nan=True)
    with pytest.raises(ValueError):
        mask_joints(batch, 1.1, torch.Generator())


def test_kl_calibration_and_sampled_joint_training_gradients():
    assert calibrated_kl_weight(.2, 100., .1) == pytest.approx(.0002)
    batch = identified_geometry(4)
    module = FidelityVAEPipeline(width=32, latent_dim=8, mode='sampled', kl_weight=.001,
                                 mask_max=.3, max_epochs=2)
    loss, metrics = module.training_objective(batch, epoch=10)
    assert torch.isfinite(loss)
    torch.testing.assert_close(loss, metrics['reconstruction']+.001*metrics['kl'])
    loss.backward()
    for layer in (module.model.encoder[0], module.model.posterior_mean,
                  module.model.posterior_logvar, module.model.decoder[0], module.model.joint_head):
        assert layer.weight.grad is not None and layer.weight.grad.abs().sum() > 0


def test_comparison_recipe_keeps_test_off_and_calibrates_kl():
    with initialize_config_dir(config_dir=str(Path('conf').resolve()), version_base=None):
        config = compose(config_name='benchmark_pose_vae_fidelity')
    assert not config.training.test_after_fit
    assert config.training.checkpoint.save_last_every_epoch
    assert config.training.max_epochs == 100 and config.training.check_val_every_n_epoch == 5
    assert config.pipeline.kl_fraction == .01 and config.pipeline.mask_max == .3
    assert config.data.batch_size == 512
    for invalid in ((float('nan'), 1, .1), (1, 0, .1), (1, 1, 2)):
        with pytest.raises(ValueError):
            calibrated_kl_weight(*invalid)


@pytest.mark.parametrize('accelerator', ['cpu', 'gpu'])
def test_fidelity_native_restore_reproduces_next_update(tmp_path, accelerator):
    if accelerator == 'gpu' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    batch = identified_geometry(8)
    samples = [{key: value[index] for key, value in batch.items()} for index in range(8)]
    module = FidelityVAEPipeline(width=32, latent_dim=8, mask_max=.3,
                                 kl_weight=.001, mode='sampled', max_epochs=2)
    trainer = pl.Trainer(accelerator=accelerator, devices=1, max_epochs=1, logger=False,
                         enable_checkpointing=False, enable_progress_bar=False, limit_val_batches=0,
                         gradient_clip_val=1.)
    trainer.fit(module, train_dataloaders=DataLoader(samples, batch_size=4))
    path = tmp_path/'fidelity.ckpt'
    trainer.save_checkpoint(path)
    restored = FidelityVAEPipeline.load_from_checkpoint(path, weights_only=False).to(module.device)
    state = torch.load(path, weights_only=False)
    configuration = restored.configure_optimizers()
    configuration['optimizer'].load_state_dict(state['optimizer_states'][0])
    configuration['lr_scheduler']['scheduler'].load_state_dict(state['lr_schedulers'][0])
    torch.testing.assert_close(module.training_generator.get_state(), restored.training_generator.get_state())
    for model, optimizer in ((module, trainer.optimizers[0]), (restored, configuration['optimizer'])):
        optimizer.zero_grad()
        moved = {key: value.to(module.device) for key, value in batch.items()}
        loss, _ = model.training_objective(moved, epoch=1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
    for key, value in module.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
