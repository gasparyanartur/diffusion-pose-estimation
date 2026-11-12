from pathlib import Path

import pytorch_lightning as pl
import pytest
import torch
from hydra import compose, initialize_config_dir

from src.models.pipelines.pose_vae.training import PoseVAEConfig, PoseVAEPipeline
from src.models.pipelines.pose_vae.evaluation import geometry_metrics
from tests.test_pose_vae import geometry


def test_recipe_and_split_contract():
    with initialize_config_dir(config_dir=str(Path('conf').resolve()), version_base=None):
        config = compose(config_name='benchmark_pose_vae')
    assert config.data.train_ann.endswith('crowdpose_train.json')
    assert config.data.val_ann.endswith('crowdpose_val.json')
    assert config.training.max_epochs == 50 and config.data.batch_size == 512
    assert config.training.precision == 32 and config.training.grad_clip == 1
    assert config.training.test_after_fit
    assert config.pipeline.latent_dim == 32 and config.pipeline.width == 256
    assert PoseVAEConfig.from_hydra_config(config.pipeline).build().model.latent_dim == 32


def test_validation_is_fixed_and_does_not_advance_training_rng():
    module = PoseVAEPipeline(width=32, latent_dim=8)
    batch = geometry()
    rng = module.training_generator.get_state().clone()
    module.on_validation_epoch_start()
    first = module.validation_outputs(batch)
    module.on_validation_epoch_start()
    second = module.validation_outputs(batch)
    torch.testing.assert_close(rng, module.training_generator.get_state())
    for variant in ('clean_mean', 'clean_sample', 'noisy_mean', 'noisy_sample'):
        for name in ('keypoints_px', 'boxes_cxcywh_px'):
            torch.testing.assert_close(first['predictions'][variant][name], second['predictions'][variant][name])


def test_metrics_masked_and_per_joint_counts():
    batch = geometry()
    prediction = {key: value.clone() for key, value in batch.items()}
    prediction['keypoints_px'][:, 0, 0] += 160
    observed = batch['joint_mask'].clone()
    observed[:, 0] = False
    metrics = geometry_metrics(prediction, batch, observed)
    numerator, denominator = metrics['masked_joint_error']
    torch.testing.assert_close(numerator / denominator, torch.tensor(1.))
    assert metrics['joint_2_error'][1] == 0
    assert metrics['bbox_iou'][0] / metrics['bbox_iou'][1] == 1


@pytest.mark.parametrize('accelerator', ['cpu', 'gpu'])
def test_native_checkpoint_optimizer_scheduler_and_next_update(tmp_path, accelerator):
    if accelerator == 'gpu' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from torch.utils.data import DataLoader
    torch.set_num_threads(2)
    batch = geometry(8)
    samples = [{key: value[index] for key, value in batch.items()} for index in range(8)]
    loader = DataLoader(samples, batch_size=4, shuffle=False)
    module = PoseVAEPipeline(width=32, latent_dim=8, max_epochs=50)
    trainer = pl.Trainer(accelerator=accelerator, devices=1, max_epochs=1,
                         logger=False, enable_checkpointing=False, enable_progress_bar=False,
                         limit_val_batches=0, gradient_clip_val=1.)
    trainer.fit(module, train_dataloaders=loader)
    path = tmp_path / 'native.ckpt'
    trainer.save_checkpoint(path)
    restored = PoseVAEPipeline.load_from_checkpoint(path, weights_only=False)
    torch.testing.assert_close(module.training_generator.get_state(), restored.training_generator.get_state())
    for key, value in module.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
    source_optimizer = trainer.optimizers[0]
    restored_config = restored.configure_optimizers()
    restored_optimizer = restored_config['optimizer']
    checkpoint = torch.load(path, weights_only=False)
    restored_optimizer.load_state_dict(checkpoint['optimizer_states'][0])
    restored_scheduler = restored_config['lr_scheduler']['scheduler']
    restored_scheduler.load_state_dict(checkpoint['lr_schedulers'][0])
    assert restored_scheduler.state_dict() == trainer.lr_scheduler_configs[0].scheduler.state_dict()
    for model, optimizer in ((module, source_optimizer), (restored, restored_optimizer)):
        model.train()
        optimizer.zero_grad()
        loss, _ = model.training_objective(geometry(4), epoch=1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
    for key, value in module.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
