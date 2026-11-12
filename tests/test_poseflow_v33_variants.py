from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.training import TrainingConfig, TrainingRunner


@pytest.mark.parametrize('variant,latent_weight', [('no_clip', 1.), ('no_clip_rescale', 10.)])
def test_v33_variant_changes_only_clipping_and_requested_latent_weight(variant, latent_weight):
    with initialize_config_dir(config_dir=str(Path('conf').resolve()), version_base=None):
        baseline = compose(config_name='benchmark_poseflow_v33')
        config = compose(config_name=f'benchmark_poseflow_v33_{variant}')
    expected = OmegaConf.to_container(baseline, resolve=True)
    expected['training']['grad_clip'] = 0
    expected['pipeline']['latent_loss_weight'] = latent_weight
    expected['run_name'] = f'poseflow_v33_{variant}'
    expected['out_dir'] = f'results/poseflow_v33_{variant}'
    assert OmegaConf.to_container(config, resolve=True) == expected


def test_native_runner_disables_clipping_for_v33_variant(tmp_path, monkeypatch):
    import src.training.runner as runner_module

    with initialize_config_dir(config_dir=str(Path('conf').resolve()), version_base=None):
        config = compose(config_name='benchmark_poseflow_v33_no_clip')
    observed = {}

    def capture_trainer(**kwargs):
        observed.update(kwargs)
        return object()

    monkeypatch.setattr(runner_module.pl, 'Trainer', capture_trainer)
    runner = TrainingRunner(TrainingConfig.from_hydra_config(config.training), tmp_path)
    monkeypatch.setattr(runner, '_loggers', lambda: [])
    runner.build_trainer(max_epochs=80)
    assert observed['gradient_clip_val'] is None
    assert observed['accumulate_grad_batches'] == 2
