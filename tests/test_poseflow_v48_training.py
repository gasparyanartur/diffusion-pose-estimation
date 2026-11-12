from copy import deepcopy
from pathlib import Path

from hydra import compose, initialize_config_dir
import pytest
import pytorch_lightning as pl
import torch

from src.models.pipelines.pose_estimators.poseflow.training_v48 import PoseFlowV48Pipeline, PoseFlowV48PipelineConfig
from src.models.pipelines.pose_estimators.poseflow.quick_eval_v48 import evaluate_v48_batches, tensorboard_metrics
from tests.test_poseflow_v44 import arguments
from tests.test_poseflow_v43_training import batch
from tests.test_poseflow_v48 import model, context


def test_single_context_and_asset_free_strict_restore(monkeypatch, tmp_path):
    config = PoseFlowV48PipelineConfig(native=arguments(), rows=7, capacity=6, codec_width=32,
        latent_dim=8, pretrained_checkpoint=None, quick_eval_enabled=False)
    instance = PoseFlowV48Pipeline(config=config, model=model()).eval()
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
    path = tmp_path/'v48.ckpt'
    torch.save(saved, path)
    monkeypatch.setattr(PoseFlowV48PipelineConfig, 'build', lambda *_, **__: pytest.fail('External initialization'))
    restored = PoseFlowV48Pipeline.load_from_checkpoint(path, weights_only=False).eval()
    sample = batch()
    torch.testing.assert_close(instance._losses_for_batch(sample)['total'], restored._losses_for_batch(sample)['total'], atol=0, rtol=0)
    invalid = deepcopy(saved)
    invalid['v48_contract']['noise'] = 'cosine'
    with pytest.raises(ValueError, match='contract'):
        restored.on_load_checkpoint(invalid)


def test_image_only_evaluator_empty_predictions_and_birth_labels(tmp_path):
    network = model()
    for head in network.eos_heads:
        head.weight.data.zero_()
        head.bias.data.fill_(10.)
    sample = batch((2,))
    sample['original_img_hw'] = sample['img_hw']
    def metrics(records, annotations, ids):
        assert records == [] and ids == [0]
        return dict(pose_AP=0., bbox_AP=0.)
    report = evaluate_v48_batches(network, [sample], tmp_path, 'unused', [0], metric_evaluator=metrics)
    assert report['count_mae'] == 2 and report['mean_generation_steps'] == 1
    assert report['birth_errors']['teacher_forced/rank_01/birth_coordinate_mse'] >= 0
    assert 'generated/rank_01/birth_coordinate_mse' not in report['birth_errors']
    assert 'quick-test/ordered_birth_only/pose_AP' in tensorboard_metrics(report)
    assert 'refinement' not in report


def test_real_model_eval_batch_independence_and_determinism():
    network = model()
    for head in network.eos_heads:
        head.weight.data.zero_()
        head.bias.data.fill_(-5.)
    image = context(network, 2)
    sizes = torch.tensor([[100., 200.], [170., 90.]])
    output = network.predict(image, sizes)
    repeated = network.predict(image, sizes)
    torch.testing.assert_close(output['pred_coordinates'], repeated['pred_coordinates'], atol=0, rtol=0)
    for index in range(2):
        one = {key: tuple(item[index:index+1] for item in value) if isinstance(value, tuple)
               else value[index:index+1] for key, value in image.items()}
        single = network.predict(one, sizes[index:index+1])
        torch.testing.assert_close(output['pred_coordinates'][index], single['pred_coordinates'][0], atol=2e-6, rtol=2e-5)


def test_full_training_preset_and_no_diffusion_fields():
    with initialize_config_dir(version_base='1.3', config_dir=str(Path('conf').resolve())):
        configuration = compose(config_name='benchmark_poseflow_v48')
    pipeline = PoseFlowV48PipelineConfig.from_hydra_config(configuration.pipeline)
    assert pipeline.capacity == 50 and pipeline.rows == 51
    assert configuration.training.max_epochs == 80 and configuration.training.precision == 32
    assert configuration.data.test_ann == configuration.data.val_ann


def test_launch_rejects_missing_verification():
    from scripts.training.train_poseflow_v48 import validate_verification
    with pytest.raises(ValueError, match='verification'):
        validate_verification(None, None, 'unused')
