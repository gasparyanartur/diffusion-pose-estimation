from copy import deepcopy

import pytest
import torch

from src.models.pipelines.pose_estimators.poseflow.model_v44 import PoseFlowV44Model
from src.models.pipelines.pose_estimators.poseflow.training_v44 import PoseFlowV44Pipeline, PoseFlowV44PipelineConfig
from src.models.pipelines.pose_estimators.poseflow.comparison_v44 import warmstart_comparison
from tests.test_poseflow_v44 import arguments, model, pipeline
from tests.test_poseflow_v43_codec import small_codec
from tests.test_poseflow_v43_training import batch


def test_velocity_target_inverse_and_low_noise_identity():
    network = PoseFlowV44Model(codec=small_codec(), prediction_type='v', **arguments()).eval()
    clean, noise = torch.randn(2, 6, 8), torch.randn(2, 6, 8)
    times = torch.tensor([0, 999])
    state = network.q_sample(clean, times, noise)
    velocity = network.velocity_target(clean, noise, times)
    torch.testing.assert_close(network.clean_from_velocity(state, velocity, times), clean, atol=1e-5, rtol=1e-5)
    predicted = network.clean_from_velocity(state, torch.zeros_like(state), times)
    alpha = network.sqrt_alphas_cumprod[times].to(state)[:, None, None]
    torch.testing.assert_close(predicted, alpha*state)
    assert network.prediction_type == 'v'


def test_mode_defaults_legacy_parity_and_semantic_checkpoint_rejection():
    source = model()
    explicit = PoseFlowV44Model(codec=small_codec(), prediction_type='x0', **arguments()).eval()
    explicit.load_state_dict(source.state_dict(), strict=True)
    images = torch.randn(1, 3, 128, 128)
    sizes = torch.tensor([[128., 100.]])
    state, times = torch.randn(1, 6, 8), torch.tensor([17])
    torch.testing.assert_close(source(images, state, times, image_wh=sizes)['pred_latents'],
        explicit(images, state, times, image_wh=sizes)['pred_latents'], rtol=0, atol=0)
    module = pipeline()
    checkpoint = {}
    module.on_save_checkpoint(checkpoint)
    checkpoint.pop('v44_prediction_type')
    module.on_load_checkpoint(checkpoint)
    module.config = module.config.model_copy(update=dict(prediction_type='v'))
    with pytest.raises(ValueError, match='prediction'):
        module.on_load_checkpoint(checkpoint)


def test_velocity_loss_uses_velocity_targets_and_frozen_codec():
    network = PoseFlowV44Model(codec=small_codec(), prediction_type='v', **arguments()).eval()
    config = PoseFlowV44PipelineConfig(native=arguments(), codec_width=32, latent_dim=8,
        pretrained_checkpoint=None, prediction_type='v', quick_eval_enabled=False)
    module = PoseFlowV44Pipeline(config=config, model=network, max_epochs=2).eval()
    before = deepcopy(network.codec.state_dict())
    calls = []
    hook = module.slot_matcher.register_forward_hook(lambda *_: calls.append(1))
    losses = module._losses_for_batch(batch((2,)))
    hook.remove()
    assert len(calls) == 1 and 'velocity/total' in losses and 'aux0/velocity/total' in losses
    expected = losses['frontend'] + 2*losses['loss_slot_focal']
    for prefix in ('', 'aux0/'):
        expected = expected + 10*losses[prefix+'velocity/total'] + .5*losses[prefix+'geometry']
    torch.testing.assert_close(losses['total'], expected)
    losses['total'].backward()
    assert network.latent_heads[0].weight.grad.abs().sum() > 0
    assert network.latent_heads[-1].weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in network.codec.parameters())
    assert all(torch.equal(value, network.codec.state_dict()[key]) for key, value in before.items())


def test_explicit_warmstart_only_negates_v_heads_and_preserves_noise_rng():
    source = pipeline()
    record = dict(state_dict=deepcopy(source.state_dict()), hyper_parameters=deepcopy(dict(source.hparams)),
        v44_codec_provenance={'test': True})
    source.on_save_checkpoint(record)
    reference = warmstart_comparison(record, prediction_type='x0', max_epochs=10)
    velocity = warmstart_comparison(record, prediction_type='v', max_epochs=10)
    for key, value in reference.state_dict().items():
        expected = -value if key.startswith('model.latent_heads.') else value
        torch.testing.assert_close(velocity.state_dict()[key], expected, rtol=0, atol=0)
    assert reference.config.lr == velocity.config.lr == 2e-5
    assert reference.config.lr_backbone == velocity.config.lr_backbone == 1e-5
    assert velocity.model.prediction_type == velocity.config.prediction_type == 'v'


def test_comparison_preset_and_evaluation_artifact_routing(monkeypatch, tmp_path):
    from pathlib import Path
    from types import SimpleNamespace
    from hydra import compose, initialize_config_dir
    from src.models.pipelines.pose_estimators.poseflow import quick_eval_v44
    with initialize_config_dir(version_base=None, config_dir=str(Path('conf').resolve())):
        config = compose(config_name='benchmark_poseflow_v44_comparison')
    assert config.training.max_epochs == 10 and config.training.accumulate_grad_batches == 2
    assert config.data.train_ann.endswith('crowdpose_train.json')
    assert config.data.test_ann == config.data.val_ann and config.training.checkpoint.save_top_k == 1
    module = pipeline()
    module.config = module.config.model_copy(update=dict(quick_eval_enabled=True))
    calls = []
    def evaluate(model, data, root, **kwargs):
        calls.append(root)
        return dict(conditions={'test': dict(metrics=dict(pose_AP=.2, bbox_AP=.3))},small_noise={})
    monkeypatch.setattr(quick_eval_v44,'run_v44_evaluation',evaluate)
    monkeypatch.setattr(module,'log',lambda *args,**kwargs:None)
    module._trainer = SimpleNamespace(sanity_checking=False,checkpoint_callback=SimpleNamespace(dirpath=tmp_path/'v/checkpoints'),
        default_root_dir=tmp_path/'wrong',datamodule=None,current_epoch=0)
    module.on_validation_epoch_end()
    assert calls == [tmp_path/'v/quick_test']
