from copy import deepcopy

import pytorch_lightning as pl
import torch

from tests.test_poseflow_v35 import batch
from tests.test_poseflow_v40 import pipeline as baseline_pipeline


def pipeline(decay=.999):
    from src.models.pipelines.pose_estimators.poseflow.training_v40_ema import PoseFlowV40EMAPipelineConfig
    return PoseFlowV40EMAPipelineConfig(backbone_pretrained=False, pretrained_checkpoint=None,
        hidden_dim=32, num_heads=4, feedforward_dim=64, num_queries=6,
        num_proposals=6, num_group=6, grid_rows=1, grid_columns=6,
        reference_candidates=12, reference_encoder_source_layers=(2,),
        quick_eval_enabled=False, ema_decay=decay).build(max_epochs=2)


def test_initial_teacher_equality_online_rng_forward_loss_and_draw_parity():
    torch.manual_seed(42)
    baseline = baseline_pipeline().eval()
    rng = torch.get_rng_state().clone()
    torch.manual_seed(42)
    module = pipeline().eval()
    assert torch.equal(rng, torch.get_rng_state())
    for name, value in baseline.model.state_dict().items():
        torch.testing.assert_close(value, module.model.state_dict()[name], atol=0, rtol=0)
    data = batch()
    with torch.no_grad():
        context = module.model.encode_context(data['image'])
        inputs = torch.rand(2, 6, 4)*100
        teacher = module.teacher(data['image'], inputs)
        online = module.model.encode_clean(inputs, context['image'])['pred_latents']
        torch.testing.assert_close(teacher, online, atol=1e-6, rtol=1e-6)
    for left, right in zip(baseline._draw_training_inputs(data), module._draw_training_inputs(data)):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    left = baseline._losses_for_batch(data, training=False)
    right = module._losses_for_batch(data, training=False)
    for name in left:
        torch.testing.assert_close(left[name], right[name], atol=1e-6, rtol=1e-6)
    assert module.ema_updates == 0


def test_teacher_is_full_eval_only_producer_and_optimizer_excludes_it():
    module = pipeline().train()
    assert not module.teacher.training and all(not p.requires_grad for p in module.teacher.parameters())
    names = set(dict(module.teacher.named_parameters()))
    for prefix in ('backbone.', 'input_proj.', 'time_embedding.', 'input_mlp.', 'clean_layers.'):
        assert any(name.startswith(prefix) for name in names)
    assert 'level_embed' in names
    assert not any(name.startswith(('proposal_', 'denoiser_', 'box_head', 'existence_head')) for name in names)
    optimizer = module.configure_optimizers()['optimizer']
    optimized = {id(p) for group in optimizer.param_groups for p in group['params']}
    assert all(id(p) not in optimized for p in module.teacher.parameters())
    module._losses_for_batch(batch())['total'].backward()
    assert all(p.grad is None for p in module.teacher.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.model.clean_layers.parameters())


def test_same_teacher_code_supplies_corruption_and_latent_target(monkeypatch):
    from src.models.pipelines.pose_estimators.poseflow.latent_fixed_source import positive_latent_mse
    import src.models.pipelines.pose_estimators.poseflow.training_v40 as training
    module = pipeline().eval()
    with torch.no_grad():
        module.model.clean_layers[0].feedforward[-1].bias[0].add_(.7)
    captured = {}
    handle = module.teacher.register_forward_hook(lambda _, args, output: captured.update(teacher=output))
    original_q = module.model.q_sample
    def q_sample(source, times, noise=None):
        captured['source'] = source
        return original_q(source, times, noise)
    module.model.q_sample = q_sample
    def latent_loss(prediction, target, positive):
        captured.update(prediction=prediction, target=target, positive=positive)
        return positive_latent_mse(prediction, target, positive)
    monkeypatch.setattr(training, 'positive_latent_mse', latent_loss)
    losses = module._losses_for_batch(batch(), training=False)
    handle.remove()
    torch.testing.assert_close(captured['source'], captured['teacher'], atol=0, rtol=0)
    torch.testing.assert_close(captured['target'], captured['teacher'], atol=0, rtol=0)
    assert not captured['source'].requires_grad and not captured['target'].requires_grad
    torch.testing.assert_close(losses['loss_latent'], positive_latent_mse(
        captured['prediction'], captured['teacher'], captured['positive']))
    assert module._diagnostics['ema/online_teacher/positive/mse'] > 0
    assert module.ema_updates == 0


def test_ema_arithmetic_after_optimizer_update_and_validation_does_not_update():
    module = pipeline(decay=.75)
    online = module.model.input_mlp[0].weight
    teacher = module.teacher.input_mlp[0].weight
    before = teacher.detach().clone()
    optimizer = torch.optim.SGD([online], lr=.1)
    def closure():
        optimizer.zero_grad()
        online.sum().backward()
        return online.sum()
    module.optimizer_step(0, 0, optimizer, closure)
    torch.testing.assert_close(online, before-.1)
    torch.testing.assert_close(teacher, .75*before+.25*online)
    assert module.ema_updates == 1
    state = deepcopy(module.teacher.state_dict())
    module._losses_for_batch(batch(), training=False)
    assert module.ema_updates == 1
    for name, value in state.items():
        torch.testing.assert_close(value, module.teacher.state_dict()[name], atol=0, rtol=0)


def test_strict_teacher_optimizer_rng_and_next_update_restore():
    module = pipeline()
    optimizer = module.configure_optimizers()['optimizer']
    data = batch()
    def step(owner, opt):
        def closure():
            opt.zero_grad()
            loss = owner._losses_for_batch(data)['total']
            loss.backward()
            return loss
        owner.optimizer_step(0, 0, opt, closure)
    step(module, optimizer)
    restored = pipeline()
    restored.load_state_dict(deepcopy(module.state_dict()), strict=True)
    other_optimizer = restored.configure_optimizers()['optimizer']
    other_optimizer.load_state_dict(deepcopy(optimizer.state_dict()))
    step(module, optimizer)
    step(restored, other_optimizer)
    assert module.ema_updates == restored.ema_updates == 2
    for name, value in module.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], atol=0, rtol=0)
    with torch.no_grad():
        for calls in (1, 4):
            module.model.flow_sampling_steps = restored.model.flow_sampling_steps = calls
            left = module.model.sample_boxes(data['image'], generator=torch.Generator().manual_seed(8))
            right = restored.model.sample_boxes(data['image'], generator=torch.Generator().manual_seed(8))
            torch.testing.assert_close(left['pred_boxes'], right['pred_boxes'], atol=0, rtol=0)


def test_cpu_lightning_accumulation_updates_once_per_optimizer_step(tmp_path):
    from torch.utils.data import DataLoader
    module = pipeline()
    observed = []
    class Probe(pl.Callback):
        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            observed.append(int(module.ema_updates))
    loader = DataLoader([batch((1,)) for _ in range(4)], batch_size=None)
    trainer = pl.Trainer(default_root_dir=tmp_path, accelerator='cpu', max_steps=2,
        max_epochs=1, accumulate_grad_batches=2, limit_train_batches=4,
        limit_val_batches=0, logger=False, enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False, callbacks=[Probe()])
    trainer.fit(module, train_dataloaders=loader)
    assert observed == [0, 1, 1, 2] and module.ema_updates == trainer.global_step
    empty = module._losses_for_batch(batch((0,)))
    assert empty['loss_latent'] == 0 and torch.isfinite(empty['total'])
