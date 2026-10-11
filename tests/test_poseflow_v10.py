import pytest
import torch

from src.models.pipelines.pose_estimators.poseflow import PoseFlowV10PipelineConfig
from tests.test_poseflow import _batch


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"))])
def test_v10_person_latents_shared_readout_and_sampling(device):
    pipeline = PoseFlowV10PipelineConfig(
        backbone_pretrained=False, num_queries=4, num_proposals=2,
        num_group=2, num_encoder_layers=1, num_decoder_layers=2,
        num_feature_levels=3, hidden_dim=32, feedforward_dim=64,
        num_heads=8, num_body_points=2, num_box_decoder_layers=1,
        dn_number=2,
    ).build(max_epochs=2)
    pipeline = pipeline.to(device)
    model = pipeline.model
    batch = {name: value.to(device) for name, value in _batch().items()} | {"image": torch.randn(1, 3, 64, 64, device=device)}
    calls = []
    hook = model.initial_coordinate_head.register_forward_hook(
        lambda module, args, output: calls.append(args[0].shape))
    losses = pipeline._losses_for_batch(batch)
    hook.remove()
    assert len(calls) == 3
    assert all(shape[-1] == 32 for shape in calls)
    assert calls[0][1] == 4
    assert calls[1][1] == 4  # two persons and two DN queries
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    assert model.dn_box_conditioning.layers[-1].weight.grad is not None
    assert model.initial_coordinate_head.layers[-1].weight.grad.abs().sum() > 0
    with torch.no_grad():
        sampled = model.sample_keypoints(batch["image"])
    assert torch.isfinite(sampled["flow_person_state"]).all()
    restored = model.state_dict()
    model.load_state_dict(restored, strict=True)


def test_v10_reference_gradients_and_denoising_mask():
    from src.models.pipelines.pose_estimators.poseflow import PoseFlowV10Model
    model = PoseFlowV10Model(
        backbone_pretrained=False, num_queries=4, num_proposals=2,
        num_group=2, num_encoder_layers=1, num_decoder_layers=2,
        num_feature_levels=3, hidden_dim=32, feedforward_dim=64,
        num_heads=8, num_body_points=2, num_box_decoder_layers=1,
        dn_number=2,
    )
    model.eval()
    references = []
    def capture(module, args):
        args[1].retain_grad()
        references.append(args[1])
    hooks = [layer.cross_attn.register_forward_pre_hook(capture)
             for layer in model.transformer.decoder.layers]
    output = model(torch.randn(1, 3, 64, 64), torch.zeros(1, 4, 4),
                   torch.tensor([0.5]), person_state=torch.randn(1, 4, 8))
    output["pred_keypoints"].sum().backward()
    for hook in hooks:
        hook.remove()
    assert len(references) == 2
    assert all(reference.grad is not None and torch.isfinite(reference.grad).all()
               and reference.grad.abs().sum() > 0 for reference in references)
    mask = model.training_attention_mask(
        [{"labels": torch.tensor([1])}], 1, torch.device("cpu"))
    assert mask.shape == (8, 4, 4)
    assert mask[:, 2:, :2].all()
    assert mask[:, 0, 1].all()
    assert not mask[:, 2:, 2:].any()


def test_v10_preset_keeps_v5_training_contract():
    from hydra import compose, initialize_config_dir
    from pathlib import Path
    from omegaconf import OmegaConf
    with initialize_config_dir(version_base=None, config_dir=str(Path(__file__).resolve().parents[1] / "conf")):
        v5 = OmegaConf.to_container(compose(config_name="benchmark_poseflow_v5"), resolve=True)
        v10 = OmegaConf.to_container(compose(config_name="benchmark_poseflow_v10"), resolve=True)
    assert v10["pipeline"].pop("decoder_loss_mode") == "all"
    v10["pipeline"]["_target_"] = v5["pipeline"]["_target_"]
    v10["run_name"] = v5["run_name"]
    v10["out_dir"] = v5["out_dir"]
    assert v10["pipeline"].pop("lr_main_warmup_epochs") == 1
    assert v10["pipeline"].pop("lr_backbone_warmup_epochs") == 10
    assert v10["pipeline"]["scheduler_type"] == "cosine"
    assert v10["pipeline"]["lr_backbone"] == 1e-5
    for name in ("scheduler_type", "lr_backbone"):
        v10["pipeline"][name] = v5["pipeline"][name]
    assert v10 == v5


def test_v10_tiny_preset_changes_only_width_and_pretraining_gate():
    from hydra import compose, initialize_config_dir
    from pathlib import Path
    from omegaconf import OmegaConf

    config_dir = Path(__file__).resolve().parents[1] / "conf"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        base = OmegaConf.to_container(compose(config_name="benchmark_poseflow_v10"), resolve=True)
        tiny = OmegaConf.to_container(compose(config_name="benchmark_poseflow_v10_tiny"), resolve=True)

    assert base["pipeline"]["hidden_dim"] == 256
    assert base["pipeline"]["feedforward_dim"] == 2048
    assert tiny["pipeline"]["hidden_dim"] == 128
    assert tiny["pipeline"]["feedforward_dim"] == 256
    assert tiny["pipeline"]["pretrained_checkpoint"] == base["pipeline"]["pretrained_checkpoint"]
    assert tiny["pipeline"]["pretrained_minimum_critical_coverage"] is None
    tiny["pipeline"].update(
        hidden_dim=256,
        feedforward_dim=2048,
        pretrained_minimum_critical_coverage=base["pipeline"]["pretrained_minimum_critical_coverage"],
        _target_=base["pipeline"]["_target_"],
    )
    tiny["run_name"] = base["run_name"]
    tiny["out_dir"] = base["out_dir"]
    assert tiny == base


def test_v10_tiny_loads_edpose_backbone_exactly():
    from hydra import compose, initialize_config_dir
    from pathlib import Path
    from src.models.pipelines.pose_estimators.poseflow import PoseFlowPipelineConfig

    config_dir = Path(__file__).resolve().parents[1] / "conf"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = compose(config_name="benchmark_poseflow_v10_tiny")
    pipeline_config = PoseFlowPipelineConfig.from_hydra_config(config.pipeline)
    with pytest.warns(UserWarning, match="shape-incompatible keys"):
        pipeline = pipeline_config.build(max_epochs=80)

    checkpoint = torch.load(
        pipeline_config.pretrained_checkpoint, map_location="cpu", weights_only=False
    )
    checkpoint_state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    checkpoint_state = {
        name.removeprefix("module."): value for name, value in checkpoint_state.items()
    }
    model_state = pipeline.model.state_dict()
    backbone_keys = [name for name in model_state if name.startswith("backbone.")]
    assert backbone_keys
    assert all(
        name in checkpoint_state and torch.equal(model_state[name], checkpoint_state[name])
        for name in backbone_keys
    )
    assert pipeline.model.transformer.decoder.d_model == 128
    assert pipeline.model.transformer.encoder.layers[0].linear1.out_features == 256
    assert pipeline.model.transformer.decoder.layers[0].linear1.out_features == 256


def test_v10_tiny_width_forward_backward():
    pipeline = PoseFlowV10PipelineConfig(
        backbone_pretrained=False, num_queries=4, num_proposals=2,
        num_group=2, num_encoder_layers=1, num_decoder_layers=2,
        num_feature_levels=3, hidden_dim=128, feedforward_dim=256,
        num_heads=8, num_body_points=2, num_box_decoder_layers=1,
        dn_number=2,
    ).build(max_epochs=2)
    batch = _batch() | {"image": torch.randn(1, 3, 64, 64)}
    losses = pipeline._losses_for_batch(batch)
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    weight = pipeline.model.transformer.decoder.layers[0].linear1.weight
    assert weight.grad is not None and torch.isfinite(weight.grad).all()


def test_v10_cosine_group_warmups_and_minima():
    pipeline = PoseFlowV10PipelineConfig(
        backbone_pretrained=False, num_queries=4, num_proposals=2,
        num_group=2, num_encoder_layers=1, num_decoder_layers=2,
        num_feature_levels=3, hidden_dim=32, feedforward_dim=64,
        num_heads=8, num_body_points=2, num_box_decoder_layers=1, dn_number=0,
    ).build(max_epochs=80)
    configured = pipeline.configure_optimizers()
    optimizer = configured['optimizer']
    scheduler = configured['lr_scheduler']['scheduler']
    trajectory = []
    for epoch in range(80):
        trajectory.append({group['schedule_group']: group['lr'] for group in optimizer.param_groups})
        optimizer.step()
        scheduler.step()
    assert trajectory[0]['main'] == pytest.approx(1e-4)
    assert trajectory[0]['backbone'] == pytest.approx(1e-6)
    assert trajectory[9]['backbone'] == pytest.approx(1e-5)
    assert trajectory[79]['main'] == pytest.approx(1e-6)
    assert trajectory[79]['backbone'] == pytest.approx(1e-7)
    assert trajectory[40]['main'] < trajectory[39]['main']


def test_v10_tiny_v5lr_preset_retains_tiny_model_and_v5_lr_policy():
    from hydra import compose, initialize_config_dir
    from pathlib import Path
    from omegaconf import OmegaConf

    config_dir = Path(__file__).resolve().parents[1] / "conf"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        tiny = OmegaConf.to_container(compose(config_name="benchmark_poseflow_v10_tiny"), resolve=True)
        v5_lr = OmegaConf.to_container(compose(config_name="benchmark_poseflow_v10_tiny_v5lr"), resolve=True)

    assert v5_lr["pipeline"]["hidden_dim"] == 128
    assert v5_lr["pipeline"]["feedforward_dim"] == 256
    assert v5_lr["pipeline"]["scheduler_type"] == "two_cycle_cosine"
    assert v5_lr["pipeline"]["lr"] == 1e-4
    assert v5_lr["pipeline"]["lr_backbone"] == 5e-5
    assert v5_lr["pipeline"]["lr_main_warmup_epochs"] == 0
    assert v5_lr["pipeline"]["lr_backbone_warmup_epochs"] == 0
    for name in ("scheduler_type", "lr_backbone", "lr_main_warmup_epochs", "lr_backbone_warmup_epochs"):
        v5_lr["pipeline"][name] = tiny["pipeline"][name]
    v5_lr["run_name"] = tiny["run_name"]
    v5_lr["out_dir"] = tiny["out_dir"]
    assert v5_lr == tiny


def test_v10_can_select_v5_two_cycle_scheduler():
    pipeline = PoseFlowV10PipelineConfig(
        backbone_pretrained=False, num_queries=4, num_proposals=2,
        num_group=2, num_encoder_layers=1, num_decoder_layers=2,
        num_feature_levels=3, hidden_dim=32, feedforward_dim=64,
        num_heads=8, num_body_points=2, num_box_decoder_layers=1, dn_number=0,
        scheduler_type="two_cycle_cosine", lr_backbone=5e-5,
        lr_main_warmup_epochs=0, lr_backbone_warmup_epochs=0,
    ).build(max_epochs=80)
    configured = pipeline.configure_optimizers()
    optimizer = configured["optimizer"]
    scheduler = configured["lr_scheduler"]["scheduler"]
    trajectory = []
    for _ in range(80):
        trajectory.append({group["schedule_group"]: group["lr"] for group in optimizer.param_groups})
        optimizer.step()
        scheduler.step()
    assert trajectory[0] == pytest.approx({"main": 1e-4, "backbone": 5e-5})
    assert trajectory[39] == pytest.approx({"main": 1e-5, "backbone": 1e-5})
    assert trajectory[40] == pytest.approx({"main": 3e-5, "backbone": 3e-5})
    assert trajectory[79] == pytest.approx({"main": 2e-6, "backbone": 2e-6})


def test_v10_rejects_two_cycle_with_group_warmup():
    with pytest.raises(ValueError, match="warmup"):
        PoseFlowV10PipelineConfig(
            scheduler_type="two_cycle_cosine", lr_main_warmup_epochs=1,
            lr_backbone_warmup_epochs=10,
        )


def test_v10_latent_metric_formulas():
    from src.eval.poseflow_diagnostics import summarize_v10_latent_update
    before = torch.ones(1, 2, 4)
    after = before * 2
    logits = torch.tensor([[[6., 0., -6., 0., 0., 0., 0., 0.]]]).expand(1, 2, -1)
    coordinates = logits.sigmoid()
    values = summarize_v10_latent_update(before, after, coordinates, logits)
    assert values['update_input_rms_ratio'] == pytest.approx(1.)
    assert values['person_centered_rms'] == 0
    assert values['geometry_logit_saturation_fraction'] == pytest.approx(.25)
    assert values['pose_movement_l1'] == 0
    zero = summarize_v10_latent_update(before * 0, before * 0, coordinates, logits)
    assert all(torch.isfinite(torch.tensor(value)) for value in zero.values())


def test_v10_capture_rejects_denoising_and_cleans_up_after_exception():
    from src.models.pipelines.pose_estimators.poseflow import PoseFlowV10Model
    from src.eval.poseflow_diagnostics import capture_v10_latents
    model = PoseFlowV10Model(
        backbone_pretrained=False, num_queries=4, num_proposals=2,
        num_group=2, num_encoder_layers=1, num_decoder_layers=2,
        num_feature_levels=3, hidden_dim=32, feedforward_dim=64,
        num_heads=8, num_body_points=2, num_box_decoder_layers=1, dn_number=2,
    )
    with pytest.raises(ValueError, match='evaluation mode'):
        with capture_v10_latents(model):
            pass
    model.eval()
    with pytest.raises(RuntimeError, match='probe failure'):
        with capture_v10_latents(model):
            raise RuntimeError('probe failure')
    assert all(not module._forward_hooks and not module._forward_pre_hooks
               for module in model.modules())
