from pathlib import Path
import json

import pytest
import torch

from src.models.pipelines.pose_estimators.poseflow.quick_eval_v21 import evaluate_v21_quick_grid
from tests.test_poseflow_v21 import configuration


@pytest.mark.parametrize('order', ['parallel', 'h_then_g'])
def test_native_sampling_parity_modes_rng_and_configuration(tmp_path, order):
    from types import SimpleNamespace
    from src.models.pipelines.pose_estimators.diffusionregpose.evaluation import crowdpose_result_records, postprocess_crowdpose
    model = configuration(dn_number=0, hypothesis_update_order=order)._build_poseflow_model().train()
    model.flow_sampling_steps = 2
    image = torch.randn(1, 3, 64, 64)
    batch = {'image':image, 'image_id':torch.tensor([17]), 'img_hw':torch.tensor([[64,64]])}
    config = SimpleNamespace(quick_eval_steps=[1,3], quick_eval_seed=42)
    modes = [module.training for module in model.modules()]
    state = torch.random.get_rng_state()
    evaluated=[]
    def metrics(records, annotations, ids):
        evaluated.append(records)
        assert ids == [17]
        return {'pose_AP':.1,'bbox_AP':.2}
    report = evaluate_v21_quick_grid(model,[batch],config,tmp_path,'test.json',[17],metric_evaluator=metrics)
    assert torch.equal(state,torch.random.get_rng_state())
    assert model.flow_sampling_steps == 2 and modes == [module.training for module in model.modules()]
    assert len(report['conditions']) == 4 and len(evaluated) == 4
    model.eval()
    for steps in [1,3]:
        model.flow_sampling_steps = steps
        native=model.sample_keypoints(image,generator=torch.Generator().manual_seed(59))
        records=crowdpose_result_records(17,postprocess_crowdpose(native,batch['img_hw'].float(),num_body_points=2,nms_target_sizes=batch['img_hw'].float()))
        row=report['conditions'][f'noise_var_100/steps_{steps}/memory_persistent']
        assert records == json.loads(Path(row['predictions']).read_text())
    first=report['conditions']['noise_var_100/steps_1/memory_persistent']
    reset=report['conditions']['noise_var_100/steps_1/memory_reset']
    assert Path(first['predictions']).read_text() == Path(reset['predictions']).read_text()


def test_failure_restores_sampler_and_rejects_incomplete_coverage(tmp_path):
    from types import SimpleNamespace
    model=configuration(dn_number=0)._build_poseflow_model().train()
    model.flow_sampling_steps=8
    with pytest.raises(ValueError,match='coverage'):
        evaluate_v21_quick_grid(model,[],SimpleNamespace(quick_eval_steps=[1],quick_eval_seed=42),tmp_path,'test.json',[5])
    assert model.training and model.flow_sampling_steps==8


def test_monitoring_fields_do_not_change_model_or_recipe():
    values=configuration().model_dump()
    from src.models.pipelines.pose_estimators.poseflow.training_v21 import PoseFlowV21PipelineConfig
    config=PoseFlowV21PipelineConfig(**values)
    assert config.quick_eval_enabled and config.quick_eval_images==32 and config.quick_eval_steps==[1,2,3]
    with pytest.raises(ValueError):
        PoseFlowV21PipelineConfig(**(values | {'quick_eval_steps':[1,1]}))
    torch.manual_seed(9)
    enabled=config._build_poseflow_model()
    torch.manual_seed(9)
    disabled=PoseFlowV21PipelineConfig(**(values | {'quick_eval_enabled':False}))._build_poseflow_model()
    assert enabled.state_dict().keys()==disabled.state_dict().keys()
    assert all(torch.equal(v,disabled.state_dict()[k]) for k,v in enabled.state_dict().items())


def test_validation_hook_guards_and_checkpoint_report_reuse(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from src.models.pipelines.pose_estimators.poseflow.training import PoseFlowPipeline
    from src.models.pipelines.pose_estimators.poseflow import quick_eval_v21
    monkeypatch.setattr(PoseFlowPipeline, 'on_validation_epoch_end', lambda self: None)
    pipeline = configuration(dn_number=0).build(max_epochs=2)
    trainer = SimpleNamespace(sanity_checking=True, is_global_zero=True, current_epoch=1,
        global_step=20, checkpoint_callback=None, default_root_dir=str(tmp_path), datamodule=None)
    pipeline._trainer = trainer
    calls, logged = [], []
    def evaluate(*args, **kwargs):
        calls.append(kwargs)
        (args[3]/'epoch001').mkdir(parents=True)
        return {'conditions': {'noise_var_100/steps_1/memory_reset': {'metrics': {'pose_AP': .3}}}}
    monkeypatch.setattr(quick_eval_v21, 'run_v21_quick_test', evaluate)
    monkeypatch.setattr(pipeline, 'log', lambda tag, value, **kwargs: logged.append((tag, value)))
    pipeline.on_validation_epoch_end()
    assert not calls
    trainer.sanity_checking = False
    pipeline.on_validation_epoch_end()
    pipeline.on_validation_epoch_end()
    assert len(calls) == 1 and calls[0]['global_step'] == 20
    assert logged == [('quick-test-reset/noise_var_100/steps_1/pose_AP', .3)]
    trainer.current_epoch = 2
    existing = tmp_path / 'quick_test/epoch002/tensorboard_logged.json'
    existing.parent.mkdir(parents=True)
    existing.write_text('{}')
    pipeline.on_validation_epoch_end()
    assert len(calls) == 1


def test_checkpoint_monitor_strict_restore_and_tensorboard_tags(tmp_path, monkeypatch):
    from omegaconf import OmegaConf
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from scripts.evaluation import watch_poseflow_v21_quick_test as monitor
    config = configuration(dn_number=0)
    model = config._build_poseflow_model()
    path = tmp_path / 'epoch001.ckpt'
    torch.save({'epoch': 1, 'global_step': 123,
        'state_dict': {'model.'+key: value for key, value in model.state_dict().items()}}, path)
    (tmp_path / '.hydra').mkdir()
    OmegaConf.save(OmegaConf.create({'pipeline': {}, 'data': {}}), tmp_path / '.hydra/config.yaml')
    monkeypatch.setattr(monitor.PoseFlowPipelineConfig, 'from_hydra_config', lambda _: config)
    class FakeDataConfig:
        def build(self, **kwargs):
            return None
    monkeypatch.setattr(monitor.DataConfig, 'from_hydra_config', lambda _: FakeDataConfig())
    def evaluate(restored, data, config, root, **kwargs):
        assert all(torch.equal(value, restored.state_dict()[key]) for key, value in model.state_dict().items())
        report = kwargs | {'conditions': {
            f'noise_var_100/steps_1/memory_{mode}': {'metrics': {'pose_AP': .4}}
            for mode in ('persistent', 'reset')}}
        directory = root / 'epoch001'
        directory.mkdir(parents=True)
        (directory / 'metrics.json').write_text(json.dumps(report))
        return report
    monkeypatch.setattr(monitor, 'run_v21_quick_test', evaluate)
    monitor.evaluate_checkpoint(tmp_path, path, torch.device('cpu'))
    monitor.evaluate_checkpoint(tmp_path, path, torch.device('cpu'))
    events = EventAccumulator(str(tmp_path / 'tb_logs')).Reload()
    for prefix in ('quick-test', 'quick-test-reset'):
        values = events.Scalars(f'{prefix}/noise_var_100/steps_1/pose_AP')
        assert len(values) == 1 and values[0].step == 123
