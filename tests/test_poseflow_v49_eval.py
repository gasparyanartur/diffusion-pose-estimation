from pathlib import Path

from hydra import compose, initialize_config_dir
import torch

from src.models.pipelines.pose_estimators.poseflow.quick_eval_v49 import evaluate_v49_batches, tensorboard_metrics
from src.models.pipelines.pose_estimators.poseflow.training_v49 import PoseFlowV49PipelineConfig
from tests.test_poseflow_v43_training import batch
from tests.test_poseflow_v49 import model


def test_empty_predictions_keep_teacher_diagnostics_and_full_refinement_tail(tmp_path):
    network = model()
    for head in network.eos_heads:
        head.weight.data.zero_()
        head.bias.data.fill_(20.)
    sample = batch((2,))
    sample['original_img_hw'] = sample['img_hw']
    def evaluator(records, annotations, ids):
        assert records == [] and ids == [0]
        return dict(pose_AP=0., bbox_AP=0.)
    report = evaluate_v49_batches(network, [sample], tmp_path, 'unused', [0], metric_evaluator=evaluator)
    assert report['count_mae'] == 2 and report['mean_generation_steps'] == 7
    assert report['refinement']['teacher_forced/rank_01/birth_coordinate_mse'] >= 0
    assert not any(k.startswith('generated_final_assignment') for k in report['refinement'])
    assert 'quick-test/token_age_diffusion/rows_7/pose_AP' in tensorboard_metrics(report)


def test_full_training_contract_uses_train_val_fp32_and_50_rows():
    with initialize_config_dir(version_base='1.3', config_dir=str(Path('conf').resolve())):
        config = compose(config_name='benchmark_poseflow_v49')
    pipeline = PoseFlowV49PipelineConfig.from_hydra_config(config.pipeline)
    assert pipeline.rows == 50 and pipeline.capacity == 49 and pipeline.scale == 6
    assert config.data.batch_size == 4 and config.training.accumulate_grad_batches == 2
    assert config.training.precision == 32 and config.training.max_epochs == 80
    assert config.data.train_ann.endswith('crowdpose_train.json')
    assert config.data.test_ann == config.data.val_ann


def test_generated_measurements_use_final_set_assignment_and_are_repeatable(tmp_path):
    network = model()
    for head in network.eos_heads:
        head.weight.data.zero_()
        head.bias.data.fill_(-20.)
    sample = batch((2,))
    sample['original_img_hw'] = sample['img_hw']
    evaluator = lambda *args: dict(pose_AP=0., bbox_AP=0.)
    first = evaluate_v49_batches(network, [sample], tmp_path/'first', 'unused', [0], metric_evaluator=evaluator)
    second = evaluate_v49_batches(network, [sample], tmp_path/'second', 'unused', [0], metric_evaluator=evaluator)
    assert first['refinement'] == second['refinement'] and first['diagnostics'] == second['diagnostics']
    assert any(key.startswith('generated_final_assignment/') for key in first['refinement'])
    assert first['diagnostics']['nearest_gt_duplicate_count'] >= 4
    assert first['capacity_stop_fraction'] == 1
