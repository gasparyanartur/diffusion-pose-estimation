from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from tests.test_poseflow_v43_training import batch
from tests.test_poseflow_v53 import model
from src.models.pipelines.pose_estimators.poseflow.quick_eval_v53 import evaluate_v53_batches, tensorboard_metrics
from src.models.pipelines.pose_estimators.poseflow import quick_eval_v53


def test_image_only_birth_eval_shared_tags_and_determinism(tmp_path):
    network = model()
    network.selector.eos_bias.data.fill_(1e6)
    sample = batch((2,))
    sample['original_img_hw'] = sample['img_hw']
    def evaluator(records, annotations, ids):
        assert records == [] and ids == [0]
        return dict(pose_AP=0., bbox_AP=0.)
    first = evaluate_v53_batches(network, [sample], tmp_path/'first', 'unused', [0], metric_evaluator=evaluator)
    second = evaluate_v53_batches(network, [deepcopy(sample)], tmp_path/'second', 'unused', [0], metric_evaluator=evaluator)
    assert first['refinement'] == second['refinement']
    assert first['mean_generation_steps'] == 1 and first['count_mae'] == 2
    assert 'quick-test/pose_AP' in tensorboard_metrics(first)
    assert not any('refine' in key or 'final' in key for key in first['refinement'])


@pytest.mark.parametrize('split,count', [('test', 2), ('test', None), ('val', 2)])
def test_evaluation_uses_requested_annotation_split(monkeypatch, tmp_path, split, count):
    data = SimpleNamespace(val_ann_path=tmp_path/'crowdpose_val.json',
        test_ann_path=tmp_path/'crowdpose_test.json', img_dir=tmp_path,
        eval_resize_scale=256, eval_max_size=427)
    annotation = getattr(data, f'{split}_ann_path')
    candidates = [11, 12, 13] if split == 'test' else [21, 22, 23]
    def dataset_from_path(path, **kwargs):
        assert path == annotation
        return SimpleNamespace(samples=[dict(image_id=value) for value in candidates])
    monkeypatch.setattr('src.data.datasets.PoseDataset.from_path', dataset_from_path)
    monkeypatch.setattr(quick_eval_v53, 'v21_test_batches', lambda *args: [])
    def evaluate(network, batches, directory, annotations, image_ids, **kwargs):
        assert annotations == annotation and kwargs['split'] == split
        assert set(image_ids) <= set(candidates)
        directory.mkdir(parents=True)
        return dict(images=len(image_ids), split=split)
    monkeypatch.setattr(quick_eval_v53, 'evaluate_v53_batches', evaluate)
    kwargs = {} if split == 'test' else dict(split=split)
    report = quick_eval_v53.run_v53_evaluation(torch.nn.Linear(1, 1), data,
        tmp_path/'evaluation', epoch=0, count=count, **kwargs)
    assert report['images'] == (3 if count is None else 2)
    assert report['full_test'] == (split == 'test' and count is None)
    assert report['full_validation'] == (split == 'val' and count is None)


def test_test_evaluation_rejects_validation_alias(tmp_path):
    data = SimpleNamespace(val_ann_path=tmp_path/'crowdpose_val.json',
        test_ann_path=tmp_path/'crowdpose_val.json')
    with pytest.raises(ValueError, match='TEST.*VAL'):
        quick_eval_v53.run_v53_evaluation(torch.nn.Linear(1, 1), data, tmp_path, epoch=0)


def test_resume_baseline_uses_restored_step_and_checkpoint_epoch(monkeypatch, tmp_path):
    from scripts.training import train_poseflow_v53
    callback = train_poseflow_v53.V53Progress(tmp_path, evaluate_on_resume=True)
    trainer = SimpleNamespace(global_step=0, current_epoch=0, ckpt_path='last.ckpt', datamodule=None)
    recorded = []
    def evaluate(*args, **kwargs):
        recorded.append(kwargs)
        return {}
    monkeypatch.setattr(train_poseflow_v53, 'run_v53_evaluation', evaluate)
    module = SimpleNamespace(model=None, config=SimpleNamespace(quick_eval_enabled=True,
        quick_eval_images=32, seed=42), log_detection_report=lambda report, **kwargs: recorded.append(kwargs))
    callback.on_fit_start(trainer, module)
    assert recorded == []
    trainer.global_step, trainer.current_epoch = 32500, 25
    callback.on_train_start(trainer, module)
    assert recorded == [dict(epoch=25, count=32, seed=42), dict(step=32500)]
