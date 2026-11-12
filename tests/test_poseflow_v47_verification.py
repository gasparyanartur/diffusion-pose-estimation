from scripts.training.verify_poseflow_v47 import learnability_passed, overfit_image_ids
import torch
from types import SimpleNamespace
from hydra import compose, initialize_config_dir
from pathlib import Path
from src.models.pipelines.pose_estimators.poseflow.training_v47 import PoseFlowV47PipelineConfig


def test_learnability_gate_requires_both_seeds_counts_and_both_AP_metrics():
    report = dict(images=4, exact_count_images=4, metrics=dict(pose_AP=.91, bbox_AP=.92))
    assert learnability_passed({'42': report, '43': report})
    assert not learnability_passed({'42': report})
    wrong_count = {**report, 'exact_count_images': 3}
    assert not learnability_passed({'42': report, '43': wrong_count})
    weak_box = {**report, 'metrics': dict(pose_AP=.99, bbox_AP=.89)}
    assert not learnability_passed({'42': weak_box, '43': report})


def test_full_training_preset_constructs_v47_without_codec_assets():
    with initialize_config_dir(version_base='1.3', config_dir=str(Path('conf').resolve())):
        configuration = compose(config_name='benchmark_poseflow_v47')
    pipeline = PoseFlowV47PipelineConfig.from_hydra_config(configuration.pipeline)
    assert pipeline.rows == 64 and pipeline.capacity == 50
    assert pipeline.coordinate_scale == 6 and not hasattr(pipeline, 'codec_checkpoint')
    assert configuration.data.train_ann.endswith('crowdpose_train.json')
    assert configuration.data.test_ann == configuration.data.val_ann
    assert configuration.training.max_epochs == 80


def test_overfit_selection_has_matching_native_box_and_pose_coverage():
    samples, annotations = [], []
    for image, count in ((1, 1), (2, 2), (3, 3), (4, 4), (5, 4)):
        samples.append(dict(image_id=image, visibility=torch.ones(count, 14), gt_boxes_pixel=torch.ones(count, 4)))
        annotations.extend(dict(image_id=image, iscrowd=0) for _ in range(count))
    annotations.append(dict(image_id=4, iscrowd=0))  # Box-only person absent from pose dataset.
    assert overfit_image_ids(SimpleNamespace(samples=samples), annotations) == [1, 2, 3, 5]
