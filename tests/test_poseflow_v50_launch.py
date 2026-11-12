from pathlib import Path

from hydra import compose, initialize_config_dir

from scripts.training.train_poseflow_v50 import validate_verification
from src.models.pipelines.pose_estimators.poseflow.training_v50 import PoseFlowV50PipelineConfig
import pytest


def test_full_recipe_and_verification_required():
    with initialize_config_dir(version_base='1.3', config_dir=str(Path('conf').resolve())):
        config = compose(config_name='benchmark_poseflow_v50')
    specification = PoseFlowV50PipelineConfig.from_hydra_config(config.pipeline)
    assert specification.rows == 50 and specification.capacity == 49
    assert config.training.max_epochs == 80 and config.training.accumulate_grad_batches == 2
    assert config.data.batch_size == 4 and config.data.test_ann == config.data.val_ann
    assert config.training.checkpoint.monitor == 'val/proposal_attention_selection_AP'
    assert config.training.precision == 32 and specification.calibration_batches == 32
    with pytest.raises(ValueError, match='verification|preflight'):
        validate_verification(None, config, 'none')


def test_initialization_metadata_counts_nested_frontend():
    from tests.test_poseflow_v44 import arguments
    specification = PoseFlowV50PipelineConfig(native=arguments(), rows=7, capacity=6,
        codec_width=32, latent_dim=8, pretrained_checkpoint=None, quick_eval_enabled=False)
    module = specification.build(load_pretrained_checkpoint=False)
    expected = sum(name.startswith(('backbone.', 'input_proj.', 'transformer.', 'bbox_embed.', 'class_embed.'))
        for name in module.model.frontend.native.state_dict())
    assert module.model.initialization_report['retained_tensor_count'] == expected > 0
