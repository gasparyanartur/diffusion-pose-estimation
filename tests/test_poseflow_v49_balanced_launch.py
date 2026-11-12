import json
from copy import deepcopy
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest
import torch

from scripts.training import train_poseflow_v49_balanced as entrypoint
from src.models.pipelines.pose_estimators.poseflow.training_v49 import PoseFlowV49PipelineConfig
from src.models.pipelines.pose_estimators.poseflow.training_v49_balanced import PoseFlowV49BalancedPipelineConfig
from tests.test_poseflow_v49_balanced_training import pipeline


def test_fresh_variant_preserves_model_initialization_and_rng():
    specification = pipeline().config.model_dump()
    common = {name:value for name,value in specification.items() if name not in ('calibration_batches','calibration_seed')}
    torch.manual_seed(42)
    baseline = PoseFlowV49PipelineConfig(**common).build(load_pretrained_checkpoint=False)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(42)
    improved = PoseFlowV49BalancedPipelineConfig(**specification).build(load_pretrained_checkpoint=False)
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(value,improved.model.state_dict()[name]) for name,value in baseline.model.state_dict().items())


def test_full_recipe_and_calibration_verification_guard(tmp_path,monkeypatch):
    with initialize_config_dir(version_base='1.3',config_dir=str(Path('conf').resolve())):
        config = compose(config_name='benchmark_poseflow_v49_balanced')
    assert config.training.max_epochs == 80 and config.data.batch_size == 4
    assert config.training.accumulate_grad_batches == 2 and config.pipeline.rows == 50
    assert config.pipeline.calibration_batches == 32 and config.data.test_ann == config.data.val_ann
    config.pipeline.calibration_batches = 2
    report = dict(passed=True,native_preflight=dict(passed=True),evaluator_smoke=dict(images=2),
        sources={'test':'sha'},initial_model_sha256='initial',inputs={},
        pipeline_config=OmegaConf.to_container(config.pipeline,resolve=True),
        data_config=OmegaConf.to_container(config.data,resolve=True),
        calibration=dict(norms=[[1.,2.,3.,4.],[3.,4.,5.,6.]],scales=[2.,3.,4.,5.],
            model_sha256_before='initial',model_sha256_after='initial'))
    monkeypatch.setattr(entrypoint,'source_hashes',lambda:{'test':'sha'})
    path = tmp_path/'verification.json'
    path.write_text(json.dumps(report))
    assert entrypoint.validate_verification(path,config,'initial')['passed']
    for field in ('norms','model_sha256_after','scales'):
        broken = deepcopy(report)
        broken['calibration'][field] = [] if field=='norms' else 'wrong'
        path.write_text(json.dumps(broken))
        with pytest.raises(ValueError,match='calibrat'):
            entrypoint.validate_verification(path,config,'initial')
