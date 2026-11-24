import argparse
import json
import os
from copy import deepcopy
from pathlib import Path
import random

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytorch_lightning as pl
import torch

from scripts.training.train_poseflow_v53 import digest, model_digest, source_hashes
from scripts.training.verify_poseflow_v53 import initialization_coverage, native_preflight, training_batches
from src.data import DataConfig
from src.models.pipelines.pose_estimators.poseflow.calibration_v53 import calibrate
from src.models.pipelines.pose_estimators.poseflow.quick_eval_v53 import run_v53_evaluation
from src.models.pipelines.pose_estimators.poseflow.training_v54 import PoseFlowV54Pipeline, PoseFlowV54PipelineConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--residual-bank', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with initialize_config_dir(version_base='1.3', config_dir=str(Path('conf').resolve())):
        config = compose(config_name='benchmark_poseflow_v54', overrides=[f'residual_bank_file={args.residual_bank.resolve()}'])
    sources = source_hashes()
    data_config = DataConfig.from_hydra_config(config.data)
    if data_config.test_ann != data_config.val_ann:
        raise ValueError('Verification must not open TEST')
    data = data_config.build()
    pl.seed_everything(42, workers=True)
    specification = PoseFlowV54PipelineConfig.from_hydra_config(config.pipeline)
    module = specification.build(max_epochs=80)
    initial_hash = model_digest(module.model)
    bank = torch.load(args.residual_bank, map_location='cpu', weights_only=False)
    provenance = bank['provenance']
    train_ids = {int(image['id']) for image in json.loads(Path(data.train_ann_path).read_text())['images']}
    val_ids = {int(image['id']) for image in json.loads(Path(data.val_ann_path).read_text())['images']}
    used = set(provenance['image_ids'])
    if (not used <= train_ids or used & val_ids or len(used) != 512
            or digest(data.train_ann_path) != provenance['train_annotations_sha256']
            or digest(provenance['source_checkpoint']) != provenance['source_checkpoint_sha256']):
        raise ValueError('V54 bank source/split provenance mismatch')
    module.set_residual_bank(bank['residuals'], bank['joint_mask'], provenance)
    frozen_bank = deepcopy(module.prefix_bank_provenance)
    try:
        module.cuda()
        calibration = calibrate(module, training_batches(data))
        calibration.update(model_sha256_before=initial_hash, model_sha256_after=model_digest(module.model),
            train_annotations=str(data.train_ann_path), train_annotations_sha256=digest(data.train_ann_path))
        if calibration['model_sha256_before'] != calibration['model_sha256_after']:
            raise AssertionError('Calibration changed initial model')
        initial_state = {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}
        initialization = deepcopy(module.model.initialization_report)
        initialization['measured_retained_coverage'] = initialization_coverage(module, specification)
        (args.output/'calibration.json').write_text(json.dumps(calibration, indent=2)+'\n')
        (args.output/'config.yaml').write_text(OmegaConf.to_yaml(config))
        preflight = native_preflight(module, args.output, pipeline_class=PoseFlowV54Pipeline)
        module.load_state_dict(initial_state, strict=True)
        if module.prefix_bank_provenance != frozen_bank:
            raise AssertionError('Frozen bank provenance changed')
        module.eval()
        smoke = run_v53_evaluation(module.model, data, args.output/'evaluator_smoke', epoch=-1, count=2)
        val = json.loads(Path(data.val_ann_path).read_text())
        ids = sorted(random.Random(42).sample(sorted(image['id'] for image in val['images']), 32))
        if source_hashes() != sources:
            raise ValueError('Sources changed during verification')
        result = dict(passed=preflight['passed'], native_preflight=preflight,
            evaluator_smoke=dict(images=smoke['images'], metrics=smoke['metrics'], architecture=smoke['architecture']),
            sources=sources, pipeline_config=OmegaConf.to_container(config.pipeline, resolve=True),
            data_config=OmegaConf.to_container(config.data, resolve=True), initial_model_sha256=initial_hash,
            initialization=initialization, calibration=calibration, contract=module.checkpoint_contract(),
            fixed_val_image_ids=ids, pid=os.getpid(), prefix_bank=module.prefix_bank_provenance,
            residual_bank_file=str(args.residual_bank.resolve()), residual_bank_file_sha256=digest(args.residual_bank),
            inputs={str(path): digest(path) for path in (data.train_ann_path, data.val_ann_path,
                specification.pretrained_checkpoint, args.residual_bank.resolve())})
        (args.output/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
        print('V54_VERIFICATION', json.dumps(dict(passed=True)), flush=True)
    except BaseException as error:
        failure = dict(passed=False, error=repr(error), resource_failure=isinstance(error, torch.cuda.OutOfMemoryError),
                       policy='FP32/50rows/49capacity/batch4/accum2 unchanged; existing jobs preserved')
        (args.output/'failure.json').write_text(json.dumps(failure, indent=2)+'\n')
        raise


if __name__ == '__main__':
    main()
