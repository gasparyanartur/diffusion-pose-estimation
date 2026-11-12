import argparse
import gc
import json
import os
import random
from copy import deepcopy
from pathlib import Path
import time

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytorch_lightning as pl
import torch

from src.data import DataConfig
from src.models.pipelines.pose_estimators.poseflow.training_v48_adaln import (
    PoseFlowV48AdaLNPipeline as PoseFlowV48Pipeline, PoseFlowV48AdaLNPipelineConfig as PoseFlowV48PipelineConfig)
from src.models.pipelines.pose_estimators.poseflow.quick_eval_v48 import run_v48_evaluation
from scripts.training.train_poseflow_v48_adaln import digest, model_digest, source_hashes
from scripts.training.verify_poseflow_v47 import synthetic_capacity_batch, optimizer_update



from scripts.training.verify_poseflow_v48 import initialization_coverage


def native_preflight(module, output):
    sample = synthetic_capacity_batch()
    configuration = module.configure_optimizers()
    optimizer = configuration['optimizer']
    scheduler = configuration['lr_scheduler']['scheduler']
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    first, norm = optimizer_update(module, optimizer, [sample, sample])
    scheduler.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-started
    peak = torch.cuda.max_memory_allocated()
    gradient_coverage = {name: float(parameter.grad.norm()) for name, parameter in module.model.named_parameters()
                         if parameter.requires_grad and parameter.grad is not None}
    if any(not torch.isfinite(parameter.grad).all() for parameter in module.parameters() if parameter.grad is not None):
        raise AssertionError('Nonfinite native gradients')
    (output/'gradient_coverage.json').write_text(json.dumps(gradient_coverage, indent=2)+'\n')
    codec_gradients = {name: float(parameter.grad.norm()) for name, parameter in module.model.person_codec.named_parameters()}
    if not all(value > 0 for value in codec_gradients.values()):
        raise AssertionError('Learned codec has an inactive native gradient path')
    checkpoint = dict(state_dict=module.state_dict(), hyper_parameters=module.hparams,
        **{'pytorch-lightning_version': pl.__version__}, optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict())
    module.on_save_checkpoint(checkpoint)
    checkpoint_path = output/'native_restore.ckpt'
    torch.save(checkpoint, checkpoint_path)
    second, _ = optimizer_update(module, optimizer, [sample, sample])
    scheduler.step()
    modulation_gradients = {name: float(parameter.grad.norm()) for name, parameter in module.model.named_parameters()
                            if 'image_modulation' in name or 'generation_step_embedding' in name}
    if not all(value > 0 for value in modulation_gradients.values()):
        raise AssertionError('AdaLN/step embedding gradient missing after second update')
    (output/'adaln_gradients.json').write_text(json.dumps(modulation_gradients, indent=2)+'\n')
    expected = {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}
    restored = PoseFlowV48Pipeline.load_from_checkpoint(checkpoint_path, map_location='cpu', weights_only=False).cuda()
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    restore_configuration = restored.configure_optimizers()
    restore_optimizer = restore_configuration['optimizer']
    restore_optimizer.load_state_dict(saved['optimizer'])
    restore_scheduler = restore_configuration['lr_scheduler']['scheduler']
    restore_scheduler.load_state_dict(saved['scheduler'])
    resumed, _ = optimizer_update(restored, restore_optimizer, [sample, sample])
    restore_scheduler.step()
    maximum = max(float((value.detach().cpu()-expected[name]).abs().max())
                  for name, value in restored.state_dict().items())
    if abs(second['total']-resumed['total']) > 1e-6 or maximum > 2e-7:
        raise AssertionError(f'Native next-update restore mismatch: {second}, {resumed}, delta={maximum}')
    if scheduler.state_dict() != restore_scheduler.state_dict():
        raise AssertionError('Scheduler restore mismatch')
    restored.eval()
    for head in restored.model.eos_heads:
        head.weight.data.zero_()
        head.bias.data.fill_(-20.)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = restored.model.sample_keypoints(sample['image'], padding_mask=sample['padding_mask'],
        image_wh=sample['img_hw'].flip(-1))
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter()-started
    assert torch.isfinite(result['pred_coordinates']).all()
    assert result['person_counts'].tolist() == [50]*4 and result['generation_steps'].tolist() == [50]*4
    report = dict(passed=True, adaln_gradients_verified=True, precision='FP32', rows=51, people=50, batch=4, accumulation=2,
        image_hw=[256, 416], optimizer_update_seconds=elapsed, train_images_per_second=8/elapsed,
        peak_training_allocated_bytes=peak, gradient_norm=norm, codec_gradient_norms=codec_gradients,
        inference_images_per_second=4/inference_seconds, inference_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        next_update_loss=second['total'], resumed_loss=resumed['total'], max_restored_state_delta=maximum,
        note='Shared-GPU measurements; inference includes two resident verification models and trace-free final outputs')
    (output/'preflight.json').write_text(json.dumps(report, indent=2)+'\n')
    print('V48_ADALN_PREFLIGHT', json.dumps(report), flush=True)
    del restored, module, optimizer, configuration, restore_configuration, restore_optimizer, sample, checkpoint, saved
    gc.collect()
    torch.cuda.empty_cache()
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline-manifest', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with initialize_config_dir(version_base='1.3', config_dir=str(Path('conf').resolve())):
        config = compose(config_name='benchmark_poseflow_v48_adaln')
    sources = source_hashes()
    data_config = DataConfig.from_hydra_config(config.data)
    if data_config.test_ann != data_config.val_ann:
        raise ValueError('Verification must not open TEST')
    data = data_config.build()
    pl.seed_everything(42, workers=True)
    specification = PoseFlowV48PipelineConfig.from_hydra_config(config.pipeline)
    module = specification.build(max_epochs=80)
    initial_hash = model_digest(module.model)
    from types import SimpleNamespace
    common = {name: value for name, value in module.model.state_dict().items()
              if 'image_modulation' not in name and 'generation_step_embedding' not in name}
    common_hash = model_digest(SimpleNamespace(state_dict=lambda: common))
    baseline_manifest = json.loads(args.baseline_manifest.read_text())
    if common_hash != baseline_manifest['initial_model_sha256']:
        raise AssertionError('Initial common weights differ from running V48 baseline')
    initial_state = deepcopy(module.state_dict())
    initialization = deepcopy(module.model.initialization_report)
    initialization['measured_retained_coverage'] = initialization_coverage(module, specification)
    (args.output/'config.yaml').write_text(OmegaConf.to_yaml(config))
    try:
        preflight = native_preflight(module.cuda(), args.output)
        module.cpu()
        module.load_state_dict(initial_state, strict=True)
        module.cuda().eval()
        smoke = run_v48_evaluation(module.model, data, args.output/'evaluator_smoke', epoch=-1, count=2)
        val = json.loads(Path(data.val_ann_path).read_text())
        ids = sorted(random.Random(42).sample(sorted(image['id'] for image in val['images']), 32))
        if source_hashes() != sources:
            raise ValueError('Sources changed during verification')
        result = dict(passed=preflight['passed'], native_preflight=preflight,
            common_V48_initialization_verified=True, common_V48_initial_model_sha256=common_hash,
            baseline_manifest=str(args.baseline_manifest),
            evaluator_smoke=dict(images=smoke['images'], metrics=smoke['metrics']),
            sources=sources, pipeline_config=OmegaConf.to_container(config.pipeline, resolve=True),
            data_config=OmegaConf.to_container(config.data, resolve=True), initial_model_sha256=initial_hash,
            initialization=initialization, fixed_val_image_ids=ids, pid=os.getpid(),
            inputs={str(path): digest(path) for path in (data.train_ann_path, data.val_ann_path,
                                                        specification.pretrained_checkpoint, args.baseline_manifest)})
        (args.output/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
        print('V48_ADALN_VERIFICATION', json.dumps(dict(passed=result['passed'])), flush=True)
    except BaseException as error:
        failure = dict(passed=False, error=repr(error), resource_failure=isinstance(error, torch.cuda.OutOfMemoryError),
            policy='Recipe unchanged: FP32/50capacity+EOS/batch4/accum2; existing jobs preserved')
        (args.output/'failure.json').write_text(json.dumps(failure, indent=2)+'\n')
        raise


if __name__ == '__main__':
    main()
