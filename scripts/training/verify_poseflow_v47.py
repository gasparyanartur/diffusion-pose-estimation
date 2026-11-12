import argparse
import gc
import json
import os
import random
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytorch_lightning as pl
import torch
from torch.utils.tensorboard import SummaryWriter

from src.data import DataConfig
from src.data.datasets import PoseDataset, SetPoseDataset, set_pose_collate
from src.models.pipelines.pose_estimators.poseflow.codec_v43 import geometry_from_batch
from src.models.pipelines.pose_estimators.poseflow.coordinates_v47 import ordered_coordinates
from src.models.pipelines.pose_estimators.diffusionregpose.evaluation import crowdpose_result_records, postprocess_crowdpose
from src.models.pipelines.pose_estimators.poseflow.quick_eval_v13 import evaluate_records
from src.models.pipelines.pose_estimators.poseflow.quick_eval_v21 import v21_test_batches
from src.models.pipelines.pose_estimators.poseflow.quick_eval_v47 import evaluate_v47_batches, tensorboard_metrics
from src.models.pipelines.pose_estimators.poseflow.training_v47 import PoseFlowV47Pipeline, PoseFlowV47PipelineConfig
from scripts.training.train_poseflow_v47 import digest, model_digest, source_hashes


def learnability_passed(reports):
    return set(reports) == {'42', '43'} and all(
        report['images'] == report['exact_count_images'] == 4
        and min(report['metrics']['pose_AP'], report['metrics']['bbox_AP']) >= .9
        for report in reports.values())


def optimizer_update(module, optimizer, samples):
    module.train()
    optimizer.zero_grad(set_to_none=True)
    values = []
    for sample in samples:
        losses = module._losses_for_batch(sample)
        if not torch.isfinite(losses['total']):
            raise FloatingPointError('Nonfinite V47 verification loss')
        (losses['total']/len(samples)).backward()
        values.append({name: float(value.detach()) for name, value in losses.items()})
    norm = torch.nn.utils.clip_grad_norm_(module.parameters(), .1, error_if_nonfinite=True)
    optimizer.step()
    return {name: sum(value[name] for value in values)/len(values) for name in values[0]}, float(norm)


def synthetic_capacity_batch():
    generator = torch.Generator().manual_seed(31415)
    batch, people, height, width = 4, 50, 256, 416
    boxes = torch.rand(batch, people, 4, generator=generator)
    boxes[..., :2] = boxes[..., :2]*torch.tensor([width-80., height-80.])+10
    boxes[..., 2:] = boxes[..., 2:]*30+30
    points = boxes[..., None, :2]+boxes[..., None, 2:]/2+torch.randn(batch, people, 14, 2, generator=generator)*8
    labeled = torch.ones(batch, people, 14, dtype=torch.bool)
    labeled[0, :, 3] = False
    sample = dict(image=torch.randn(batch, 3, height, width, generator=generator),
        padding_mask=torch.zeros(batch, height, width, dtype=torch.bool),
        gt_keypoints=2*points/torch.tensor([width, height])-1, gt_boxes_pixel=boxes,
        gt_labeled=labeled, gt_box_labeled=torch.ones(batch, people, dtype=torch.bool),
        person_mask=torch.ones(batch, people, dtype=torch.bool),
        img_hw=torch.tensor([[height, width]]*batch, dtype=torch.float32), image_id=torch.arange(batch))
    return {name: value.cuda() for name, value in sample.items()}


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
    expected = {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}
    restored = PoseFlowV47Pipeline.load_from_checkpoint(checkpoint_path, map_location='cpu', weights_only=False).cuda()
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
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = restored.model.sample_keypoints(sample['image'], padding_mask=sample['padding_mask'],
        image_wh=sample['img_hw'].flip(-1), generator=[torch.Generator().manual_seed(i) for i in range(4)])
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter()-started
    assert torch.isfinite(result['pred_coordinates']).all()
    report = dict(passed=True, precision='FP32', rows=64, people=50, batch=4, accumulation=2,
        image_hw=[256, 416], optimizer_update_seconds=elapsed, train_images_per_second=8/elapsed,
        peak_training_allocated_bytes=peak, gradient_norm=norm, codec_gradient_norms=codec_gradients,
        inference_images_per_second=4/inference_seconds, inference_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        next_update_loss=second['total'], resumed_loss=resumed['total'], max_restored_state_delta=maximum,
        note='Shared-GPU measurements; inference includes two resident verification models and trace-free final outputs')
    (output/'preflight.json').write_text(json.dumps(report, indent=2)+'\n')
    print('V47_PREFLIGHT', json.dumps(report), flush=True)
    del restored, module, optimizer, configuration, restore_configuration, restore_optimizer, sample, checkpoint, saved
    gc.collect()
    torch.cuda.empty_cache()
    return report


def overfit_image_ids(dataset, annotations):
    box_counts = Counter(person['image_id'] for person in annotations if not person.get('iscrowd', 0))
    selected = {}
    for sample in sorted(dataset.samples, key=lambda value: value['image_id']):
        boxes = sample['gt_boxes_pixel']
        valid = (sample['visibility'] > 0).any(-1) & torch.isfinite(boxes).all(-1) & (boxes[:, 2:] > 0).all(-1)
        count = int(valid.sum())
        if count in (1, 2, 3, 4) and count not in selected and count == box_counts[sample['image_id']]:
            selected[count] = int(sample['image_id'])
        if len(selected) == 4:
            return [selected[count] for count in (1, 2, 3, 4)]
    raise ValueError('Cannot find four TRAIN images with valid counts1/2/3/4')


@torch.no_grad()
def verify_subset_oracle(model, data, dataset, ids):
    records = []
    for batch in v21_test_batches(data, dataset, ids, 'cuda'):
        geometry, locations = geometry_from_batch(batch)
        sources = ordered_coordinates(geometry, locations, batch_size=1, capacity=model.capacity)
        sizes = batch['img_hw'].flip(-1).to(batch['image'])
        output = model._geometry_output(sources['coordinates'], sources['counts'],
            sizes.new_ones(1, model.capacity), sizes)
        valid = output['valid_mask'][0]
        processed = postprocess_crowdpose({name: output[name][:, valid] for name in
            ('pred_logits', 'pred_boxes', 'pred_keypoints')}, batch['original_img_hw'].to(sizes),
            nms_target_sizes=batch['img_hw'].to(sizes))
        records.extend(crowdpose_result_records(int(batch['image_id'][0]), processed))
    return evaluate_records(records, data.train_ann_path, ids)


def fixed_training_batch(data, dataset, ids):
    indices = {int(sample['image_id']): index for index, sample in enumerate(dataset.samples)}
    wrapped = SetPoseDataset(dataset)
    batch = set_pose_collate([wrapped[indices[image_id]] for image_id in ids])
    batch = {key: [image.cuda() for image in value] if key == 'image'
             else value.cuda() if torch.is_tensor(value) else value for key, value in batch.items()}
    return data.on_after_batch_transfer(batch, 0)


def overfit(module, data, output, *, max_updates):
    dataset = PoseDataset.from_path(data.train_ann_path, img_root=data.img_dir, num_keypoints=14)
    annotations = json.loads(Path(data.train_ann_path).read_text())['annotations']
    ids = overfit_image_ids(dataset, annotations)
    data.trainer = SimpleNamespace(training=False)
    oracle = verify_subset_oracle(module.model, data, dataset, ids)
    (output/'subset_oracle.json').write_text(json.dumps(dict(image_ids=ids, metrics=oracle), indent=2)+'\n')
    if min(oracle['pose_AP'], oracle['bbox_AP']) < .99:
        raise AssertionError(f'Overfit subset cannot pass native AP gate even with oracle outputs: {oracle}')
    sample = fixed_training_batch(data, dataset, ids)
    _, locations = geometry_from_batch(sample)
    counts = torch.bincount(locations[:, 0], minlength=4).tolist()
    if counts != [1, 2, 3, 4]:
        raise AssertionError(f'Overfit selection differs from transformed validity: {counts}')
    optimizer = module.configure_optimizers()['optimizer']
    writer = SummaryWriter(str(output/'tb_logs'))
    (output/'subset.json').write_text(json.dumps(dict(split='train', image_ids=ids, counts=counts,
        augmentation=False, max_updates=max_updates, note='Memorization probe; not validation/generalization'), indent=2)+'\n')
    history = []
    try:
        for update in range(max_updates+1):
            if update:
                losses, gradient_norm = optimizer_update(module, optimizer, [sample, sample])
                if update % 10 == 0:
                    for name, value in losses.items():
                        writer.add_scalar(f'overfit/train/{name}', value, update)
                    writer.add_scalar('overfit/train/gradient_norm', gradient_norm, update)
                    writer.flush()
                    print('V47_OVERFIT_TRAIN', update, json.dumps(losses), flush=True)
            if update % 100:
                continue
            reports = {}
            for seed in (42, 43):
                report = evaluate_v47_batches(module.model,
                    v21_test_batches(data, dataset, ids, 'cuda'), output/f'update{update:04d}_seed{seed}',
                    data.train_ann_path, ids, seed=seed, split='train_overfit')
                reports[str(seed)] = report
                for name, value in tensorboard_metrics(report).items():
                    writer.add_scalar(f'overfit/seed{seed}/{name}', value, update)
            passed = learnability_passed(reports)
            summary = dict(update=update, passed=passed, reports={seed: dict(metrics=report['metrics'],
                exact_count_images=report['exact_count_images'], count_mae=report['count_mae']) for seed, report in reports.items()})
            history.append(summary)
            result = dict(passed=passed, updates=update, image_ids=ids, history=history)
            (output/'overfit.json').write_text(json.dumps(result, indent=2)+'\n')
            writer.flush()
            print('V47_OVERFIT_EVAL', json.dumps(summary), flush=True)
            if passed:
                torch.save(module.state_dict(), output/'overfit_passed_weights.pt')
                return result
    finally:
        writer.close()
    torch.save(module.state_dict(), output/'overfit_failed_weights.pt')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-overfit-updates', type=int, default=2000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with initialize_config_dir(version_base='1.3', config_dir=str(Path('conf').resolve())):
        config = compose(config_name='benchmark_poseflow_v47')
    sources = source_hashes()
    data_config = DataConfig.from_hydra_config(config.data)
    if data_config.test_ann != data_config.val_ann:
        raise ValueError('Verification must not open TEST')
    data = data_config.build()
    pl.seed_everything(42, workers=True)
    specification = PoseFlowV47PipelineConfig.from_hydra_config(config.pipeline)
    module = specification.build(max_epochs=80)
    initial_hash = model_digest(module.model)
    initial_state = deepcopy(module.state_dict())
    initialization = deepcopy(module.model.initialization_report)
    (args.output/'config.yaml').write_text(OmegaConf.to_yaml(config))
    try:
        preflight = native_preflight(module.cuda(), args.output)
        module.cpu()
        module.load_state_dict(initial_state, strict=True)
        module.cuda()
        gate = overfit(module, data, args.output, max_updates=args.max_overfit_updates)
        val = json.loads(Path(data.val_ann_path).read_text())
        ids = sorted(random.Random(42).sample(sorted(image['id'] for image in val['images']), 32))
        if source_hashes() != sources:
            raise ValueError('Sources changed during verification')
        result = dict(passed=preflight['passed'] and gate['passed'], native_preflight=preflight,
            overfit=gate, sources=sources, pipeline_config=OmegaConf.to_container(config.pipeline, resolve=True),
            data_config=OmegaConf.to_container(config.data, resolve=True), initial_model_sha256=initial_hash,
            initialization=initialization, fixed_val_image_ids=ids, pid=os.getpid(),
            inputs={str(path): digest(path) for path in (data.train_ann_path, data.val_ann_path,
                                                        specification.pretrained_checkpoint)})
        (args.output/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
        print('V47_VERIFICATION', json.dumps(dict(passed=result['passed'], updates=gate['updates'])), flush=True)
    except BaseException as error:
        failure = dict(passed=False, error=repr(error), resource_failure=isinstance(error, torch.cuda.OutOfMemoryError),
            policy='Recipe unchanged: FP32/64rows/50capacity/batch4/accum2; existing jobs preserved')
        (args.output/'failure.json').write_text(json.dumps(failure, indent=2)+'\n')
        raise


if __name__ == '__main__':
    main()
