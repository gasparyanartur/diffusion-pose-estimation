import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import random

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytorch_lightning as pl
import torch

from scripts.training.train_poseflow_v53 import digest, model_digest, source_hashes
from scripts.training.verify_full_depth_birth import capture_layers, record_predictions, reference_decode
from scripts.training.verify_poseflow_v53 import initialization_coverage, native_preflight, training_batches
from scripts.training.verify_poseflow_v47 import synthetic_capacity_batch
from scripts.training.verify_poseflow_v51_learning import optimizer_update
from src.data import DataConfig
from src.data.datasets import PoseDataset
from src.models.pipelines.pose_estimators.poseflow.calibration_v53 import calibrate
from src.models.pipelines.pose_estimators.poseflow.full_depth_birth import FullDepthBirthDecoder
from src.models.pipelines.pose_estimators.poseflow.quick_eval_v13 import evaluate_records
from src.models.pipelines.pose_estimators.poseflow.quick_eval_v21 import v21_test_batches
from src.models.pipelines.pose_estimators.poseflow.quick_eval_v53 import run_v53_evaluation
from src.models.pipelines.pose_estimators.poseflow.training_v56 import PoseFlowV56Pipeline, PoseFlowV56PipelineConfig


def conditional_parity(module, data, ids, output):
    from types import SimpleNamespace
    reference = FullDepthBirthDecoder(native=module.config.native_config().build(max_epochs=80).model).cuda().eval()
    dataset = PoseDataset.from_path(data.val_ann_path, img_root=data.img_dir, num_keypoints=14)
    previous = getattr(data, 'trainer', None)
    data.trainer = SimpleNamespace(training=False)
    records = []
    module.eval()
    try:
        with torch.no_grad():
            for sample in v21_test_batches(data, dataset, ids, 'cuda'):
                context = module.model.prepare_context(sample['image'], sample.get('padding_mask'))
                native = reference.prepare_context(sample['image'], sample.get('padding_mask'))
                for name in ('memory', 'features', 'boxes', 'indices', 'pred_logits'):
                    torch.testing.assert_close(context[name], native[name], atol=0, rtol=0)
                types = torch.full((1, 50, 1), 2, device='cuda', dtype=torch.long)
                identities = torch.arange(50, device='cuda')[None, :, None]
                with capture_layers(reference.pose_decoder) as expected_trace:
                    expected = reference_decode(reference.pose_decoder, native)
                with capture_layers(module.model.pose_decoder) as actual_trace:
                    actual = module.model.forward_rows(context, context['features'].new_zeros(1, 50, 1, 32),
                        types, types == 0, identities)['pred_coordinates']
                for left, right in zip(actual_trace, expected_trace):
                    for name in ('input', 'references', 'output'):
                        torch.testing.assert_close(left[name], right[name], atol=0, rtol=0)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                records.extend(record_predictions(actual[-1], native['pred_logits'], sample))
                print('V56_CONDITIONAL_PARITY',int(sample['image_id'][0]),flush=True)
    finally:
        data.trainer = previous
    metrics = evaluate_records(records, data.val_ann_path, ids)
    if abs(metrics['pose_AP']-.5415092168543765) > 1e-7 or abs(metrics['bbox_AP']-.3910077906526713) > 1e-7:
        raise AssertionError(f'V56 original independent full-depth AP changed: {metrics}')
    result = dict(passed=True, images=len(ids), layer_maxabs=0., metrics=metrics,
        scope='Supplied all50 proposal IDs; not learned sequential selection/EOS')
    (output/'conditional_parity.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def gradient_routes(module, sample):
    module.train()
    module.zero_grad(set_to_none=True)
    losses = module._losses_for_batch(sample)
    probes = dict(prefix=module.model.coordinate_input[0].weight,
        context=module.model.proposal_context.attention.out_proj.weight,
        pose=module.model.pose_decoder.pose_heads[-1].layers[-1].weight,
        image=module.model.frontend.native.input_proj[0][0].weight)
    results = {}
    for task in ('pose_birth', 'box_birth', 'selection', 'eos'):
        gradients = torch.autograd.grad(losses[task], tuple(probes.values()), retain_graph=True, allow_unused=True)
        norms = {name:0. if gradient is None else float(gradient.norm()) for name, gradient in zip(probes, gradients)}
        if norms['prefix'] <= 0 or norms['context'] <= 0:
            raise AssertionError(f'V56 inactive prefix/context route: {task}: {norms}')
        if task in ('selection', 'eos') and (norms['image'] != 0 or norms['pose'] != 0):
            raise AssertionError(f'V56 classification crossed image/pose boundary: {task}: {norms}')
        results[task] = dict(loss=float(losses[task].detach()), norms=norms)
    module.zero_grad(set_to_none=True)
    return dict(passed=True, tasks=results, scope='Post-warmup native TRAIN sample; no new optimization policy')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--residual-bank', type=Path, required=True)
    parser.add_argument('--fixed-subset', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with initialize_config_dir(version_base='1.3', config_dir=str(Path('conf').resolve())):
        config = compose(config_name='benchmark_poseflow_v56', overrides=[f'residual_bank_file={args.residual_bank.resolve()}'])
    sources = source_hashes()
    data_config = DataConfig.from_hydra_config(config.data)
    if data_config.test_ann != data_config.val_ann:
        raise ValueError('V56 verification must not open TEST')
    data = data_config.build()
    pl.seed_everything(42, workers=True)
    specification = PoseFlowV56PipelineConfig.from_hydra_config(config.pipeline)
    module = specification.build(max_epochs=80)
    initial_hash = model_digest(module.model)
    bank = torch.load(args.residual_bank, map_location='cpu', weights_only=False)
    provenance = bank['provenance']
    train_ids = {int(image['id']) for image in json.loads(Path(data.train_ann_path).read_text())['images']}
    val_ids = {int(image['id']) for image in json.loads(Path(data.val_ann_path).read_text())['images']}
    if (not set(provenance['image_ids']) <= train_ids or set(provenance['image_ids']) & val_ids
            or len(set(provenance['image_ids'])) != 512
            or digest(data.train_ann_path) != provenance['train_annotations_sha256']
            or digest(provenance['source_checkpoint']) != provenance['source_checkpoint_sha256']):
        raise ValueError('V56 TRAIN residual-bank provenance mismatch')
    module.set_residual_bank(bank['residuals'], bank['joint_mask'], provenance)
    ids = json.loads(args.fixed_subset.read_text())['image_ids']
    if ids != sorted(random.Random(42).sample(sorted(val_ids), 32)):
        raise ValueError('V56 fixed32VAL differs from previous comparisons')
    try:
        module.cuda()
        parity = conditional_parity(module, data, ids, args.output)
        assert model_digest(module.model) == initial_hash
        calibration = calibrate(module, training_batches(data))
        calibration.update(model_sha256_before=initial_hash, model_sha256_after=model_digest(module.model),
            train_annotations=str(data.train_ann_path), train_annotations_sha256=digest(data.train_ann_path))
        initial_state = {name:value.detach().cpu().clone() for name,value in module.state_dict().items()}
        initialization = deepcopy(module.model.initialization_report)
        initialization['measured_retained_coverage'] = initialization_coverage(module, specification)
        (args.output/'calibration.json').write_text(json.dumps(calibration, indent=2)+'\n')
        (args.output/'config.yaml').write_text(OmegaConf.to_yaml(config))
        sample = synthetic_capacity_batch()
        for name in ('gt_keypoints', 'gt_boxes_pixel', 'gt_labeled', 'gt_box_labeled', 'person_mask'):
            sample[name] = sample[name][:, :49]
        warmup_optimizer = module.configure_optimizers()['optimizer']
        optimizer_update(module, warmup_optimizer, [sample, sample])
        del warmup_optimizer, sample
        preflight = native_preflight(module, args.output, pipeline_class=PoseFlowV56Pipeline)
        batches = training_batches(data)
        sample = next(batches)
        batches.close()
        routes = gradient_routes(module, sample)
        (args.output/'gradient_routes.json').write_text(json.dumps(routes, indent=2)+'\n')
        del sample
        module.load_state_dict(initial_state, strict=True)
        assert model_digest(module.model) == initial_hash
        module.eval()
        smoke = run_v53_evaluation(module.model, data, args.output/'evaluator_smoke', epoch=-1, count=2)
        if source_hashes() != sources:
            raise ValueError('Sources changed during V56 verification')
        result = dict(passed=True, native_preflight=preflight, gradient_routes=routes, conditional_parity=parity,
            evaluator_smoke=dict(images=smoke['images'], metrics=smoke['metrics'], architecture=smoke['architecture']),
            sources=sources, pipeline_config=OmegaConf.to_container(config.pipeline, resolve=True),
            data_config=OmegaConf.to_container(config.data, resolve=True), initial_model_sha256=initial_hash,
            initialization=initialization, calibration=calibration, contract=module.checkpoint_contract(),
            fixed_val_image_ids=ids, pid=os.getpid(), prefix_bank=module.prefix_bank_provenance,
            residual_bank_file=str(args.residual_bank.resolve()), residual_bank_file_sha256=digest(args.residual_bank),
            inputs={str(path):digest(path) for path in (data.train_ann_path, data.val_ann_path,
                specification.pretrained_checkpoint, args.residual_bank.resolve())},
            warmup='One synthetic update opens zero projection before gradient parity checks; all initial weights restored for launch')
        (args.output/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
        print('V56_VERIFICATION',json.dumps(dict(passed=True, initial_model_sha256=initial_hash)),flush=True)
    except BaseException as error:
        failure = dict(passed=False, error=repr(error), resource_failure=isinstance(error, torch.cuda.OutOfMemoryError),
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
            policy='FP32/50rows/49capacity/batch4/accum2/full6encoder_2box_4pose unchanged; existing jobs preserved')
        (args.output/'failure.json').write_text(json.dumps(failure, indent=2)+'\n')
        raise


if __name__ == '__main__':
    main()
