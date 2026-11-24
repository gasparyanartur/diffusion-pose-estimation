import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

from omegaconf import OmegaConf
import torch

from src.data import DataConfig
from src.data.datasets import PoseDataset
from src.models.pipelines.pose_estimators.diffusionregpose.evaluation import (
    crowdpose_result_records, postprocess_crowdpose,
)
from src.models.pipelines.pose_estimators.diffusionregpose.training import DiffusionRegPosePipelineConfig
from src.models.pipelines.pose_estimators.diffusionregpose.transformer import _load_native_msda
from src.models.pipelines.pose_estimators.poseflow.expanded_pose_v53 import ExpandedPoseDecoder
from src.models.pipelines.pose_estimators.poseflow.full_depth_birth import FullDepthBirthDecoder
from src.models.pipelines.pose_estimators.poseflow.model_v50 import ProposalFrontend
from src.models.pipelines.pose_estimators.poseflow.quick_eval_v13 import evaluate_records
from src.models.pipelines.pose_estimators.poseflow.quick_eval_v21 import v21_test_batches


PRETRAINED_SHA256 = '0f50dae126a73ad2a15c41906c9ba6d8efa3421ef6cc2fbdc878238bd26b8f93'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_digest(model):
    result = hashlib.sha256()
    for name, value in model.state_dict().items():
        result.update(name.encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


@contextmanager
def capture_layers(decoder):
    original = decoder._layer
    trace = []

    def capture(*args):
        state, references = args[1:3]
        output = original(*args)
        trace.append(dict(input=state.detach().clone(), references=references.detach().clone(),
                          output=output.detach().clone()))
        return output

    decoder._layer = capture
    try:
        yield trace
    finally:
        decoder._layer = original


def reference_decode(decoder, context):
    features, boxes = context['features'], context['boxes']
    batch, rows, width = features.shape
    inputs = context['inputs']
    return decoder(features[:, :, None], boxes[:, :, None], features.new_zeros(batch, rows, 0, width),
        boxes.new_zeros(batch, rows, 0, 4), torch.zeros(batch, rows, 0, dtype=torch.bool, device=features.device),
        decoder.project_memory(context['memory'], inputs.padding_mask), inputs.spatial_shapes,
        inputs.level_start_index, inputs.valid_ratios)


def record_predictions(coordinates, logits, batch):
    person_logit = torch.logit(logits[..., 1].sigmoid().clamp(1e-8, 1-1e-7))
    logits = torch.stack((torch.full_like(person_logit, -30.), person_logit), -1)
    output = dict(pred_boxes=coordinates[..., 28:], pred_logits=logits,
        pred_keypoints=torch.cat((coordinates[..., :28], torch.ones_like(coordinates[..., :14])), -1))
    results = postprocess_crowdpose(output, batch['original_img_hw'].to(coordinates), num_body_points=14,
        nms_target_sizes=batch['img_hw'].to(coordinates))
    return crowdpose_result_records(int(batch['image_id'][0]), results)


def capacity_measurements(model):
    images = torch.randn(4, 3, 256, 416, device='cuda')
    identities = torch.arange(50, device='cuda').expand(4, -1)
    prefix = torch.randn(4, 50, 49, 256, device='cuda')
    ranks = torch.arange(50, device='cuda')
    prefix_valid = torch.arange(49, device='cuda')[None, None] < ranks[None, :, None]
    prefix_valid = prefix_valid.expand(4, -1, -1)
    target = torch.rand(4, 50, 32, device='cuda')
    coordinate_mask = torch.ones_like(target, dtype=torch.bool)
    coordinate_mask[:, :, :28:7] = False
    model.train()
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(2):
        context = model.prepare_context(images)
        output = model.decode_proposals(context, identities, prefix=prefix,
            prefix_valid=prefix_valid, collect_trace=True)
        losses = [((item['pred_coordinates']-target).square()*coordinate_mask).sum()/coordinate_mask.sum()
                  for item in output['trace']]
        loss = torch.stack(losses).mean()/2
        loss.backward()
        del output, losses, loss, context
    torch.cuda.synchronize()
    seconds = time.perf_counter()-started
    peak = torch.cuda.max_memory_allocated()
    gradients = dict(model.named_parameters())
    required = ['proposal_context.attention.out_proj.weight', 'pose_decoder.expansion_head.layers.0.weight']
    required += [f'pose_decoder.layers.{index}.cross_attn.sampling_offsets.weight' for index in range(4)]
    norms = {}
    for name in required:
        gradient = gradients[name].grad
        if gradient is None or not torch.isfinite(gradient).all() or gradient.norm() == 0:
            raise AssertionError(f'Missing/nonfinite gradient: {name}')
        norms[name] = float(gradient.norm())
    if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
           for parameter in model.parameters()):
        raise AssertionError('Nonfinite full-capacity gradient')
    model.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad():
        torch.cuda.synchronize()
        started = time.perf_counter()
        context = model.prepare_context(images)
        torch.cuda.synchronize()
        frontend_seconds = time.perf_counter()-started
        started = time.perf_counter()
        batched = model.decode_proposals(context, identities)['pred_coordinates']
        torch.cuda.synchronize()
        batched_seconds = time.perf_counter()-started
        started = time.perf_counter()
        sequential = [model.decode_proposals(context, identities[:, row:row+1],
            prefix=prefix[:, row:row+1, :row], prefix_valid=prefix_valid[:, row:row+1, :row])['pred_coordinates']
                      for row in range(50)]
        torch.cuda.synchronize()
        sequential_seconds = time.perf_counter()-started
        torch.testing.assert_close(torch.cat(sequential, 1), batched, atol=2e-5, rtol=2e-5)
    return dict(batch_size=4, accumulation=2, proposals=50, precision='FP32', image_hw=[256, 416],
        backward_seconds=seconds, backward_images_per_second=8/seconds, peak_cuda_allocated_bytes=peak,
        cached_frontend_seconds=frontend_seconds, batched50_decode_seconds=batched_seconds,
        sequential50_decode_seconds=sequential_seconds,
        sequential50_images_per_second=4/(frontend_seconds+sequential_seconds),
        gradient_norms=norms, no_optimizer_updates=True,
        scope='Synthetic masked-coordinate gradient/resource diagnostic; no trainer recipe or optimizer verification')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pretrained', type=Path,
        default=Path('external/DiffusionRegPose/pretrain_models/edpose_r50_crowdpose.pth'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source_paths = [Path(__file__), Path('tests/test_full_depth_birth.py')]
    source_paths += [Path('src/models/pipelines/pose_estimators')/name for name in (
        'poseflow/full_depth_birth.py', 'poseflow/expanded_pose_v53.py', 'poseflow/human_context.py',
        'poseflow/model_v50.py', 'poseflow/proposal_context_v55.py', 'diffusionregpose/model.py',
        'diffusionregpose/transformer.py', 'diffusionregpose/training.py', 'diffusionregpose/evaluation.py')]
    sources = {str(path):digest(path) for path in source_paths}
    if not torch.cuda.is_available() or _load_native_msda() is None:
        raise RuntimeError('Native CUDA is required; fallback is forbidden')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if digest(args.pretrained) != PRETRAINED_SHA256:
        raise ValueError('Original ED-Pose asset hash mismatch')
    manifest = json.loads((args.reference_run/'seed42/manifest.json').read_text())
    audit_path = args.reference_run/'diagnostics/pose_bottleneck_audit/frontend_transfer_results.json'
    audit = json.loads(audit_path.read_text())
    ids = json.loads((args.reference_run/'seed42/quick_test/subset.json').read_text())['image_ids']
    config = DiffusionRegPosePipelineConfig(**dict(manifest['config']['pipeline']['native'],
        num_encoder_layers=6, num_decoder_layers=6, num_box_decoder_layers=2,
        backbone_pretrained=False, pretrained_checkpoint=str(args.pretrained),
        pretrained_minimum_critical_coverage=.99))
    native = config.build(max_epochs=80).model
    reference_decoder = ExpandedPoseDecoder(native).cuda().eval()
    reference_frontend = ProposalFrontend(deepcopy(native)).cuda().eval()
    model = FullDepthBirthDecoder(native=native).cuda().eval()
    assert model.pose_decoder.expansion_head is model.pose_decoder.pose_heads[0]
    initial_hash = model_digest(model)
    data = DataConfig.from_hydra_config(OmegaConf.create(manifest['config']['data'])).build()
    data.trainer = SimpleNamespace(training=False)
    if digest(data.val_ann_path) != manifest['inputs'][str(args.reference_run/'workspace/data/crowdpose/annotations/crowdpose_val.json')]:
        raise ValueError('VAL annotation hash mismatch')
    dataset = PoseDataset.from_path(data.val_ann_path, img_root=data.img_dir, num_keypoints=14)
    predictions, reference_predictions = [], []
    layer_error = 0.
    with torch.no_grad():
        for batch in v21_test_batches(data, dataset, ids, 'cuda'):
            context = model.prepare_context(batch['image'], batch.get('padding_mask'))
            original = reference_frontend.prepare_context(batch['image'], batch.get('padding_mask'))
            for name in ('memory', 'features', 'boxes', 'indices', 'pred_logits'):
                torch.testing.assert_close(context[name], original[name], atol=0, rtol=0)
            identities = torch.arange(50, device='cuda')[None]
            with capture_layers(reference_decoder) as expected_trace:
                expected = reference_decode(reference_decoder, original)
            with capture_layers(model.pose_decoder) as actual_trace:
                output = model.decode_proposals(context, identities, collect_trace=True)
            for actual, wanted in zip(actual_trace, expected_trace):
                for name in ('input', 'references', 'output'):
                    torch.testing.assert_close(actual[name], wanted[name], atol=0, rtol=0)
            actual = torch.stack([item['pred_coordinates'] for item in output['trace']])
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            layer_error = max(layer_error, float((actual-expected).abs().max()))
            predictions.extend(record_predictions(output['pred_coordinates'], output['pred_logits'], batch))
            reference_predictions.extend(record_predictions(expected[-1], original['pred_logits'], batch))
            print('PARITY',int(batch['image_id'][0]),flush=True)
    del context, original, expected_trace, actual_trace, output, actual, expected
    metrics = evaluate_records(predictions, data.val_ann_path, ids)
    reference_metrics = evaluate_records(reference_predictions, data.val_ann_path, ids)
    expected_metrics = audit['results']['original_full_frontend_four_pose_layers_independent_layer4']['metrics']
    for name, value in metrics.items():
        if abs(value-reference_metrics[name]) > 1e-10 or abs(value-expected_metrics[name]) > 1e-7:
            raise AssertionError(f'Frozen reference AP mismatch: {name}: {value} vs {expected_metrics[name]}')
    del reference_decoder, reference_frontend
    torch.cuda.empty_cache()
    try:
        resource = capacity_measurements(model)
    except torch.cuda.OutOfMemoryError as error:
        failure = dict(passed=False, resource_failure=str(error), batch_size=4, accumulation=2,
            proposals=50, precision='FP32', peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
        (args.output/'resource_failure.json').write_text(json.dumps(failure, indent=2)+'\n')
        raise
    assert model_digest(model) == initial_hash
    assert sources == {str(path):digest(path) for path in source_paths}
    result = dict(passed=True, images=len(ids), image_ids=ids, split='val', resize=[256, 427],
        metrics=metrics, reference_metrics=reference_metrics, layer_maxabs=layer_error,
        model_sha256=initial_hash, unchanged=True, pretrained_sha256=PRETRAINED_SHA256,
        native_config=config.model_dump(), resources=resource, reference_report=str(audit_path),
        annotations_sha256=digest(data.val_ann_path),
        sources=sources,
        scope='Frozen transfer preservation/all50 supplied proposals; not learned sequential selection/EOS or certified held-out AP')
    (args.output/'predictions.json').write_text(json.dumps(predictions, allow_nan=False)+'\n')
    (args.output/'verification.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2),flush=True)


if __name__ == '__main__':
    main()
