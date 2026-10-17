"""Fixed test-subset diagnostics for GT-conditioned and Gaussian-start sampling."""

import builtins
import contextlib
import io
import json
import math
import random
from pathlib import Path

import numpy as np
import torch

from .latent_v12 import build_v12_targets
from ..diffusionregpose.evaluation import crowdpose_result_records
from src.eval.metrics import CROWDPOSE_SIGMAS
from src.eval.poseflow_diagnostics import stage_pose_oks
from src.eval.poseflow_training_diagnostics import _image_targets


def select_test_images(candidates, path, *, count, seed, annotations):
    path = Path(path)
    candidates = sorted(set(candidates))
    contract = {'seed':seed, 'requested_images':count, 'annotations':str(annotations)}
    if path.exists():
        saved = json.loads(path.read_text())
        if any(saved.get(key) != value for key,value in contract.items()):
            raise ValueError('Quick evaluation subset contract changed on resume')
        if not set(saved['image_ids']).issubset(candidates):
            raise ValueError('Quick evaluation test images are missing')
        return saved['image_ids']
    selected = sorted(random.Random(seed).sample(candidates, min(count,len(candidates))))
    if not selected:
        raise ValueError('Quick evaluation requires a nonempty test set')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(contract | {'image_ids':selected}, indent=2))
    return selected


def noise_fraction_time(fraction):
    if not 0 < fraction <= 1:
        raise ValueError('Noise variance fraction must be in (0,1]')
    if fraction == 1:
        return 1.
    normalization = math.cos(.008/1.008*math.pi/2)**2
    return math.acos(math.sqrt((1-fraction)*normalization))*2/math.pi*1.008-.008


@contextlib.contextmanager
def evaluation_context(model):
    modes = [(module,module.training) for module in model.modules()]
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    device = next(model.parameters()).device
    devices = [device.index] if device.type == 'cuda' else []
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            model.eval()
            yield
    finally:
        for module,training in modes:
            module.training = training
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)


def evaluate_records(records, annotations, image_ids):
    from crowdposetools.coco import COCO
    from crowdposetools.cocoeval import COCOeval

    had_unicode, had_float = hasattr(builtins,'unicode'), hasattr(np,'float')
    try:
        if not had_unicode:
            builtins.unicode = str
        if not had_float:
            np.float = float
        with contextlib.redirect_stdout(io.StringIO()):
            ground_truth = COCO(str(annotations))
            selected = set(image_ids)
            ground_truth.dataset['images'] = [image for image in ground_truth.dataset['images']
                                               if image['id'] in selected]
            ground_truth.dataset['annotations'] = [person for person in ground_truth.dataset['annotations']
                                                    if person['image_id'] in selected]
            ground_truth.createIndex()
            if records:
                detections = COCO.loadRes(ground_truth,records)
            else:
                detections = COCO()
                detections.dataset = {**ground_truth.dataset,'annotations':[]}
                detections.createIndex()
            metrics = {}
            for kind in ('keypoints','bbox'):
                evaluator = COCOeval(ground_truth,detections,kind)
                evaluator.params.imgIds = image_ids
                evaluator.evaluate()
                evaluator.accumulate()
                evaluator.summarize()
                if kind == 'keypoints':
                    metrics.update({name:float(evaluator.stats[index]) for name,index in
                                    [('pose_AP',0),('pose_AP50',1),('pose_AP75',2),('pose_AR',5)]})
                else:
                    metrics['bbox_AP'] = float(evaluator.stats[0])
        return metrics
    finally:
        if not had_unicode:
            del builtins.unicode
        if not had_float:
            del np.float


def raw_pose_quality(output, batch, sigmas):
    target = _image_targets(batch,0)
    oks = stage_pose_oks(output,target,sigmas)
    if not oks.numel():
        return 0., 0., oks.shape[1]
    best = oks.amax(0)
    return float(best.sum()), float((best >= .75).sum()), best.numel()


def evaluate_quick_grid(model, batches, config, directory, annotations, image_ids,
                        *, metric_evaluator=evaluate_records, hypothesis_modes=None):
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    conditions = {}
    for fraction in config.quick_eval_noise_fractions:
        for steps in config.quick_eval_steps:
            name = f'noise_var_{fraction*100:g}/steps_{steps}'
            start_time = noise_fraction_time(fraction)
            base = {'noise_variance':fraction,'steps':steps,
                'times':[start_time*(1-index/steps) for index in range(steps)],
                'next_times':[start_time*(1-(index+1)/steps) for index in range(steps)],
                'oks_sum':0.,'covered':0.,'gt_people':0}
            for mode in hypothesis_modes or (None,):
                if mode not in (None, 'persistent', 'reset'):
                    raise ValueError('Unknown hypothesis memory evaluation mode')
                key = name if mode is None else f'{name}/memory_{mode}'
                conditions[key] = {**base, 'records':[]}
                if mode is not None:
                    conditions[key].update(hypothesis_mode=mode, box_quality=[], steps_quality=[])
    seen_ids, overflow = [], 0
    with evaluation_context(model):
        for batch in batches:
            image = batch['image']
            if image.shape[0] != 1:
                raise ValueError('Quick evaluation expects one image per batch')
            image_id = int(batch['image_id'][0])
            seen_ids.append(image_id)
            seed = config.quick_eval_seed+image_id
            targets = build_v12_targets(batch,groups=model.num_proposals,
                generator=torch.Generator(device=image.device).manual_seed(seed))
            overflow += int(targets['overflow'].sum())
            clean = model.codec.encode(targets)
            noise = torch.randn(clean.shape,device=image.device,dtype=image.dtype,
                                generator=torch.Generator(device=image.device).manual_seed(seed))
            memory = model.encode_image(image,batch.get('padding_mask'))
            sigmas = image.new_tensor(CROWDPOSE_SIGMAS[:model.num_body_points])
            for condition in conditions.values():
                fraction = condition['noise_variance']
                initial = noise if fraction == 1 else math.sqrt(1-fraction)*clean+math.sqrt(fraction)*noise
                sampler_options = {}
                if hypothesis_modes:
                    sampler_options = {'persist_hypotheses': condition['hypothesis_mode'] == 'persistent',
                                       'collect_step_diagnostics': True}
                output = model.sample_keypoints(image,padding_mask=batch.get('padding_mask'),
                    initial_latents=initial,start_time=condition['times'][0],
                    sampling_steps=condition['steps'],encoded_image=memory, **sampler_options)
                original_sizes = batch.get('original_img_hw',batch['img_hw']).to(image)
                results = model.postprocess(output,original_sizes,nms_target_sizes=batch['img_hw'].to(image))
                condition['records'].extend(crowdpose_result_records(image_id,results))
                oks_sum, covered, people = raw_pose_quality(output,batch,sigmas)
                condition['oks_sum'] += oks_sum
                condition['covered'] += covered
                condition['gt_people'] += people
                if hypothesis_modes:
                    from .quick_eval_v14 import box_quality, step_quality
                    condition['box_quality'].append(box_quality(output, batch))
                    condition['steps_quality'].append({'image_id': image_id,
                        'steps': step_quality(output, batch, sigmas)})
    if seen_ids != image_ids:
        raise ValueError('Quick evaluation did not cover its exact fixed image manifest')
    for name,condition in conditions.items():
        records = condition.pop('records')
        record_path = directory/(name.replace('/','_')+'_predictions.json')
        record_path.write_text(json.dumps(records,allow_nan=False))
        metrics = metric_evaluator(records,annotations,image_ids)
        count = condition['gt_people']
        metrics.update(oracle_oks=condition.pop('oks_sum')/max(count,1),
                       coverage75=condition.pop('covered')/max(count,1))
        if hypothesis_modes:
            from .quick_eval_v14 import aggregate_box_quality
            metrics.update(aggregate_box_quality(condition.pop('box_quality')))
        condition['metrics'] = metrics
        condition['predictions'] = str(record_path)
    return {'image_ids':image_ids,'seed':config.quick_eval_seed,'per_image_seed':'seed + image_id',
            'noise_definition':'variance: sqrt(1-p)*clean + sqrt(p)*Gaussian',
            'annotations':str(annotations),'gt_people':next(iter(conditions.values()))['gt_people'],
            'overflow_people':overflow,'sampler':'latent DDIM eta0','conditions':conditions,
            'conditioning':'25/50% use GT-derived clean states;100% is Gaussian-start generation',
            'evaluator':'crowdposetools COCOeval keypoints/bbox; corrected native pose NMS'}
