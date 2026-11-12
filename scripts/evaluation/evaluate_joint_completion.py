#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np

from src.eval.joint_completion import (
    build_prediction_record,
    joint_errors,
    select_annotations,
    summarize_predictions,
    unflip_keypoints,
)
from src.eval.metrics import CROWDPOSE_SIGMAS
from src.data.skeleton import CROWDPOSE_JOINT_NAMES, CROWDPOSE_EDGE_LIST


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / ('external/mmpose/configs/body_2d_keypoint/rtmpose/crowdpose/'
                         'rtmpose-m_8xb64-210e_crowdpose-256x192.py')


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')


def parse_args():
    parser = argparse.ArgumentParser(description='GT-box-assisted, frozen CrowdPose joint diagnostics.')
    parser.add_argument('--annotations', type=Path,
                        default=ROOT / 'data/crowdpose/annotations/crowdpose_val.json')
    parser.add_argument('--images', type=Path, default=ROOT / 'data/crowdpose/images')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'checkpoints/rtmpose-m_crowdpose.pth')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit-images', type=int, default=256, help='0 selects all images.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--reuse-predictions', type=Path)
    parser.add_argument('--bootstrap-draws', type=int, default=500)
    parser.add_argument('--review-people', type=int, default=48)
    return parser.parse_args()


def predict_people(model, image, boxes, batch_size):
    from mmpose.apis import inference_topdown

    coordinates, scores = [], []
    for start in range(0, len(boxes), batch_size):
        batch_boxes = np.asarray(boxes[start:start + batch_size], dtype=np.float32)
        samples = inference_topdown(model, image, batch_boxes, bbox_format='xywh')
        if len(samples) != len(batch_boxes):
            raise ValueError('native prediction count does not match input boxes')
        for sample, box in zip(samples, batch_boxes):
            prediction = sample.pred_instances
            np.testing.assert_allclose(prediction.bboxes[0],
                                       [box[0], box[1], box[0] + box[2], box[1] + box[3]],
                                       atol=1e-3)
            if prediction.keypoints.shape != (1, 14, 2):
                raise ValueError('expected native CrowdPose14 head')
            coordinates.append(prediction.keypoints[0])
            scores.append(prediction.keypoint_scores[0])
    return np.asarray(coordinates), np.asarray(scores)


def infer_image(model, image, annotations, batch_size):
    from mmpose.structures.bbox import bbox_xywh2xyxy, bbox_xyxy2xywh, flip_bbox

    boxes = np.array([person['bbox'] for person in annotations], dtype=np.float32)
    if not np.isfinite(boxes).all() or (boxes[:, 2:] <= 0).any():
        raise ValueError('invalid person boxes')
    coordinates, scores = predict_people(model, image, boxes, batch_size)
    height, width = image.shape[:2]
    flipped_boxes = bbox_xyxy2xywh(flip_bbox(bbox_xywh2xyxy(boxes), (width, height),
                                           bbox_format='xyxy'))
    # Primary predictions keep native flip TTA. Consistency uses independent single passes.
    native_flip_test = model.test_cfg.get('flip_test', False)
    try:
        model.test_cfg['flip_test'] = False
        single_xy, _ = predict_people(model, image, boxes, batch_size)
        flipped_xy, _ = predict_people(model, image[:, ::-1].copy(), flipped_boxes, batch_size)
    finally:
        model.test_cfg['flip_test'] = native_flip_test
    restored = unflip_keypoints(flipped_xy, width, model.dataset_meta['flip_indices'])
    return coordinates, scores, single_xy, restored


def render_reviews(records, images, image_root: Path, output: Path, count: int, seed: int):
    from PIL import Image, ImageDraw

    missing = [record for record in records
               if 0 in record['original_annotation']['keypoints'][2::3]]
    rng = np.random.default_rng(seed)
    random_indices = rng.choice(len(missing), min(count, len(missing)), replace=False)
    random_records = [missing[index] for index in random_indices]
    worst = sorted(records, key=lambda record: np.nanmean(joint_errors(
        record['predicted_xy'], record['original_annotation']['keypoints'],
        record['original_annotation']['bbox'], CROWDPOSE_SIGMAS)['normalized_error']), reverse=True)[:12]
    review_rows = []
    joint_names = list(CROWDPOSE_JOINT_NAMES)
    for category, selection in [('random_missing', random_records), ('labeled_failures', worst)]:
        for page_start in range(0, len(selection), 12):
            sheet = Image.new('RGB', (1440, 1440), 'white')
            for cell, record in enumerate(selection[page_start:page_start + 12]):
                person = record['original_annotation']
                image_meta = images[record['image_id']]
                with Image.open(image_root / image_meta['file_name']) as source:
                    source = source.convert('RGB')
                    x, y, width, height = person['bbox']
                    margin = max(width, height) * .25
                    crop_box = (int(x - margin), int(y - margin),
                                int(x + width + margin), int(y + height + margin))
                    crop = source.crop(crop_box)
                crop.thumbnail((350, 390))
                tile = Image.new('RGB', (360, 480), 'white')
                tile.paste(crop, (0, 60))
                draw = ImageDraw.Draw(tile)
                scale_x = crop.width / (crop_box[2] - crop_box[0])
                scale_y = crop.height / (crop_box[3] - crop_box[1])

                def point(xy):
                    return ((xy[0] - crop_box[0]) * scale_x,
                            (xy[1] - crop_box[1]) * scale_y + 60)

                predicted = np.asarray(record['predicted_xy'])
                truth = np.asarray(person['keypoints']).reshape(14, 3)
                draw.text((5, 4), f"image {record['image_id']} / ann {record['annotation_id']}", fill='black')
                draw.text((5, 20), 'green=GT; orange=missing prediction', fill='black')
                draw.text((5, 36), 'blue=prediction at labeled joint; numbers=joint', fill='black')
                for first, second in CROWDPOSE_EDGE_LIST:
                    draw.line([point(predicted[first]), point(predicted[second])], fill='#aaaaaa', width=1)
                draw.rectangle([point([x, y]), point([x + width, y + height])], outline='#990099', width=1)
                for joint in range(14):
                    color = '#ed6500' if truth[joint, 2] == 0 else '#0066dd'
                    px, py = point(predicted[joint])
                    draw.ellipse((px-3, py-3, px+3, py+3), fill=color)
                    if truth[joint, 2] > 0:
                        tx, ty = point(truth[joint, :2])
                        draw.ellipse((tx-3, ty-3, tx+3, ty+3), outline='#00bb33', width=2)
                    else:
                        draw.text((px+4, py-4), str(joint), fill='#b34800')
                        review_rows.append(dict(category=category, image_id=record['image_id'],
                            annotation_id=record['annotation_id'], joint_index=joint,
                            joint_name=joint_names[joint], predicted_x=predicted[joint, 0],
                            predicted_y=predicted[joint, 1], score=record['keypoint_scores'][joint],
                            adjudication='', reference_x='', reference_y='', notes=''))
                tile_draw = ImageDraw.Draw(tile)
                tile_draw.text((5, 455), 'Predicted outside crop/image points may be clipped in view.', fill='black')
                sheet.paste(tile, ((cell % 4) * 360, (cell // 4) * 480))
            sheet.save(output / f'{category}_{page_start // 12 + 1:02d}.jpg', quality=92)
    if review_rows:
        with (output / 'human_review.csv').open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(review_rows[0]))
            writer.writeheader()
            writer.writerows(review_rows)


def main():
    args = parse_args()
    if args.limit_images < 0 or args.batch_size < 1:
        raise ValueError('invalid image limit or batch size')
    args.output.mkdir(parents=True, exist_ok=False)
    data_hash = sha256(args.annotations)
    data = json.loads(args.annotations.read_text())
    images = {image['id']: image for image in data['images']}
    image_ids = sorted(images)
    if args.limit_images:
        image_ids = sorted(np.random.default_rng(args.seed).choice(
            image_ids, min(args.limit_images, len(image_ids)), replace=False).tolist())
    annotations = select_annotations(data, image_ids)

    import cv2
    import torch
    from mmpose.apis import init_model

    torch.set_num_threads(2)
    cv2.setNumThreads(0)
    torch.manual_seed(args.seed)
    model = init_model(str(args.config), str(args.checkpoint), device=args.device)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(state['state_dict'], strict=True)
    joint_names = list(CROWDPOSE_JOINT_NAMES)
    native_names = [model.dataset_meta['keypoint_id2name'][index] for index in range(14)]
    if native_names != joint_names[:12] + ['top_head', 'neck']:
        raise ValueError('checkpoint joint definitions do not match CrowdPose')
    if model.head.out_channels != 14:
        raise ValueError('not a CrowdPose14 checkpoint')
    teacher_id = sha256(args.checkpoint)
    resolved_config = model.cfg.pretty_text
    (args.output / 'resolved_config.py').write_text(resolved_config)
    config_hash = hashlib.sha256(resolved_config.encode()).hexdigest()
    source_paths = [Path(__file__), ROOT / 'src/eval/joint_completion.py',
                    ROOT / 'src/eval/metrics.py', ROOT / 'src/data/skeleton.py',
                    ROOT / 'external/mmpose/mmpose/apis/inference.py',
                    ROOT / 'external/mmpose/mmpose/models/pose_estimators/topdown.py',
                    ROOT / 'external/mmpose/mmpose/structures/bbox/transforms.py']
    manifest = dict(teacher='RTMPose-M CrowdPose14', teacher_id=teacher_id,
                    config_sha256=config_hash, config_source_sha256=sha256(args.config),
                    annotation_sha256=data_hash, annotations=str(args.annotations.resolve()),
                    image_root=str(args.images.resolve()), image_ids=image_ids,
                    coordinate_space='original image pixels; no clipping',
                    person_source='GT boxes; iscrowd=0 and num_keypoints>0',
                    split_caveat='Official teacher trained on TRAIN+VAL; VAL is in-sample.',
                    joint_names=joint_names, native_joint_names=native_names,
                    primary_flip_tta=bool(model.test_cfg.get('flip_test', False)),
                    consistency='independent original/flip single passes without TTA',
                    versions={name: importlib.metadata.version(name) for name in
                              ['torch', 'numpy', 'mmcv', 'mmengine', 'mmpose', 'mmdet']},
                    checkpoint_meta={name: state.get('meta', {}).get(name) for name in ['epoch', 'iter']},
                    checkpoint_training_config=state.get('meta', {}).get('cfg'),
                    sources={str(path.relative_to(ROOT)): sha256(path) for path in source_paths},
                    batch_size=args.batch_size, device=args.device, seed=args.seed)
    write_json(args.output / 'manifest.json', manifest)
    del state
    reused = {}
    if args.reuse_predictions:
        previous = json.loads((args.reuse_predictions.parent / 'manifest.json').read_text())
        for key in ['teacher_id', 'config_sha256', 'annotation_sha256', 'sources', 'consistency']:
            if previous[key] != manifest[key]:
                raise ValueError(f'incompatible reuse manifest: {key}')
        for line in args.reuse_predictions.read_text().splitlines():
            record = json.loads(line)
            reused[record['annotation_id']] = record
    start = time.monotonic()
    records = []
    cache = args.output / 'predictions.jsonl'
    with cache.open('w') as handle:
        for image_number, image_id in enumerate(image_ids, 1):
            people = annotations[image_id]
            if not people:
                continue
            old = [reused.get(person['id']) for person in people]
            if all(record is not None for record in old):
                for person, record in zip(people, old):
                    if record['original_annotation'] != person or record['teacher_id'] != teacher_id:
                        raise ValueError('cached source annotation or teacher mismatch')
                image_records = old
            else:
                image_path = args.images / images[image_id]['file_name']
                image = cv2.imread(str(image_path))
                if image is None or image.shape[:2] != (images[image_id]['height'], images[image_id]['width']):
                    raise ValueError(f'bad image or annotation dimensions: {image_path}')
                predicted, scores, single, flipped = infer_image(model, image, people, args.batch_size)
                image_records = []
                for person, xy, score, single_xy, flipped_xy in zip(people, predicted, scores, single, flipped):
                    record = build_prediction_record(person, xy, score, flipped_xy, teacher_id)
                    record['single_pass_xy'] = single_xy.tolist()
                    record['image_wh'] = [image.shape[1], image.shape[0]]
                    image_records.append(record)
            for record in image_records:
                handle.write(json.dumps(record, allow_nan=False) + '\n')
                records.append(record)
            handle.flush()
            if image_number % 50 == 0 or image_number == len(image_ids):
                print(f'{image_number}/{len(image_ids)} images, {len(records)} people, '
                      f'{time.monotonic() - start:.1f}s', flush=True)
    if args.device.startswith('cuda'):
        torch.cuda.synchronize()
        manifest['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated()
    manifest['inference_wall_seconds'] = time.monotonic() - start
    del model
    summary = summarize_predictions(records, joint_names, CROWDPOSE_SIGMAS,
                                    args.bootstrap_draws, args.seed)
    write_json(args.output / 'summary.json', summary)
    render_reviews(records, images, args.images, args.output,
                   args.review_people, args.seed)
    if sha256(args.annotations) != data_hash or sha256(args.checkpoint) != teacher_id:
        raise ValueError('source annotations or teacher changed during evaluation')
    manifest.update(status='complete', people=len(records),
                    predictions_sha256=sha256(cache),
                    finished_wall_seconds=time.monotonic() - start)
    write_json(args.output / 'manifest.json', manifest)
    print(json.dumps(summary['groups'], indent=2), flush=True)


if __name__ == '__main__':
    main()
