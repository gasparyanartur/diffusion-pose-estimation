#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import torch

from src.models.pipelines.pose_estimators.poseflow.control_v43 import evaluate_cached, tiny_set_gate
from src.models.pipelines.pose_estimators.poseflow.training_v43 import PoseFlowV43PipelineConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    record = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if record.get('format') != 'v43_small_noise_control_v1':
        raise ValueError('Expected a cached small-noise control checkpoint')
    specification = PoseFlowV43PipelineConfig(**record['settings']['pipeline'])
    if specification.prediction_type != record['prediction_type']:
        raise ValueError('Checkpoint prediction metadata mismatch')
    if hashlib.sha256(args.cache.read_bytes()).hexdigest() != record['cache_sha256']:
        raise ValueError('Checkpoint cache identity mismatch')
    model = specification.create_model().to(args.device).eval()
    model.load_state_dict(record['model_state'], strict=True)
    cache = torch.load(args.cache, map_location='cpu', weights_only=False)
    control = record['settings']['control']
    metrics = {split: evaluate_cached(model, rows, sigmas=control['evaluation_sigmas'],
        seeds=control['evaluation_seeds'], batch_size=control['batch_size']) for split, rows in cache.items()}
    report = dict(step=record['step'], prediction_type=record['prediction_type'], metrics=metrics,
        train_gate=tiny_set_gate(metrics['train']), cache_sha256=record['cache_sha256'],
        scope='Cached conditional denoising only; frozen components have prior trainval exposure; not detector AP')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
