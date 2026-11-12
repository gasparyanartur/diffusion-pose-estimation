from copy import deepcopy
import hashlib
import random

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from src.models.pipelines.pose_autoencoder.evaluation import paired_metrics
from src.models.pipelines.pose_autoencoder.model import geometry_loss
from .latent_v43 import balanced_latent_loss
from .noise_v43 import velocity_target


DENOISER_MODULES = ('latent_input', 'time_embedding', 'denoiser_layers', 'latent_output', 'reference_box_embedding')


def configure_cached_model(model):
    model.requires_grad_(False).eval()
    for name in DENOISER_MODULES:
        getattr(model, name).requires_grad_(True)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def frozen_state_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        if name.split('.')[0] not in DENOISER_MODULES:
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def pack_cached_batch(cache, indices, device):
    rows = [cache[int(index)] for index in indices]
    memory = pad_sequence([row['memory'] for row in rows], batch_first=True).to(device)
    position = pad_sequence([row['position'] for row in rows], batch_first=True).to(device)
    padding = pad_sequence([row['padding_mask'] for row in rows], batch_first=True, padding_value=True).to(device)
    return dict(encoded_image=(memory, position, padding),
        references={name: torch.stack([row['references'][name] for row in rows]).to(device)
                    for name in rows[0]['references']},
        clean=torch.stack([row['clean'] for row in rows]).to(device),
        positive=torch.stack([row['positive'] for row in rows]).to(device),
        image_wh=torch.stack([row['image_wh'] for row in rows]).to(device),
        target={name: torch.stack([row['target'][name] for row in rows]).to(device)
                for name in rows[0]['target']},
        image_ids=[row['image_id'] for row in rows])


def control_step(model, optimizer, cache, generator, *, batch_size, noise_levels,
                 geometry_weight=0., center_min_side_px=1., grad_clip=1.):
    indices = torch.randint(len(cache), (batch_size,), generator=generator)
    batch = pack_cached_batch(cache, indices, next(model.parameters()).device)
    levels = torch.tensor(noise_levels)
    sigma = levels[torch.randint(len(levels), (batch_size,), generator=generator)].to(batch['clean'])
    noise = torch.randn(batch['clean'].shape, generator=generator).to(batch['clean'])
    noisy = batch['clean']+sigma[:, None, None]*noise
    output = model.denoise_at_sigma(noisy, sigma, batch['encoded_image'],
        references=batch['references'], image_wh=batch['image_wh'])
    prediction, target = output['pred_latents'], batch['clean']
    if model.prediction_type == 'v':
        prediction, target = output['pred_velocity'], velocity_target(batch['clean'], noise, sigma)
    regression = balanced_latent_loss(prediction, target, batch['positive'])
    loss = regression['total']
    geometry = loss.new_zeros(())
    if geometry_weight and batch['positive'].any():
        positive = batch['positive']
        geometry = geometry_loss({name: output[name][positive] for name in ('keypoints_px', 'boxes_cxcywh_px')},
            {name: value[positive] for name, value in batch['target'].items()},
            center_min_side_px=center_min_side_px)['loss']
        loss = loss+geometry_weight*geometry
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite small-noise control loss')
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    norm = torch.nn.utils.clip_grad_norm_(parameters, grad_clip, error_if_nonfinite=True)
    optimizer.step()
    clean_error = (output['pred_latents'].detach()-batch['clean']).square().mean(-1)
    return dict(loss=float(loss.detach()), regression_positive=float(regression['positive'].detach()),
        regression_background=float(regression['background'].detach()), geometry_loss=float(geometry.detach()),
        foreground_latent_mse=float(clean_error[batch['positive']].mean()) if batch['positive'].any() else 0.,
        gradient_norm=float(norm), sample=dict(indices=indices.tolist(), image_ids=batch['image_ids'], sigma=sigma.tolist()))


@torch.no_grad()
def evaluate_cached(model, cache, *, sigmas=(0., .01, .025, .05, .1), seeds=(42, 43, 44), batch_size=4):
    results = {}
    device = next(model.parameters()).device
    for sigma in sigmas:
        predictions = {label: [] for label in ('clean_codec', 'raw_noisy', 'denoised')}
        targets, errors, maximum_difference = [], [], 0.
        for seed in seeds:
            for start in range(0, len(cache), batch_size):
                indices = list(range(start, min(start+batch_size, len(cache))))
                batch = pack_cached_batch(cache, indices, device)
                noise = torch.stack([torch.randn(cache[index]['clean'].shape,
                    generator=torch.Generator().manual_seed(seed+cache[index]['image_id'])) for index in indices]).to(device)
                noisy = batch['clean']+sigma*noise
                output = model.denoise_at_sigma(noisy, torch.full((len(indices),), sigma, device=device),
                    batch['encoded_image'], references=batch['references'], image_wh=batch['image_wh'])
                sizes = batch['image_wh'][:, None].expand(-1, model.num_proposals, -1)
                clean = model.codec.decode_latents(batch['clean'], sizes)
                raw = model.codec.decode_latents(noisy, sizes)
                positive = batch['positive']
                targets.append({name: value[positive].cpu() for name, value in batch['target'].items()})
                for label, prediction in (('clean_codec', clean), ('raw_noisy', raw), ('denoised', output)):
                    predictions[label].append({name: prediction[name][positive].cpu()
                        for name in ('keypoints_px', 'boxes_cxcywh_px')})
                errors.append(((output['pred_latents']-batch['clean']).square().mean(-1))[positive].cpu())
                if sigma == 0. and positive.any():
                    maximum_difference = max(maximum_difference, *(float((output[name][positive]-clean[name][positive]).abs().max())
                        for name in ('keypoints_px', 'boxes_cxcywh_px')))
        target = {name: torch.cat([row[name] for row in targets]) for name in targets[0]}
        if not len(target['joint_mask']):
            raise ValueError('Cached evaluation requires at least one labeled person')
        result = {}
        for label, rows in predictions.items():
            prediction = {name: torch.cat([row[name] for row in rows]) for name in rows[0]}
            result[label] = paired_metrics(prediction, target)
        result.update(foreground_latent_mse=float(torch.cat(errors).mean()),
                      max_clean_decode_difference_px=maximum_difference if sigma == 0. else None)
        results[str(sigma)] = result
    return results


def tiny_set_gate(metrics):
    clean = metrics['0.0']['max_clean_decode_difference_px'] <= 1e-4
    positive = {}
    for sigma in (.01, .025, .05):
        row = metrics[str(sigma)]
        positive[str(sigma)] = (row['foreground_latent_mse'] <= 1e-5 and all(
            row['denoised'][name] >= row['clean_codec'][name]-.005
            for name in ('paired_OKS_mean', 'paired_IoU_mean')))
    return dict(passed=clean and all(positive.values()), exact_clean_parity=clean, positive_noise=positive)


def make_control_checkpoint(model, optimizer, generator, *, step, cache_sha256, settings):
    return dict(format='v43_small_noise_control_v1', prediction_type=model.prediction_type,
        step=step, cache_sha256=cache_sha256, settings=settings,
        model_state=deepcopy(model.state_dict()), optimizer=deepcopy(optimizer.state_dict()),
        generator_state=generator.get_state().clone(), global_rng=dict(python=random.getstate(),
            numpy=np.random.get_state(), torch=torch.get_rng_state(),
            cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []))


def restore_control_checkpoint(record, model, optimizer, generator, *, cache_sha256):
    if record.get('format') != 'v43_small_noise_control_v1' or record['prediction_type'] != model.prediction_type:
        raise ValueError('Checkpoint prediction type or format mismatch')
    if record['cache_sha256'] != cache_sha256:
        raise ValueError('Checkpoint cache identity mismatch')
    model.load_state_dict(record['model_state'], strict=True)
    optimizer.load_state_dict(deepcopy(record['optimizer']))
    generator.set_state(record['generator_state'].cpu())
    state = record['global_rng']
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])
    return record['step']
