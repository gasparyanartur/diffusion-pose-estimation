from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.data.pose_geometry import PoseGeometryDataset
from src.models.pipelines.pose_autoencoder.evaluation import (
    CodecAPEvaluator, EVALUATION_SEEDS, export_reconstructions,
)
from src.models.pipelines.pose_autoencoder.experiment import latent_statistics, noise_curve, verify_ceilings
from src.models.pipelines.pose_autoencoder.model import geometry_loss, joint_exponents
from src.models.pipelines.pose_autoencoder.training import (
    GEOMETRY_FIELDS, TARGET_FIELDS, PoseAutoencoderPipeline, concatenate_batches,
)
from src.training import TrainingConfig, TrainingRunner
from src.training.callbacks import TrustedCheckpointIO
from .fidelity import FidelityPoseVAE, FidelityVAEPipeline, calibrated_kl_weight, mask_joints, posterior_statistics
from .model import prior_kl
from .training import PoseGeometryDataModule


class PosteriorMeanCodec:
    def __init__(self, model):
        self.model, self.latent_dim = model, model.latent_dim

    def to(self, device):
        self.model.to(device)
        return self

    def eval(self):
        self.model.eval()
        return self

    def encode(self, **geometry):
        return self.model.encode(**geometry)[0]

    def decode(self, latent, image_wh):
        return self.model.decode(latent, image_wh)


@torch.inference_mode()
def collect_posterior(model, loader, device):
    model.to(device).eval()
    means, logvars, targets = [], [], []
    for batch in loader:
        mean, logvar = model.encode(**{name: batch[name].to(device) for name in GEOMETRY_FIELDS})
        means.append(mean.cpu())
        logvars.append(logvar.cpu())
        targets.append({name: batch[name].cpu() for name in TARGET_FIELDS})
    return torch.cat(means), torch.cat(logvars), concatenate_batches(targets)


@torch.inference_mode()
def decode_cpu(model, latent, target, device):
    return {name: value.cpu() for name, value in model.decode(latent.to(device), target['image_wh'].to(device)).items()}


@torch.inference_mode()
def prepare_initialization(checkpoint, train_ann, output, device='cuda'):
    autoencoder = PoseAutoencoderPipeline.load_from_checkpoint(checkpoint, weights_only=False).model
    loader = DataLoader(PoseGeometryDataset(train_ann), batch_size=512, shuffle=False)
    mean, std = latent_statistics(autoencoder, loader, device)
    model = FidelityPoseVAE(autoencoder.encoder[0].out_features, autoencoder.latent_dim).to(device)
    model.initialize_from_autoencoder(autoencoder, std.to(device))
    means, logvars, target = collect_posterior(model, loader, device)
    prediction = decode_cpu(model, means, target, device)
    noise = torch.randn(means.shape, generator=torch.Generator().manual_seed(1042))
    sampled = decode_cpu(model, means+(.5*logvars).exp()*noise, target, device)
    reconstruction = float(geometry_loss(sampled, target)['loss'])
    kl = float(prior_kl(means, logvars))
    first = next(iter(loader))
    arguments = {name: first[name].to(device) for name in GEOMETRY_FIELDS}
    expected_mean = autoencoder.encode(**arguments)
    actual_mean, _ = model.encode(**arguments)
    if not torch.equal(expected_mean, actual_mean):
        raise RuntimeError('AE-to-VAE posterior-mean transfer is not exact')
    expected = autoencoder.decode(expected_mean, arguments['image_wh'])
    actual = model.decode(actual_mean, arguments['image_wh'])
    if any(not torch.equal(expected[name], actual[name]) for name in expected):
        raise RuntimeError('AE-to-VAE decoded transfer is not exact')
    record = dict(state_dict={name: value.cpu() for name, value in model.state_dict().items()},
                  latent_mean=mean.cpu(), latent_std=std.cpu(), width=autoencoder.encoder[0].out_features,
                  latent_dim=model.latent_dim, checkpoint=str(checkpoint),
                  checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                  training_annotation_sha256=hashlib.sha256(Path(train_ann).read_bytes()).hexdigest(),
                  people=len(loader.dataset), reconstruction=reconstruction, kl=kl,
                  mean_reconstruction=float(geometry_loss(prediction, target)['loss']),
                  exact_mean_parity=True, posterior_std_fraction=.05, test_access=False)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(record, output)
    summary = {name: value for name, value in record.items() if name not in ('state_dict', 'latent_mean', 'latent_std')}
    output.with_suffix('.json').write_text(json.dumps(summary, indent=2))
    return record


def hidden_joint_metrics(prediction, target, observed):
    error = (prediction['keypoints_px']-target['keypoints_px']).norm(dim=-1)
    normalized = error/target['boxes_cxcywh_px'][:, 2:].amax(-1)[:, None]
    exponents = joint_exponents(prediction, target)
    result = {}
    for name, selected in dict(hidden=target['joint_mask'] & ~observed,
                               observed=target['joint_mask'] & observed).items():
        values = normalized[selected]
        result[name] = dict(joints=int(selected.sum()),
                            error_median=float(values.quantile(.5)) if len(values) else None,
                            error_p95=float(values.quantile(.95)) if len(values) else None,
                            mean_joint_OKS=float((-exponents[selected]).exp().mean()) if len(values) else None)
    return result


def error_slices(prediction, target):
    error = (prediction['keypoints_px']-target['keypoints_px']).norm(dim=-1)
    labels = target['joint_mask']
    scale = target['boxes_cxcywh_px'][:, 2:].amax(-1)
    relative_size = scale/target['image_wh'].amax(-1)
    groups = {}
    for name, selected in dict(small=relative_size < .1,
                               medium=(relative_size >= .1)&(relative_size < .3),
                               large=relative_size >= .3).items():
        values = (error/scale[:, None])[selected][labels[selected]]
        groups[name] = dict(people=int(selected.sum()),
                            median=float(values.quantile(.5)) if len(values) else None,
                            p95=float(values.quantile(.95)) if len(values) else None)
    per_joint = []
    for index in range(14):
        values = error[:, index][labels[:, index]]
        per_joint.append(dict(joint_index=index, labeled=len(values),
                              median_px=float(values.quantile(.5)) if len(values) else None,
                              p95_px=float(values.quantile(.95)) if len(values) else None))
    return dict(person_size=groups, per_joint=per_joint,
                size_definition='max box extent / max image extent; boundaries .1 and .3')


def export_prior(path, prediction, image_wh):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from src.data.skeleton import CROWDPOSE_EDGE_LIST

    figure, axes = plt.subplots(4, 4, figsize=(12, 12))
    for index, axis in enumerate(axes.flat):
        points, box = prediction['keypoints_px'][index], prediction['boxes_cxcywh_px'][index]
        for first, second in CROWDPOSE_EDGE_LIST:
            axis.plot(points[[first, second], 0], points[[first, second], 1], color='#176ab4', lw=1)
        axis.scatter(points[:, 0], points[:, 1], color='#176ab4', s=7)
        axis.add_patch(Rectangle((box[0]-box[2]/2, box[1]-box[3]/2), box[2], box[3], fill=False))
        width, height = image_wh[index].tolist()
        axis.set_xlim(-.1*width, 1.1*width)
        axis.set_ylim(1.1*height, -.1*height)
        axis.set_aspect('equal')
        axis.axis('off')
    figure.suptitle('Standard-normal prior draws | No matched ground truth')
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


@torch.inference_mode()
def evaluate_masking(module, target, evaluator, device):
    rows = []
    for probability in (.1, .3, .5):
        for seed in EVALUATION_SEEDS:
            corrupted = mask_joints(target, probability, torch.Generator().manual_seed(seed), fixed_probability=True)
            mean, logvar = module.model.encode(**{name: corrupted[name].to(device) for name in GEOMETRY_FIELDS})
            noise = torch.randn(mean.shape, generator=torch.Generator().manual_seed(seed+10000)).to(device)
            latents = dict(mean=mean, sample=mean+(.5*logvar).exp()*noise)
            if module.hparams.mode == 'deterministic':
                latents['sample'] = mean
            for variant, latent in latents.items():
                prediction = decode_cpu(module.model, latent, target, device)
                rows.append(dict(mask_probability=probability, seed=seed, variant=variant,
                                 **evaluator.evaluate(prediction, target),
                                 joint_groups=hidden_joint_metrics(prediction, target, corrupted['joint_mask'])))
    return rows


@torch.inference_mode()
def assess(module, data, directory, device):
    mean, logvar, target = collect_posterior(module.model, data.val_dataloader(), device)
    evaluator = CodecAPEvaluator(data.paths['val'], target['annotation_id'].tolist())
    mean_prediction = decode_cpu(module.model, mean, target, device)
    validation = dict(mean=evaluator.evaluate(mean_prediction, target), posterior=posterior_statistics(mean, logvar))
    (directory/'error_slices.json').write_text(json.dumps(error_slices(mean_prediction, target), indent=2))
    sampled_rows = []
    for seed in EVALUATION_SEEDS:
        noise = torch.randn(mean.shape, generator=torch.Generator().manual_seed(seed))
        latent = mean+(.5*logvar).exp()*noise if module.hparams.mode == 'sampled' else mean
        prediction = decode_cpu(module.model, latent, target, device)
        sampled_rows.append(dict(seed=seed, **evaluator.evaluate(prediction, target)))
        if seed == EVALUATION_SEEDS[0]:
            export_reconstructions(directory/'posterior_reconstructions.png', prediction, target,
                                   title='Ground truth: gray | Posterior reconstruction: blue | Labeled joints only')
    validation['samples'] = sampled_rows
    export_reconstructions(directory/'mean_reconstructions.png', mean_prediction, target)
    worst = joint_exponents(mean_prediction, target).mul(target['joint_mask']).sum(-1)/target['joint_mask'].sum(-1)
    selected = worst.argsort(descending=True)[:16]
    export_reconstructions(directory/'worst_reconstructions.png',
                           {name: value[selected] for name, value in mean_prediction.items()},
                           {name: value[selected] for name, value in target.items()})
    torch.save(dict(mean=mean, logvar=logvar, target=target, prediction=mean_prediction), directory/'validation_geometry.pt')
    train_loader = DataLoader(data.train_dataloader().dataset, batch_size=512, shuffle=False)
    train_mean, train_logvar, train_target = collect_posterior(module.model, train_loader, device)
    train_evaluator = CodecAPEvaluator(data.paths['train'], train_target['annotation_id'].tolist())
    training = dict(mean=train_evaluator.evaluate(decode_cpu(module.model, train_mean, train_target, device), train_target),
                    posterior=posterior_statistics(train_mean, train_logvar))
    train_noise = torch.randn(train_mean.shape, generator=torch.Generator().manual_seed(1042))
    train_sample = train_mean+(.5*train_logvar).exp()*train_noise if module.hparams.mode == 'sampled' else train_mean
    training['sample'] = train_evaluator.evaluate(decode_cpu(module.model, train_sample, train_target, device), train_target)
    std = train_mean.std(0, unbiased=False)
    torch.save(dict(mean=train_mean.mean(0), std=std, initial_std=module.initial_latent_std.cpu(),
                    people=len(train_mean), source='train-only unflipped population'), directory/'latent_statistics.pt')
    codec = PosteriorMeanCodec(module.model)
    for label, scale in dict(initial_reference=module.initial_latent_std.cpu(), current_training=std).items():
        rows = noise_curve(codec, target, mean, scale, evaluator, device)
        (directory/f'noise_curve_{label}.json').write_text(json.dumps(rows, indent=2))
    masking = evaluate_masking(module, target, evaluator, device)
    (directory/'masking.json').write_text(json.dumps(masking, indent=2))
    prior = torch.randn((16, module.model.latent_dim), generator=torch.Generator().manual_seed(3042))
    prior_target = {name: value[:16] for name, value in target.items()}
    prior_prediction = decode_cpu(module.model, prior, prior_target, device)
    torch.save(dict(prediction=prior_prediction, image_wh=prior_target['image_wh']), directory/'prior_samples.pt')
    export_prior(directory/'prior_samples.png', prior_prediction, prior_target['image_wh'])
    return dict(validation=validation, training=training, test_access=False)


def run_fidelity_experiment(config, directory, source_root):
    runtime = TrainingConfig.from_hydra_config(config.training)
    if runtime.test_after_fit or runtime.skip_training:
        raise ValueError('Fidelity comparison uses train/val only')
    directory.mkdir(parents=True, exist_ok=True)
    runner = TrainingRunner(runtime, directory)
    runner.seed_everything()
    data = PoseGeometryDataModule(**OmegaConf.to_container(config.data, resolve=True))
    data.setup('fit')
    device = 'cuda' if torch.cuda.is_available() and config.hardware.accelerator != 'cpu' else 'cpu'
    initial_path = Path(config.initialization_file) if config.initialization_file else directory/'initialization.pt'
    if not initial_path.exists():
        prepare_initialization(config.initial_checkpoint, data.paths['train'], initial_path, device)
    initial = torch.load(initial_path, map_location='cpu', weights_only=False)
    if hashlib.sha256(data.paths['train'].read_bytes()).hexdigest() != initial['training_annotation_sha256']:
        raise ValueError('Initialization statistics must match training annotations')
    settings = OmegaConf.to_container(config.pipeline, resolve=True)
    fraction = settings.pop('kl_fraction')
    coefficient = calibrated_kl_weight(initial['reconstruction'], initial['kl'], fraction)
    val_loader = data.val_dataloader()
    val_ids = [int(record['annotation_id']) for record in val_loader.dataset]
    evaluator = CodecAPEvaluator(data.paths['val'], val_ids)
    (directory/'ceilings.json').write_text(json.dumps(verify_ceilings(evaluator, val_loader), indent=2))
    module = FidelityVAEPipeline(**settings, width=initial['width'], latent_dim=initial['latent_dim'],
                                 kl_weight=coefficient, max_epochs=runtime.max_epochs,
                                 val_ann=str(data.paths['val']), val_annotation_ids=val_ids,
                                 artifact_directory=str(directory/'validation'))
    module.model.load_state_dict(initial['state_dict'], strict=True)
    module.initial_latent_std.copy_(initial['latent_std'])
    sources = [path for folder in ('src', 'conf', 'scripts/training') for path in (source_root/folder).rglob('*')
               if path.suffix in ('.py', '.yaml')]
    manifest = dict(initialization=str(initial_path), initialization_sha256=hashlib.sha256(initial_path.read_bytes()).hexdigest(),
                    initial_checkpoint=initial['checkpoint'], initial_checkpoint_sha256=initial['checkpoint_sha256'],
                    parameters=sum(p.numel() for p in module.parameters()), kl_fraction=fraction,
                    calibrated_kl_weight=coefficient, initial_reconstruction=initial['reconstruction'], initial_kl=initial['kl'],
                    source_root=str(source_root), source_sha256={str(p.relative_to(source_root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                    annotations={name:dict(path=str(data.paths[name].resolve()),sha256=hashlib.sha256(data.paths[name].read_bytes()).hexdigest()) for name in ('train','val')},
                    people={name:len(dataset) for name,dataset in data.datasets.items()},
                    train_val_image_overlap=len(data.datasets['train'].image_ids & data.datasets['val'].image_ids), test_access=False)
    (directory/'manifest.json').write_text(json.dumps(manifest, indent=2))
    OmegaConf.save(config, directory/'resolved.yaml', resolve=True)
    result = runner.run(module, data, max_epochs=runtime.max_epochs, plugins=[TrustedCheckpointIO()])
    best = FidelityVAEPipeline.load_from_checkpoint(result.checkpoint, weights_only=False)
    summary = dict(checkpoint=result.checkpoint, checkpoint_sha256=hashlib.sha256(Path(result.checkpoint).read_bytes()).hexdigest(),
                   mode=best.hparams.mode, mask_max=best.hparams.mask_max, kl_weight=coefficient,
                   completed_epochs=runtime.max_epochs, test_access=False, integration_ready=False)
    (directory/'fit_complete.json').write_text(json.dumps(summary, indent=2))
    summary.update(assess(best, data, directory, device))
    (directory/'completion.json').write_text(json.dumps(summary, indent=2))
    return summary
