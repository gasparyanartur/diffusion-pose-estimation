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
from src.models.pipelines.pose_estimators.poseflow.training_v55 import PoseFlowV55Pipeline, PoseFlowV55PipelineConfig


def gradient_routes(module, sample):
    module.zero_grad(set_to_none=True)
    parameters = [(name, value) for name, value in module.model.named_parameters() if value.requires_grad]
    groups = module.gradient_clipping_families()
    if len({id(value) for group in groups.values() for value in group}) != len(parameters):
        raise AssertionError('V55 gradient-family coverage is incomplete')
    losses = module._losses_for_batch(sample)
    results, context_gradients = {}, {}
    for task in ('pose_birth', 'box_birth', 'pose_birth_contribution', 'box_birth_contribution', 'selection', 'eos', 'frontend', 'total'):
        gradients = torch.autograd.grad(losses[task], [value for _, value in parameters], retain_graph=True, allow_unused=True)
        norms = dict(native=0., shared_context=0., selector=0., geometry=0.)
        shared = []
        for (name, parameter), gradient in zip(parameters, gradients):
            group = ('native' if name.startswith('frontend.') else 'shared_context' if name.startswith('proposal_context.')
                     else 'selector' if name.startswith('selector.') else 'geometry')
            if gradient is not None:
                norms[group] += float(gradient.detach().double().square().sum())
            if group == 'shared_context':
                shared.append(torch.zeros(parameter.numel()) if gradient is None else gradient.detach().cpu().flatten())
        results[task] = dict(loss=float(losses[task].detach()), norms={name: value**.5 for name, value in norms.items()})
        context_gradients[task] = torch.cat(shared)
        del gradients
    for task in ('pose_birth', 'box_birth', 'selection', 'eos'):
        if results[task]['norms']['shared_context'] <= 0:
            raise AssertionError(f'V55 {task} does not train context')
    for task in ('selection', 'eos'):
        if results[task]['norms']['geometry'] != 0 or results[task]['norms']['native'] != 0:
            raise AssertionError('V55 classifier crossed detached context input boundary')
    pairs = [('pose_birth', 'box_birth'), ('pose_birth_contribution', 'selection'), ('box_birth_contribution', 'selection'),
             ('pose_birth_contribution', 'eos'), ('box_birth_contribution', 'eos')]
    cosines = {first+'__'+second: float(torch.nn.functional.cosine_similarity(
        context_gradients[first][None].double(), context_gradients[second][None].double())) for first, second in pairs}
    combined = context_gradients['total']
    geometry_norm = (results['total']['norms']['shared_context']**2+results['total']['norms']['geometry']**2)**.5
    losses['total'].backward()
    clipping = {name: float(value) for name, value in module.configure_gradient_clipping(None, .1, 'norm').items()}
    return dict(passed=True, objectives=results, shared_context_cosines=cosines,
        combined_geometry_preclip=geometry_norm, clipping_budgets=module.gradient_clipping_budgets(),
        clipping=clipping,
        context_combined_norm=float(combined.double().norm()),
        context_postclip_norm=clipping['shared_context/postclip'],
        note='Same-forward post-two-update native diagnostic; shared block receives classification and geometry; not learned AP evidence')


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
        config = compose(config_name='benchmark_poseflow_v55', overrides=[f'residual_bank_file={args.residual_bank.resolve()}'])
    sources = source_hashes()
    data_config = DataConfig.from_hydra_config(config.data)
    if data_config.test_ann != data_config.val_ann:
        raise ValueError('Verification must not open TEST')
    data = data_config.build()
    pl.seed_everything(42, workers=True)
    specification = PoseFlowV55PipelineConfig.from_hydra_config(config.pipeline)
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
        raise ValueError('V55 bank source/split provenance mismatch')
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
        preflight = native_preflight(module, args.output, pipeline_class=PoseFlowV55Pipeline)
        batches = training_batches(data)
        sample = next(batches)
        batches.close()
        routes = gradient_routes(module, sample)
        (args.output/'gradient_routes.json').write_text(json.dumps(routes, indent=2)+'\n')
        del sample
        module.load_state_dict(initial_state, strict=True)
        if module.prefix_bank_provenance != frozen_bank:
            raise AssertionError('Frozen bank provenance changed')
        module.eval()
        smoke = run_v53_evaluation(module.model, data, args.output/'evaluator_smoke', epoch=-1, count=2)
        val = json.loads(Path(data.val_ann_path).read_text())
        ids = sorted(random.Random(42).sample(sorted(image['id'] for image in val['images']), 32))
        if source_hashes() != sources:
            raise ValueError('Sources changed during verification')
        result = dict(passed=preflight['passed'], native_preflight=preflight, gradient_routes=routes,
            evaluator_smoke=dict(images=smoke['images'], metrics=smoke['metrics'], architecture=smoke['architecture']),
            sources=sources, pipeline_config=OmegaConf.to_container(config.pipeline, resolve=True),
            data_config=OmegaConf.to_container(config.data, resolve=True), initial_model_sha256=initial_hash,
            initialization=initialization, calibration=calibration, contract=module.checkpoint_contract(),
            fixed_val_image_ids=ids, pid=os.getpid(), prefix_bank=module.prefix_bank_provenance,
            residual_bank_file=str(args.residual_bank.resolve()), residual_bank_file_sha256=digest(args.residual_bank),
            inputs={str(path): digest(path) for path in (data.train_ann_path, data.val_ann_path,
                specification.pretrained_checkpoint, args.residual_bank.resolve())})
        (args.output/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
        print('V55_VERIFICATION', json.dumps(dict(passed=True)), flush=True)
    except BaseException as error:
        failure = dict(passed=False, error=repr(error), resource_failure=isinstance(error, torch.cuda.OutOfMemoryError),
                       policy='FP32/50rows/49capacity/batch4/accum2 unchanged; existing jobs preserved')
        (args.output/'failure.json').write_text(json.dumps(failure, indent=2)+'\n')
        raise


if __name__ == '__main__':
    main()
