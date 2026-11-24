import argparse
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
import pytorch_lightning as pl
import torch

from scripts.training.train_poseflow_v46 import digest
from scripts.training.verify_poseflow_v53 import training_batches
from src.data import DataConfig
from src.models.pipelines.pose_estimators.poseflow.prefix_augmentation_v54 import (
    bank_digest, bounded_residuals, geometric_residuals,
)
from src.models.pipelines.pose_estimators.poseflow.training_v53 import PoseFlowV53Pipeline


def collect_residual_bank(module, batches, *, images=512, capacity=2048, seed=42054):
    residuals, masks, people_ids, image_ids = [], [], [], []
    module.eval()
    with torch.no_grad():
        for index, batch in enumerate(batches):
            from src.models.pipelines.pose_estimators.poseflow.latent_v43 import geometry_targets
            _, targets = geometry_targets(batch)
            context = module.model.prepare_context(batch['image'], batch.get('padding_mask'), targets=targets)
            rows, sources, _, _ = module.matched_rows(batch, context, generator=torch.Generator().manual_seed(seed+index))
            output = module.model.forward_rows(context, **{name: rows[name] for name in module.model.row_input_fields})
            selected = rows['birth_rows']
            image, rank = selected.nonzero(as_tuple=True)
            mask = sources['coordinate_mask'][image, rank, :28].unflatten(-1, (14, 2)).all(-1)
            error = geometric_residuals(output['pred_coordinates'][-1, image, rank], sources['coordinates'][image, rank], mask)
            if not torch.isfinite(error).all():
                raise ValueError('Nonfinite TRAIN residuals')
            remaining = images-len(image_ids)
            eligible = image < remaining
            residuals.append(error[eligible].cpu())
            masks.append(mask[eligible].cpu())
            people_ids.extend(batch['image_id'][image[eligible]].cpu().tolist())
            image_ids.extend(batch['image_id'][:remaining].cpu().tolist())
            if len(image_ids) % 64 == 0:
                print('V54_RESIDUAL_IMAGES', len(image_ids), flush=True)
            if len(image_ids) >= images:
                break
    if len(image_ids) != images or len(set(image_ids)) != images:
        raise ValueError('Residual collection requires the declared unique TRAIN images')
    residuals, masks = torch.cat(residuals), torch.cat(masks)
    if not len(residuals):
        raise ValueError('Empty TRAIN residual bank')
    selected = torch.randperm(len(residuals), generator=torch.Generator().manual_seed(seed))[:capacity]
    residuals, masks = residuals[selected], masks[selected]
    people_ids = [people_ids[index] for index in selected.tolist()]
    bounded = bounded_residuals(residuals)
    statistics = dict(people=len(residuals),
        capped_vector_fraction=float(((bounded-residuals).abs().amax(-1) > 1e-6).float().mean()),
        center_box_unit_norm_quantiles=torch.quantile(residuals[:, 28:30].norm(dim=-1), torch.tensor([.5, .9, .99])).tolist(),
        absolute_log_scale_quantiles=torch.quantile(residuals[:, 30:].abs().flatten(), torch.tensor([.5, .9, .99])).tolist())
    return dict(residuals=residuals, joint_mask=masks, provenance=dict(split='train', seed=seed,
        image_ids=image_ids, person_image_ids=people_ids, statistics=statistics,
        pairing='GT_assigned_proposal_newborn_prediction;clean_random_GT_prefix;last_pose_layer',
        residual_sha256=bank_digest(residuals, masks)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from src.models.pipelines.pose_estimators.diffusionregpose.transformer import _load_native_msda
    if _load_native_msda() is None:
        raise RuntimeError('Residual collection requires native CUDA')
    with initialize_config_dir(version_base='1.3', config_dir=str(Path('conf').resolve())):
        config = compose(config_name='benchmark_poseflow_v54')
    data_config = DataConfig.from_hydra_config(config.data)
    if data_config.test_ann != data_config.val_ann:
        raise ValueError('Residual collection must not open TEST')
    data = data_config.build()
    checksum = digest(args.checkpoint)
    module = PoseFlowV53Pipeline.load_from_checkpoint(args.checkpoint, weights_only=False).cuda().eval()
    pl.seed_everything(42054, workers=True)
    bank = collect_residual_bank(module, training_batches(data), capacity=config.pipeline.residual_bank_capacity)
    train_ids = {int(image['id']) for image in json.loads(Path(data.train_ann_path).read_text())['images']}
    val_ids = {int(image['id']) for image in json.loads(Path(data.val_ann_path).read_text())['images']}
    used = set(bank['provenance']['image_ids'])
    if not used <= train_ids or used & val_ids or checksum != digest(args.checkpoint):
        raise ValueError('Residual bank provenance/source changed or split contamination detected')
    record = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    bank['provenance'].update(source_checkpoint=str(args.checkpoint.resolve()), source_checkpoint_sha256=checksum,
        source_epoch=record['epoch'], source_step=record['global_step'],
        train_annotations=str(data.train_ann_path), train_annotations_sha256=digest(data.train_ann_path))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bank, args.output)
    args.output.with_suffix('.json').write_text(json.dumps(bank['provenance'], indent=2)+'\n')
    print(json.dumps(bank['provenance']['statistics']), flush=True)


if __name__ == '__main__':
    main()
