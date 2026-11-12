import torch

from src.data.datasets import PoseDataset
from .latent_v43 import geometry_targets, assign_latent_sources
from .quick_eval_v21 import v21_test_batches


def select_cache_ids(dataset, count, seed):
    candidates = sorted(int(sample['image_id']) for sample in dataset.samples)
    if len(candidates) < count or count < 1:
        raise ValueError('Invalid cached image count')
    order = torch.randperm(len(candidates), generator=torch.Generator().manual_seed(seed))[:count]
    return sorted(candidates[int(index)] for index in order)


@torch.no_grad()
def build_context_cache(module, data, *, images_per_split=16, seed=42):
    data.trainer = type('CachedEvaluationContext', (), {'training': False})()
    module.eval()
    model = module.model
    cache, manifests = {}, {}
    for split, annotations in (('train', data.train_ann_path), ('val', data.val_ann_path)):
        dataset = PoseDataset.from_path(annotations, img_root=data.img_dir, num_keypoints=14)
        image_ids = select_cache_ids(dataset, images_per_split, seed+(0 if split == 'train' else 100))
        rows = []
        for batch in v21_test_batches(data, dataset, image_ids, next(model.parameters()).device):
            geometry, targets = geometry_targets(batch)
            context = model.encode_context(batch['image'], batch.get('padding_mask'))
            references = model.select_references(model.propose(context))
            matches = module.matcher(dict(pred_boxes=references['boxes'], pred_logits=references['pred_logits']),
                                     targets, include_keypoints=False)
            sources = assign_latent_sources(model.codec, geometry, targets, matches, slots=model.num_proposals,
                generator=torch.Generator().manual_seed(seed), posterior_mode='mean')
            positive = sources['positive'][0]
            if not positive.any():
                raise ValueError('Selected cache image has no valid labeled people')
            indices = sources['target_indices'][0, positive]
            sizes = batch['img_hw'].flip(-1).to(geometry['image_wh'])
            target = dict(keypoints_px=torch.zeros(model.num_proposals, 14, 2, device=positive.device),
                boxes_cxcywh_px=torch.zeros(model.num_proposals, 4, device=positive.device),
                joint_mask=torch.zeros(model.num_proposals, 14, dtype=torch.bool, device=positive.device))
            for name in target:
                target[name][positive] = geometry[name][indices]
            row = dict(image_id=int(batch['image_id'][0]), memory=context['image'][0][0].cpu(),
                position=context['image'][1][0].cpu(), padding_mask=context['image'][2][0].cpu(),
                references={name: references[name][0].cpu() for name in ('features', 'boxes', 'existence_logits')},
                reference_indices=references['indices'][0].cpu(),
                source_target_indices=sources['target_indices'][0].cpu(),
                clean=sources['latents'][0].cpu(), positive=positive.cpu(), image_wh=sizes[0].cpu(),
                target={name: value.cpu() for name, value in target.items()})
            rows.append(row)
        cache[split] = rows
        manifests[split] = dict(annotation_path=str(annotations), image_ids=image_ids,
            people=sum(int(row['positive'].sum()) for row in rows))
    if set(manifests['train']['image_ids']) & set(manifests['val']['image_ids']):
        raise ValueError('Train and validation cache images overlap')
    return cache, manifests
