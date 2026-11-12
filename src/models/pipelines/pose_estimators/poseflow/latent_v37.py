import math

import torch
from torch import nn
from torch.nn import functional


def pairwise_latent_distance(left, right):
    # left <N,D>, right <M,D>, result <N,M>
    return (left[:, None]-right[None]).square().mean(-1)


@torch.no_grad()
def latent_ota_assignment(cost, people, *, candidate_topk=5):
    query_count, target_count = cost.shape
    if query_count < 1 or target_count < 1 or not torch.isfinite(cost).all():
        raise ValueError('Latent OTA requires nonempty finite costs')
    if people.shape != (target_count,) or candidate_topk < 1:
        raise ValueError('Latent OTA target IDs/top-k disagree with costs')
    real_people = people[people >= 0].unique(sorted=True)
    if len(real_people) > query_count:
        raise ValueError('Latent OTA needs at least one prediction per real person')
    quality = (-cost).exp()
    counts = quality.topk(min(candidate_topk, query_count), dim=0).values.sum(0).long().clamp_min(1)
    matching = torch.zeros_like(cost, dtype=torch.bool)
    order = cost.argsort(dim=0, stable=True)
    for target, count in enumerate(counts.tolist()):
        matching[order[:count, target], target] = True
    selected = cost.masked_fill(~matching, float('inf')).argmin(-1)
    unselected = ~matching.any(-1)
    selected[unselected] = cost[unselected].argmin(-1)

    for person in real_people:
        assigned_people = people[selected]
        if (assigned_people == person).any():
            continue
        # Fake assignments and duplicate real-person assignments can move safely.
        same_person = assigned_people[:, None] == assigned_people[None]
        movable = (assigned_people < 0) | (same_person.sum(-1) > 1)
        queries = movable.nonzero().flatten()
        targets = (people == person).nonzero().flatten()
        candidate_cost = cost[queries[:, None], targets[None]]
        best = candidate_cost.flatten().argmin()
        selected[queries[best//len(targets)]] = targets[best % len(targets)]
    return selected, counts


@torch.no_grad()
def assign_latent_targets(clean, recovered, constructed, *, candidate_topk=5):
    latents = clean['pred_latents'].detach()
    selections, dynamic_counts, costs = [], [], []
    for clean_image, recovered_image, people in zip(latents, recovered['pred_latents'], constructed['target_indices']):
        cost = pairwise_latent_distance(recovered_image, clean_image)
        selected, counts = latent_ota_assignment(cost, people, candidate_topk=candidate_topk)
        selections.append(selected)
        dynamic_counts.append(counts.float().mean())
        costs.append(cost[torch.arange(len(selected), device=cost.device), selected].mean())
    selected = torch.stack(selections)
    images = torch.arange(len(latents), device=latents.device)[:, None]
    assigned = {name: constructed[name][images, selected] for name in ('positive', 'boxes')}
    assigned_people = constructed['target_indices'][images, selected]
    coverage, fake_utilization = [], []
    for image in range(len(latents)):
        real = assigned_people[image]
        covered = real[real >= 0].unique().numel()
        count = int(constructed['unique_people'][image])
        coverage.append(latents.new_tensor(covered/max(count, 1) if count else 1.))
        fake = ~constructed['positive'][image]
        utilized = selected[image].unique()
        fake_utilization.append(fake[utilized].sum()/fake.sum().clamp_min(1))
    metadata = dict(matched_person=assigned_people, selected_slots=selected,
        assigned_positive=assigned['positive'].sum(-1).float().mean(),
        unique_gt_coverage=torch.stack(coverage).mean(),
        fake_target_utilization=torch.stack(fake_utilization).mean(),
        target_utilization=latents.new_tensor(sum(row.unique().numel() for row in selected)/selected.numel()),
        dynamic_k_mean=torch.stack(dynamic_counts).mean(), matched_distance=torch.stack(costs).mean(),
        reassignment_fraction=(selected != torch.arange(selected.shape[1], device=selected.device)).float().mean(),
        original_source_latent_mse=(recovered['pred_latents']-latents).square().mean())
    return assigned, latents[images, selected], metadata


def distance_correlation(left, right):
    left = left-left.mean()
    right = right-right.mean()
    return (left*right).sum()/(left.square().sum()*right.square().sum()).sqrt().clamp_min(1e-8)


class LatentMetricAlignment(nn.Module):
    def __init__(self, *, position_scale=.25, size_scale=.25, weights=(1., 1., 4.)):
        super().__init__()
        self.position_scale, self.size_scale = position_scale, size_scale
        self.weights = weights
        self.raw_scale = nn.Parameter(torch.full((3,), math.log(math.expm1(1.))))
        self.bias = nn.Parameter(torch.zeros(3))

    def forward(self, latents, constructed):
        positive = constructed['positive']
        geometry = torch.where(positive[..., None], constructed['boxes'], constructed['input_boxes']).detach()
        rows, columns = torch.triu_indices(latents.shape[1], latents.shape[1], offset=1, device=latents.device)
        distances = (latents[:, rows]-latents[:, columns]).square().mean(-1)
        position_distance = (geometry[:, rows, :2]-geometry[:, columns, :2]).square().sum(-1)
        size_distance = (geometry[:, rows, 2:]-geometry[:, columns, 2:]).square().sum(-1)
        left_positive, right_positive = positive[:, rows], positive[:, columns]
        targets = torch.stack(((-position_distance/self.position_scale**2).exp(),
            (-size_distance/self.size_scale**2).exp(), (left_positive == right_positive).to(latents)), -1)
        logits = self.bias-functional.softplus(self.raw_scale)*distances[..., None]
        elements = functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        groups = dict(real_real=left_positive & right_positive,
            fake_fake=~left_positive & ~right_positive, mixed=left_positive != right_positive)
        image_losses = []
        for image in range(len(latents)):
            values = [elements[image, mask[image]].mean(0) for mask in groups.values() if mask[image].any()]
            image_losses.append(torch.stack(values).mean(0) if values else logits[image].sum(0)*0)
        components = torch.stack(image_losses).mean(0)
        loss = (components*latents.new_tensor(self.weights)).sum()
        metrics = {f'loss/{name}': value.detach() for name, value in zip(('position', 'size', 'existence'), components)}
        with torch.no_grad():
            for name, mask in groups.items():
                metrics[f'pairs/{name}'] = mask.sum().to(latents)/len(latents)
                metrics[f'distance/{name}'] = distances[mask].mean() if mask.any() else distances.new_zeros(())
                if name != 'mixed':
                    for property_name, actual in (('position', position_distance), ('size', size_distance)):
                        metrics[f'correlation/{name}/{property_name}'] = (distance_correlation(distances[mask], actual[mask])
                            if mask.any() else distances.new_zeros(()))
            for index, name in enumerate(('position', 'size', 'existence')):
                metrics[f'scale/{name}'] = functional.softplus(self.raw_scale[index]).detach()
                metrics[f'bias/{name}'] = self.bias[index].detach()
        return loss, metrics
