import torch


@torch.no_grad()
def build_v29_ota_targets(clean, recovered, constructed, matcher):
    latents = clean['pred_latents'].detach()  # <B, P, D>
    batch_size, proposals, _ = latents.shape
    people = constructed['unique_people'].tolist()
    unique_targets = []
    for image, count in enumerate(people):
        boxes = constructed['boxes'][image, :count]
        unique_targets.append(dict(
            labels=torch.ones(count, dtype=torch.long, device=latents.device),
            boxes=boxes,
            keypoints=torch.cat((constructed['joints'][image, :count].flatten(-2),
                                 constructed['labeled'][image, :count].float()), -1),
            area=boxes[:, 2:].prod(-1).clamp_min(1e-6)))
    matches = matcher(recovered, unique_targets)
    assigned = {name:torch.zeros_like(constructed[name])
                for name in ('positive', 'joints', 'boxes', 'labeled')}
    assigned['target_indices'] = torch.full_like(constructed['target_indices'], -1)
    latent_indices = torch.full((batch_size, proposals), -1, dtype=torch.long, device=latents.device)
    matched_person = torch.full_like(latent_indices, -1)
    bank_used, bank_size = 0, 0
    for image, (queries, targets) in enumerate(matches):
        queries, targets = queries.to(latents.device), targets.to(latents.device)
        assigned['positive'][image, queries] = True
        matched_person[image, queries] = targets
        latent_indices[image, queries] = targets
        for name in ('joints', 'boxes', 'labeled', 'target_indices'):
            assigned[name][image, queries] = constructed[name][image, targets]
        unmatched = (~assigned['positive'][image]).nonzero().flatten()
        negative_bank = (~constructed['positive'][image]).nonzero().flatten()
        bank_size += len(negative_bank)
        if len(unmatched):
            if not len(negative_bank):
                raise ValueError('V29 unmatched queries require a clean negative latent bank')
            distance = torch.cdist(recovered['pred_latents'][image, unmatched],
                                   latents[image, negative_bank]).square()
            selected = negative_bank[distance.argmin(-1)]
            latent_indices[image, unmatched] = selected
            bank_used += selected.unique().numel()
    if (latent_indices < 0).any():
        raise ValueError('V29 requires a detached latent target for every query')
    latent_targets = latents.gather(1, latent_indices[..., None].expand_as(latents))
    source_positive = constructed['positive']
    changed = source_positive & (assigned['target_indices'] != constructed['target_indices'])
    metadata = dict(latent_indices=latent_indices, matched_person=matched_person,
        assigned_positive=assigned['positive'].sum(-1).float().mean(),
        unmatched=(~assigned['positive']).sum(-1).float().mean(),
        unique_gt_coverage=latents.new_tensor(sum(t.unique().numel() for _, t in matches)/max(sum(people), 1)),
        dynamic_k_mean=latents.new_tensor(sum(len(q) for q, _ in matches)/max(sum(people), 1)),
        source_positive_reassigned=changed.sum()/source_positive.sum().clamp_min(1),
        source_negative_to_positive=(~source_positive & assigned['positive']).sum(-1).float().mean(),
        negative_bank_utilization=latents.new_tensor(bank_used/max(bank_size, 1)))
    return assigned, latent_targets, metadata
