import json
from copy import deepcopy

import torch
from torch.nn import functional as F

from src.models.pipelines.pose_estimators.poseflow.balance_v51 import gradient_families
from src.models.pipelines.pose_estimators.poseflow.losses_v49_balanced import row_losses
from src.models.pipelines.pose_estimators.poseflow.proposals_v50 import residual_coordinates
from src.models.pipelines.pose_estimators.poseflow.rows_token_age import PERSON
from scripts.training.verify_poseflow_v49_balanced import training_batches


def optimizer_update(module, optimizer, samples):
    module.train()
    optimizer.zero_grad(set_to_none=True)
    records = []
    for sample in samples:
        losses = module._losses_for_batch(sample)
        if not torch.isfinite(losses['total']):
            raise FloatingPointError('Nonfinite V51 verification objective')
        (losses['total']/len(samples)).backward()
        records.append({name:float(value.detach()) for name,value in losses.items()})
    module.on_before_optimizer_step(optimizer)
    clipped = module.configure_gradient_clipping(optimizer,.1,'norm')
    norm = sum(float(value.square()) for name,value in clipped.items() if name.endswith('/preclip'))**.5
    optimizer.step()
    return {name:sum(record[name] for record in records)/len(records) for name in records[0]},norm


def verify_learning(module, data, output, *, steps=1000):
    trial = deepcopy(module).train()
    trial.model.activation_checkpointing = False
    trial.model.frontend.requires_grad_(False)
    batches = training_batches(data)
    batch = next(batches)
    batches.close()
    with torch.no_grad():
        context = trial.model.prepare_context(batch['image'],batch.get('padding_mask'))
        rows,sources,geometry,_ = trial.matched_rows(batch,context,generator=torch.Generator().manual_seed(420510))
    sizes = batch['img_hw'].flip(-1)
    person = rows['token_types'] == PERSON
    sigma = rows['noise_sigmas']
    truth = F.pad(sources['coordinates'],(0,0,0,rows['token_types'].shape[-1]-sources['coordinates'].shape[1]))[:,None]
    scaled = trial.model.scale*(2*truth-1)
    alpha = (1-sigma.square()).sqrt()
    valid = ~rows['eos_targets']
    optimizer = trial.configure_optimizers()['optimizer']
    train_rng = torch.Generator(device=sigma.device).manual_seed(420511)

    def objective(generator):
        noise = torch.randn(rows['noisy_coordinates'].shape,device=sigma.device,generator=generator)
        noisy = (alpha[...,None]*scaled+sigma[...,None]*noise).masked_fill(rows['padding_mask'][...,None],0.)
        inputs = dict(rows,noisy_coordinates=noisy)
        prediction = trial.model.forward_rows(context,**{name:inputs[name] for name in trial.model.row_input_fields})
        losses = row_losses(prediction,rows,sources,geometry,sizes,scales=torch.ones_like(trial.gradient_scales))
        losses['selection'] = F.cross_entropy(prediction['selection_logits'][valid],rows['selection_targets'][valid]) if valid.any() else prediction['selection_logits'].sum()*0
        losses['frontend'] = losses['selection']*0
        bypass = residual_coordinates(noisy,sigma.masked_fill(~person,0.),torch.zeros_like(noisy),scale=trial.model.scale)
        control = dict(prediction,pred_coordinates=torch.where(person[None,...,None],bypass[None],prediction['pred_coordinates']))
        with torch.no_grad():
            baseline = row_losses(control,rows,sources,geometry,sizes,scales=torch.ones_like(trial.gradient_scales))
        return losses,baseline

    def score():
        values=[]
        rng=torch.Generator(device=sigma.device).manual_seed(420512)
        with torch.no_grad():
            for _ in range(8):
                learned,bypass=objective(rng)
                values.append({f'{label}/{name}':float(record[name]) for label,record in (('learned',learned),('bypass',bypass))
                               for name in ('pose_refine','coordinate_joint_refine','box_refine')})
        return {name:sum(value[name] for value in values)/len(values) for name in values[0]}

    curve = [dict(step=0,**score())]
    for step in range(steps):
        losses,_ = objective(train_rng)
        if step % trial.config.gradient_probe_interval == 0:
            trial.balance.update(trial.measure_gradients(losses))
        losses = trial._apply_balance(losses)
        optimizer.zero_grad(set_to_none=True)
        losses['total'].backward()
        trial.on_before_optimizer_step(optimizer)
        trial.configure_gradient_clipping(optimizer,.1,'norm')
        optimizer.step()
        if step+1 in (200,600,steps):
            point=dict(step=step+1,**score())
            curve.append(point)
            print('V51_FRESH_NOISE',json.dumps(point),flush=True)
    final=curve[-1]
    passed=all(final[f'learned/{name}'] < .9*final[f'bypass/{name}'] for name in ('pose_refine','coordinate_joint_refine'))
    report=dict(passed=passed,steps=steps,image_ids=batch['image_id'].tolist(),people=int(sources['counts'].sum()),curve=curve,
        scope='Fixed augmented TRAIN images and sample-local identities; fresh training noise and8 independent evaluation draws; full row birth/refine/EOS/selection objectives; frozen native context/KV; not AP/generalization evidence.')
    (output/'fresh_noise_learning.json').write_text(json.dumps(report,indent=2)+'\n')
    if not passed:
        raise AssertionError('V51 independent-noise refinement did not beat bypass by10%; full launch blocked')
    return report
