import pytest
import torch
from tests.test_poseflow import _batch
from src.models.pipelines.pose_estimators.poseflow.latent_v12 import V12StateCodec, build_v12_targets
from src.models.pipelines.pose_estimators.poseflow.losses_v13 import matched_layer_losses
from src.models.pipelines.pose_estimators.diffusionregpose.training import DiffusionRegPoseMatcher, DiffusionRegPoseLossWeights


def test_matched_losses_follow_prediction_not_corruption_slot_and_are_permutation_invariant():
    torch.manual_seed(42)
    codec = V12StateCodec(width=32, joints=2)
    targets = build_v12_targets(_batch(), groups=3)
    clean = codec.encode(targets)
    matcher = DiffusionRegPoseMatcher(num_body_points=2)
    weights = DiffusionRegPoseLossWeights()
    # Exact geometry gives an unambiguous matching, regardless of corruption source.
    output = {'pred_latents': clean.detach().clone(),
              'pred_logits': (targets['existence'] * 2 - 1) * 8,
              'pred_boxes': targets['boxes'].clone(),
              'pred_keypoints': torch.cat((targets['joints'].flatten(-2), targets['labeled'].float()), -1)}
    original = matched_layer_losses(output, targets, clean, codec, matcher, weights)
    order = torch.tensor([2,0,1])
    changed = {key:value[:,order] for key,value in output.items()}
    permuted = matched_layer_losses(changed, targets, clean, codec, matcher, weights)
    for name in original:
        torch.testing.assert_close(original[name], permuted[name])
    assert original['loss_latent'].abs() < 1e-6
    target_order = {key: value[:,order] if value.ndim >= 2 else value for key,value in targets.items()}
    reordered = matched_layer_losses(output, target_order, clean[:,order], codec, matcher, weights)
    for name in original:
        torch.testing.assert_close(original[name], reordered[name])
    assert permuted['loss_keypoints'] == 0


def test_unmatched_geometry_is_ignored_but_existence_has_gradients_and_empty_images_are_finite():
    codec = V12StateCodec(width=32, joints=2)
    targets = build_v12_targets(_batch(), groups=3)
    clean = codec.encode(targets)
    matcher = DiffusionRegPoseMatcher(num_body_points=2)
    output = codec.decode(clean.detach()) | {'pred_latents':clean.detach().clone().requires_grad_()}
    # No real people: all slots are unmatched, only existence latent is supervised.
    empty = {**targets, 'real': torch.zeros_like(targets['real']), 'existence':torch.zeros_like(targets['existence'])}
    first = matched_layer_losses(output, empty, clean, codec, matcher, DiffusionRegPoseLossWeights())
    changed = {**output, 'pred_latents':output['pred_latents'].detach().clone(),
               'pred_boxes':torch.rand_like(output['pred_boxes']), 'pred_keypoints':torch.rand_like(output['pred_keypoints'])}
    changed['pred_latents'][:,:,1:] += 100
    second = matched_layer_losses(changed, empty, clean, codec, matcher, DiffusionRegPoseLossWeights())
    for name in first:
        assert torch.isfinite(first[name])
        torch.testing.assert_close(first[name],second[name])
    first['loss_latent'].backward()
    assert output['pred_latents'].grad[:,:,0].abs().sum() > 0
    assert output['pred_latents'].grad[:,:,1:].abs().sum() == 0
    assert codec.existence_encoder.layers[0].weight.grad is None
    assert first['loss_bbox'] == first['loss_keypoints'] == 0


def _config(**overrides):
    from src.models.pipelines.pose_estimators.poseflow import PoseFlowV13PipelineConfig
    from tests.test_poseflow_v12 import _config as v12_config
    return PoseFlowV13PipelineConfig(**(v12_config().model_dump() | {
        'matching_strategy':'hungarian', 'quick_eval_enabled':False,
    } | overrides))


def test_v13_pipeline_layer_matching_gradients_logs_and_strict_restore(tmp_path):
    from omegaconf import OmegaConf
    from scripts.evaluation.evaluate_poseflow import load_poseflow_model
    from src.models.pipelines.pose_estimators.poseflow.model_v13 import PoseFlowV13Model
    pipeline = _config().build(max_epochs=2)
    assert isinstance(pipeline.model, PoseFlowV13Model)
    calls = []
    original = pipeline.matcher.forward
    def capture(*args, **kwargs):
        matches = original(*args, **kwargs)
        calls.append(matches)
        return matches
    pipeline.matcher.forward = capture
    batch = _batch() | {'image':torch.randn(1,3,64,64)}
    losses = pipeline._losses_for_batch(batch)
    assert len(calls) == 2
    torch.testing.assert_close(losses['total'],losses['loss_latent_mean']+
                               losses['loss_prediction']+losses['loss_reconstruction'])
    torch.testing.assert_close(losses['loss_latent'],losses['loss_latent_1'])
    torch.testing.assert_close(losses['loss_keypoints'], losses['loss_keypoints_1'])
    torch.testing.assert_close(losses['loss_keypoints_mean'],
                               (losses['loss_keypoints_0']+losses['loss_keypoints_1'])/2)
    losses['loss_latent_mean'].backward(retain_graph=True)
    assert pipeline.model.codec.joint_encoder.layers[0].weight.grad is None
    assert pipeline.model.layers[0].latent_head.weight.grad.abs().sum() > 0
    pipeline.zero_grad(set_to_none=True)
    losses['total'].backward()
    assert pipeline.model.codec.joint_encoder.layers[0].weight.grad.abs().sum() > 0
    pipeline.eval()
    torch.save({'state_dict':pipeline.state_dict()},tmp_path/'checkpoint.ckpt')
    OmegaConf.save(OmegaConf.create({'pipeline':{'_target_':
        'src.models.pipelines.pose_estimators.poseflow.PoseFlowV13PipelineConfig',
        **pipeline.v12_config.model_dump(mode='json')}}),tmp_path/'config.yaml')
    loaded, config = load_poseflow_model(tmp_path/'checkpoint.ckpt',tmp_path/'config.yaml')
    loaded.eval()
    one = pipeline.model.sample_keypoints(batch['image'],generator=torch.Generator().manual_seed(42))
    two = loaded.sample_keypoints(batch['image'],generator=torch.Generator().manual_seed(42))
    torch.testing.assert_close(one['latent_state'],two['latent_state'],rtol=0,atol=0)
    assert config.matching_strategy == 'hungarian'


def test_v13_sampler_explicit_start_cached_memory_and_readout_only_states():
    from src.models.pipelines.pose_estimators.poseflow.model_v12 import PoseFlowV12Model
    model = _config()._build_poseflow_model().eval()
    image = torch.randn(1,3,64,64)
    generator = torch.Generator().manual_seed(42)
    expected = PoseFlowV12Model.sample_keypoints(model,image,generator=generator)
    actual = model.sample_keypoints(image,generator=torch.Generator().manual_seed(42))
    torch.testing.assert_close(expected['latent_state'],actual['latent_state'],rtol=0,atol=0)
    memory = model.encode_image(image)
    state = torch.randn(1,3,5,32)
    original = model.sample_keypoints(image,initial_latents=state,start_time=.6,
        sampling_steps=2,encoded_image=memory,collect_step_diagnostics=True)
    assert [round(row['time'],3) for row in original['latent_step_diagnostics']] == [.6,.3]
    before = state.clone()
    with torch.no_grad():
        for parameter in model.codec.point_head.parameters():parameter.add_(1)
    changed = model.sample_keypoints(image,initial_latents=state,start_time=.6,
        sampling_steps=2,encoded_image=memory,collect_step_diagnostics=True)
    torch.testing.assert_close(original['latent_state'],changed['latent_state'],rtol=0,atol=0)
    torch.testing.assert_close(before,state,rtol=0,atol=0)
    with pytest.raises(ValueError):model.sample_keypoints(image,start_time=0)


def test_v13_preset_and_quick_eval_validation():
    from hydra import compose, initialize_config_dir
    from pathlib import Path
    from src.models.pipelines.pose_estimators.poseflow import PoseFlowPipelineConfig
    with initialize_config_dir(config_dir=str(Path('conf').resolve()),version_base=None):
        config=compose(config_name='benchmark_poseflow_v13')
    parsed=PoseFlowPipelineConfig.from_hydra_config(config.pipeline)
    assert parsed.matching_strategy == 'hungarian'
    assert parsed.quick_eval_images == 32 and parsed.quick_eval_enabled
    assert parsed.quick_eval_noise_fractions == [.25,.5,1.]
    assert parsed.quick_eval_steps == [1,2,3]
    assert config.out_dir == 'results/poseflow_v13_training'
    with pytest.raises(ValueError):_config(quick_eval_noise_fractions=[1.1])


def test_quick_eval_manifest_and_noise_time_round_trip(tmp_path):
    from src.models.pipelines.pose_estimators.poseflow.quick_eval_v13 import select_test_images, noise_fraction_time
    from src.models.pipelines.pose_estimators.poseflow.latent_v12 import cosine_alpha
    path=tmp_path/'subset.json'
    selected=select_test_images(list(range(100)),path,count=32,seed=42,annotations='test.json')
    assert len(selected)==32 and selected==sorted(selected)
    assert select_test_images(list(reversed(range(100))),path,count=32,seed=42,annotations='test.json')==selected
    for fraction in [.25,.5,1.]:
        time=noise_fraction_time(fraction)
        torch.testing.assert_close(cosine_alpha(torch.tensor([time])),torch.tensor([1-fraction]),atol=1e-6,rtol=1e-6)
    assert noise_fraction_time(1)==1
    with pytest.raises(ValueError):select_test_images(list(range(100)),path,count=32,seed=7,annotations='test.json')


def test_quick_eval_nine_conditions_share_starts_cache_image_and_restore_rng_mode(tmp_path):
    from src.models.pipelines.pose_estimators.poseflow.quick_eval_v13 import evaluate_quick_grid
    model=_config()._build_poseflow_model().train()
    batch=_batch() | {'image':torch.randn(1,3,64,64),'image_id':torch.tensor([7]),
        'img_hw':torch.tensor([[64.,64.]]),'person_mask':torch.ones(1,2,dtype=torch.bool)}
    boxes=batch['gt_boxes_canon']
    batch['gt_boxes_pixel']=torch.cat((boxes[...,:2]-boxes[...,2:]/2,boxes[...,2:]),-1)*64
    starts=[]; encodings=[]
    original_sample=model.sample_keypoints
    def capture(image,**kwargs):
        starts.append((kwargs['start_time'],kwargs['sampling_steps'],kwargs['initial_latents'].clone()))
        return original_sample(image,**kwargs)
    original_encode=model.encode_image
    def encode(*args,**kwargs):
        encodings.append(1)
        return original_encode(*args,**kwargs)
    model.sample_keypoints=capture
    model.encode_image=encode
    rng=torch.get_rng_state().clone()
    result=evaluate_quick_grid(model,[batch],_config(),tmp_path,'test.json',[7],
        metric_evaluator=lambda records,annotations,ids:{'pose_AP':.1,'pose_AP50':.2,'pose_AP75':.05,'pose_AR':.3,'bbox_AP':.1})
    assert model.training and torch.equal(torch.get_rng_state(),rng)
    assert len(starts)==9 and len(encodings)==1 and len(result['conditions'])==9
    for offset in [0,3,6]:
        for entry in starts[offset:offset+3]:torch.testing.assert_close(entry[2],starts[offset][2],rtol=0,atol=0)
    expected=torch.randn(starts[6][2].shape,generator=torch.Generator().manual_seed(49))
    torch.testing.assert_close(starts[6][2],expected,rtol=0,atol=0)
    assert result['gt_people']==2
    assert all('oracle_oks' in row['metrics'] for row in result['conditions'].values())


def test_each_decoder_layer_can_choose_a_different_permutation():
    pipeline=_config().build(max_epochs=2)
    targets=build_v12_targets(_batch(),groups=3)
    clean=pipeline.model.codec.encode(targets)
    output={'pred_latents':clean.detach(), 'pred_logits':(targets['existence']*2-1)*8,
        'pred_boxes':targets['boxes'],
        'pred_keypoints':torch.cat((targets['joints'].flatten(-2),targets['labeled'].float()),-1)}
    order=torch.tensor([1,2,0])
    first={key:value[:,order] for key,value in output.items()}
    combined=output | {'aux_outputs':[first]}
    losses=pipeline._denoising_losses(combined,targets,clean,pipeline.v12_config.weights.to_dataclass())
    assert losses['loss_latent_0'].abs()<1e-6 and losses['loss_latent_1'].abs()<1e-6
    assert losses['loss_keypoints_0']==losses['loss_keypoints_1']==0


def test_quick_eval_failure_restores_all_module_modes_and_python_numpy_rng():
    import random
    import numpy as np
    from src.models.pipelines.pose_estimators.poseflow.quick_eval_v13 import evaluation_context
    model=_config()._build_poseflow_model().train()
    model.backbone.eval()
    modes=[module.training for module in model.modules()]
    torch_state=torch.get_rng_state().clone()
    python_state=random.getstate()
    numpy_state=np.random.get_state()
    with pytest.raises(RuntimeError,match='probe'):
        with evaluation_context(model):
            torch.randn(2);random.random();np.random.random()
            assert not model.training
            raise RuntimeError('probe')
    assert [module.training for module in model.modules()]==modes
    assert torch.equal(torch_state,torch.get_rng_state())
    assert random.getstate()==python_state
    actual=np.random.get_state()
    assert actual[0]==numpy_state[0] and np.array_equal(actual[1],numpy_state[1])
    assert actual[2:]==numpy_state[2:]


def test_quick_metrics_keep_missed_images_in_denominator_and_handle_empty_predictions():
    import json
    from pathlib import Path
    from src.models.pipelines.pose_estimators.poseflow.quick_eval_v13 import evaluate_records
    annotation_path=Path('tests/fixtures/mini_crowdpose_val.json')
    annotations=json.loads(annotation_path.read_text())
    image_ids=sorted(image['id'] for image in annotations['images'])
    first=image_ids[0]
    records=[{'image_id':first,'category_id':1,'bbox':person['bbox'],
              'keypoints':person['keypoints'],'score':1.}
             for person in annotations['annotations'] if person['image_id']==first]
    one=evaluate_records(records,annotation_path,[first])
    full=evaluate_records(records,annotation_path,image_ids)
    assert full['pose_AR']<one['pose_AR']
    assert full['pose_AP']<one['pose_AP']
    empty=evaluate_records([],annotation_path,image_ids)
    assert empty['pose_AP']==empty['bbox_AP']==0


def test_quick_hook_runs_once_each_epoch_and_skips_sanity(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from src.data.datasets import PoseDataset
    from src.models.pipelines.pose_estimators.poseflow import quick_eval_v13
    pipeline=_config(quick_eval_enabled=True).build(max_epochs=2)
    data=SimpleNamespace(test_ann_path='test.json',img_dir='images',num_keypoints=2,
                         eval_resize_scale=256,eval_max_size=427)
    trainer=SimpleNamespace(current_epoch=0,global_step=12,max_epochs=2,sanity_checking=True,
        is_global_zero=True,gradient_clip_val=.1,datamodule=data,
        checkpoint_callback=SimpleNamespace(dirpath=tmp_path/'checkpoints'))
    pipeline._trainer=trainer
    calls=[]
    monkeypatch.setattr(PoseDataset,'from_path',lambda *args,**kwargs:SimpleNamespace(samples=[{'image_id':7}]))
    def evaluate(*args,**kwargs):
        (tmp_path/'quick_test'/f'epoch{trainer.current_epoch:03d}').mkdir(parents=True)
        calls.append(1)
        return {'conditions':{'noise_var_100/steps_3':{'metrics':{'pose_AP':.1}}}}
    monkeypatch.setattr(quick_eval_v13,'evaluate_quick_grid',evaluate)
    monkeypatch.setattr(pipeline,'log',lambda *args,**kwargs:None)
    pipeline.on_validation_epoch_end()
    assert calls==[]
    trainer.sanity_checking=False
    pipeline.on_validation_epoch_end()
    pipeline.on_validation_epoch_end()
    assert len(calls)==1
    trainer.current_epoch=1
    pipeline.on_validation_epoch_end()
    assert len(calls)==2


@pytest.mark.parametrize('empty',[False,True])
def test_v13_classification_uses_person_normalizer_and_weighted_total_once(empty):
    import math
    from src.models.pipelines.pose_estimators.poseflow.latent_v12 import prediction_loss
    codec=V12StateCodec(width=32,joints=2)
    targets=build_v12_targets(_batch(),groups=3)
    if empty:
        targets['real'].zero_()
        targets['existence'].zero_()
    clean=codec.encode(targets)
    logits=torch.zeros(1,3,1,requires_grad=True)
    output=codec.decode(clean.detach()) | {'pred_logits':logits,'pred_latents':clean.detach()}
    weights=DiffusionRegPoseLossWeights()
    values=matched_layer_losses(output,targets,clean,codec,DiffusionRegPoseMatcher(num_body_points=2),weights)
    count=0 if empty else 2
    expected=(count*.25+(3-count)*.75)*.25*math.log(2)/max(count,1)
    assert values['loss_ce'].item()==pytest.approx(expected)
    values['loss_ce'].backward()
    derivative = .125 + .25*math.log(2)
    expected_gradients = [derivative*weight/max(count,1) for weight in
                          ([.75,.75,.75] if empty else [.25,.25,.75])]
    torch.testing.assert_close(logits.grad.abs().flatten().sort().values,
                               torch.tensor(sorted(expected_gradients)))
    coefficients={'loss_ce':weights.cls,'loss_bbox':weights.bbox,'loss_giou':weights.giou,
                  'loss_keypoints':weights.keypoints,'loss_oks':weights.oks}
    torch.testing.assert_close(values['total'],sum(coefficient*values[name] for name,coefficient in coefficients.items()))
    # V12/reconstruction keeps its original slot-mean reduction.
    old=prediction_loss(output,targets,weights=weights)
    torch.testing.assert_close(values['loss_ce'],old['loss_ce']*3/max(count,1))
