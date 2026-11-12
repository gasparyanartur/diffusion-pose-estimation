from copy import deepcopy

import pytest
import torch

from src.models.pipelines.pose_autoencoder.model import geometry_loss
from src.models.pipelines.pose_estimators.poseflow.noise_v43 import (
    noise_coefficients, velocity_target, clean_from_velocity,
)
from src.models.pipelines.pose_estimators.poseflow.training_v43 import PoseFlowV43PipelineConfig
from tests.test_poseflow_v43 import model
from tests.test_poseflow_v43_training import batch, geometry_targets, assign_latent_sources


def cache_context(network, count=2):
    return (torch.randn(count, 9, network.hidden_dim), torch.randn(count, 9, network.hidden_dim),
            torch.zeros(count, 9, dtype=torch.bool)), dict(
        features=torch.randn(count, network.num_proposals, network.hidden_dim),
        boxes=torch.rand(count, network.num_proposals, 4),
        existence_logits=torch.randn(count, network.num_proposals))


def test_velocity_oracle_zero_noise_identity_and_endpoint_gradients():
    clean = torch.randn(3, 6, 8, requires_grad=True)
    noise = torch.randn_like(clean)
    sigma = torch.tensor([0., .01, 1000.])
    alpha, beta = noise_coefficients(sigma, clean)
    state = alpha*clean+beta*noise
    velocity = velocity_target(clean, noise, sigma)
    recovered = clean_from_velocity(state, velocity, sigma)
    torch.testing.assert_close(recovered, clean, rtol=1e-6, atol=1e-6)
    assert torch.equal(recovered[0], clean[0])
    recovered.square().sum().backward()
    assert torch.isfinite(clean.grad).all()
    for invalid in (torch.tensor([-1., 0., 1.]), torch.tensor([float('nan')]*3)):
        with pytest.raises(ValueError, match='sigma'):
            noise_coefficients(invalid, clean)


def test_continuous_sigma_v_head_identity_and_frozen_decoder_input_gradients():
    network = model()
    network.prediction_type = 'v'
    torch.nn.init.zeros_(network.latent_output.weight)
    torch.nn.init.zeros_(network.latent_output.bias)
    context, references = cache_context(network)
    clean = torch.randn(2, 6, 8, requires_grad=True)
    sizes = torch.tensor([[120., 80.], [90., 128.]])
    sigma = torch.tensor([0., .025])
    output = network.denoise_at_sigma(clean, sigma, context, references=references, image_wh=sizes)
    expected = clean/(1+sigma[:, None, None].square())
    torch.testing.assert_close(output['pred_latents'], expected, rtol=1e-6, atol=5e-7)
    assert torch.equal(output['pred_latents'][0], clean[0])
    decoded = network.codec.decode_latents(expected, sizes[:, None].expand(-1, 6, -1))
    torch.testing.assert_close(output['keypoints_px'], decoded['keypoints_px'])
    output['keypoints_px'].square().mean().backward()
    assert torch.isfinite(clean.grad).all() and clean.grad.abs().sum()>0
    assert network.latent_output.weight.grad.abs().sum()>0
    assert all(p.grad is None for p in network.codec.parameters())
    one = network.denoise_at_sigma(clean[:1], torch.zeros(1), tuple(v[:1] for v in context),
        references={k:v[:1] for k,v in references.items()}, image_wh=sizes[:1])
    assert torch.equal(one['pred_latents'], clean[:1])


def test_center_floor_preserves_legacy_and_only_changes_thin_box_center_term():
    target = dict(keypoints_px=torch.zeros(1,14,2), joint_mask=torch.ones(1,14,dtype=torch.bool),
                  boxes_cxcywh_px=torch.tensor([[20.,30.,10.,40.]]))
    prediction = dict(keypoints_px=torch.ones(1,14,2), boxes_cxcywh_px=target['boxes_cxcywh_px']+torch.tensor([[1.,0.,0.,0.]]))
    original = geometry_loss(prediction,target)
    for floor in (0.,1.):
        current = geometry_loss(prediction,target,center_min_side_px=floor)
        for key in original:assert torch.equal(current[key],original[key])
    target['boxes_cxcywh_px'][:,2]=.02333798
    prediction['boxes_cxcywh_px']=target['boxes_cxcywh_px']+torch.tensor([[1.,0.,0.,0.]])
    current=geometry_loss(prediction,target,center_min_side_px=1.)
    assert current['box_center']==.5 and geometry_loss(prediction,target)['box_center']>900
    for key in ('pose','box_size','giou'):
        assert torch.equal(current[key],geometry_loss(prediction,target)[key])


def test_mean_source_codes_preserve_identity_and_do_not_consume_posterior_rng():
    network=model();geometry,targets=geometry_targets(batch((2,)))
    generator=torch.Generator().manual_seed(42);state=generator.get_state().clone()
    matches=[(torch.tensor([4,1]),torch.tensor([1,0]))]
    source=assign_latent_sources(network.codec,geometry,targets,matches,slots=6,
        generator=generator,posterior_mode='mean')
    mean,_=network.codec.encode_posterior(geometry)
    assert torch.equal(generator.get_state(),state)
    torch.testing.assert_close(source['latents'][0,4],network.codec.standardize(mean)[1])
    assert not source['latents'][~source['positive']].any()


def test_checkpoint_mode_metadata_rejects_semantic_head_mismatch():
    from tests.test_poseflow_v43_training import pipeline
    original=pipeline();record={};original.on_save_checkpoint(record)
    assert record['v43_prediction_type']=='x0'
    restored=pipeline();restored.on_load_checkpoint(record)
    with pytest.raises(ValueError,match='prediction'):
        restored.on_load_checkpoint(dict(record,v43_prediction_type='v'))
    legacy={k:v for k,v in record.items() if k!='v43_prediction_type'}
    restored.on_load_checkpoint(legacy)
    config=PoseFlowV43PipelineConfig(prediction_type='v',noise_policy='small_noise',posterior_mode='mean',center_min_side_px=1.)
    assert config.create_model().prediction_type=='v'


def test_cached_control_restore_replays_next_update_and_preserves_frozen_state():
    from src.models.pipelines.pose_estimators.poseflow.control_v43 import (
        configure_cached_model, pack_cached_batch, control_step, make_control_checkpoint,
        restore_control_checkpoint, frozen_state_digest, evaluate_cached,
    )
    network=model();network.prediction_type='v'
    torch.nn.init.zeros_(network.latent_output.weight);torch.nn.init.zeros_(network.latent_output.bias)
    context,references=cache_context(network)
    rows=[]
    for image in range(2):
        clean=torch.randn(6,8);positive=torch.tensor([True,False,True,False,False,True])
        clean[~positive]=0
        wh=torch.tensor([120.+image,80.])
        geometry=network.codec.decode_latents(clean,wh.expand(6,-1))
        rows.append(dict(image_id=image,memory=context[0][image,:7+image],position=context[1][image,:7+image],
            padding_mask=context[2][image,:7+image],references={k:v[image] for k,v in references.items()},
            clean=clean,positive=positive,image_wh=wh,target=dict(**geometry,joint_mask=positive[:,None].expand(-1,14))))
    packed=pack_cached_batch(rows,[0,1],torch.device('cpu'))
    assert packed['encoded_image'][0].shape==(2,8,32)
    assert packed['encoded_image'][2][0,-1] and not packed['encoded_image'][2][1,-1]
    parameters=configure_cached_model(network)
    optimizer=torch.optim.AdamW(parameters,lr=2e-4,weight_decay=1e-4)
    generator=torch.Generator().manual_seed(10042)
    frozen=frozen_state_digest(network)
    control_step(network,optimizer,rows,generator,batch_size=2,noise_levels=(.01,.025,.05))
    record=make_control_checkpoint(network,optimizer,generator,step=1,cache_sha256='cache',settings={})
    restored=model();restored.prediction_type='v'
    restored_optimizer=torch.optim.AdamW(configure_cached_model(restored),lr=2e-4,weight_decay=1e-4)
    restored_generator=torch.Generator()
    assert restore_control_checkpoint(record,restored,restored_optimizer,restored_generator,cache_sha256='cache')==1
    left=control_step(network,optimizer,rows,generator,batch_size=2,noise_levels=(.01,.025,.05))
    right=control_step(restored,restored_optimizer,rows,restored_generator,batch_size=2,noise_levels=(.01,.025,.05))
    assert left['sample']==right['sample'] and left['loss']==right['loss']
    for key,value in network.state_dict().items():
        torch.testing.assert_close(value,restored.state_dict()[key],rtol=0,atol=1e-7)
    assert frozen_state_digest(network)==frozen
    state=generator.get_state().clone()
    result=evaluate_cached(network,rows,sigmas=(0.,.01),seeds=(42,),batch_size=2)
    assert result['0.0']['max_clean_decode_difference_px']==0
    assert torch.equal(generator.get_state(),state)
    with pytest.raises(ValueError,match='prediction'):
        restore_control_checkpoint(dict(record,prediction_type='x0'),restored,restored_optimizer,
            restored_generator,cache_sha256='cache')


def test_small_noise_pipeline_supervises_velocity_and_keeps_mean_posterior_rng_fixed():
    from tests.test_poseflow_v43_training import pipeline
    from src.models.pipelines.pose_estimators.poseflow.training_v43 import PoseFlowV43Pipeline
    original=pipeline()
    config=original.config.model_copy(update=dict(prediction_type='v',noise_policy='small_noise',
        posterior_mode='mean',center_min_side_px=1.))
    network=config.create_model()
    network.codec.load_state_dict(original.model.codec.state_dict())
    module=PoseFlowV43Pipeline(model=network,config=config,max_epochs=2).eval()
    posterior=module.posterior_rng_state.clone()
    losses=module._losses_for_batch(batch((2,1)))
    assert torch.equal(posterior,module.posterior_rng_state)
    assert losses['velocity/total']>.1 and losses['latent/total']<.01
    assert losses['debug/noise_sigma_min']>=.01 and losses['debug/noise_sigma_max']<=.05
    torch.testing.assert_close(losses['loss_latent'],losses['velocity/total'])
    losses['total'].backward()
    assert network.latent_output.weight.grad.abs().sum()>0
    assert all(p.grad is None for p in network.codec.parameters())
