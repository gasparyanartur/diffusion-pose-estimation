from copy import deepcopy

import torch

from src.models.pipelines.pose_estimators.poseflow.model_v51 import PoseFlowV51Model
from src.models.pipelines.pose_estimators.poseflow.proposals_v51 import clean_anchor_residual
from tests.test_poseflow_v50 import model, context, inputs


def instance():
    from src.models.pipelines.pose_estimators.diffusionregpose.model import DiffusionRegPoseModel
    from tests.test_poseflow_v44 import arguments
    return PoseFlowV51Model(native=DiffusionRegPoseModel(**arguments()),rows=7,capacity=6,
        hidden_dim=32,num_heads=4,feedforward_dim=64,codec_width=32,latent_dim=8)


def test_classifier_reads_prefix_without_training_geometry():
    network = instance().train()
    cached = context(network)
    output = network.forward_rows(cached, **inputs())
    loss = output['selection_logits'].square().mean()+output['eos_logits'].square().mean()
    loss.backward()
    assert network.selector.key.weight.grad.abs().sum() > 0
    for component in (network.person_codec.encoder,network.latent_input,network.reference_input):
        assert all(parameter.grad is None or not parameter.grad.any() for parameter in component.parameters())


def test_time_is_bounded_and_clean_anchor_has_an_analytic_coordinate_path():
    network = instance()
    with torch.no_grad():
        network.time_embedding[-1].weight.fill_(100.)
        network.time_embedding[-1].bias.fill_(1000.)
    sigma = torch.tensor([[[.1,.05,0.]]])
    time = network._time_features(sigma)
    assert time.square().mean(-1).sqrt().max() <= 1.00001
    noisy = torch.randn(1,1,3,32,requires_grad=True)
    anchor = torch.rand_like(noisy)
    anchor[...,30:] += .2
    output = clean_anchor_residual(noisy,sigma,anchor,scale=6.)
    base = (noisy/6.+1)/2
    torch.testing.assert_close(output[...,2,:],base[...,2,:],rtol=0,atol=0)
    gradient = torch.autograd.grad(output[...,:30].sum(),noisy)[0]
    torch.testing.assert_close(gradient[...,:30],torch.tensor([0.,.5/(1-.05**2)**.5,1.])[None,None,:,None].expand(1,1,3,30)/12.)


def test_sampling_identity_determinism_and_zero_time_finiteness():
    network=instance().eval()
    with torch.no_grad():
        for parameter in network.time_embedding.parameters(): parameter.zero_()
    time=network._time_features(torch.zeros(1,1,2))
    time.sum().backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in network.time_embedding.parameters() if parameter.grad is not None)
    cached=context(network,1)
    with torch.no_grad(): network.selector.eos_bias.fill_(-1e6)
    a=network.predict(cached,torch.tensor([[80.,64.]]),generator=torch.Generator().manual_seed(42),collect_trace=True)
    b=network.predict(cached,torch.tensor([[80.,64.]]),generator=torch.Generator().manual_seed(42))
    assert a['person_counts'].tolist()==[6] and a['proposal_indices'][0].unique().numel()==6
    torch.testing.assert_close(a['pred_coordinates'],b['pred_coordinates'],rtol=0,atol=0)
