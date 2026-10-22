import pytest
import torch

from tests.test_poseflow import _batch
from tests.test_poseflow_v21 import configuration


@pytest.mark.parametrize('dn_number', [0, 2])
def test_current_criterion_reaches_exact_returned_hypothesis(dn_number):
    torch.manual_seed(42)
    pipeline = configuration(dn_number=dn_number, hypothesis_update_order='h_then_g').build(max_epochs=2).train()
    captured = []
    hook = pipeline.model.register_forward_hook(lambda module, args, output: captured.append(output))
    losses = pipeline._losses_for_batch(_batch() | {'image': torch.randn(1, 3, 64, 64)})
    hook.remove()
    gradient = torch.autograd.grad(losses['total'], captured[0]['hypothesis_memory'], allow_unused=True)[0]
    assert gradient is not None, 'The current pose loss must read the H returned for carry'
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_geometry_reads_fully_updated_h_before_prediction():
    model = configuration(dn_number=0, hypothesis_update_order='h_then_g')._build_poseflow_model().eval()
    image = torch.randn(1, 3, 64, 64)
    inputs = model.prepare_transformer_inputs(image, torch.zeros(1, 64, 64, dtype=torch.bool))
    memory = model.encode_features(inputs, torch.tensor([.3]))
    geometry, hypotheses = torch.randn(1, 3, 32), torch.randn(1, 3, 32)
    references = torch.rand(1, 3, 4)*.5+.25
    layer = model.transformer.decoder.layers[0]
    calls = []
    hook = layer.self_attn.register_forward_pre_hook(lambda module, args: calls.append(tuple(t.detach().clone() for t in args[:3])))
    mask = model.hypothesis_attention_mask(batch_size=1, dn_count=0, expanded=False, device=image.device)
    _, returned = model.refine_with_hypotheses(layer, geometry, references, hypotheses, inputs, memory, mask)
    hook.remove()
    assert len(calls) == 2
    torch.testing.assert_close(calls[0][0], hypotheses.transpose(0, 1), atol=0, rtol=0)
    for keys_or_values in calls[1][1:]:
        torch.testing.assert_close(keys_or_values[-3:], returned.transpose(0, 1), atol=0, rtol=0)


def test_legacy_order_default_and_no_checkpoint_parameter_changes():
    legacy = configuration()
    assert legacy.hypothesis_update_order == 'parallel'
    with pytest.raises(ValueError):
        configuration(hypothesis_update_order='unknown')
    old_model = legacy._build_poseflow_model()
    new_model = configuration(hypothesis_update_order='h_then_g')._build_poseflow_model()
    new_model.load_state_dict(old_model.state_dict(), strict=True)
    assert new_model.state_dict().keys() == old_model.state_dict().keys()
