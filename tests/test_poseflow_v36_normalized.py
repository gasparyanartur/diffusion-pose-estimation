from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from tests.test_poseflow_v35 import batch
from tests.test_poseflow_v36 import pipeline


def normalized_pipeline():
    return pipeline(clean_box_input_format='normalized')


def test_normalized_input_and_resize_invariance_without_double_scaling():
    module = normalized_pipeline()
    model = module.model
    boxes = torch.tensor([.3, .4, .2, .1]).expand(1, 6, 4).clone()
    actual = model.encode_clean(boxes)['pred_latents']
    expected = model.clean_norm(model.input_mlp(boxes))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    pixels = boxes*torch.tensor([384., 256., 384., 256.])
    resized_pixels = pixels*2
    torch.testing.assert_close(model.encode_clean(pixels/torch.tensor([384., 256., 384., 256.]))['pred_latents'],
        model.encode_clean(resized_pixels/torch.tensor([768., 512., 768., 512.]))['pred_latents'],atol=0,rtol=0)
    same_pixels = torch.tensor([128., 128., 64., 64.]).expand(1, 6, 4)
    landscape = model.encode_clean(same_pixels/torch.tensor([384., 256., 384., 256.]))['pred_latents']
    portrait = model.encode_clean(same_pixels/torch.tensor([256., 384., 256., 384.]))['pred_latents']
    assert not torch.equal(landscape, portrait)


def test_pipeline_selects_unmasked_input_boxes_and_detached_routes():
    module = normalized_pipeline()
    data = batch()
    targets,times,noise = module._draw_training_inputs(data)
    captured = {}
    original = module.model.training_outputs
    def capture(images, coordinates, **kwargs):
        captured['coordinates'] = coordinates
        result = original(images,coordinates,**kwargs)
        assert not result[2].requires_grad
        return result
    module.model.training_outputs = capture
    module._training_outputs(data,targets,times,noise)
    torch.testing.assert_close(captured['coordinates'],targets['input_boxes'],atol=0,rtol=0)
    assert (captured['coordinates'][~targets['positive']][:,2:] > 0).all()
    losses=module._losses_for_batch(data)
    assert not losses['denoised/total'].requires_grad
    losses['loss_latent'].backward()
    for component in (module.model.input_mlp,module.model.box_head,module.model.existence_head):
        assert all(p.grad is None for p in component.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.model.backbone.parameters())


def test_config_and_initialization_only_input_format_changes():
    with initialize_config_dir(config_dir=str(Path('conf').resolve()),version_base=None):
        old=OmegaConf.to_container(compose(config_name='benchmark_poseflow_v36_box'),resolve=True)
        new=OmegaConf.to_container(compose(config_name='benchmark_poseflow_v36_box_normalized'),resolve=True)
    old['pipeline']['clean_box_input_format']='normalized'
    old.update(run_name='poseflow_v36_box_normalized',out_dir='results/poseflow_v36_box_normalized')
    assert old == new
    torch.manual_seed(42); baseline=pipeline()
    torch.manual_seed(42); normalized=normalized_pipeline()
    for name,value in baseline.state_dict().items():
        torch.testing.assert_close(value,normalized.state_dict()[name],atol=0,rtol=0)
    restored=normalized.v26_config.build(max_epochs=2)
    restored.load_state_dict(normalized.state_dict(),strict=True)
    data=batch();normalized._draw_training_inputs(data)
    restored.load_state_dict(normalized.state_dict(),strict=True)
    left=normalized._draw_training_inputs(data);right=restored._draw_training_inputs(data)
    for a,b in zip(left,right):
        if isinstance(a,dict):
            for key in a:torch.testing.assert_close(a[key],b[key],atol=0,rtol=0)
        else:torch.testing.assert_close(a,b,atol=0,rtol=0)
