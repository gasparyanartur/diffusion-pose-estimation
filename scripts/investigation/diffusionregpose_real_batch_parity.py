#!/usr/bin/env python3
"""Compare one real CrowdPose training batch with the external implementation."""

from __future__ import annotations

import argparse
import sys
import types
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn.functional as functional
from torchvision.transforms import InterpolationMode

ROOT = Path(__file__).resolve().parents[2]
EXTERNAL = ROOT / "external" / "DiffusionRegPose"
CHECKPOINT = EXTERNAL / "logs" / "crowdpose_r50_effective_batch8_40epoch" / "checkpoint.pth"


def install_deformable_attention_fallback() -> None:
    operator = types.ModuleType("MultiScaleDeformableAttention")

    def forward(value, spatial_shapes, level_start_index, sampling_locations, attention_weights, _step):
        batch_size, _, num_heads, head_dim = value.shape
        _, query_count, _, level_count, point_count, _ = sampling_locations.shape
        values = value.split(tuple(height * width for height, width in spatial_shapes.tolist()), dim=1)
        sampled = []
        for level, (height, width) in enumerate(spatial_shapes.tolist()):
            level_value = values[level].flatten(2).transpose(1, 2).reshape(
                batch_size * num_heads, head_dim, height, width
            )
            grid = (2.0 * sampling_locations[:, :, :, level] - 1.0).transpose(1, 2).flatten(0, 1)
            sampled.append(functional.grid_sample(
                level_value, grid, mode="bilinear", padding_mode="zeros", align_corners=False
            ))
        weights = attention_weights.transpose(1, 2).reshape(
            batch_size * num_heads, 1, query_count, level_count * point_count
        )
        output = (torch.stack(sampled, dim=-2).flatten(-2) * weights).sum(-1)
        return output.view(batch_size, num_heads * head_dim, query_count).transpose(1, 2).contiguous()

    operator.ms_deform_attn_forward = forward
    operator.ms_deform_attn_backward = lambda *args: (_ for _ in ()).throw(
        RuntimeError("backward is not supported by the parity probe")
    )
    sys.modules[operator.__name__] = operator


def load_external(query_count: int, group_count: int):
    sys.path.insert(0, str(EXTERNAL))
    sys.path.insert(0, str(EXTERNAL / "models/diffusionregpose/ops"))
    from models.diffusionregpose.backbones import backbone as reference_backbone
    from models.diffusionregpose.diffusionregpose import build_diffusionregpose
    from util.config import Config

    reference_backbone.is_main_process = lambda: False
    config = Config.fromfile(str(EXTERNAL / "config" / "diffusionregpose.cfg.py"))
    values = config._cfg_dict.to_dict()
    values.update(
        device="cuda",
        eval=False,
        num_queries=query_count,
        num_group=group_count,
        num_proposals=group_count,
        use_dn=True,
        dn_number=100,
    )
    model, criterion, _ = build_diffusionregpose(Namespace(**values))

    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state = {
        key: value for key, value in checkpoint["model"].items()
        if key not in {
            "betas", "alphas_cumprod", "alphas_cumprod_prev", "sqrt_alphas_cumprod",
            "sqrt_one_minus_alphas_cumprod", "log_one_minus_alphas_cumprod",
            "sqrt_recip_alphas_cumprod", "sqrt_recipm1_alphas_cumprod", "posterior_variance",
            "posterior_log_variance_clipped", "posterior_mean_coef1", "posterior_mean_coef2",
        }
    }
    model.load_state_dict(state, strict=True)
    return model.cuda().train(), Namespace(**values), criterion.cuda()


def make_real_batch(height: int, width: int, device: torch.device) -> tuple[dict, dict]:
    from src.data.datasets import PoseDataset, SetPoseDataset
    from src.data.data_module import _normalize_imagenet_image

    dataset = PoseDataset.from_path(
        ROOT / "data/crowdpose/annotations/crowdpose_train.json",
        img_root=ROOT / "data/crowdpose/images",
    )
    sample = next(item for item in dataset.samples if item["keypoints"].shape[0] > 0)
    item = SetPoseDataset(PoseDataset([sample]))[0]
    image = item["image"].float().div(255.0)
    original_height, original_width = image.shape[-2:]
    image = functional.interpolate(
        image[None], size=(height, width), mode="bilinear", align_corners=False
    )[0]
    keypoints_pixel = (item["gt_keypoints"] + 1.0) / 2.0
    keypoints_pixel = keypoints_pixel * torch.tensor(
        [original_width, original_height], dtype=torch.float32
    )
    keypoints_pixel = keypoints_pixel * torch.tensor(
        [width / original_width, height / original_height]
    )
    keypoints = keypoints_pixel / torch.tensor([width, height]) * 2.0 - 1.0
    boxes = item["gt_boxes_pixel"].clone()
    boxes = boxes * torch.tensor([
        width / original_width, height / original_height,
        width / original_width, height / original_height,
    ])
    x, y, box_width, box_height = boxes.unbind(-1)
    boxes_normalized = torch.stack(
        [(x + box_width / 2) / width, (y + box_height / 2) / height,
         box_width / width, box_height / height], dim=-1
    )
    batch = {
        "image": _normalize_imagenet_image(image)[None].to(device),
        "padding_mask": torch.zeros(1, height, width, dtype=torch.bool, device=device),
        "gt_keypoints": keypoints[None].to(device),
        "gt_diffusion_keypoints": keypoints[None].to(device),
        "gt_labeled": item["gt_labeled"][None].to(device),
        "gt_visible": item["gt_visible"][None].to(device),
        "person_mask": torch.ones(1, keypoints.shape[0], device=device),
        "gt_boxes_canon": boxes_normalized[None].to(device),
    }
    external_target = {
        "labels": torch.ones(keypoints.shape[0], dtype=torch.long, device=device),
        "boxes": boxes_normalized.to(device),
        "keypoints": torch.cat(
            ((keypoints_pixel / torch.tensor([width, height])).flatten(1), item["gt_labeled"]), dim=1
        ).to(device),
        "keypoints_buquan": torch.cat(
            ((keypoints_pixel / torch.tensor([width, height])).flatten(1), item["gt_labeled"]), dim=1
        ).to(device),
        "area": (boxes_normalized[:, 2] * boxes_normalized[:, 3]).to(device),
        "size": torch.tensor([height, width], device=device),
        "orig_size": torch.tensor([height, width], device=device),
        "iscrowd": torch.zeros(keypoints.shape[0], dtype=torch.long, device=device),
        "image_id": torch.tensor([sample["image_id"]], device=device),
    }
    return batch, external_target


def report(name: str, native: torch.Tensor, external: torch.Tensor) -> float:
    difference = (native.float() - external.float()).abs()
    maximum = difference.max().item()
    print(f"{name}: shape={tuple(native.shape)} max_abs={maximum:.6g} mean_abs={difference.mean().item():.6g}")
    return maximum


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--queries", type=int, default=32)
    parser.add_argument("--groups", type=int, default=8)
    args = parser.parse_args()
    device = torch.device("cuda")
    torch.manual_seed(123)

    from src.models.pipelines.pose_estimators.diffusionregpose.training import (
        DiffusionRegPosePipelineConfig, build_diffusionregpose_targets,
    )
    sys.path.insert(0, str(EXTERNAL))
    from util.misc import NestedTensor

    batch, external_target = make_real_batch(args.height, args.width, device)
    native_pipeline = DiffusionRegPosePipelineConfig(
        backbone_pretrained=False,
        pretrained_checkpoint=str(CHECKPOINT),
        num_queries=args.queries,
        num_proposals=args.groups,
        num_group=args.groups,
        sampling_timesteps=1,
    ).build(max_epochs=80).to(device).train()
    external_model, external_args, external_criterion = load_external(args.queries, args.groups)
    print("training_flags:", native_pipeline.model.training, external_model.training, external_model.transformer.training)
    external_parameters = dict(external_model.named_parameters())
    parameter_errors = []
    for name, parameter in native_pipeline.model.named_parameters():
        if name in external_parameters:
            parameter_errors.append((parameter - external_parameters[name]).abs().max().item())
    print(f"parameter_max_abs={max(parameter_errors):.6g}")
    native_features, _, _ = native_pipeline.model.backbone(
        batch["image"], batch["padding_mask"]
    )
    external_features, _ = external_model.backbone(
        NestedTensor(batch["image"], batch["padding_mask"])
    )
    for index, (native_feature, external_feature) in enumerate(zip(native_features, external_features)):
        report(f"train_backbone[{index}]", native_feature, external_feature.tensors)
    noise = torch.randn(1, args.groups, 28, device=device)
    native_diffusion, timesteps, native_targets = build_diffusionregpose_targets(
        batch, native_pipeline.model, num_proposals=args.groups, num_body_points=14, noise=noise
    )
    external_x = ((native_diffusion / native_pipeline.model.scale) + 1.0) / 2.0
    torch.manual_seed(456)
    external_label, external_box, _, external_mask, _ = external_model.prepare_for_dn2(
        [external_target]
    )
    torch.manual_seed(456)
    native_label, native_box, _ = native_pipeline.model._prepare_denoising_queries(
        native_targets, device
    )
    native_mask = native_pipeline.model.training_attention_mask(native_targets, 1, device)
    report("dn_label_input", native_label, external_label)
    report("dn_box_input", native_box, external_box)
    print("dn_mask_equal:", torch.equal(native_mask, external_mask))

    def deterministic_prepare(_args, _targets):
        return _targets, external_x, noise, timesteps[:, None]

    external_model.prepare_targets = deterministic_prepare
    diagnostics: dict[str, torch.Tensor] = {}
    native_handle = native_pipeline.model.transformer.encoder.register_forward_hook(
        lambda _module, _inputs, output: diagnostics.update(native_memory=output.detach())
    )
    external_handle = external_model.transformer.encoder.register_forward_hook(
        lambda _module, _inputs, output: diagnostics.update(external_memory=output[0].detach())
    )
    native_decoder_handle = native_pipeline.model.transformer.decoder.layers[0].register_forward_pre_hook(
        lambda _module, inputs, kwargs: diagnostics.update(
            native_decoder_tgt=inputs[0].detach(),
            native_decoder_ref=kwargs["tgt_reference_points"].detach(),
        ), with_kwargs=True,
    )
    external_decoder_handle = external_model.transformer.decoder.layers[0].register_forward_pre_hook(
        lambda _module, _inputs, kwargs: diagnostics.update(
            external_decoder_tgt=kwargs["tgt"].detach(),
            external_decoder_ref=kwargs["tgt_reference_points"].detach(),
        ), with_kwargs=True,
    )
    native_decoder_output_handle = native_pipeline.model.transformer.decoder.register_forward_hook(
        lambda _module, _inputs, output: diagnostics.update(native_decoder_output=output)
    )
    external_decoder_output_handle = external_model.transformer.decoder.register_forward_hook(
        lambda _module, _inputs, output: diagnostics.update(external_decoder_output=output)
    )
    native_decoder_input_handle = native_pipeline.model.transformer.decoder.register_forward_pre_hook(
        lambda _module, _inputs, kwargs: diagnostics.update(
            native_decoder_diffusion=kwargs["diffusion_keypoints"].detach()
        ), with_kwargs=True,
    )
    external_decoder_input_handle = external_model.transformer.decoder.register_forward_pre_hook(
        lambda _module, _inputs, kwargs: diagnostics.update(
            external_decoder_diffusion=kwargs["x_boxes"].detach()
        ), with_kwargs=True,
    )
    torch.manual_seed(456)
    native_output = native_pipeline.model(
        batch["image"], native_diffusion, timesteps, native_targets, batch["padding_mask"]
    )
    torch.manual_seed(456)
    external_output = external_model(
        external_args,
        NestedTensor(batch["image"], batch["padding_mask"]),
        [external_target],
    )

    def retain_output_gradients(output: dict, prefix: str) -> list[tuple[str, torch.Tensor]]:
        retained = []
        for name, value in output.items():
            if isinstance(value, torch.Tensor) and value.requires_grad:
                value.retain_grad()
                retained.append((name, value))
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    if not isinstance(item, dict):
                        continue
                    for item_name, item_value in item.items():
                        if isinstance(item_value, torch.Tensor) and item_value.requires_grad:
                            item_value.retain_grad()
                            retained.append((f"{name}[{index}].{item_name}", item_value))
        return retained

    native_retained = retain_output_gradients(native_output, "native")
    external_retained = retain_output_gradients(external_output, "external")
    native_handle.remove()
    external_handle.remove()
    native_decoder_handle.remove()
    external_decoder_handle.remove()
    native_decoder_output_handle.remove()
    external_decoder_output_handle.remove()
    native_decoder_input_handle.remove()
    external_decoder_input_handle.remove()
    report("train_encoder_memory", diagnostics["native_memory"], diagnostics["external_memory"])
    report(
        "train_decoder_diffusion",
        diagnostics["native_decoder_diffusion"],
        diagnostics["external_decoder_diffusion"],
    )
    report(
        "train_decoder_tgt",
        diagnostics["native_decoder_tgt"].transpose(0, 1),
        diagnostics["external_decoder_tgt"].transpose(0, 1),
    )
    report(
        "train_decoder_ref",
        diagnostics["native_decoder_ref"].transpose(0, 1),
        diagnostics["external_decoder_ref"].transpose(0, 1),
    )
    native_decoder_tgt = diagnostics["native_decoder_tgt"].transpose(0, 1)
    external_decoder_tgt = diagnostics["external_decoder_tgt"].transpose(0, 1)
    native_decoder_ref = diagnostics["native_decoder_ref"].transpose(0, 1)
    external_decoder_ref = diagnostics["external_decoder_ref"].transpose(0, 1)
    print("decoder_input_shapes:", tuple(native_decoder_tgt.shape), tuple(external_decoder_tgt.shape))
    report("train_decoder_dn_tgt", native_decoder_tgt[:, :100], external_decoder_tgt[:, :100])
    report("train_decoder_normal_tgt", native_decoder_tgt[:, 100:], external_decoder_tgt[:, 100:])
    report("train_decoder_dn_ref", native_decoder_ref[:, :100], external_decoder_ref[:, :100])
    report("train_decoder_normal_ref", native_decoder_ref[:, 100:], external_decoder_ref[:, 100:])
    native_decoder_layers, native_references = diagnostics["native_decoder_output"]
    external_decoder_layers, external_references = diagnostics["external_decoder_output"]
    for index, (native_layer, external_layer) in enumerate(zip(native_decoder_layers, external_decoder_layers)):
        report(f"train_decoder_layer[{index}]", native_layer, external_layer)

    for name in ("pred_logits", "pred_boxes", "pred_keypoints"):
        report(name, native_output[name], external_output[name])
        native_output[name].retain_grad()
        external_output[name].retain_grad()
    native_losses = native_pipeline.criterion(native_output, native_targets)
    external_losses = external_criterion(external_output, [external_target])
    common_losses = sorted(set(native_losses) & set(external_losses) - {"total"})
    for name in common_losses:
        report(f"loss/{name}", native_losses[name], external_losses[name])
    external_total = sum(
        value * external_criterion.weight_dict[name]
        for name, value in external_losses.items()
        if name in external_criterion.weight_dict
    )
    report("loss/total", native_losses["total"], external_total)
    native_losses["total"].backward()
    external_total.backward()
    external_retained_by_name = dict(external_retained)
    for name, native_value in native_retained:
        external_value = external_retained_by_name.get(name)
        if external_value is not None and native_value.grad is not None and external_value.grad is not None:
            report(f"gradient_output/{name}", native_value.grad, external_value.grad)
    external_parameters = dict(external_model.named_parameters())
    gradient_max = 0.0
    gradient_name = ""
    for name, parameter in native_pipeline.model.named_parameters():
        if name not in external_parameters:
            continue
        external_parameter = external_parameters[name]
        if parameter.grad is None or external_parameter.grad is None:
            continue
        error = (parameter.grad - external_parameter.grad).abs().max().item()
        if error > gradient_max:
            gradient_max, gradient_name = error, name
    print(f"gradient_max_abs={gradient_max:.6g} name={gradient_name}")
    if gradient_name:
        native_gradient = dict(native_pipeline.model.named_parameters())[gradient_name].grad
        external_gradient = external_parameters[gradient_name].grad
        print(
            "gradient_norms:", gradient_name,
            native_gradient.norm().item(), external_gradient.norm().item()
        )
    for name in ("pred_logits", "pred_boxes", "pred_keypoints"):
        report(f"gradient/{name}", native_output[name].grad, external_output[name].grad)

    from util.get_param_dicts import get_param_dict

    native_spec = native_pipeline.configure_optimizers()
    native_optimizer = native_spec["optimizer"]
    external_optimizer = torch.optim.AdamW(
        get_param_dict(external_args, external_model),
        lr=external_args.lr,
        weight_decay=external_args.weight_decay,
    )
    native_before = {
        name: parameter.detach().clone()
        for name, parameter in native_pipeline.model.named_parameters()
    }
    external_before = {
        name: parameter.detach().clone()
        for name, parameter in external_model.named_parameters()
    }
    native_optimizer.step()
    external_optimizer.step()
    external_after = dict(external_model.named_parameters())
    native_after = dict(native_pipeline.model.named_parameters())
    update_max = max(
        (native_after[name] - external_after[name]).abs().max().item()
        for name in native_before if name in external_after
    )
    print(f"adamw_update_max_abs={update_max:.6g}")
    print("native_target_count:", native_targets[0]["labels"].numel())
    print("external_target_count:", external_target["labels"].numel())
    print("real-batch forward parity: COMPLETE")


if __name__ == "__main__":
    main()
