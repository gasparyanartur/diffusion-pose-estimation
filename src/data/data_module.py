"""Lightning DataModules for pose estimation.

The DataConfig.build() factory returns a concrete PoseDataModule. The training
module reads the resolved annotation paths and prediction-cache paths from the
data module so the lightning module never has to know the on-disk layout.
"""

from __future__ import annotations

import logging
import random
from abc import ABC
from pathlib import Path
from typing import Literal

import pytorch_lightning as pl
import torch
import torchvision.transforms.functional as TF
from torch import Tensor
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader

from src.data.datasets import PoseDataset, SetPoseDataset, compute_person_scale, set_pose_collate
from src.data.transforms import pixel_to_canonical

log = logging.getLogger(__name__)


def resize_and_pad_to_square(img: Tensor, size: int, mask_value: float = 0.0) -> tuple[Tensor, Tensor]:
    """Resize longest side to `size`, pad the shorter side to make a square.

    Args:
        img:  float [3, H, W]
        size: target side length
        mask_value: value to fill the padding area

    Returns:
        image:        [3, size, size]  float, zero-padded
        padding_mask: [size, size]    1 = real pixel, 0 = padding
    """
    _, h, w = img.shape
    scale = size / max(h, w)
    new_h = round(h * scale)
    new_w = round(w * scale)
    img = TF.resize(img, [new_h, new_w], interpolation=InterpolationMode.BICUBIC, antialias=True)
    # pad right and bottom to reach size × size
    padded = TF.pad(img, [0, 0, size - new_w, size - new_h], fill=mask_value)
    mask = torch.zeros(size, size, dtype=torch.float32, device=img.device)
    mask[:new_h, :new_w] = 1.0
    return padded, mask


def _boxes_to_padded_canonical_batch(
    boxes_xywh_pixel: Tensor,  # [B, P, 4]
    img_hw: Tensor,            # [B, 2]   (H, W) original
) -> Tensor:
    """Vectorised batched form of :func:`boxes_xywh_to_padded_canonical`.

    Each row uses its own original ``max(H, W)``. Padding entries (all zeros)
    stay all zeros after this transform (they get mapped to ``(-1, -1, 0, 0)``
    which is still a sentinel-only region; the ``gt_box_labeled`` mask is the
    source of truth for which slots are real).
    """
    side = img_hw.amax(dim=-1).clamp_min(1.0).unsqueeze(-1)  # [B, 1]
    x = boxes_xywh_pixel[..., 0]
    y = boxes_xywh_pixel[..., 1]
    w = boxes_xywh_pixel[..., 2]
    h = boxes_xywh_pixel[..., 3]
    cx = (x + w / 2.0) / side * 2.0 - 1.0
    cy = (y + h / 2.0) / side * 2.0 - 1.0
    cw = w / side * 2.0
    ch = h / side * 2.0
    return torch.stack([cx, cy, cw, ch], dim=-1)


def _boxes_to_image_canonical_batch(boxes_xywh_pixel: Tensor, img_hw: Tensor) -> Tensor:
    """Convert pixel boxes to normalized ``cxcywh`` coordinates.

    DiffusionRegPose predicts boxes with sigmoid outputs, so its targets use
    the same ``[0, 1]`` image-coordinate convention as the external model.
    The padded ``[-1, 1]`` convention remains available through
    ``_boxes_to_padded_canonical_batch`` for the other set pipelines.
    """
    height = img_hw[:, 0].clamp_min(1.0).unsqueeze(-1)
    width = img_hw[:, 1].clamp_min(1.0).unsqueeze(-1)
    x, y, box_width, box_height = boxes_xywh_pixel.unbind(-1)
    return torch.stack(
        ((x + box_width / 2) / width,
         (y + box_height / 2) / height,
         box_width / width,
         box_height / height),
        dim=-1,
    )


def _external_resize_size(height: int, width: int, short_side: int, max_size: int) -> tuple[int, int]:
    """Return the integer ``(height, width)`` used by the external transforms."""
    if max(height, width) / min(height, width) * short_side > max_size:
        short_side = int(round(max_size * min(height, width) / max(height, width)))
    if width < height:
        return short_side * height // width, short_side
    return short_side, short_side * width // height


def _resize_set_sample(
    image: Tensor,
    keypoints: Tensor,
    labeled: Tensor,
    visible: Tensor,
    boxes: Tensor,
    box_labeled: Tensor,
    short_side: int,
    max_size: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Resize one set sample and update its pixel-space targets."""
    height, width = image.shape[-2:]
    new_height, new_width = _external_resize_size(height, width, short_side, max_size)
    image = TF.resize(
        image, [new_height, new_width], interpolation=InterpolationMode.BILINEAR, antialias=True,
    )
    scale_x, scale_y = new_width / width, new_height / height
    keypoints = keypoints * keypoints.new_tensor([scale_x, scale_y])
    boxes = boxes * boxes.new_tensor([scale_x, scale_y, scale_x, scale_y])
    return image, keypoints, labeled, visible, boxes, box_labeled, image.new_tensor([new_height, new_width])


def _resize_set_sample_with_diffusion_keypoints(
    image: Tensor,
    keypoints: Tensor,
    diffusion_keypoints: Tensor,
    labeled: Tensor,
    visible: Tensor,
    boxes: Tensor,
    box_labeled: Tensor,
    short_side: int,
    max_size: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Resize clean and diffusion keypoint streams with the same geometry."""
    height, width = image.shape[-2:]
    new_height, new_width = _external_resize_size(height, width, short_side, max_size)
    diffusion_keypoints = diffusion_keypoints * diffusion_keypoints.new_tensor([
        new_width / width,
        new_height / height,
    ])
    image, keypoints, labeled, visible, boxes, box_labeled, image_hw = _resize_set_sample(
        image,
        keypoints,
        labeled,
        visible,
        boxes,
        box_labeled,
        short_side,
        max_size,
    )
    return (
        image,
        keypoints,
        diffusion_keypoints,
        labeled,
        visible,
        boxes,
        box_labeled,
        image_hw,
    )


def _external_train_transform(
    image: Tensor,
    keypoints_canon: Tensor,
    labeled: Tensor,
    visible: Tensor,
    boxes: Tensor,
    box_labeled: Tensor,
    *,
    diffusion_keypoints_canon: Tensor | None = None,
    scales: tuple[int, ...],
    max_size: int,
    scales2_resize: tuple[int, ...],
    scales2_crop: tuple[int, int],
    random_flip: bool = True,
    random_crop: bool = True,
    rng: random.Random | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Apply the external CrowdPose train transform to one set sample."""
    height, width = image.shape[-2:]
    keypoints = keypoints_canon.clone()
    keypoints = (keypoints + 1.0) / 2.0 * keypoints.new_tensor([width, height])
    if diffusion_keypoints_canon is None:
        diffusion_keypoints_canon = keypoints_canon
    diffusion_keypoints = diffusion_keypoints_canon.clone()
    diffusion_keypoints = (
        (diffusion_keypoints + 1.0) / 2.0 * diffusion_keypoints.new_tensor([width, height])
    )
    boxes = boxes.clone()
    labeled, visible, box_labeled = labeled.clone(), visible.clone(), box_labeled.clone()

    random_source = random if rng is None else rng
    if random_flip and random_source.random() < 0.5:
        image = image.flip(-1)
        keypoints[..., 0] = width - keypoints[..., 0] - 1
        diffusion_keypoints[..., 0] = width - diffusion_keypoints[..., 0] - 1
        boxes[..., 0] = width - boxes[..., 0] - boxes[..., 2]
        for first, second in ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)):
            keypoints[:, [first, second]] = keypoints[:, [second, first]].clone()
            diffusion_keypoints[:, [first, second]] = (
                diffusion_keypoints[:, [second, first]].clone()
            )
            labeled[:, [first, second]] = labeled[:, [second, first]].clone()
            visible[:, [first, second]] = visible[:, [second, first]].clone()

    if random_crop and random_source.random() < 0.5:
        short_side = random_source.choice(scales)
        image, keypoints, diffusion_keypoints, labeled, visible, boxes, box_labeled, image_hw = (
            _resize_set_sample_with_diffusion_keypoints(
                image,
                keypoints,
                diffusion_keypoints,
                labeled,
                visible,
                boxes,
                box_labeled,
                short_side,
                max_size,
            )
        )
    elif random_crop:
        image, keypoints, diffusion_keypoints, labeled, visible, boxes, box_labeled, image_hw = (
            _resize_set_sample_with_diffusion_keypoints(
                image,
                keypoints,
                diffusion_keypoints,
                labeled,
                visible,
                boxes,
                box_labeled,
                random_source.choice(scales2_resize),
                max_size=10_000,
            )
        )
        height, width = image.shape[-2:]
        crop_width = random_source.randint(min(scales2_crop[0], width), min(width, scales2_crop[1]))
        crop_height = random_source.randint(min(scales2_crop[0], height), min(height, scales2_crop[1]))
        top = random_source.randint(0, max(height - crop_height, 0))
        left = random_source.randint(0, max(width - crop_width, 0))
        image = image[..., top:top + crop_height, left:left + crop_width]
        keypoints[..., 0] = (keypoints[..., 0] - left).clamp(0, crop_width)
        keypoints[..., 1] = (keypoints[..., 1] - top).clamp(0, crop_height)
        diffusion_keypoints[..., 0] = (
            diffusion_keypoints[..., 0] - left
        ).clamp(0, crop_width)
        diffusion_keypoints[..., 1] = (
            diffusion_keypoints[..., 1] - top
        ).clamp(0, crop_height)
        x1 = (boxes[..., 0] - left).clamp(0, crop_width)
        y1 = (boxes[..., 1] - top).clamp(0, crop_height)
        x2 = (boxes[..., 0] + boxes[..., 2] - left).clamp(0, crop_width)
        y2 = (boxes[..., 1] + boxes[..., 3] - top).clamp(0, crop_height)
        boxes = torch.stack((x1, y1, x2 - x1, y2 - y1), dim=-1)
        keep = (boxes[..., 2] > 0) & (boxes[..., 3] > 0)
        box_labeled = box_labeled * keep.float()
        labeled = labeled * keep[..., None].float()
        visible = visible * keep[..., None].float()
        image_hw = image.new_tensor([crop_height, crop_width])
        image, keypoints, diffusion_keypoints, labeled, visible, boxes, box_labeled, image_hw = (
            _resize_set_sample_with_diffusion_keypoints(
                image,
                keypoints,
                diffusion_keypoints,
                labeled,
                visible,
                boxes,
                box_labeled,
                random_source.choice(scales),
                max_size,
            )
        )

    else:
        image, keypoints, diffusion_keypoints, labeled, visible, boxes, box_labeled, image_hw = (
            _resize_set_sample_with_diffusion_keypoints(
                image,
                keypoints,
                diffusion_keypoints,
                labeled,
                visible,
                boxes,
                box_labeled,
                random_source.choice(scales),
                max_size,
            )
        )

    height, width = image.shape[-2:]
    keypoints = pixel_to_canonical(keypoints, height, width)
    diffusion_keypoints = pixel_to_canonical(diffusion_keypoints, height, width)
    person_scale = compute_person_scale(keypoints, labeled)
    return (
        image,
        keypoints,
        diffusion_keypoints,
        labeled,
        visible,
        boxes,
        box_labeled,
        image.new_tensor([height, width]),
        person_scale,
    )


def _external_eval_transform(
    image: Tensor,
    keypoints_canon: Tensor,
    labeled: Tensor,
    visible: Tensor,
    boxes: Tensor,
    box_labeled: Tensor,
    *,
    short_side: int = 800,
    max_size: int = 1333,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Apply the deterministic evaluation resize and update pixel targets."""
    height, width = image.shape[-2:]
    keypoints = (keypoints_canon.clone() + 1.0) / 2.0 * keypoints_canon.new_tensor([width, height])
    image, keypoints, labeled, visible, boxes, box_labeled, image_hw = _resize_set_sample(
        image,
        keypoints,
        labeled.clone(),
        visible.clone(),
        boxes.clone(),
        box_labeled.clone(),
        short_side=short_side,
        max_size=max_size,
    )
    height, width = image.shape[-2:]
    keypoints = pixel_to_canonical(keypoints, height, width)
    person_scale = compute_person_scale(keypoints, labeled)
    return image, keypoints, labeled, visible, boxes, box_labeled, image_hw, person_scale


def _pad_variable_images(images: list[Tensor]) -> tuple[Tensor, Tensor]:
    """Pad transformed images bottom/right and mark padded pixels as True."""
    max_height = max(image.shape[-2] for image in images)
    max_width = max(image.shape[-1] for image in images)
    padded, masks = [], []
    for image in images:
        height, width = image.shape[-2:]
        padded.append(TF.pad(image, [0, 0, max_width - width, max_height - height]))
        mask = torch.ones(max_height, max_width, dtype=torch.bool, device=image.device)
        mask[:height, :width] = False
        masks.append(mask)
    return torch.stack(padded), torch.stack(masks)


def _normalize_imagenet_image(image: Tensor) -> Tensor:
    """Normalize one RGB image before zero-padding a variable-size batch."""
    mean = image.new_tensor([0.485, 0.456, 0.406])[:, None, None]
    std = image.new_tensor([0.229, 0.224, 0.225])[:, None, None]
    return (image - mean) / std


class PoseDataModule(pl.LightningDataModule, ABC):
    """Contract: active pipelines read these attributes after construction."""

    val_ann_path: Path
    test_ann_path: Path
    val_pred_cache_resolved: str | None
    test_pred_cache_resolved: str | None
    num_keypoints: int
    batch_size: int
    num_workers: int
    ap_batch_size: int


class CrowdPoseDataModule(PoseDataModule):
    """CrowdPose train/val/test loaders backed by :class:`PoseDataset`."""

    def __init__(
        self,
        train_ann: Path,
        val_ann: Path,
        test_ann: Path,
        img_dir: Path,
        val_pred_cache: str | None,
        test_pred_cache: str | None,
        batch_size: int,
        num_workers: int,
        ap_batch_size: int,
        num_keypoints: int = 14,
        subset_frac: float = 1.0,
        pin_memory: bool = False,
        prefetch_factor: int = 2,
        target_format: Literal["refinement", "set"] = "refinement",
        image_h: int | None = 256,
        image_w: int | None = 192,
        max_persons: int | None = None,
        max_size: int | None = 512,
        external_train_transforms: bool = False,
        train_random_flip: bool = True,
        train_random_crop: bool = True,
        train_transform_seed: int | None = None,
        train_resize_scales: tuple[int, ...] = (480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
        train_max_size: int = 1333,
        train_crop_resize_scales: tuple[int, ...] = (400, 500, 600),
        train_crop_size: tuple[int, int] = (384, 600),
        eval_resize_scale: int = 800,
        eval_max_size: int = 1333,
    ) -> None:
        super().__init__()
        self.train_ann_path = Path(train_ann)
        self.val_ann_path = Path(val_ann)
        self.test_ann_path = Path(test_ann)
        self.img_dir = Path(img_dir)
        self.val_pred_cache_resolved = self._resolve_cache(val_pred_cache, "val_pred_cache")
        self.test_pred_cache_resolved = self._resolve_cache(test_pred_cache, "test_pred_cache")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.ap_batch_size = ap_batch_size
        self.num_keypoints = num_keypoints
        self.subset_frac = subset_frac
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.target_format = target_format
        self.image_h = image_h
        self.image_w = image_w
        self.max_persons = max_persons
        self.max_size = max_size
        self.external_train_transforms = external_train_transforms
        self.train_random_flip = train_random_flip
        self.train_random_crop = train_random_crop
        self.train_transform_seed = train_transform_seed
        self.train_resize_scales = train_resize_scales
        self.train_max_size = train_max_size
        self.train_crop_resize_scales = train_crop_resize_scales
        self.train_crop_size = train_crop_size
        self.eval_resize_scale = eval_resize_scale
        self.eval_max_size = eval_max_size

        self._train_ds: PoseDataset | None = None
        self._val_ds: PoseDataset | None = None
        self._test_ds: PoseDataset | None = None

    @staticmethod
    def _resolve_cache(raw: str | None, label: str) -> str | None:
        if not raw:
            return None
        path = Path(raw)
        if not path.exists():
            log.warning(
                "%s not found at %s — AP eval will be skipped for this split.",
                label, path,
            )
            return None
        return str(path)

    def setup(self, stage: str | None = None) -> None:
        if self._train_ds is None:
            self._train_ds = PoseDataset.from_path(
                self.train_ann_path, img_root=self.img_dir, num_keypoints=self.num_keypoints,
            )
            self._sort_set_samples(self._train_ds)
            self._maybe_subset(self._train_ds, "train")
            self._maybe_filter_persons(self._train_ds, "train")
        if self._val_ds is None:
            self._val_ds = PoseDataset.from_path(
                self.val_ann_path, img_root=self.img_dir, num_keypoints=self.num_keypoints,
            )
            self._sort_set_samples(self._val_ds)
            self._maybe_subset(self._val_ds, "val")
            self._maybe_filter_persons(self._val_ds, "val")
        if self._test_ds is None:
            self._test_ds = PoseDataset.from_path(
                self.test_ann_path, img_root=self.img_dir, num_keypoints=self.num_keypoints,
            )
            self._sort_set_samples(self._test_ds)
            self._maybe_subset(self._test_ds, "test")
            self._maybe_filter_persons(self._test_ds, "test")

    def _sort_set_samples(self, dataset: PoseDataset) -> None:
        """Match the external set-dataset image-ID ordering."""
        if self.target_format == "set":
            dataset.samples.sort(key=lambda sample: sample["image_id"])

    def _maybe_subset(self, ds: PoseDataset, split: str) -> None:
        if self.subset_frac >= 1.0:
            return
        n_full = len(ds.samples)
        n_keep = max(1, round(n_full * self.subset_frac))
        ds.samples = ds.samples[:n_keep]
        log.warning(
            "subset_frac=%.4g: %s split reduced %d -> %d samples.",
            self.subset_frac, split, n_full, n_keep,
        )

    def _maybe_filter_persons(self, ds: PoseDataset, split: str) -> None:
        if self.max_persons is None:
            return
        n_full = len(ds.samples)
        ds.samples = [
            s for s in ds.samples
            if s["keypoints"].shape[0] <= self.max_persons
        ]
        n_keep = len(ds.samples)
        log.warning(
            "max_persons=%d: %s split reduced %d -> %d samples.",
            self.max_persons, split, n_full, n_keep,
        )

    def _resolve_target_size(self) -> tuple[int, int] | None:
        h, w = self.image_h, self.image_w
        if h is None and w is None:
            return None
        if h is None:
            h = w
        if w is None:
            w = h
        return (h, w)  # type: ignore[return-value]

    def _worker_kwargs(self) -> dict:
        """Keyword arguments shared by all DataLoader constructions."""
        kwargs: dict = {
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.num_workers > 0,
        }
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = self.prefetch_factor
        return kwargs

    def _make_loader(self, ds: PoseDataset, shuffle: bool, drop_last: bool) -> DataLoader:
        if self.target_format == "set":
            wrapped = SetPoseDataset(ds)
            return DataLoader(
                wrapped,
                batch_size=self.batch_size,
                shuffle=shuffle,
                drop_last=drop_last,
                collate_fn=set_pose_collate,
                **self._worker_kwargs(),
            )
        return ds.create_dataloader(
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=self.num_workers,
            load_images=False,
            pin_memory=self.pin_memory,
        )

    def on_after_batch_transfer(self, batch: dict, dataloader_idx: int) -> dict:
        """Resize and normalize the per-sample uint8 images as a GPU batch."""
        if "image" not in batch or not isinstance(batch["image"], list):
            return batch

        imgs: list[Tensor] = batch.pop("image")  # list of uint8 [C, H, W] on device
        target_size = self._resolve_target_size()
        normalized = []
        for img in imgs:
            if img.shape[0] == 1:
                img = img.expand(3, -1, -1)       # grayscale → 3-channel
            normalized.append(img.float().div_(255.0))

        if self.external_train_transforms:
            # Keep annotation-space dimensions for AP while `img_hw` below
            # records the transformed, unpadded model-input dimensions.
            batch["original_img_hw"] = batch["img_hw"].clone()
            if getattr(self.trainer, "training", False):
                if "person_mask" in batch:
                    batch["diffusion_person_mask"] = batch["person_mask"].clone()
                transformed = [
                    _external_train_transform(
                        image,
                        batch["gt_keypoints"][index],
                        batch["gt_labeled"][index],
                        batch["gt_visible"][index],
                        batch["gt_boxes_pixel"][index],
                        batch["gt_box_labeled"][index],
                        diffusion_keypoints_canon=batch["gt_keypoints"][index],
                        scales=self.train_resize_scales,
                        max_size=self.train_max_size,
                        scales2_resize=self.train_crop_resize_scales,
                        scales2_crop=self.train_crop_size,
                        random_flip=self.train_random_flip,
                        random_crop=self.train_random_crop,
                        rng=(
                            random.Random(self.train_transform_seed + index)
                            if self.train_transform_seed is not None else None
                        ),
                    )
                    for index, image in enumerate(normalized)
                ]
                (
                    images,
                    keypoints,
                    diffusion_keypoints,
                    labeled,
                    visible,
                    boxes,
                    box_labeled,
                    image_hw,
                    scales,
                ) = zip(*transformed)
            else:
                transformed = [
                    _external_eval_transform(
                        image,
                        batch["gt_keypoints"][index],
                        batch["gt_labeled"][index],
                        batch["gt_visible"][index],
                        batch["gt_boxes_pixel"][index],
                        batch["gt_box_labeled"][index],
                        short_side=self.eval_resize_scale,
                        max_size=self.eval_max_size,
                    )
                    for index, image in enumerate(normalized)
                ]
                images, keypoints, labeled, visible, boxes, box_labeled, image_hw, scales = zip(*transformed)
                diffusion_keypoints = keypoints
            normalized_images = [_normalize_imagenet_image(image) for image in images]
            batch["image"], batch["padding_mask"] = _pad_variable_images(normalized_images)
            batch["gt_keypoints"] = torch.stack(list(keypoints))
            batch["gt_diffusion_keypoints"] = torch.stack(list(diffusion_keypoints))
            batch["gt_labeled"] = torch.stack(list(labeled))
            batch["gt_visible"] = torch.stack(list(visible))
            batch["gt_boxes_pixel"] = torch.stack(list(boxes))
            transformed_box_labeled = torch.stack(list(box_labeled))
            batch["gt_box_labeled"] = transformed_box_labeled
            # Cropping removes people from supervised targets but the external
            # proposal stream retains their clipped keypoint rows.
            if "person_mask" in batch:
                batch["person_mask"] = transformed_box_labeled.gt(0).to(
                    dtype=batch["person_mask"].dtype,
                )
            batch["img_hw"] = torch.stack(list(image_hw))
            batch["person_scale"] = torch.stack(list(scales))
            batch["gt_boxes_canon"] = _boxes_to_image_canonical_batch(
                batch["gt_boxes_pixel"], batch["img_hw"],
            )
            return batch

        if target_size is not None:
            batch["image"] = torch.stack([
                TF.resize(img, list(target_size), interpolation=InterpolationMode.BILINEAR, antialias=True)
                for img in normalized
            ])  # [B, 3, H, W]
        elif self.max_size is not None:
            pairs = [resize_and_pad_to_square(img, self.max_size) for img in normalized]
            batch["image"]        = torch.stack([p for p, _ in pairs])   # [B, 3, S, S]
            batch["padding_mask"] = torch.stack([m for _, m in pairs])   # [B, S, S]
            if "gt_boxes_pixel" in batch:
                batch["gt_boxes_canon"] = _boxes_to_padded_canonical_batch(
                    batch["gt_boxes_pixel"], batch["img_hw"],
                )
        else:
            batch["image"] = normalized
        return batch

    def train_dataloader(self) -> DataLoader:
        self.setup()
        assert self._train_ds is not None
        return self._make_loader(self._train_ds, shuffle=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        self.setup()
        assert self._val_ds is not None
        return self._make_loader(self._val_ds, shuffle=False, drop_last=False)

    def test_dataloader(self) -> DataLoader:
        self.setup()
        assert self._test_ds is not None
        return self._make_loader(self._test_ds, shuffle=False, drop_last=False)
