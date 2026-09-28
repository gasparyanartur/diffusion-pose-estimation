"""Run one CPU-only SHARP depth and Gaussian feasibility example.

Requires the official SHARP source on PYTHONPATH and its checkpoint locally.
The output is exploratory; it does not change the pose pipeline.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.patches import Rectangle

from sharp.models import PredictorParams, create_predictor
from sharp.utils import io
from sharp.utils.color_space import linearRGB2sRGB
from sharp.utils.gaussians import (
    Gaussians3D,
    get_unprojection_matrix,
    save_ply,
    unproject_gaussians,
)


def sample_gaussians(gaussians: Gaussians3D, maximum: int) -> Gaussians3D:
    """Take an evenly spaced sample across both SHARP layers."""
    count = gaussians.mean_vectors.shape[1]
    indices = torch.linspace(0, count - 1, min(count, maximum)).long()
    return Gaussians3D(*(tensor[:, indices] for tensor in gaussians))


def plot_diagnostics(
    image: np.ndarray,
    depth: np.ndarray,
    centers: np.ndarray,
    colors: np.ndarray,
    opacities: np.ndarray,
    intrinsics: np.ndarray,
    output_path: Path,
    foreground_max_m: float = 12.0,
) -> None:
    """Show both metric depth layers and two RGB Gaussian center views."""
    valid = np.isfinite(centers).all(axis=1) & (centers[:, 2] > 0)
    valid &= centers[:, 2] < foreground_max_m
    valid &= np.isfinite(opacities) & (opacities >= 0.5)
    indices = np.flatnonzero(valid)
    if len(indices) > 30_000:
        indices = indices[np.linspace(0, len(indices) - 1, 30_000).astype(int)]
    points = centers[indices]
    point_colors = colors[indices]

    figure = plt.figure(figsize=(16, 10), constrained_layout=True)
    image_ax = figure.add_subplot(2, 3, 1)
    image_ax.imshow(image)
    image_ax.set_title("Input RGB, stretched to SHARP's square resolution")
    image_ax.axis("off")
    for layer in range(2):
        ax = figure.add_subplot(2, 3, layer + 2)
        display_max_m = foreground_max_m if layer == 0 else 150.0
        image_artist = ax.imshow(
            depth[layer], cmap="turbo", vmin=1.0, vmax=display_max_m,
        )
        ax.set_title(f"SHARP depth layer {layer + 1} (1–{display_max_m:g} m display)")
        ax.axis("off")
        figure.colorbar(image_artist, ax=ax, fraction=0.046, pad=0.04)

    front_ax = figure.add_subplot(2, 3, 4)
    pixel_u = intrinsics[0, 0] * points[:, 0] / points[:, 2] + intrinsics[0, 2]
    pixel_v = intrinsics[1, 1] * points[:, 1] / points[:, 2] + intrinsics[1, 2]
    front_ax.scatter(pixel_u, pixel_v, c=point_colors, s=0.4,
                     marker=".", linewidths=0, rasterized=True)
    front_ax.set_xlim(0, image.shape[1])
    front_ax.set_ylim(image.shape[0], 0)
    front_ax.set_aspect("equal")
    front_ax.set_title(f"Perspective reprojection (<{foreground_max_m:g} m)")
    front_ax.set_xlabel("u (processed pixels)")
    front_ax.set_ylabel("v (processed pixels)")

    for position, elevation, azimuth, title in [
        (5, 25, -65, "Gaussian centers, oblique view"),
        (6, 5, -90, "Gaussian centers, side view"),
    ]:
        ax = figure.add_subplot(2, 3, position, projection="3d")
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=point_colors,
                   s=0.8, marker=".", linewidths=0, depthshade=False,
                   rasterized=True)
        ax.view_init(elev=elevation, azim=azimuth)
        ax.set_title(title)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y down (m)")
        ax.set_zlabel("z (m)")

    figure.savefig(output_path, dpi=140)
    plt.close(figure)


def plot_person_region(
    image: np.ndarray,
    depth: np.ndarray,
    centers: np.ndarray,
    colors: np.ndarray,
    opacities: np.ndarray,
    intrinsics: np.ndarray,
    boxes: list[list[float]],
    original_hw: tuple[int, int],
    output_path: Path,
) -> int:
    """Inspect points projecting inside annotated person boxes."""
    scale_x = image.shape[1] / original_hw[1]
    scale_y = image.shape[0] / original_hw[0]
    scaled_boxes = [(x * scale_x, y * scale_y, w * scale_x, h * scale_y)
                    for x, y, w, h in boxes]
    z = centers[:, 2]
    projected_u = intrinsics[0, 0] * centers[:, 0] / z + intrinsics[0, 2]
    projected_v = intrinsics[1, 1] * centers[:, 1] / z + intrinsics[1, 2]
    inside_box = np.zeros(len(centers), dtype=bool)
    for x, y, width, height in scaled_boxes:
        inside_box |= ((projected_u >= x) & (projected_u <= x + width)
                       & (projected_v >= y) & (projected_v <= y + height))
    valid = inside_box & np.isfinite(centers).all(axis=1)
    valid &= (z > 0) & (z < 12) & (opacities >= 0.5)
    points = centers[valid]
    point_colors = colors[valid]
    if len(points) == 0:
        raise ValueError("No foreground Gaussians project inside person boxes")

    figure = plt.figure(figsize=(12, 9), constrained_layout=True)
    image_ax = figure.add_subplot(2, 2, 1)
    image_ax.imshow(image)
    for x, y, width, height in scaled_boxes:
        image_ax.add_patch(Rectangle((x, y), width, height, fill=False,
                                     edgecolor="yellow", linewidth=1.2))
    image_ax.set_title("CrowdPose person boxes on SHARP input")
    image_ax.axis("off")

    depth_ax = figure.add_subplot(2, 2, 2)
    depth_ax.imshow(depth[0], cmap="turbo", vmin=1, vmax=8)
    for x, y, width, height in scaled_boxes:
        depth_ax.add_patch(Rectangle((x, y), width, height, fill=False,
                                     edgecolor="yellow", linewidth=1.2))
    depth_ax.set_title("Layer 1 depth, 1–8 m display")
    depth_ax.axis("off")

    for position, elevation, azimuth, title in [
        (3, 25, -65, "Person-box points, oblique view"),
        (4, 5, -90, "Person-box points, side view"),
    ]:
        ax = figure.add_subplot(2, 2, position, projection="3d")
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=point_colors,
                   s=1.2, marker=".", linewidths=0, depthshade=False,
                   rasterized=True)
        ax.view_init(elev=elevation, azim=azimuth)
        ax.set_title(title)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y down (m)")
        ax.set_zlabel("z (m)")
    figure.savefig(output_path, dpi=140)
    plt.close(figure)
    return len(points)


def run(
    image_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    threads: int,
    save_all_centers: bool = False,
) -> None:
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    device = torch.device("cpu")
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()

    image, _, focal_px = io.load_rgb(image_path)
    height, width = image.shape[:2]
    internal_size = 1536
    image_tensor = torch.from_numpy(image.copy()).float().permute(2, 0, 1)[None] / 255.0
    resized = F.interpolate(image_tensor, size=(internal_size, internal_size),
                            mode="bilinear", align_corners=True)
    disparity_factor = torch.tensor([focal_px / width], dtype=torch.float32)
    resized_rgb = (resized[0].permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
    print(f"Image {image_path.name}: {width}x{height}, assumed focal {focal_px:.1f} px", flush=True)

    predictor = create_predictor(PredictorParams())
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    predictor.load_state_dict(state_dict)
    del state_dict
    predictor.eval().to(device)
    print(f"Model loaded after {time.monotonic() - start:.1f} s", flush=True)

    captured_disparity = {}

    def capture_disparity(_module, _inputs, output):
        captured_disparity["value"] = output.disparity.detach().clone()

    hook = predictor.monodepth_model.register_forward_hook(capture_disparity)
    with torch.inference_mode():
        gaussians_ndc = predictor(resized, disparity_factor)
    hook.remove()
    print(f"Inference finished after {time.monotonic() - start:.1f} s", flush=True)

    disparity = captured_disparity["value"].clamp(min=1e-4, max=1e4)
    depth = (disparity_factor[:, None, None, None] / disparity)[0].numpy()
    assert depth.shape == (2, internal_size, internal_size)
    assert np.isfinite(depth).all() and (depth > 0).all()
    depth_path = output_dir / f"{image_path.stem}_sharp_depth.npz"
    np.savez_compressed(depth_path, depth_m=depth, rgb=resized_rgb,
                        focal_px=np.float32(focal_px), original_hw=np.array([height, width]))
    print(f"Depth saved after {time.monotonic() - start:.1f} s", flush=True)

    intrinsics = torch.tensor([
        [focal_px * internal_size / width, 0, internal_size / 2, 0],
        [0, focal_px * internal_size / height, internal_size / 2, 0],
        [0, 0, 1, 0], [0, 0, 0, 1],
    ], dtype=torch.float32)
    centers_path = None
    if save_all_centers:
        # The official unprojection applies this same affine transform to
        # Gaussian means. It also transforms covariance, which is not needed
        # for a center-depth diagnostic.
        transform = get_unprojection_matrix(
            torch.eye(4), intrinsics, (internal_size, internal_size)
        )
        centers_m = (
            gaussians_ndc.mean_vectors @ transform[:3, :3].T + transform[:3, 3]
        )[0]
        colors_srgb = linearRGB2sRGB(gaussians_ndc.colors[0]).clamp(0, 1)
        centers_path = output_dir / f"{image_path.stem}_sharp_all_centers.npz"
        np.savez_compressed(
            centers_path,
            xyz_m=centers_m.numpy(),
            rgb=(colors_srgb.numpy() * 255).round().astype(np.uint8),
            opacity=gaussians_ndc.opacities[0].numpy(),
            intrinsics=intrinsics[:3, :3].numpy(),
        )
        print(f"All Gaussian centers saved after {time.monotonic() - start:.1f} s", flush=True)

    sampled_ndc = sample_gaussians(gaussians_ndc, maximum=200_000)
    with torch.inference_mode():
        gaussians = unproject_gaussians(
            sampled_ndc, torch.eye(4), intrinsics, (internal_size, internal_size)
        )
    if save_all_centers:
        sample_indices = torch.linspace(
            0, len(gaussians_ndc.mean_vectors[0]) - 1,
            len(gaussians.mean_vectors[0]),
        ).long()
        np.testing.assert_allclose(
            centers_m[sample_indices].numpy(), gaussians.mean_vectors[0].numpy(),
            rtol=1e-5, atol=2e-5,
        )
    ply_path = output_dir / f"{image_path.stem}_sharp_gaussians_sampled.ply"
    save_ply(gaussians, focal_px, (height, width), ply_path)
    print(f"Gaussians saved after {time.monotonic() - start:.1f} s", flush=True)

    centers = gaussians.mean_vectors[0].numpy()
    colors = linearRGB2sRGB(gaussians.colors[0]).clamp(0, 1).numpy()
    opacities = gaussians.opacities[0].numpy()
    figure_path = output_dir / f"{image_path.stem}_sharp_diagnostic.png"
    plot_diagnostics(resized_rgb, depth, centers, colors, opacities,
                     intrinsics.numpy(), figure_path)

    annotations_path = Path("data/crowdpose/annotations/crowdpose_val.json")
    annotations = json.loads(annotations_path.read_text())["annotations"]
    person_boxes = [item["bbox"] for item in annotations
                    if item["image_id"] == int(image_path.stem) and not item["iscrowd"]]
    people_figure_path = output_dir / f"{image_path.stem}_sharp_people.png"
    person_point_count = plot_person_region(
        resized_rgb, depth, centers, colors, opacities, intrinsics.numpy(),
        person_boxes, (height, width), people_figure_path,
    )

    metadata = {
        "image": str(image_path), "checkpoint": str(checkpoint_path),
        "device": "cpu", "threads": threads, "original_hw": [height, width],
        "internal_hw": [internal_size, internal_size], "focal_source": "SHARP fallback 30 mm equivalent",
        "focal_px": focal_px, "gaussians_sampled": len(centers),
        "opacity_ge_0_5": int(np.count_nonzero(opacities >= 0.5)),
        "depth_layer_median_m": [float(np.median(layer)) for layer in depth],
        "elapsed_seconds": round(time.monotonic() - start, 1),
        "depth_npz": str(depth_path), "gaussian_ply": str(ply_path),
        "figure": str(figure_path), "people_figure": str(people_figure_path),
        "person_box_gaussians": person_point_count,
        "all_centers_npz": str(centers_path) if centers_path is not None else None,
    }
    (output_dir / f"{image_path.stem}_sharp_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--save-all-centers", action="store_true")
    arguments = parser.parse_args()
    run(arguments.image, arguments.checkpoint, arguments.output_dir,
        arguments.threads, arguments.save_all_centers)
