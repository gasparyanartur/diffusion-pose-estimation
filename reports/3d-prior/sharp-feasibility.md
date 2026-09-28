# SHARP depth and Gaussian pilot on CrowdPose

## Pilot observation

SHARP is a better *depth visualization* candidate than the failed DA3-BASE
example on image `107292`: its first depth layer separates the three people,
foreground gear, truck walls, and distant background. The 3D Gaussian centers
inside the person boxes still form overlapping visible surfaces. This pilot
does **not** establish usable 3D body geometry for a skeletal prior. Stop the
point-cloud replacement here; do not integrate SHARP Gaussians into the pose
model on this evidence.

The original sampled-Gaussian view was limited to one CrowdPose image and
200,000 splats. The later RGBD notebook also inspects all final Gaussian
centers on that image. Neither view assesses photorealistic Gaussian rendering
or SHARP's published nearby-view synthesis task.

## Model contract

- [SHARP](https://arxiv.org/html/2512.10685) takes one RGB image at inference.
  It uses an adapted Depth Pro backbone to predict two depth layers, then
  predicts roughly 1.2 million 3D Gaussians. Its depth adjustment module is
  the identity function at inference. Training uses paired views; published
  comparisons primarily evaluate rendered image fidelity.
- The [released predictor](https://github.com/apple-aiml-research/ml-sharp/blob/main/src/sharp/models/predictor.py)
  forms metric depth as `(focal_px / image_width) / predicted_disparity` before
  Gaussian prediction. The [CLI](https://github.com/apple-aiml-research/ml-sharp/blob/main/src/sharp/cli/predict.py)
  exports Gaussians as PLY; this pilot captures the two intermediate depth
  layers during the same model forward pass.
- The [image loader](https://github.com/apple-aiml-research/ml-sharp/blob/main/src/sharp/utils/io.py)
  uses EXIF focal length or a 30 mm equivalent fallback. CrowdPose image
  `107292.jpg` has no focal-length EXIF, so SHARP assumed `531.9 px` at the
  original `640×423` resolution. Metric values inherit this camera assumption.
- The official [model license](https://github.com/apple-aiml-research/ml-sharp/blob/main/LICENSE_MODEL)
  restricts the released checkpoint to research purposes.

## Pilot artifacts

| Artifact | Contents |
|---|---|
| [Scene diagnostic](assets/107292_sharp_diagnostic.png) | Source RGB, both full-resolution depth layers, foreground Gaussian centers from camera and angled views. |
| [Person-box diagnostic](assets/107292_sharp_people.png) | Annotated person boxes, layer-one depth, and Gaussian centers whose projections fall inside those boxes. Boxes include background; they are not masks. |
| [Depth and RGB arrays](assets/107292_sharp_depth.npz) | Two `1536×1536` float32 metric depth layers, aligned stretched RGB, original image size, assumed focal length. |
| [Gaussian PLY](assets/107292_sharp_gaussians_sampled.ply) | Evenly spaced 200,000-splat sample of the full approximately 1.2 million output, using SHARP's own PLY exporter. |
| [All final Gaussian centers](assets/107292_sharp_all_centers.npz) | Metric camera-frame centers, RGB, opacity, and processed-image intrinsics for the full output. |
| [RGBD point-cloud notebook](sharp-rgbd-pointcloud.ipynb) | Executed notebook projecting final Gaussian centers to a sparse depth raster, pairing valid pixels with source RGB, and back-projecting a colored point cloud. Includes interactive scene and person-box views. |
| [RGBD overview](assets/107292_sharp_refined_rgbd_overview.png) and [fixed 3D views](assets/107292_sharp_refined_rgbd_orthographic.png) | Intermediate depth beside derived sparse depth; top and side views of the person-box points. |
| [Run metadata](assets/107292_sharp_metadata.json) | Input, output, focal assumption, sample count, and timing. |
| [DA3 comparison](assets/107292_da3base_pointcloud.png) | Earlier RGBD and point-cloud diagnostic on the same image. |

The pilot used the official source at commit
`aed6527499ef91cba3b54c18d49a870f25947190` and checkpoint
`sharp_2572gikvuh.pt`. It ran on CPU with two threads and no visible CUDA
device; total runtime was 190 seconds. The training GPU remained busy with
its existing job. The 200,000-splat PLY has 33,992 points with opacity at
least 0.5 that project inside a person box and lie within 12 m.

Layer one depth has foreground values of roughly 1–5 m in this scene. Its
far sky reaches the disparity-clamp ceiling of about 8,311 m; that ceiling
is not a measured sky distance. Diagnostic views cap the displayed depth
range and restrict 3D scatter to the foreground so distant points do not
hide the people. Layer two places much of the sky near 100 m. Neither layer
reveals the backs or internal structure of the people.

The notebook uses the **final Gaussian centers** for its primary RGBD cloud.
It projects centers with opacity at least 0.5 into the source camera and keeps
the nearest center per pixel. This sparse z-buffer covers 37.4% of processed
pixels; 17,950 displayed points lie within 12 m at stride four, including
5,817 selected by person boxes. Pixel-to-3D-to-pixel round-trip error is below
0.0001 pixel. This is a center-based approximation, not a SHARP-rendered depth
map: Gaussian extent and alpha compositing are absent. The first depth layer
is a separate intermediate visible-surface prediction. The fixed top and side
views show layered surfaces around the people, without clear skeletal geometry.

## Reproduction

The adapter is [`scripts/investigation/sharp_feasibility.py`](../../scripts/investigation/sharp_feasibility.py).
It expects a local official SHARP checkout, checkpoint, and import-only
dependencies. From the repository root:

```bash
git clone --depth 1 https://github.com/apple/ml-sharp.git tmp/ml-sharp
venv/bin/python -m pip install --target tmp/sharp-deps --no-deps pillow-heif imageio plyfile
mkdir -p tmp/sharp-checkpoints
curl -L --fail -o tmp/sharp-checkpoints/sharp_2572gikvuh.pt https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TMPDIR=$PWD/tmp PYTHONPATH=$PWD/tmp/sharp-deps:$PWD/tmp/ml-sharp/src venv/bin/python scripts/investigation/sharp_feasibility.py --image data/crowdpose/images/107292.jpg --checkpoint tmp/sharp-checkpoints/sharp_2572gikvuh.pt --output-dir reports/3d-prior/assets --threads 2 --save-all-centers
```

## Decision

The depth-based 3D-prior route is closed for this project. The inspected DA3
point clouds and SHARP final-geometry RGBD view do not provide reliable enough
human geometry for the intended skeletal reasoning. This is a task-feasibility
decision from the inspected visualizations, not a quantitative benchmark of
either model's general depth accuracy. The notebooks and exported artifacts
remain as the evidence behind the decision. Future experiments in this thread
will use DINO image features without relying on inferred 3D geometry.
