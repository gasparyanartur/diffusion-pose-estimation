# DINOv3 multi-scale deformable-attention CUDA sources

This directory vendors the following CUDA sources from Meta's DINOv3
repository:

- Repository: https://github.com/facebookresearch/dinov3
- Pinned revision: `6876159a11b4df116f30f667f8c9888617df0751`
- Source path:
  `dinov3/eval/segmentation/models/utils/ops/src/cuda/`
- Imported files: `ms_deform_attn_cuda.cu`, `cuda/ms_deform_attn_cuda.h`,
  and `cuda/ms_deform_im2col_cuda.cuh`

The imported DINOv3 material is used under the DINOv3 License Agreement.
A copy of that agreement is included in `DINOv3-LICENSE.md`. The imported
source also contains the upstream Deformable DETR/SenseTime attribution and
license notice, which is preserved verbatim.

`binding.cpp` is a repository-local adapter and is not part of DINOv3.
The adapter exposes the existing project-level forward/backward interface.

