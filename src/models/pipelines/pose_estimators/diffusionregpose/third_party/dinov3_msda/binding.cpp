#include <torch/extension.h>

#include <vector>

at::Tensor ms_deform_attn_cuda_forward(
    const at::Tensor& value,
    const at::Tensor& spatial_shapes,
    const at::Tensor& level_start_index,
    const at::Tensor& sampling_locations,
    const at::Tensor& attention_weights,
    const int im2col_step);

std::vector<at::Tensor> ms_deform_attn_cuda_backward(
    const at::Tensor& value,
    const at::Tensor& spatial_shapes,
    const at::Tensor& level_start_index,
    const at::Tensor& sampling_locations,
    const at::Tensor& attention_weights,
    const at::Tensor& grad_output,
    const int im2col_step);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("forward", &ms_deform_attn_cuda_forward,
               "DINOv3 multi-scale deformable attention forward",
               py::arg("value"), py::arg("spatial_shapes"),
               py::arg("level_start_index"), py::arg("sampling_locations"),
               py::arg("attention_weights"), py::arg("im2col_step") = 64);
    module.def("backward", &ms_deform_attn_cuda_backward,
               "DINOv3 multi-scale deformable attention backward",
               py::arg("value"), py::arg("spatial_shapes"),
               py::arg("level_start_index"), py::arg("sampling_locations"),
               py::arg("attention_weights"), py::arg("grad_output"),
               py::arg("im2col_step") = 64);
}

