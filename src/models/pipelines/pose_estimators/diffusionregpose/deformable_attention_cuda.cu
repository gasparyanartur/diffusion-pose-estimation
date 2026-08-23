#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <vector>

// Repository-owned CUDA implementation of multi-scale deformable attention.
// The sampling operation follows the Deformable DETR operator (SenseTime,
// Apache-2.0); this implementation is maintained locally and does not load
// the external project's compiled extension.
// Sampling uses zero-padded bilinear interpolation with align_corners=False.

namespace {

template <typename scalar_t>
__device__ inline scalar_t bilinear_value(
    const scalar_t* value,
    int height,
    int width,
    int heads,
    int channels,
    scalar_t y,
    scalar_t x,
    int head,
    int channel) {
    const int y_low = static_cast<int>(floor(y));
    const int x_low = static_cast<int>(floor(x));
    const scalar_t y_frac = y - y_low;
    const scalar_t x_frac = x - x_low;
    const scalar_t y_weight = scalar_t(1) - y_frac;
    const scalar_t x_weight = scalar_t(1) - x_frac;
    const int stride = width * heads * channels;
    const int channel_offset = head * channels + channel;

    scalar_t top_left = scalar_t(0);
    scalar_t top_right = scalar_t(0);
    scalar_t bottom_left = scalar_t(0);
    scalar_t bottom_right = scalar_t(0);
    if (y_low >= 0 && y_low < height && x_low >= 0 && x_low < width) {
        top_left = value[y_low * stride + x_low * heads * channels + channel_offset];
    }
    if (y_low >= 0 && y_low < height && x_low + 1 >= 0 && x_low + 1 < width) {
        top_right = value[y_low * stride + (x_low + 1) * heads * channels + channel_offset];
    }
    if (y_low + 1 >= 0 && y_low + 1 < height && x_low >= 0 && x_low < width) {
        bottom_left = value[(y_low + 1) * stride + x_low * heads * channels + channel_offset];
    }
    if (y_low + 1 >= 0 && y_low + 1 < height && x_low + 1 >= 0 && x_low + 1 < width) {
        bottom_right = value[(y_low + 1) * stride + (x_low + 1) * heads * channels + channel_offset];
    }
    return y_weight * x_weight * top_left
        + y_weight * x_frac * top_right
        + y_frac * x_weight * bottom_left
        + y_frac * x_frac * bottom_right;
}

template <typename scalar_t>
__global__ void deformable_attention_forward_kernel(
    int count,
    const scalar_t* value,
    const int64_t* spatial_shapes,
    const int64_t* level_start_index,
    const scalar_t* sampling_locations,
    const scalar_t* attention_weights,
    int spatial_size,
    int heads,
    int channels,
    int levels,
    int queries,
    int points,
    scalar_t* output) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < count;
         index += blockDim.x * gridDim.x) {
        int remainder = index;
        const int channel = remainder % channels;
        remainder /= channels;
        const int head = remainder % heads;
        remainder /= heads;
        const int query = remainder % queries;
        const int batch = remainder / queries;
        const int value_batch_offset = batch * spatial_size * heads * channels;
        const int location_base = ((batch * queries + query) * heads + head)
            * levels * points * 2;
        const int weight_base = ((batch * queries + query) * heads + head)
            * levels * points;
        scalar_t result = scalar_t(0);
        int sample_index = 0;
        for (int level = 0; level < levels; ++level) {
            const int height = spatial_shapes[2 * level];
            const int width = spatial_shapes[2 * level + 1];
            const scalar_t* level_value = value
                + value_batch_offset + level_start_index[level] * heads * channels;
            for (int point = 0; point < points; ++point, ++sample_index) {
                const scalar_t x = sampling_locations[location_base + 2 * sample_index];
                const scalar_t y = sampling_locations[location_base + 2 * sample_index + 1];
                const scalar_t sample_x = x * width - scalar_t(0.5);
                const scalar_t sample_y = y * height - scalar_t(0.5);
                if (sample_x > -1 && sample_y > -1 && sample_x < width && sample_y < height) {
                    result += bilinear_value(
                        level_value, height, width, heads, channels,
                        sample_y, sample_x, head, channel
                    ) * attention_weights[weight_base + sample_index];
                }
            }
        }
        output[index] = result;
    }
}

template <typename scalar_t>
__global__ void deformable_attention_backward_kernel(
    int count,
    const scalar_t* value,
    const int64_t* spatial_shapes,
    const int64_t* level_start_index,
    const scalar_t* sampling_locations,
    const scalar_t* attention_weights,
    const scalar_t* grad_output,
    int spatial_size,
    int heads,
    int channels,
    int levels,
    int queries,
    int points,
    scalar_t* grad_value,
    scalar_t* grad_sampling_locations,
    scalar_t* grad_attention_weights) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < count;
         index += blockDim.x * gridDim.x) {
        int remainder = index;
        const int point = remainder % points;
        remainder /= points;
        const int level = remainder % levels;
        remainder /= levels;
        const int head = remainder % heads;
        remainder /= heads;
        const int query = remainder % queries;
        const int batch = remainder / queries;

        const int sample_index = level * points + point;
        const int location_base = ((batch * queries + query) * heads + head)
            * levels * points * 2 + sample_index * 2;
        const int weight_base = ((batch * queries + query) * heads + head)
            * levels * points + sample_index;
        const scalar_t x = sampling_locations[location_base];
        const scalar_t y = sampling_locations[location_base + 1];
        const int height = spatial_shapes[2 * level];
        const int width = spatial_shapes[2 * level + 1];
        const scalar_t sample_x = x * width - scalar_t(0.5);
        const scalar_t sample_y = y * height - scalar_t(0.5);
        const scalar_t attention_weight = attention_weights[weight_base];
        const scalar_t* level_value = value
            + batch * spatial_size * heads * channels
            + level_start_index[level] * heads * channels;
        scalar_t* level_grad_value = grad_value
            + batch * spatial_size * heads * channels
            + level_start_index[level] * heads * channels;

        const int y_low = static_cast<int>(floor(sample_y));
        const int x_low = static_cast<int>(floor(sample_x));
        const scalar_t y_frac = sample_y - y_low;
        const scalar_t x_frac = sample_x - x_low;
        const scalar_t y_weight = scalar_t(1) - y_frac;
        const scalar_t x_weight = scalar_t(1) - x_frac;
        const scalar_t* output_gradient = grad_output
            + ((batch * queries + query) * heads + head) * channels;
        const int stride = width * heads * channels;
        scalar_t location_gradient_x = scalar_t(0);
        scalar_t location_gradient_y = scalar_t(0);
        scalar_t weight_gradient = scalar_t(0);

        for (int channel = 0; channel < channels; ++channel) {
            const scalar_t gradient = output_gradient[channel];
            const int channel_offset = head * channels + channel;
            scalar_t top_left = scalar_t(0);
            scalar_t top_right = scalar_t(0);
            scalar_t bottom_left = scalar_t(0);
            scalar_t bottom_right = scalar_t(0);
            if (y_low >= 0 && y_low < height && x_low >= 0 && x_low < width) {
                top_left = level_value[y_low * stride + x_low * heads * channels + channel_offset];
            }
            if (y_low >= 0 && y_low < height && x_low + 1 >= 0 && x_low + 1 < width) {
                top_right = level_value[y_low * stride + (x_low + 1) * heads * channels + channel_offset];
            }
            if (y_low + 1 >= 0 && y_low + 1 < height && x_low >= 0 && x_low < width) {
                bottom_left = level_value[(y_low + 1) * stride + x_low * heads * channels + channel_offset];
            }
            if (y_low + 1 >= 0 && y_low + 1 < height && x_low + 1 >= 0 && x_low + 1 < width) {
                bottom_right = level_value[(y_low + 1) * stride + (x_low + 1) * heads * channels + channel_offset];
            }
            const scalar_t interpolated = y_weight * x_weight * top_left
                + y_weight * x_frac * top_right
                + y_frac * x_weight * bottom_left
                + y_frac * x_frac * bottom_right;
            const scalar_t gradient_times_weight = gradient * attention_weight;
            weight_gradient += gradient * interpolated;
            location_gradient_x += gradient_times_weight * (
                -y_weight * top_left + y_weight * top_right
                -y_frac * bottom_left + y_frac * bottom_right
            );
            location_gradient_y += gradient_times_weight * (
                -x_weight * top_left - x_frac * top_right
                + x_weight * bottom_left + x_frac * bottom_right
            );

            if (y_low >= 0 && y_low < height && x_low >= 0 && x_low < width) {
                atomicAdd(level_grad_value + y_low * stride + x_low * heads * channels + channel_offset,
                          gradient_times_weight * y_weight * x_weight);
            }
            if (y_low >= 0 && y_low < height && x_low + 1 >= 0 && x_low + 1 < width) {
                atomicAdd(level_grad_value + y_low * stride + (x_low + 1) * heads * channels + channel_offset,
                          gradient_times_weight * y_weight * x_frac);
            }
            if (y_low + 1 >= 0 && y_low + 1 < height && x_low >= 0 && x_low < width) {
                atomicAdd(level_grad_value + (y_low + 1) * stride + x_low * heads * channels + channel_offset,
                          gradient_times_weight * y_frac * x_weight);
            }
            if (y_low + 1 >= 0 && y_low + 1 < height && x_low + 1 >= 0 && x_low + 1 < width) {
                atomicAdd(level_grad_value + (y_low + 1) * stride + (x_low + 1) * heads * channels + channel_offset,
                          gradient_times_weight * y_frac * x_frac);
            }
        }
        grad_attention_weights[weight_base] = weight_gradient;
        grad_sampling_locations[location_base] = location_gradient_x * width;
        grad_sampling_locations[location_base + 1] = location_gradient_y * height;
    }
}

template <typename scalar_t>
void launch_forward(
    const at::Tensor& value,
    const at::Tensor& spatial_shapes,
    const at::Tensor& level_start_index,
    const at::Tensor& sampling_locations,
    const at::Tensor& attention_weights,
    at::Tensor& output) {
    const int batch = value.size(0);
    const int spatial_size = value.size(1);
    const int heads = value.size(2);
    const int channels = value.size(3);
    const int levels = spatial_shapes.size(0);
    const int queries = sampling_locations.size(1);
    const int points = sampling_locations.size(4);
    const int count = batch * queries * heads * channels;
    const int blocks = std::max(1, (count + 255) / 256);
    deformable_attention_forward_kernel<scalar_t><<<blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        count, value.data_ptr<scalar_t>(), spatial_shapes.data_ptr<int64_t>(),
        level_start_index.data_ptr<int64_t>(), sampling_locations.data_ptr<scalar_t>(),
        attention_weights.data_ptr<scalar_t>(), spatial_size, heads, channels,
        levels, queries, points, output.data_ptr<scalar_t>()
    );
}

template <typename scalar_t>
void launch_backward(
    const at::Tensor& value,
    const at::Tensor& spatial_shapes,
    const at::Tensor& level_start_index,
    const at::Tensor& sampling_locations,
    const at::Tensor& attention_weights,
    const at::Tensor& grad_output,
    at::Tensor& grad_value,
    at::Tensor& grad_sampling_locations,
    at::Tensor& grad_attention_weights) {
    const int batch = value.size(0);
    const int spatial_size = value.size(1);
    const int heads = value.size(2);
    const int channels = value.size(3);
    const int levels = spatial_shapes.size(0);
    const int queries = sampling_locations.size(1);
    const int points = sampling_locations.size(4);
    const int count = batch * queries * heads * levels * points;
    const int blocks = std::max(1, (count + 255) / 256);
    deformable_attention_backward_kernel<scalar_t><<<blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        count, value.data_ptr<scalar_t>(), spatial_shapes.data_ptr<int64_t>(),
        level_start_index.data_ptr<int64_t>(), sampling_locations.data_ptr<scalar_t>(),
        attention_weights.data_ptr<scalar_t>(), grad_output.data_ptr<scalar_t>(),
        spatial_size, heads, channels, levels, queries, points,
        grad_value.data_ptr<scalar_t>(), grad_sampling_locations.data_ptr<scalar_t>(),
        grad_attention_weights.data_ptr<scalar_t>()
    );
}

void check_inputs(
    const at::Tensor& value,
    const at::Tensor& spatial_shapes,
    const at::Tensor& level_start_index,
    const at::Tensor& sampling_locations,
    const at::Tensor& attention_weights) {
    TORCH_CHECK(value.is_cuda(), "value must be CUDA");
    TORCH_CHECK(spatial_shapes.is_cuda() && spatial_shapes.scalar_type() == torch::kInt64,
                "spatial_shapes must be CUDA int64");
    TORCH_CHECK(level_start_index.is_cuda() && level_start_index.scalar_type() == torch::kInt64,
                "level_start_index must be CUDA int64");
    TORCH_CHECK(sampling_locations.is_cuda(), "sampling_locations must be CUDA");
    TORCH_CHECK(attention_weights.is_cuda(), "attention_weights must be CUDA");
    TORCH_CHECK(value.is_contiguous() && spatial_shapes.is_contiguous()
                && level_start_index.is_contiguous() && sampling_locations.is_contiguous()
                && attention_weights.is_contiguous(), "all inputs must be contiguous");
    TORCH_CHECK(value.scalar_type() == sampling_locations.scalar_type()
                && value.scalar_type() == attention_weights.scalar_type(),
                "value, sampling locations, and attention weights must have the same dtype");
    TORCH_CHECK(value.dim() == 4 && sampling_locations.dim() == 6
                && attention_weights.dim() == 5, "invalid deformable-attention input rank");
}

}  // namespace

at::Tensor forward(
    at::Tensor value,
    at::Tensor spatial_shapes,
    at::Tensor level_start_index,
    at::Tensor sampling_locations,
    at::Tensor attention_weights) {
    check_inputs(value, spatial_shapes, level_start_index, sampling_locations, attention_weights);
    auto output = at::zeros(
        {value.size(0), sampling_locations.size(1), value.size(2), value.size(3)},
        value.options()
    );
    AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "diffusionregpose_msda_forward", [&] {
        launch_forward<scalar_t>(value, spatial_shapes, level_start_index,
                                 sampling_locations, attention_weights, output);
    });
    return output.reshape({value.size(0), sampling_locations.size(1), value.size(2) * value.size(3)});
}

std::vector<at::Tensor> backward(
    at::Tensor value,
    at::Tensor spatial_shapes,
    at::Tensor level_start_index,
    at::Tensor sampling_locations,
    at::Tensor attention_weights,
    at::Tensor grad_output) {
    check_inputs(value, spatial_shapes, level_start_index, sampling_locations, attention_weights);
    TORCH_CHECK(grad_output.is_cuda() && grad_output.is_contiguous(),
                "grad_output must be contiguous CUDA");
    auto grad_value = at::zeros_like(value);
    auto grad_sampling_locations = at::zeros_like(sampling_locations);
    auto grad_attention_weights = at::zeros_like(attention_weights);
    AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "diffusionregpose_msda_backward", [&] {
        launch_backward<scalar_t>(value, spatial_shapes, level_start_index,
                                  sampling_locations, attention_weights, grad_output,
                                  grad_value, grad_sampling_locations, grad_attention_weights);
    });
    return {grad_value, grad_sampling_locations, grad_attention_weights};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("forward", &forward, "multi-scale deformable attention forward");
    module.def("backward", &backward, "multi-scale deformable attention backward");
}
