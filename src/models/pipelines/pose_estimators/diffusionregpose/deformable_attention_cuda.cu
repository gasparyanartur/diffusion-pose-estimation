#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>

namespace {

template <typename scalar_t>
__device__ scalar_t bilinear_sample(
    const scalar_t* value,
    int height,
    int width,
    int heads,
    int channels,
    scalar_t y,
    scalar_t x,
    int head,
    int channel) {
    const int y_low = floor(y);
    const int x_low = floor(x);
    const int y_high = y_low + 1;
    const int x_high = x_low + 1;
    const scalar_t y_fraction = y - y_low;
    const scalar_t x_fraction = x - x_low;
    const scalar_t y_complement = 1 - y_fraction;
    const scalar_t x_complement = 1 - x_fraction;
    const int pixel_stride = heads * channels;
    const int row_stride = width * pixel_stride;
    const int channel_offset = head * channels + channel;

    scalar_t top_left = 0;
    if (y_low >= 0 && x_low >= 0) {
        top_left = value[y_low * row_stride + x_low * pixel_stride + channel_offset];
    }
    scalar_t top_right = 0;
    if (y_low >= 0 && x_high <= width - 1) {
        top_right = value[y_low * row_stride + x_high * pixel_stride + channel_offset];
    }
    scalar_t bottom_left = 0;
    if (y_high <= height - 1 && x_low >= 0) {
        bottom_left = value[y_high * row_stride + x_low * pixel_stride + channel_offset];
    }
    scalar_t bottom_right = 0;
    if (y_high <= height - 1 && x_high <= width - 1) {
        bottom_right = value[y_high * row_stride + x_high * pixel_stride + channel_offset];
    }

    const scalar_t top_left_weight = y_complement * x_complement;
    const scalar_t top_right_weight = y_complement * x_fraction;
    const scalar_t bottom_left_weight = y_fraction * x_complement;
    const scalar_t bottom_right_weight = y_fraction * x_fraction;
    return top_left_weight * top_left
        + top_right_weight * top_right
        + bottom_left_weight * bottom_left
        + bottom_right_weight * bottom_right;
}

template <typename scalar_t>
__global__ void deformable_attention_kernel(
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
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < count; index += blockDim.x * gridDim.x) {
        int channel = index % channels;
        int remainder = index / channels;
        const int head = remainder % heads;
        remainder /= heads;
        const int query = remainder % queries;
        const int batch = remainder / queries;
        const scalar_t* batch_value = value + batch * spatial_size * heads * channels;
        const scalar_t* query_locations = sampling_locations
            + ((batch * queries + query) * heads + head) * levels * points * 2;
        const scalar_t* query_weights = attention_weights
            + ((batch * queries + query) * heads + head) * levels * points;
        scalar_t result = 0;
        int point_index = 0;
        for (int level = 0; level < levels; ++level) {
            const int height = spatial_shapes[2 * level];
            const int width = spatial_shapes[2 * level + 1];
            const scalar_t* level_value = batch_value + level_start_index[level] * heads * channels;
            for (int point = 0; point < points; ++point) {
                const scalar_t x = query_locations[2 * point_index];
                const scalar_t y = query_locations[2 * point_index + 1];
                const scalar_t sample_y = y * height - static_cast<scalar_t>(0.5);
                const scalar_t sample_x = x * width - static_cast<scalar_t>(0.5);
                if (sample_y > -1 && sample_x > -1 && sample_y < height && sample_x < width) {
                    result += bilinear_sample(
                        level_value,
                        height,
                        width,
                        heads,
                        channels,
                        sample_y,
                        sample_x,
                        head,
                        channel
                    ) * query_weights[point_index];
                }
                ++point_index;
            }
        }
        output[index] = result;
    }
}

}  // namespace

torch::Tensor forward(
    torch::Tensor value,
    torch::Tensor spatial_shapes,
    torch::Tensor level_start_index,
    torch::Tensor sampling_locations,
    torch::Tensor attention_weights) {
    TORCH_CHECK(value.is_cuda(), "value must be a CUDA tensor");
    TORCH_CHECK(value.is_contiguous(), "value must be contiguous");
    TORCH_CHECK(spatial_shapes.is_cuda() && spatial_shapes.is_contiguous(), "spatial shapes must be contiguous CUDA data");
    TORCH_CHECK(level_start_index.is_cuda() && level_start_index.is_contiguous(), "level starts must be contiguous CUDA data");
    TORCH_CHECK(sampling_locations.is_cuda() && sampling_locations.is_contiguous(), "sampling locations must be contiguous CUDA data");
    TORCH_CHECK(attention_weights.is_cuda() && attention_weights.is_contiguous(), "attention weights must be contiguous CUDA data");
    TORCH_CHECK(value.scalar_type() == sampling_locations.scalar_type(), "value and sampling locations must have the same dtype");
    TORCH_CHECK(value.scalar_type() == attention_weights.scalar_type(), "value and attention weights must have the same dtype");
    TORCH_CHECK(spatial_shapes.scalar_type() == torch::kInt64, "spatial shapes must be int64");
    TORCH_CHECK(level_start_index.scalar_type() == torch::kInt64, "level starts must be int64");

    const int batch = value.size(0);
    const int spatial_size = value.size(1);
    const int heads = value.size(2);
    const int channels = value.size(3);
    const int levels = spatial_shapes.size(0);
    const int queries = sampling_locations.size(1);
    const int points = sampling_locations.size(4);
    auto output = torch::zeros({batch, queries, heads, channels}, value.options());
    const int count = batch * queries * heads * channels;
    const int blocks = std::max(1, (count + 255) / 256);

    AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "diffusionregpose_msda_forward", [&] {
        deformable_attention_kernel<scalar_t><<<blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            count,
            value.data_ptr<scalar_t>(),
            spatial_shapes.data_ptr<int64_t>(),
            level_start_index.data_ptr<int64_t>(),
            sampling_locations.data_ptr<scalar_t>(),
            attention_weights.data_ptr<scalar_t>(),
            spatial_size,
            heads,
            channels,
            levels,
            queries,
            points,
            output.data_ptr<scalar_t>()
        );
    });
    return output.reshape({batch, queries, heads * channels});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("forward", &forward, "DiffusionRegPose deformable attention forward");
}
