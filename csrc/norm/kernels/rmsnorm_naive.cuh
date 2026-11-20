#pragma once

#include <cuda_runtime.h>
#include <cstddef>

namespace kernforge {
namespace detail {

// RMSNorm Naive Kernel
// 每个 Block 处理一行
// 固定使用 256 Threads / Block
__global__ void rmsnorm_naive_kernel(
        const float* input,
        const float* weight,
        float* output,
        int cols,
        float eps) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    __shared__ float smem[256];

    const float* x = input + static_cast<size_t>(row) * cols;
    float* y = output + static_cast<size_t>(row) * cols;

    float sum = 0.0f;

    for (int i = tid; i < cols; i += blockDim.x) {
        float v = x[i];
        sum += v * v;
    }

    smem[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    float inv_rms = rsqrtf(smem[0] / cols + eps);

    for (int i = tid; i < cols; i += blockDim.x) {
        y[i] = x[i] * inv_rms * weight[i];
    }
}

} // namespace detail
} // namespace kernforge
