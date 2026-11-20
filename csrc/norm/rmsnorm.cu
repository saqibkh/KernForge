#include "kernforge/norm/rmsnorm.h"

#include <cuda_runtime.h>
#include <cmath>

namespace kernforge {
namespace {

// naive
// 每个 Block 处理一行。
// 暂时固定 256 线程

__global__ void rmsnorm_kernel(
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

    // 1. 每个线程计算自己的局部平方和
    float sum = 0.0f;

    for (int i = tid; i < cols; i += blockDim.x) {
        float v = x[i];
        sum += v * v;
    }

    smem[tid] = sum;
    __syncthreads();

    // 2. Block 内规约
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        } 
        __syncthreads();
    }

    // 3. 计算归一化系数
    float inv_rms = rsqrtf(smem[0] / cols + eps);

    // 4. 写入结果
    for (int i = tid; i < cols; i += blockDim.x) {
        y[i] = x[i] * inv_rms * weight[i];
    }
}

}

// Host 端 Launcher
cudaError_t rmsnorm(
        const float* input,
        const float* weight,
        float* output,
        int rows,
        int cols,
        float eps,
        cudaStream_t stream) {

    if (input == nullptr ||
        weight == nullptr ||
        output == nullptr ||
        rows <= 0 ||
        cols <= 0 ||
        !std::isfinite(eps) ||
        eps <= 0.0f) {
            return cudaErrorInvalidValue;
    }

    constexpr int threads = 256;
    dim3 block(threads);
    dim3 grid(rows);

    rmsnorm_kernel<<<grid, block, 0, stream>>>(
        input, weight, output, cols, eps);

    return cudaPeekAtLastError();
}

}