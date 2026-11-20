#pragma once

#include <cuda_runtime_api.h>

namespace kernforge {

// 第一版仅支持：
// 1. FP32
// 2. 连续存储的二维矩阵 [rows, cols]
// 3. weight 形状为 [cols]
// 4. input、weight、output 均为有效的 GPU 指针
// 5. output 不与 input 或 weight 重叠
//
// 调用是异步的，使用调用者提供的 CUDA Stream。

cudaError_t rmsnorm(
    const float* input,
    const float* weight,
    float* output,
    int rows,
    int cols,
    float eps,
    cudaStream_t stream = nullptr
);

} // namespace kernforge