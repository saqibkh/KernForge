
#include "kernforge/norm/rmsnorm.h"

#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                    \
    do {                                                    \
        cudaError_t err = (call);                           \
        if (err != cudaSuccess) {                           \
            std::cerr << "CUDA error: "                      \
                      << cudaGetErrorString(err)             \
                      << std::endl;                         \
            return 1;                                       \
        }                                                   \
    } while (0)

int main() {
    constexpr int rows = 2;
    constexpr int cols = 4;
    constexpr float eps = 1e-6f;

    std::vector<float> h_input = {
        1.0f, 2.0f, 3.0f, 4.0f,
        2.0f, 3.0f, 4.0f, 5.0f
    };

    std::vector<float> h_weight(cols, 1.0f);
    std::vector<float> h_output(rows * cols, 0.0f);
    std::vector<float> h_reference(rows * cols, 0.0f);

    // CPU 参考实现
    for (int r = 0; r < rows; ++r) {
        float sum = 0.0f;

        for (int c = 0; c < cols; ++c) {
            float v = h_input[r * cols + c];
            sum += v * v;
        }

        float inv_rms = 1.0f /
            std::sqrt(sum / cols + eps);

        for (int c = 0; c < cols; ++c) {
            h_reference[r * cols + c] =
                h_input[r * cols + c] *
                inv_rms * h_weight[c];
        }
    }

    // GPU 内存
    float *d_input = nullptr;
    float *d_weight = nullptr;
    float *d_output = nullptr;

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_input),
        rows * cols * sizeof(float)));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_weight),
        cols * sizeof(float)));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_output),
        rows * cols * sizeof(float)));

    // Host → Device
    CUDA_CHECK(cudaMemcpy(
        d_input, h_input.data(),
        rows * cols * sizeof(float),
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        d_weight, h_weight.data(),
        cols * sizeof(float),
        cudaMemcpyHostToDevice));

    // 调用 KernForge 共享库
    CUDA_CHECK(kernforge::rmsnorm(
        d_input,
        d_weight,
        d_output,
        rows,
        cols,
        eps
    ));

    // 等待 GPU 完成，检查异步执行错误
    CUDA_CHECK(cudaDeviceSynchronize());

    // Device → Host
    CUDA_CHECK(cudaMemcpy(
        h_output.data(),
        d_output,
        rows * cols * sizeof(float),
        cudaMemcpyDeviceToHost));

    // 正确性验证
    bool passed = true;
    constexpr float atol = 1e-5f;

    for (int i = 0; i < rows * cols; ++i) {
        float diff = std::abs(
            h_output[i] - h_reference[i]);

        if (diff > atol || !std::isfinite(diff)) {
            passed = false;
            std::cerr << "Mismatch at " << i
                      << ": GPU=" << h_output[i]
                      << ", CPU=" << h_reference[i]
                      << std::endl;
        }
    }

    std::cout << (passed
        ? "RMSNorm Test PASSED"
        : "RMSNorm Test FAILED") << std::endl;

    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);

    return passed ? 0 : 1;
}
