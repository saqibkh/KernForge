#include "kernforge/norm/rmsnorm.h"
#include "kernels/rmsnorm_naive.cuh"

#include <cuda_runtime.h>
#include <cmath>

namespace kernforge {

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

    dim3 grid(rows);
    dim3 block(threads);

    detail::rmsnorm_naive_kernel<<<grid, block, 0, stream>>>(
        input, weight, output, cols, eps);

    return cudaPeekAtLastError();
}

} // namespace kernforge