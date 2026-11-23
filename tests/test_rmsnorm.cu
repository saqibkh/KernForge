
#include "kernforge/norm/rmsnorm.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

// CUDA 错误检查
bool check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::cerr << "[CUDA ERROR] "
                  << operation << ": "
                  << cudaGetErrorString(status)
                  << std::endl;
        return false;
    }
    return true;
}

struct TestCase {
    int rows;
    int cols;
    const char* name = "random";
    float eps = 1e-6f;
    float input_scale = 1.0f;
    bool capture_stream = false;
};

bool has_kernel_node(cudaGraph_t graph) {
    size_t count = 0;
    if (!check_cuda(cudaGraphGetNodes(graph, nullptr, &count), "cudaGraphGetNodes(count)")) {
        return false;
    }

    if (count > 0) {
        std::vector<cudaGraphNode_t> nodes(count);
        if (!check_cuda(cudaGraphGetNodes(graph, nodes.data(), &count), "cudaGraphGetNodes")) {
            return false;
        }
        for (cudaGraphNode_t node : nodes) {
            cudaGraphNodeType type;
            if (!check_cuda(cudaGraphNodeGetType(node, &type), "cudaGraphNodeGetType")) {
                return false;
            }
            if (type == cudaGraphNodeTypeKernel) return true;
        }
    }

    std::cerr << "RMSNorm did not enqueue a kernel on the supplied stream\n";
    return false;
}

bool run_invalid_arguments_test() {
    float* buffers = nullptr;
    if (!check_cuda(cudaMalloc(&buffers, 3 * sizeof(float)), "cudaMalloc(invalid arguments)")) {
        return false;
    }

    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    constexpr float eps = 1e-6f;
    const struct {
        const char* name;
        const float* input;
        const float* weight;
        float* output;
        int rows;
        int cols;
        float eps;
    } cases[] = {
        {"null input", nullptr, buffers + 1, buffers + 2, 1, 1, eps},
        {"null weight", buffers, nullptr, buffers + 2, 1, 1, eps},
        {"null output", buffers, buffers + 1, nullptr, 1, 1, eps},
        {"zero rows", buffers, buffers + 1, buffers + 2, 0, 1, eps},
        {"negative rows", buffers, buffers + 1, buffers + 2, -1, 1, eps},
        {"zero cols", buffers, buffers + 1, buffers + 2, 1, 0, eps},
        {"negative cols", buffers, buffers + 1, buffers + 2, 1, -1, eps},
        {"zero eps", buffers, buffers + 1, buffers + 2, 1, 1, 0.0f},
        {"negative eps", buffers, buffers + 1, buffers + 2, 1, 1, -eps},
        {"NaN eps", buffers, buffers + 1, buffers + 2, 1, 1, nan},
        {"infinite eps", buffers, buffers + 1, buffers + 2, 1, 1, inf},
    };

    bool passed = true;
    for (const auto& test : cases) {
        const cudaError_t status = kernforge::rmsnorm(
            test.input, test.weight, test.output, test.rows, test.cols, test.eps);
        if (status != cudaErrorInvalidValue) {
            std::cerr << "Invalid argument (" << test.name << "): expected cudaErrorInvalidValue, got "
                      << cudaGetErrorName(status) << '\n';
            passed = false;
        }
    }
    return check_cuda(cudaFree(buffers), "cudaFree(invalid arguments)") && passed;
}

// 执行一组 RMSNorm 正确性测试
bool run_rmsnorm_test(const TestCase& test,
                      std::mt19937& rng) {
    const int rows = test.rows;
    const int cols = test.cols;
    const float eps = test.eps;
    constexpr float atol = 2e-5f;
    constexpr float rtol = 2e-4f;

    const size_t input_size =
        static_cast<size_t>(rows) * cols;

    // Host 数据
    std::vector<float> h_input(input_size);
    std::vector<float> h_weight(cols);
    std::vector<float> h_output(input_size);
    std::vector<float> h_reference(input_size);

    // 生成随机数据
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (auto& x : h_input) {
        x = test.input_scale == 0.0f ? 0.0f : dist(rng) * test.input_scale;
    }

    for (auto& w : h_weight) {
        w = dist(rng);
    }

    // CPU 参考实现，使用 double 累加平方和
    for (int r = 0; r < rows; ++r) {
        double sum = 0.0;

        for (int c = 0; c < cols; ++c) {
            double x = h_input[static_cast<size_t>(r) * cols + c];
            sum += x * x;
        }

        double inv_rms =
            1.0 / std::sqrt(sum / cols + eps);

        for (int c = 0; c < cols; ++c) {
            const size_t idx = static_cast<size_t>(r) * cols + c;

            h_reference[idx] = static_cast<float>(
                static_cast<double>(h_input[idx]) *
                inv_rms *
                static_cast<double>(h_weight[c])
            );
        }
    }

    // Device 内存
    float* d_input = nullptr;
    float* d_weight = nullptr;
    float* d_output = nullptr;
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;

    auto cleanup = [&]() {
        if (graph_exec) cudaGraphExecDestroy(graph_exec);
        if (graph) cudaGraphDestroy(graph);
        if (stream) cudaStreamDestroy(stream);
        if (d_input) cudaFree(d_input);
        if (d_weight) cudaFree(d_weight);
        if (d_output) cudaFree(d_output);
    };

    const size_t input_bytes = input_size * sizeof(float);
    const size_t weight_bytes = cols * sizeof(float);

    if (!check_cuda(
        cudaMalloc(reinterpret_cast<void**>(&d_input),
                   input_bytes),
        "cudaMalloc(d_input)")) {
        cleanup();
        return false;
    }

    if (!check_cuda(
        cudaMalloc(reinterpret_cast<void**>(&d_weight),
                   weight_bytes),
        "cudaMalloc(d_weight)")) {
        cleanup();
        return false;
    }

    if (!check_cuda(
        cudaMalloc(reinterpret_cast<void**>(&d_output),
                   input_bytes),
        "cudaMalloc(d_output)")) {
        cleanup();
        return false;
    }

    // Host -> Device
    if (!check_cuda(
        cudaMemcpy(d_input, h_input.data(),
                   input_bytes, cudaMemcpyHostToDevice),
        "cudaMemcpy(input)")) {
        cleanup();
        return false;
    }

    if (!check_cuda(
        cudaMemcpy(d_weight, h_weight.data(),
                   weight_bytes, cudaMemcpyHostToDevice),
        "cudaMemcpy(weight)")) {
        cleanup();
        return false;
    }

    // FP32 的全 1 位模式是 NaN，漏写元素会触发后面的有限值检查。
    if (!check_cuda(cudaMemset(d_output, 0xff, input_bytes), "cudaMemset(output)")) {
        cleanup();
        return false;
    }

    // Non-blocking stream 与默认 stream 独立，先等待输入拷贝完成。
    if (test.capture_stream &&
        (!check_cuda(cudaStreamSynchronize(nullptr), "cudaStreamSynchronize(input copies)") ||
         !check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags") ||
         !check_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "cudaStreamBeginCapture"))) {
        cleanup();
        return false;
    }

    const cudaError_t launch_status = kernforge::rmsnorm(
        d_input, d_weight, d_output, rows, cols, eps, stream);
    const bool launched = check_cuda(launch_status, "kernforge::rmsnorm");

    if (test.capture_stream) {
        // 即使 API 返回错误，也要先结束 capture；图中只应有该 API 提交的工作。
        const bool captured = check_cuda(cudaStreamEndCapture(stream, &graph), "cudaStreamEndCapture");
        if (!launched || !captured || !has_kernel_node(graph) ||
            !check_cuda(cudaGraphInstantiate(&graph_exec, graph, 0), "cudaGraphInstantiate") ||
            !check_cuda(cudaGraphLaunch(graph_exec, stream), "cudaGraphLaunch")) {
            cleanup();
            return false;
        }
    }

    if (!launched || !check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(RMSNorm)")) {
        cleanup();
        return false;
    }

    // Device -> Host
    if (!check_cuda(
        cudaMemcpy(h_output.data(), d_output,
                   input_bytes, cudaMemcpyDeviceToHost),
        "cudaMemcpy(output)")) {
        cleanup();
        return false;
    }

    // 结果对比
    bool passed = true;
    float max_abs_error = 0.0f;
    size_t error_count = 0;

    for (size_t i = 0; i < input_size; ++i) {
        float diff = std::abs(h_output[i] - h_reference[i]);

        max_abs_error = std::max(max_abs_error, diff);

        float tolerance =
            atol + rtol * std::abs(h_reference[i]);

        if (!std::isfinite(h_output[i]) ||
            diff > tolerance) {
            passed = false;
            ++error_count;

            if (error_count <= 5) {
                std::cerr
                    << "Mismatch at index " << i
                    << ", GPU = " << h_output[i]
                    << ", CPU = " << h_reference[i]
                    << ", diff = " << diff
                    << std::endl;
            }
        }
    }

    cleanup();

    std::cout
        << "[RMSNorm] " << test.name << ", Shape = ["
        << rows << ", " << cols << "] "
        << (passed ? "PASSED" : "FAILED")
        << ", Max Error = "
        << max_abs_error
        << ", Error Count = "
        << error_count
        << std::endl;

    return passed;
}

int main() {
    // 测试用例
    const std::vector<TestCase> test_cases = {
        {1, 1},
        {1, 7},
        {2, 4},
        {3, 127},
        {4, 128},
        {2, 255},
        {2, 256},
        {3, 257},
        {8, 1024},
        {32, 4096},
        {3, 33, "zero input", 1e-6f, 0.0f},
        {3, 257, "tiny input, small eps", 1e-12f, 1e-6f},
        {3, 257, "large eps", 0.5f},
        {3, 257, "non-blocking stream", 1e-6f, 1.0f, true},
    };

    std::mt19937 rng(1234);

    size_t passed_count = 0;

    for (const auto& test : test_cases) {
        if (run_rmsnorm_test(test, rng)) {
            ++passed_count;
        } else {
            std::cerr << "[RMSNorm] Failed case: " << test.name
                      << " [" << test.rows << ", " << test.cols << "]\n";
        }
    }

    const bool invalid_arguments_passed = run_invalid_arguments_test();
    std::cout << "[RMSNorm] Invalid arguments "
              << (invalid_arguments_passed ? "PASSED" : "FAILED") << '\n';

    std::cout << "\n================================\n";
    std::cout << "RMSNorm Test Summary\n";
    std::cout << "Passed: " << passed_count
              << " / " << test_cases.size()
              << std::endl;
    std::cout << "================================\n";

    return passed_count == test_cases.size() && invalid_arguments_passed
        ? 0 : 1;
}
