# CMake 版本至少是 3.24
cmake_minimum_required(VERSION 3.24)

project(KernForge
    VERSION 0.1.0
    LANGUAGES CXX CUDA
)

# C++ / CUDA 编译标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 默认针对 RTX 5070 Ti (SM 120)
# 允许用户在 CMake 配置阶段覆盖
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 120)
endif()

# CUDA 算子库
add_library(kernforge SHARED
    csrc/norm/rmsnorm.cu
)

# 暴露公共头文件目录
target_include_directories(kernforge
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# CUDA 编译选项
set_target_properties(kernforge PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# C++ / CUDA 单元测试
add_executable(test_rmsnorm
    tests/test_rmsnorm.cu
)

target_link_libraries(test_rmsnorm
    PRIVATE
        kernforge
)