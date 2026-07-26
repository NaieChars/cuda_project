cmake_minimum_required(VERSION 3.24)

project(cuda_project LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# 新增：明确指定目标架构为 sm_120（RTX 5060 / Blackwell架构）
# 这样nvcc会直接生成原生机器码(SASS)嵌入可执行文件，
# 不再依赖运行时驱动做JIT编译，从根本上避开"PTX工具链不兼容"的问题
set(CMAKE_CUDA_ARCHITECTURES 120)

add_executable(cuda_project
    src/main.cu
)

set_target_properties(cuda_project PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)