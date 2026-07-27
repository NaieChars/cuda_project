cmake_minimum_required(VERSION 3.24)

project(cuda_project LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 120)

add_executable(cuda_project
    src/main.cu
)

# CUDA源码要用 -Xcompiler 把参数透传给底层的cl.exe
target_compile_options(cuda_project PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
    $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
)

set_target_properties(cuda_project PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)