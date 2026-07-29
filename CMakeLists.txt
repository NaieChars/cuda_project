cmake_minimum_required(VERSION 3.24)

project(cuda_project LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 120)

# Day29
add_executable(cuda_project
    src/main.cu
)
target_compile_options(cuda_project PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
    $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
)
set_target_properties(cuda_project PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# N体引力模拟 naive版本
add_executable(nbody_naive
    src/nbody_naive.cu
)
target_compile_options(nbody_naive PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
    $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
)
set_target_properties(nbody_naive PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# tiled 版本
add_executable(nbody_tiled
    src/nbody_tiled.cu
)
target_compile_options(nbody_tiled PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
    $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
)
set_target_properties(nbody_tiled PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# uniform grid 空间划分 - 用thrust按cellIndex排序
add_executable(grid_sort
    src/grid_sort.cu
)
target_compile_options(grid_sort PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8 -Xcompiler=/Zc:preprocessor>
    $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
)
set_target_properties(grid_sort PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)