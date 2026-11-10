cmake_minimum_required(VERSION 3.24)
project(hello_cuda LANGUAGES CXX CUDA)

# Standalone smoke/benchmark target. This project is intentionally NOT wired
# into the root CMakeLists.txt; it is built on the remote CUDA box only.

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Hard-pin to the RTX 3060 target. CMake's CUDA detection initialises the
# cache to sm_52 in this container, and a plain set(... CACHE ...) would lose
# to that existing entry, leaving the binary to PTX-JIT from compute_52.
set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures" FORCE)

find_package(CUDAToolkit REQUIRED)

add_executable(hello_cuda_bench
  src/main.cu
  src/kernels.cu
)

target_include_directories(hello_cuda_bench PRIVATE src)

target_compile_options(hello_cuda_bench PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
)

target_link_libraries(hello_cuda_bench PRIVATE CUDA::cudart)
