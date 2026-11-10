cmake_minimum_required(VERSION 3.24)

# Standalone DCT precision/performance benchmark.  Deliberately not wired into
# the root CMake project; build it on the CUDA box only.
project(cuda_dct_perf LANGUAGES CXX CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "86" CACHE STRING
      "CUDA architectures (RTX 3060 primary target: 86)")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(CUDAToolkit REQUIRED)

add_executable(cuda_dct_perf
  src/main.cu
  src/kernels.cu
  src/cpu_reference.cpp
)

# The reference tables are read from the repository, never modified here.
target_include_directories(cuda_dct_perf PRIVATE
  src
  "${CMAKE_CURRENT_SOURCE_DIR}/../.."
)

target_compile_options(cuda_dct_perf PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3;-ffp-contract=off;-Wall;-Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo;-fmad=false;-Xptxas=-v>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
)

target_link_libraries(cuda_dct_perf PRIVATE CUDA::cudart)
