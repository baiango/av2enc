cmake_minimum_required(VERSION 3.24)

# Default to the primary (RTX 3060, sm_86) and secondary (RTX 5090, sm_120)
# targets. This must be set before project() so CMake's CUDA detection does not
# seed the cache with its own default (sm_52 on this box). An explicit
# -DCMAKE_CUDA_ARCHITECTURES=... still overrides it.
# The source is baseline CUDA only; sm_120 is compiled from the same code and
# enables no Blackwell-specific features.
set(CMAKE_CUDA_ARCHITECTURES "86;120" CACHE STRING
    "CUDA architectures (default: RTX 3060/3080 Ti sm_86 + RTX 5090 sm_120)")

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
