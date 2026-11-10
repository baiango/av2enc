cmake_minimum_required(VERSION 3.24)

# Match the encoder targets: RTX 3060 (sm_86) primary, RTX 5070 (sm_120)
# secondary. An explicit -DCMAKE_CUDA_ARCHITECTURES=... overrides this default.
set(CMAKE_CUDA_ARCHITECTURES "86;120" CACHE STRING
    "CUDA architectures (default: RTX 3060 sm_86 + RTX 5070 sm_120)")

project(vram_config LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_executable(vram_config src/main.cu)

target_compile_options(vram_config PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
)

target_link_libraries(vram_config PRIVATE CUDA::cudart)
