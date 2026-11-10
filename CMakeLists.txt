cmake_minimum_required(VERSION 3.24)

project(av2enc
  VERSION 0.1.0
  DESCRIPTION "From-scratch AV2 encoder"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# CUDA is the product target. The CPU-only build exists only for reference
# tests and should not be used to judge encoder support.
option(AV2ENC_ENABLE_CUDA "Build the CUDA encoder backend (requires nvcc)" ON)
option(AV2ENC_ALLOW_CPU_ONLY
  "Allow a CPU-only reference/test build (not a supported encoder target)" OFF)

include(CheckLanguage)
check_language(CUDA)
set(AV2ENC_CUDA_AVAILABLE OFF)
if(CMAKE_CUDA_COMPILER)
  set(AV2ENC_CUDA_AVAILABLE ON)
endif()

set(AV2ENC_HAVE_CUDA OFF)
if(AV2ENC_ENABLE_CUDA)
  if(AV2ENC_CUDA_AVAILABLE)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      # 86 = RTX 3060 (primary target), 120 = RTX 5070 / Blackwell (secondary).
      # The code must use the sm_86 feature baseline; sm_120 is only a build
      # target, not permission to use Blackwell-only instructions.
      set(CMAKE_CUDA_ARCHITECTURES "86;120" CACHE STRING
          "CUDA architectures (86 = RTX 3060 primary, 120 = RTX 5070 secondary)")
    endif()
    enable_language(CUDA)
    set(AV2ENC_HAVE_CUDA ON)
    message(STATUS "av2enc: CUDA backend enabled (architectures: ${CMAKE_CUDA_ARCHITECTURES})")
  else()
    if(NOT AV2ENC_ALLOW_CPU_ONLY)
      message(FATAL_ERROR
        "av2enc: CUDA is required for the encoder target, but no CUDA compiler "
        "was found. Pass -DAV2ENC_ENABLE_CUDA=OFF -DAV2ENC_ALLOW_CPU_ONLY=ON "
        "only for the CPU reference/test path.")
    endif()
    message(WARNING
      "av2enc: no CUDA compiler; building the CPU-only reference path. "
      "This is NOT a supported encoder target. The product build is Linux + CUDA.")
  endif()
else()
  if(NOT AV2ENC_ALLOW_CPU_ONLY)
    message(FATAL_ERROR
      "av2enc: AV2ENC_ENABLE_CUDA=OFF requires AV2ENC_ALLOW_CPU_ONLY=ON. "
      "The encoder target is Linux + CUDA.")
  endif()
  message(STATUS "av2enc: CPU-only reference build (unsupported encoder target)")
endif()

function(av2enc_apply_warnings target)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4)
  else()
    target_compile_options(${target} PRIVATE
      -Wall
      -Wextra
      -Wpedantic
      -Wconversion
      -Wshadow)
  endif()
endfunction()

# Host-side code shared by the CLI and the tests. GPU code will be added as a
# separate target once CUDA kernels exist.
add_library(av2enc_core STATIC
  src/io/y4m.cpp
  src/io/ivf.cpp
  src/bitstream/bitwriter.cpp
  src/bitstream/obu.cpp
  src/bitstream/m0_headers.cpp
  src/encoder/m0_still.cpp
  src/encoder/partition/partition.cpp
  src/encoder/intra_modes/intra_modes.cpp
  src/encoder/tx_variants/tx_variants.cpp
  src/encoder/tx_variants/tx_select.cpp
  src/encoder/cfl/cfl.cpp
  src/encoder/filters/deblock/deblock.cpp
  src/encoder/rdo/rdo.cpp
  src/encoder/m1/m1_headers.cpp
  src/encoder/m1/m1_still.cpp
  src/entropy/av2/av2_entropy.cpp
  src/profile/profiler.cpp)
target_include_directories(av2enc_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_compile_features(av2enc_core PUBLIC cxx_std_20)
av2enc_apply_warnings(av2enc_core)

if(AV2ENC_HAVE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  # Host-side CUDA runtime used by the CLI (device query and arena cap policy).
  target_sources(av2enc_core PRIVATE
    src/cuda/arena.cpp
    src/cuda/context.cpp)
  target_include_directories(av2enc_core PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  target_compile_definitions(av2enc_core PUBLIC AV2ENC_HAVE_CUDA=1)
  target_link_libraries(av2enc_core PUBLIC CUDA::cudart)
else()
  target_compile_definitions(av2enc_core PUBLIC AV2ENC_HAVE_CUDA=0)
endif()

add_executable(av2enc src/main.cpp)
target_link_libraries(av2enc PRIVATE av2enc_core)
av2enc_apply_warnings(av2enc)

include(CTest)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
