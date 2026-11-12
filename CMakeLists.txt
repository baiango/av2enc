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
      # 86 = RTX 3060 (primary, feature baseline), 89 = RTX 4070 Super
      # (secondary cost-efficiency target), 120 = RTX 5070 / Blackwell (smoke).
      # No arch-specific accelerators: the code uses the sm_86 feature set.
      set(CMAKE_CUDA_ARCHITECTURES "86;89;120" CACHE STRING
          "CUDA architectures (86 = RTX 3060 primary, 89 = RTX 4070S secondary, 120 = RTX 5070 smoke)")
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
  src/core/io/y4m.cpp
  src/core/io/ivf.cpp
  src/core/bitstream/bitwriter.cpp
  src/core/bitstream/obu.cpp
  src/core/bitstream/basic_headers.cpp
  src/pipeline/still/basic_encoder.cpp
  src/codec/partition/tree/partition.cpp
  src/codec/intra/modes/intra_modes.cpp
  src/codec/rate/rdoq/rdoq.cpp
  src/codec/rate/rdoq/rdoq_av2_model.cpp
  src/codec/transform/tx_type/tx_type.cpp
  src/codec/transform/variants/tx_variants.cpp
  src/codec/transform/variants/tx_select.cpp
  src/codec/intra/cfl/cfl.cpp
  src/codec/filters/deblock/deblock.cpp
  src/codec/filters/cdef/cdef.cpp
  src/codec/filters/loop_restoration/restoration.cpp
  src/codec/rate/rdo/rdo.cpp
  src/codec/rate/rdo/intra_decision.cpp
  src/pipeline/still/still_headers.cpp
  src/pipeline/still/still_syntax.cpp
  src/pipeline/still/still_encoder.cpp
  src/pipeline/still/stream/still_stream_bridge.cpp
  src/codec/entropy/av2/av2_entropy.cpp
  src/core/telemetry/profiler.cpp
  # M1 CDEF stage host half (header packing, MI grids, strength search).  It is
  # CUDA-free and the CPU M1 path needs it; the M1 GPU OBJECT target excludes
  # this file so the symbols resolve from this archive.
  src/cuda/pipeline/still/cdef/still_cdef.cpp)
target_include_directories(av2enc_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_compile_features(av2enc_core PUBLIC cxx_std_20)
av2enc_apply_warnings(av2enc_core)
# The CPU reference must be bit-identical across compilers and platforms. The
# M0/M1 forward transforms use binary64; FMA contraction changes coefficients
# and therefore the bitstream.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(av2enc_core PRIVATE -ffp-contract=off)
endif()

if(AV2ENC_HAVE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  # Host-side CUDA runtime used by the CLI (device query and arena cap policy).
  target_sources(av2enc_core PRIVATE
    src/cuda/arena.cpp
    src/cuda/context.cpp)
  target_include_directories(av2enc_core PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  target_compile_definitions(av2enc_core PUBLIC AV2ENC_HAVE_CUDA=1)
  target_link_libraries(av2enc_core PUBLIC CUDA::cudart)

  # M1 GPU pipeline (src/cuda/pipeline/still/, owner: orchestrator).  Several agents land
  # pieces of this tree, so only pull it in once the public contract header and
  # the orchestrator entry point exist.  GLOB_RECURSE covers the component
  # subdirectories (plan/search/tx/recon/chroma/...); per-component tests,
  # dumps and AppleDouble files from the macOS sync tar are not library sources.
  set(AV2ENC_STILL_GPU_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/pipeline/still")
  set(AV2ENC_HAVE_STILL_GPU OFF)
  if(EXISTS "${AV2ENC_STILL_GPU_DIR}/still_gpu.h" AND
     (EXISTS "${AV2ENC_STILL_GPU_DIR}/still_gpu.cu" OR
      EXISTS "${AV2ENC_STILL_GPU_DIR}/still_gpu.cpp"))
    # Explicit source list.  A GLOB with CONFIGURE_DEPENDS re-runs configure
    # on every new directory, which makes refactors and feature branches churn.
    # Keep this list explicit; tests/dumps/benchmarks are deliberately absent.
    set(AV2ENC_STILL_GPU_SOURCES
      src/cuda/pipeline/still/still_gpu.cu
      src/cuda/pipeline/still/cdef/still_cdef.cu
      src/cuda/pipeline/still/cfl/still_cfl.cu
      src/cuda/pipeline/still/chroma/still_chroma.cu
      src/cuda/pipeline/still/hbd/hbd_cpu.cpp
      src/cuda/pipeline/still/hbd/hbd_frame.cu
      src/cuda/pipeline/still/hbd/hbd_tx.cu
      src/cuda/pipeline/still/plan/plan.cpp
      src/cuda/pipeline/still/plan/plan_gpu_bridge.cpp
      src/cuda/pipeline/still/recon/recon.cu
      src/cuda/pipeline/still/restoration/still_restoration.cu
      src/cuda/pipeline/still/sched/sched.cpp
      src/cuda/pipeline/still/sched/sched_bridge.cpp
      src/cuda/pipeline/still/sched/sched_exec.cu
      src/cuda/pipeline/still/search/luma_search.cu
      src/cuda/pipeline/still/tx/still_tx.cu)
    if(AV2ENC_STILL_GPU_SOURCES)
      set(AV2ENC_HAVE_STILL_GPU ON)
      # Host-side support for the M1 filter/format stages.
      foreach(host_support
          src/formats/format.cpp)
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${host_support}")
          target_sources(av2enc_core PRIVATE "${host_support}")
        endif()
      endforeach()
      # Shared CUDA components the M1 stages call directly.  These are only
      # pulled in with the M1 GPU target, not with the CPU core library.
      set(AV2ENC_STILL_GPU_SUPPORT_SOURCES)
      foreach(support_source
          src/cuda/codec/intra/cfl/cfl.cu
          src/cuda/codec/filters/deblock/deblock.cu
          src/cuda/codec/intra/modes/intra_modes.cu
          src/cuda/codec/transform/variants/tx_variants.cu
          src/cuda/codec/filters/cdef/cdef.cu
          src/cuda/codec/filters/loop_restoration/ccso.cu
          src/cuda/codec/filters/loop_restoration/wiener.cu)
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${support_source}")
          list(APPEND AV2ENC_STILL_GPU_SUPPORT_SOURCES "${support_source}")
        endif()
      endforeach()
      # Separate object target: the stage adapters register through weak hook
      # references in still_gpu.cu, so a static archive would drop objects that
      # nothing strongly references.  Linking these objects directly into the
      # executable keeps every stage present without whole-archiving unrelated
      # av2enc_core objects.
      add_library(av2enc_still_gpu OBJECT
        ${AV2ENC_STILL_GPU_SOURCES}
        ${AV2ENC_STILL_GPU_SUPPORT_SOURCES})
      target_link_libraries(av2enc_still_gpu PUBLIC av2enc_core CUDA::cudart)
      target_include_directories(av2enc_still_gpu PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS})
      target_compile_definitions(av2enc_still_gpu PRIVATE
        AV2ENC_HAVE_CUDA=1
        AV2ENC_HAVE_STILL_GPU=1)
      av2enc_apply_warnings(av2enc_still_gpu)
      # The M1 gates require byte-identical output; FP contraction changes the
      # double-precision transform results (docs/negative-results.md).
      foreach(still_source IN LISTS AV2ENC_STILL_GPU_SOURCES
                             AV2ENC_STILL_GPU_SUPPORT_SOURCES)
        if(still_source MATCHES "\\.cu$")
          set_source_files_properties("${still_source}" PROPERTIES
            COMPILE_OPTIONS "--fmad=false")
        endif()
      endforeach()
      message(STATUS
        "av2enc: M1 GPU pipeline enabled (${AV2ENC_STILL_GPU_DIR})")
    endif()
  endif()
  if(NOT AV2ENC_HAVE_STILL_GPU)
    message(STATUS
      "av2enc: M1 GPU entry point not present; --still-gpu will be unavailable")
  endif()
  if(AV2ENC_HAVE_STILL_GPU)
    target_compile_definitions(av2enc_core PUBLIC AV2ENC_HAVE_STILL_GPU=1)
  else()
    target_compile_definitions(av2enc_core PUBLIC AV2ENC_HAVE_STILL_GPU=0)
  endif()
else()
  target_compile_definitions(av2enc_core PUBLIC
    AV2ENC_HAVE_CUDA=0
    AV2ENC_HAVE_STILL_GPU=0)
endif()

add_executable(av2enc src/main.cpp)
target_link_libraries(av2enc PRIVATE av2enc_core)
if(TARGET av2enc_still_gpu)
  # OBJECT library: all M1 GPU stage objects are linked directly, so weak
  # stage-hook references resolve without --whole-archive on av2enc_core.
  target_link_libraries(av2enc PRIVATE av2enc_still_gpu)
endif()
av2enc_apply_warnings(av2enc)

include(CTest)
if(BUILD_TESTING)
  add_subdirectory(tests)

  # Host-only GpuOptions plumbing test (no CUDA toolkit or device needed):
  # every M3 flag must be mapped into the CPU parity configuration and every
  # unported tool must be rejected by name, never silently ignored.
  add_executable(still_gpu_options_test src/cuda/pipeline/still/still_gpu_options_test.cpp)
  target_include_directories(still_gpu_options_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_compile_features(still_gpu_options_test PRIVATE cxx_std_20)
  av2enc_apply_warnings(still_gpu_options_test)
  add_test(NAME still_gpu_options_test COMMAND still_gpu_options_test)

  # Opt-in M1 tools: pinned IVF MD5s + dav2d --strict 1 decode.  The script
  # exits 77 when no decoder is available, which CTest reports as a skip.
  add_test(NAME still_feature_gate
    COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/tools/still_verify/feature_gate.sh"
            --bin $<TARGET_FILE:av2enc>)
  set_tests_properties(still_feature_gate PROPERTIES
    SKIP_RETURN_CODE 77
    TIMEOUT 300)
endif()
