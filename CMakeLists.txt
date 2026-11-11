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
  src/io/y4m.cpp
  src/io/ivf.cpp
  src/bitstream/bitwriter.cpp
  src/bitstream/obu.cpp
  src/bitstream/m0_headers.cpp
  src/encoder/m0_still.cpp
  src/encoder/partition/partition.cpp
  src/encoder/intra_modes/intra_modes.cpp
  src/encoder/rdoq/rdoq.cpp
  src/encoder/rdoq/rdoq_av2_model.cpp
  src/encoder/tx_type/tx_type.cpp
  src/encoder/tx_variants/tx_variants.cpp
  src/encoder/tx_variants/tx_select.cpp
  src/encoder/cfl/cfl.cpp
  src/encoder/filters/deblock/deblock.cpp
  src/encoder/filters/cdef/cdef.cpp
  src/encoder/filters/restoration/restoration.cpp
  src/encoder/rdo/rdo.cpp
  src/encoder/rdo/intra_decision.cpp
  src/encoder/m1/m1_headers.cpp
  src/encoder/m1/m1_syntax.cpp
  src/encoder/m1/m1_still.cpp
  src/encoder/m1/stream/m1_stream_bridge.cpp
  src/entropy/av2/av2_entropy.cpp
  src/profile/profiler.cpp
  # M1 CDEF stage host half (header packing, MI grids, strength search).  It is
  # CUDA-free and the CPU M1 path needs it; the M1 GPU OBJECT target excludes
  # this file so the symbols resolve from this archive.
  src/cuda/m1/cdef/m1_cdef.cpp)
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

  # M1 GPU pipeline (src/cuda/m1/, owner: orchestrator).  Several agents land
  # pieces of this tree, so only pull it in once the public contract header and
  # the orchestrator entry point exist.  GLOB_RECURSE covers the component
  # subdirectories (plan/search/tx/recon/chroma/...); per-component tests,
  # dumps and AppleDouble files from the macOS sync tar are not library sources.
  set(AV2ENC_M1_GPU_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/m1")
  set(AV2ENC_HAVE_M1_GPU OFF)
  if(EXISTS "${AV2ENC_M1_GPU_DIR}/m1_gpu.h" AND
     (EXISTS "${AV2ENC_M1_GPU_DIR}/m1_gpu.cu" OR
      EXISTS "${AV2ENC_M1_GPU_DIR}/m1_gpu.cpp"))
    file(GLOB_RECURSE AV2ENC_M1_GPU_SOURCES CONFIGURE_DEPENDS
      "${AV2ENC_M1_GPU_DIR}/*.cu"
      "${AV2ENC_M1_GPU_DIR}/*.cpp")
    list(FILTER AV2ENC_M1_GPU_SOURCES EXCLUDE REGEX "/\\._[^/]*$")
    list(FILTER AV2ENC_M1_GPU_SOURCES EXCLUDE REGEX
      "/(run_[^/]*|[^/]*_test|[^/]*_dump|[^/]*_bench|[^/]*_shim)\\.(cu|cpp)$")
    # m1_cdef.cpp is compiled into av2enc_core (the CPU M1 path needs it);
    # linking it again from the OBJECT library would duplicate its symbols.
    list(FILTER AV2ENC_M1_GPU_SOURCES EXCLUDE REGEX "/m1_cdef\\.cpp$")
    if(AV2ENC_M1_GPU_SOURCES)
      set(AV2ENC_HAVE_M1_GPU ON)
      # Host-side support for the M1 filter/format stages.
      foreach(host_support
          src/encoder/format/format.cpp)
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${host_support}")
          target_sources(av2enc_core PRIVATE "${host_support}")
        endif()
      endforeach()
      # Shared CUDA components the M1 stages call directly.  These are only
      # pulled in with the M1 GPU target, not with the CPU core library.
      set(AV2ENC_M1_GPU_SUPPORT_SOURCES)
      foreach(support_source
          src/cuda/cfl/cfl.cu
          src/cuda/deblock/deblock.cu
          src/cuda/intra_modes/intra_modes.cu
          src/cuda/tx_variants/tx_variants.cu
          src/cuda/cdef/cdef.cu
          src/cuda/restoration/ccso.cu
          src/cuda/restoration/wiener.cu)
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${support_source}")
          list(APPEND AV2ENC_M1_GPU_SUPPORT_SOURCES "${support_source}")
        endif()
      endforeach()
      # Separate object target: the stage adapters register through weak hook
      # references in m1_gpu.cu, so a static archive would drop objects that
      # nothing strongly references.  Linking these objects directly into the
      # executable keeps every stage present without whole-archiving unrelated
      # av2enc_core objects.
      add_library(av2enc_m1_gpu OBJECT
        ${AV2ENC_M1_GPU_SOURCES}
        ${AV2ENC_M1_GPU_SUPPORT_SOURCES})
      target_link_libraries(av2enc_m1_gpu PUBLIC av2enc_core CUDA::cudart)
      target_include_directories(av2enc_m1_gpu PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS})
      target_compile_definitions(av2enc_m1_gpu PRIVATE
        AV2ENC_HAVE_CUDA=1
        AV2ENC_HAVE_M1_GPU=1)
      av2enc_apply_warnings(av2enc_m1_gpu)
      # The M1 gates require byte-identical output; FP contraction changes the
      # double-precision transform results (docs/negative-results.md).
      foreach(m1_source IN LISTS AV2ENC_M1_GPU_SOURCES
                             AV2ENC_M1_GPU_SUPPORT_SOURCES)
        if(m1_source MATCHES "\\.cu$")
          set_source_files_properties("${m1_source}" PROPERTIES
            COMPILE_OPTIONS "--fmad=false")
        endif()
      endforeach()
      message(STATUS
        "av2enc: M1 GPU pipeline enabled (${AV2ENC_M1_GPU_DIR})")
    endif()
  endif()
  if(NOT AV2ENC_HAVE_M1_GPU)
    message(STATUS
      "av2enc: M1 GPU entry point not present; --m1-gpu will be unavailable")
  endif()
  if(AV2ENC_HAVE_M1_GPU)
    target_compile_definitions(av2enc_core PUBLIC AV2ENC_HAVE_M1_GPU=1)
  else()
    target_compile_definitions(av2enc_core PUBLIC AV2ENC_HAVE_M1_GPU=0)
  endif()
else()
  target_compile_definitions(av2enc_core PUBLIC
    AV2ENC_HAVE_CUDA=0
    AV2ENC_HAVE_M1_GPU=0)
endif()

add_executable(av2enc src/main.cpp)
target_link_libraries(av2enc PRIVATE av2enc_core)
if(TARGET av2enc_m1_gpu)
  # OBJECT library: all M1 GPU stage objects are linked directly, so weak
  # stage-hook references resolve without --whole-archive on av2enc_core.
  target_link_libraries(av2enc PRIVATE av2enc_m1_gpu)
endif()
av2enc_apply_warnings(av2enc)

include(CTest)
if(BUILD_TESTING)
  add_subdirectory(tests)

  # Host-only GpuOptions plumbing test (no CUDA toolkit or device needed):
  # every M3 flag must be mapped into the CPU parity configuration and every
  # unported tool must be rejected by name, never silently ignored.
  add_executable(m1_gpu_options_test src/cuda/m1/m1_gpu_options_test.cpp)
  target_include_directories(m1_gpu_options_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_compile_features(m1_gpu_options_test PRIVATE cxx_std_20)
  av2enc_apply_warnings(m1_gpu_options_test)
  add_test(NAME m1_gpu_options_test COMMAND m1_gpu_options_test)

  # Opt-in M1 tools: pinned IVF MD5s + dav2d --strict 1 decode.  The script
  # exits 77 when no decoder is available, which CTest reports as a skip.
  add_test(NAME m1_feature_gate
    COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/tools/m1_verify/feature_gate.sh"
            --bin $<TARGET_FILE:av2enc>)
  set_tests_properties(m1_feature_gate PROPERTIES
    SKIP_RETURN_CODE 77
    TIMEOUT 300)
endif()
