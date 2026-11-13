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
      # 86 = RTX 3060 12 GB (development and feature baseline) and
      # RTX 3080 Ti (production), 120 = RTX 5090 (maximum-speed target).
      # No arch-specific accelerators are required: the baseline kernels use
      # the sm_86 feature set and also run on sm_120.
      set(CMAKE_CUDA_ARCHITECTURES "86;120" CACHE STRING
          "CUDA architectures (86 = RTX 3060/3080 Ti, 120 = RTX 5090)")
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
  src/codec/partition/ext/partition_ext.cpp
  src/codec/partition/ext/ext_syntax.cpp
  src/codec/partition/sdp/sdp.cpp
  src/codec/partition/sdp/sdp_validate.cpp
  src/codec/partition/sdp/sdp_syntax.cpp
  src/codec/partition/stream/partition_stream.cpp
  src/codec/intra/modes/intra_modes.cpp
  src/codec/rate/rdoq/rdoq.cpp
  src/codec/rate/rdoq/rdoq_av2_model.cpp
  src/codec/transform/delta_q/quant.cpp
  src/codec/transform/delta_q/search.cpp
  src/codec/transform/delta_q/syntax.cpp
  src/codec/transform/qm/qm.cpp
  src/codec/transform/secondary/tx_secondary.cpp
  src/codec/transform/tcq/cctx.cpp
  src/codec/transform/tcq/tcq.cpp
  src/codec/transform/tcq/tcq_av2_model.cpp
  src/codec/transform/tx_partition/tx_partition.cpp
  src/codec/transform/tx_partition/tx_scan.cpp
  src/codec/transform/tx_type/tx_type.cpp
  src/codec/transform/variants/tx_variants.cpp
  src/codec/transform/variants/tx_select.cpp
  # §5.20.8.2 inter TX-type sets/symbols, shared by the M2 CPU reference and
  # the opt-in TX search (the module under src/cuda/pipeline/video/tx/ links
  # the same component).
  src/codec/transform/inter_residual/inter_tx_type.cpp
  src/codec/intra/cfl/cfl.cpp
  src/codec/intra/cfl/mhccp/mhccp.cpp
  src/codec/intra/cfl/mhccp/mhccp_syntax.cpp
  src/codec/intra/dip/dip.cpp
  src/codec/intra/dip/dip_syntax.cpp
  src/codec/intra/ibp/ibp.cpp
  src/codec/intra/palette/core/palette.cpp
  src/codec/intra/palette/core/scc_stream.cpp
  src/codec/intra/palette/stream/palette_stream.cpp
  # Intra block copy (§5.20.5.3/.4): CPU reference, exact §7.12.2 BV stack and
  # the §5.20.5.4 syntax writer (opt-in `--still-ibc`).
  src/codec/intra/ibc/core/ibc.cpp
  src/codec/intra/ibc/stream/ibc_stack.cpp
  src/codec/intra/ibc/stream/ibc_syntax.cpp
  # Exact §8.3.2 use_intrabc/skip_flag context helpers (NPos/NPosBuf).
  src/codec/intra/ibc/stream/still_ibc_stream.cpp
  # §5.4.7/§5.18.3.3/.4 screen-content header bits + frame-scc validation.
  src/codec/intra/screen_content/scc_flags.cpp
  src/codec/intra/mrl/core/mrl.cpp
  src/codec/intra/mrl/stream/mrl_syntax.cpp
  src/codec/intra/mrl/stream/mrl_interaction.cpp
  src/codec/filters/deblock/deblock.cpp
  src/codec/filters/cdef/cdef.cpp
  # §5.18.7.10/§5.20.10.1 M2 CDEF adapter (strength plan + header bits).
  src/codec/filters/cdef/syntax/video_cdef.cpp
  src/codec/filters/loop_restoration/restoration.cpp
  # Encoder-side Wiener-NS search/syntax (§5.18.7.11, §5.20.10.4-.6, §7.20.3).
  src/codec/filters/loop_restoration/classes/wiener_class.cpp
  src/codec/filters/loop_restoration/classes/wiener_fit.cpp
  src/codec/filters/loop_restoration/classes/wiener_syntax.cpp
  src/codec/filters/loop_restoration/classes/wiener_decision.cpp
  # PC-Wiener (§7.20.4) stream syntax + decision, with the recovered tables.
  src/codec/filters/pc_wiener/pc_wiener_stream.cpp
  src/codec/filters/tables/pc_wiener.cpp
  src/codec/filters/tables/luma_source.cpp
  # GDF (§5.18.7.9, §5.20.10.3, §7.20.5) syntax + decision.
  src/codec/filters/gdf/gdf_syntax.cpp
  src/codec/filters/gdf/gdf_decision.cpp
  src/codec/filters/tables/gdf.cpp
  # Film grain signalling (§5.14, §5.18.10) + OBU placement/slot state.
  src/codec/filters/tables/film_grain.cpp
  src/pipeline/grain/film_grain_syntax.cpp
  src/pipeline/grain/grain_stream.cpp
  src/codec/rate/rdo/rdo.cpp
  src/codec/rate/rdo/intra_decision.cpp
  src/pipeline/still/still_headers.cpp
  src/pipeline/still/still_syntax.cpp
  src/pipeline/still/still_encoder.cpp
  src/pipeline/still/stream/still_stream_bridge.cpp
  # Multi-frame inter (P-frame) path, opt-in via --video.
  src/codec/inter/me/me.cpp
  src/codec/inter/mc/mc.cpp
  src/codec/inter/merge/merge.cpp
  # §7.10/§7.13.3.2 TIP and the §7.9/§7.22 temporal motion field
  # (opt-in --video-tip).  Off the default path, so every existing stream is
  # byte-identical.
  src/codec/inter/temporal/motion_field.cpp
  src/codec/inter/temporal/temporal_common.cpp
  src/codec/inter/temporal/tip.cpp
  src/codec/inter/temporal/temporal_syntax.cpp
  src/codec/inter/temporal/refinemv.cpp
  src/codec/inter/tip/tip_signalling.cpp
  src/codec/inter/tip/tip_candidates.cpp
  src/codec/inter/tip/tip_build.cpp
  src/codec/inter/tip/tip_output.cpp
  src/codec/inter/drl_reorder/drl_reorder.cpp
  src/codec/inter/drl_reorder/video_stack_adapter.cpp
  src/codec/inter/mv_precision/mv_precision.cpp
  # Long-term references, open-loop keys and the reference pool (opt-in,
  # --video-long-term / --video-olk).
  src/codec/inter/longterm/longterm_headers.cpp
  src/codec/inter/longterm/longterm_pool.cpp
  src/codec/inter/longterm/longterm_stream.cpp
  # Leading pictures, output suppression and the open-GOP plan (opt-in,
  # --video-leading).
  src/codec/inter/leading/leading.cpp
  # Show-existing-frame syntax and output bookkeeping (opt-in, --video-sef).
  src/codec/inter/sef/sef.cpp
  # SWITCH / RAS frame syntax, planning and validation (opt-in,
  # --video-switch / --video-ras).
  src/codec/inter/random_access/random_access.cpp
  # Hierarchical-P temporal layers (opt-in, --video-tlayers).
  src/codec/inter/tlayers/tlayers.cpp
  src/codec/inter/tlayers/tlayers_syntax.cpp
  # B-frame schedules (opt-in, --b-frames).
  src/codec/inter/advanced/b_frame.cpp
  src/codec/inter/bframes/bframes.cpp
  # Compound prediction (opt-in, --video-compound).
  src/codec/inter/multiref/compound_predict.cpp
  src/codec/inter/advanced/compound.cpp
  # §7.13.3.25 block adaptive weighted prediction (opt-in, --video-bawp).
  src/codec/inter/advanced/bawp.cpp
  # Global motion estimation + §7.13.3.19/.20 warp prediction (opt-in,
  # --video-gme / --video-warp).
  src/codec/inter/gme/gme.cpp
  src/codec/inter/gme/gme_syntax.cpp
  src/codec/inter/advanced/warp.cpp
  src/pipeline/video/video_gme.cpp
  src/pipeline/video/video_warp.cpp
  src/pipeline/video/video_headers.cpp
  src/pipeline/video/video_syntax.cpp
  src/pipeline/video/video_chroma_search.cpp
  src/pipeline/video/video_frame.cpp
  # Format video core (10-bit 4:2:0/4:2:2/4:4:4/4:0:0 and 8-bit 4:2:2/4:4:4/
  # 4:0:0).  The CPU reference and the shared orchestration live here; the CUDA
  # backend is under src/cuda/pipeline/video/hbd/.
  src/pipeline/video/hbd/video_hbd.cpp
  # Luma TX size/type/partition search (opt-in --video-tx-search): the CPU
  # reference for src/cuda/pipeline/video/tx/.  Both paths share the same
  # candidate sets, arithmetic and tie-breaks.
  src/pipeline/video/video_tx_search.cpp
  # Wave-3 frame types (long-term/OLK, leading, SEF, SWITCH/RAS).  Guards the
  # video_frame.cpp reference so standalone component gates stay linkable.
  src/pipeline/video/video_frames.cpp
  src/codec/entropy/av2/av2_entropy.cpp
  # §8.2.4 / §7.5 average-CDF + frame-end bookkeeping (opt-in --avg-cdf).
  src/codec/entropy/avg_cdf/avg_cdf_bank.cpp
  # delta-Q / segmentation syntax (neutral, opt-in still integration).
  src/codec/entropy/deltaq/still/deltaq.cpp
  src/codec/rate/rc/rc.cpp
  # Lossless (Q=0, DPCM, FSC) still-picture path; opt-in via --lossless.
  src/codec/entropy/lossless/lossless.cpp
  # Content interpretation OBU (§5.15); opt-in, default parameters.
  src/frame/metadata/content_interpretation/content_interpretation.cpp
  # Neutral metadata OBUs (§5.17): timecode / scan type, opt-in via
  # --metadata-timecode / --metadata-scan-type.  Decoder-invariant.
  src/frame/metadata/hdr/hdr_metadata.cpp
  src/frame/metadata/types/metadata_types.cpp
  # Multi-frame header (§5.7) on the non-single-picture video path
  # (--video-mfh).
  src/frame/mfh/mfh.cpp
  src/core/telemetry/profiler.cpp
  # M1 CDEF stage host half (header packing, MI grids, strength search).  It is
  # CUDA-free and the CPU M1 path needs it; the M1 GPU OBJECT target excludes
  # this file so the symbols resolve from this archive.
  src/cuda/pipeline/still/cdef/still_cdef.cpp)
target_include_directories(av2enc_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_compile_features(av2enc_core PUBLIC cxx_std_20)
# Enable the wave-3 frame-type dispatch in video_frame.cpp.  Standalone tool
# gate scripts compile video_frame.cpp without this define and keep the plain
# path only, so they do not need the inter module sources.
target_compile_definitions(av2enc_core PRIVATE AV2ENC_VIDEO_FRAMES=1)
# Opt-in video CDF-continuity / timing wiring (avg_cdf bank + CI timing OBU).
# Standalone component gates compile video_frame.cpp without these modules and
# keep the pre-wave code path; the product binary always has them.
target_compile_definitions(av2enc_core PRIVATE AV2ENC_VIDEO_CDF_CHAIN=1)
# Opt-in global motion / warped prediction wiring (§5.18.9, §7.13.3.19-.24).
# Standalone gates that compile the video writer without the GME sources keep
# the historical "enable_global_motion = 0" path.
target_compile_definitions(av2enc_core PRIVATE AV2ENC_VIDEO_GME=1)
# Format video core dispatch for 10-bit / 4:2:2 / 4:4:4 / 4:0:0.  Standalone
# component gates keep the 8-bit 4:2:0 path without the extra source.
target_compile_definitions(av2enc_core PRIVATE AV2ENC_VIDEO_FORMATS=1)
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
      src/cuda/formats/chroma/chroma_formats.cpp
      src/cuda/formats/chroma/chroma_formats.cu
      src/cuda/formats/format_device.cpp
      src/cuda/formats/high_bitdepth_gpu.cu
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
      src/cuda/pipeline/still/intra/intra_extra.cu
      src/cuda/pipeline/still/lossless/still_lossless_gpu.cu
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
          src/cuda/codec/intra/modes/intra_modes.cu
          src/cuda/codec/intra/mrl/mrl.cu
          src/cuda/codec/intra/dip/dip.cu
          src/cuda/codec/intra/ibp/ibp.cu
          src/cuda/codec/intra/palette/palette.cu
          src/cuda/codec/intra/ibc/ibc.cu
          src/cuda/codec/filters/deblock/deblock.cu
          src/cuda/codec/intra/modes/intra_modes.cu
          src/cuda/codec/transform/variants/tx_variants.cu
          src/cuda/codec/transform/tx_type/tx_type.cu
          src/cuda/codec/transform/secondary/tx_secondary_gpu.cu
          src/cuda/codec/filters/cdef/cdef.cu
          src/cuda/codec/filters/loop_restoration/ccso.cu
          src/cuda/codec/filters/loop_restoration/wiener.cu
          src/cuda/codec/filters/loop_restoration/chroma/lr_chroma.cu
          src/cuda/codec/filters/pc_wiener/pc_wiener.cu
          src/cuda/codec/filters/gdf/gdf.cu)
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

  # M2 video GPU pipeline (src/cuda/pipeline/video/, owner: orchestrator).
  # Opt-in product path behind --video-gpu.  The CPU --video path stays the
  # default reference; this target only adds the device backend and the
  # orchestration that mirrors the CPU decisions.
  set(AV2ENC_VIDEO_GPU_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/pipeline/video")
  set(AV2ENC_HAVE_VIDEO_GPU OFF)
  if(EXISTS "${AV2ENC_VIDEO_GPU_DIR}/video_gpu.h" AND
     EXISTS "${AV2ENC_VIDEO_GPU_DIR}/video_pipeline.cpp" AND
     EXISTS "${AV2ENC_VIDEO_GPU_DIR}/video_backend_cuda.cu")
    set(AV2ENC_VIDEO_GPU_SOURCES
      src/cuda/pipeline/video/video_pipeline.cpp
      src/cuda/pipeline/video/video_frame_modes.cpp
      src/cuda/pipeline/video/intra/video_intra.cu
      src/cuda/pipeline/video/video_backend_cuda.cu
      # Format video core (10-bit / chroma) CUDA backend.  Shares the CPU
      # orchestration and `hbd_tx_batch` with the still HBD module.
      src/cuda/pipeline/video/hbd/video_hbd_cuda.cu)
    # Opt-in luma TX size/type/partition search (--video-tx-search): the
    # verified device module, consumed by video_backend_cuda.cu.
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/pipeline/video/tx/video_tx.cu")
      list(APPEND AV2ENC_VIDEO_GPU_SOURCES
        src/cuda/pipeline/video/tx/video_tx.cu)
    endif()
    # Shared CUDA components the M2 stages call.  Link each translation unit
    # once: the M1 GPU target may already provide some of them, and duplicate
    # object symbols would break the executable link.
    set(AV2ENC_VIDEO_GPU_SUPPORT_SOURCES)
    foreach(support_source
        src/cuda/codec/inter/me/me.cu
        src/cuda/codec/inter/mc/mc.cu
        src/cuda/codec/inter/compound/compound_full.cu
        src/cuda/codec/inter/warp_stack/warp_stack.cu
        src/cuda/codec/inter/gme/gme.cu
        src/cuda/codec/inter/gme/gme_plan.cpp
        src/cuda/codec/intra/modes/intra_modes.cu
        src/cuda/pipeline/still/tx/still_tx.cu
        src/cuda/pipeline/still/hbd/hbd_tx.cu
        src/cuda/codec/transform/variants/tx_variants.cu)
      if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${support_source}")
        list(FIND AV2ENC_STILL_GPU_SUPPORT_SOURCES "${support_source}" _dup)
        list(FIND AV2ENC_STILL_GPU_SOURCES "${support_source}" _dup_primary)
        if(_dup EQUAL -1 AND _dup_primary EQUAL -1)
          list(APPEND AV2ENC_VIDEO_GPU_SUPPORT_SOURCES "${support_source}")
        endif()
      endif()
    endforeach()
    set(AV2ENC_HAVE_VIDEO_GPU ON)
    add_library(av2enc_video_gpu OBJECT
      ${AV2ENC_VIDEO_GPU_SOURCES}
      ${AV2ENC_VIDEO_GPU_SUPPORT_SOURCES})
    target_link_libraries(av2enc_video_gpu PUBLIC av2enc_core CUDA::cudart)
    target_include_directories(av2enc_video_gpu PRIVATE
      ${CUDAToolkit_INCLUDE_DIRS})
    target_compile_definitions(av2enc_video_gpu PRIVATE
      AV2ENC_HAVE_CUDA=1
      AV2ENC_HAVE_VIDEO_GPU=1)
    av2enc_apply_warnings(av2enc_video_gpu)
    foreach(video_source IN LISTS AV2ENC_VIDEO_GPU_SOURCES
                            AV2ENC_VIDEO_GPU_SUPPORT_SOURCES)
      if(video_source MATCHES "\\.cu$")
        set_source_files_properties("${video_source}" PROPERTIES
          COMPILE_OPTIONS "--fmad=false")
      endif()
    endforeach()
    message(STATUS
      "av2enc: M2 GPU video pipeline enabled (--video-gpu)")
  endif()
  # av2enc_core compiles main.cpp users against this macro; --video-gpu is
  # rejected by name when the target is absent (CPU-only build or missing tree).
  if(AV2ENC_HAVE_VIDEO_GPU)
    target_compile_definitions(av2enc_core PUBLIC AV2ENC_HAVE_VIDEO_GPU=1)
  else()
    target_compile_definitions(av2enc_core PUBLIC AV2ENC_HAVE_VIDEO_GPU=0)
  endif()
else()
  target_compile_definitions(av2enc_core PUBLIC
    AV2ENC_HAVE_CUDA=0
    AV2ENC_HAVE_STILL_GPU=0
    AV2ENC_HAVE_VIDEO_GPU=0)
endif()

# Optical formats: 10-bit 4:2:0 (M10-derived) and 8-bit 4:2:2/4:4:4/4:0:0.
# Both are opt-in product paths (--bit-depth 10 / --chroma ...), default off;
# see src/formats/README.md.  The components are CPU reference code and are
# always linked so the CPU-only build can gate them.
target_sources(av2enc_core PRIVATE
  src/formats/format_y4m.cpp
  src/formats/format_encode.cpp
  src/formats/high_bitdepth/hbd_depth.cpp
  src/formats/high_bitdepth/hbd_headers.cpp
  src/formats/high_bitdepth/hbd_reader.cpp
  src/formats/high_bitdepth/hbd_still.cpp
  src/pipeline/still/high_bitdepth/high_bitdepth_still.cpp
  src/pipeline/still/high_bitdepth/high_bitdepth_entropy.cpp
  src/pipeline/still/high_bitdepth/high_bitdepth_headers.cpp
  src/formats/chroma/chroma_format.cpp
  src/formats/chroma/chroma_residual.cpp
  src/formats/chroma/chroma_headers.cpp
  src/formats/chroma/chroma_still.cpp)
if(NOT AV2ENC_HAVE_STILL_GPU)
  # In a CUDA build with the M1 GPU pipeline, `format.cpp` is already added by
  # the still-GPU block above.  The CPU-only reference build needs it here.
  target_sources(av2enc_core PRIVATE
    src/formats/format.cpp)
endif()
# The format video core uses the verified 10-bit block arithmetic
# (`hbd::dequantize` / `hbd::inverse_block`) in both builds, so the oracle is
# part of the core archive; the M1 GPU target links it from there.
target_sources(av2enc_core PRIVATE
  src/cuda/pipeline/still/hbd/hbd_cpu.cpp)

# Multi-tile stills (--tiles): geometry planner, tile-group framing and the
# tiled writer.  Opt-in; the 1x1 default still stream is unchanged.
target_sources(av2enc_core PRIVATE
  src/pipeline/tiles/tile_layout.cpp
  src/pipeline/tiles/tile_group.cpp
  src/pipeline/tiles/stream/tiles_stream.cpp)

# Per-frame rate control for --video (--rc crf|vbr).  The default fixed-QP path
# does not call into these; they are always linked so the CPU gate can build.
target_sources(av2enc_core PRIVATE
  src/codec/rate/rc/rc.cpp
  src/pipeline/video/post/rc_adapter.cpp)

# Multi-tile video (--video --tiles).  The splitter re-enters the video syntax
# writer per tile; the default 1x1 --video stream never calls these.
target_sources(av2enc_core PRIVATE
  src/codec/filters/stream/filters_stream.cpp
  src/pipeline/video/post/bit_depth_checklist.cpp
  src/pipeline/video/post/loop_filters.cpp
  src/pipeline/video/post/tiled_frame.cpp
  # Opt-in M2 loop filters (--video-cdef, ...): CDEF/CCSO/LR decision + pixel
  # stage.  All flags default off, so the default stream is unchanged.
  src/pipeline/video/post/video_loop_filters.cpp)

# SVC layer encode (--video --svc-layers): per-layer sequences, global LCR/OPS
# and Annex F.3.2 selection.  Default off; the single-layer --video path never
# calls it.
target_sources(av2enc_core PRIVATE
  src/pipeline/tiles/svc_layers.cpp
  src/pipeline/svc/svc_config.cpp
  src/pipeline/svc/svc_pipeline.cpp
  src/pipeline/svc/svc_framing.cpp
  src/pipeline/svc/svc_stream.cpp)

# Host-only multi-GPU scheduling contract (--gpu-plan). Pure planning: no CUDA,
# no allocations.
target_sources(av2enc_core PRIVATE
  src/pipeline/dispatch/planner.cpp)

# Effort/cpu-used feature ranking (--effort). Host-only selector; the measured
# table is compiled in from src/pipeline/planner/effort_table.inc, which is
# generated by tools/effort/gen_table.py. Never called unless --effort is given.
target_sources(av2enc_core PRIVATE
  src/pipeline/planner/effort.cpp)

add_executable(av2enc src/main.cpp)
target_link_libraries(av2enc PRIVATE av2enc_core)
if(TARGET av2enc_still_gpu)
  # OBJECT library: all M1 GPU stage objects are linked directly, so weak
  # stage-hook references resolve without --whole-archive on av2enc_core.
  target_link_libraries(av2enc PRIVATE av2enc_still_gpu)
endif()
if(TARGET av2enc_video_gpu)
  # M2 video backend (--video-gpu).  The M1 target may already supply the shared
  # intra/transform components; CMake links each object once.
  target_link_libraries(av2enc PRIVATE av2enc_video_gpu)
endif()
av2enc_apply_warnings(av2enc)

# Encoder-reconstruction dump harness for the intra gates.  Test-only: it links
# the same core library and is not registered with CTest, so the pinned ctest
# inventory is unchanged.
add_executable(still_recon_dump src/pipeline/still/stream/still_recon_dump.cpp)
target_link_libraries(still_recon_dump PRIVATE av2enc_core)
av2enc_apply_warnings(still_recon_dump)

# Encoder-reconstruction dump harness for the loop-restoration gate.  The
# documented command is `cmake --build <build> --target still_dump`, so the
# target and the produced binary share that name.  Test-only: it links the same
# core library and is not registered with CTest (the gate is; see
# tests/gates/CMakeLists.txt), so the pinned ctest inventory is unchanged.
add_executable(still_dump tools/still_verify/still_dump.cpp)
target_link_libraries(still_dump PRIVATE av2enc_core)
av2enc_apply_warnings(still_dump)

# Multi-frame P-frame correctness gate (encode -> dav2d -> recon compare).  Not
# registered with CTest so the pinned test inventory is unchanged.
add_executable(video_gate src/pipeline/video/video_gate.cpp)
target_link_libraries(video_gate PRIVATE av2enc_core)
av2enc_apply_warnings(video_gate)

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
