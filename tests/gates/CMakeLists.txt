# Product gates introduced during integration (wave 4).
#
# Every gate is a script that takes an explicit --bin, uses the pinned dav2d
# discovery from tools/validate_dav2d.sh, and exits 77 when a prerequisite is
# missing (decoder, python3, gitignored testdata fixtures). CTest reports 77 as
# a clean skip, so a bare CPU checkout keeps a green test run while a machine
# with tools/bin and testdata exercises the real product paths.

# Encoder-reconstruction dump harness for tools/still_verify/loop_restoration_gate.sh.
# The executable is defined at the top level as `still_dump` so the documented
# `cmake --build <build> --target still_dump` works whether or not CTest is
# enabled.  Keep `gate_still_dump` as an alias: earlier gate wiring and the
# build tree used that name.  Neither name is a CTest test.
if(NOT TARGET gate_still_dump)
  add_executable(gate_still_dump ALIAS still_dump)
endif()

# Guard against a future top-level target rename: the gate always points at the
# documented binary name, not at the compatibility alias.
if(NOT TARGET still_dump)
  message(FATAL_ERROR
    "av2enc: still_dump target is missing; gate_loop_restoration needs it")
endif()

# 10-bit 4:2:0 and 8-bit 4:2:2 / 4:4:4 / 4:0:0 still paths.
add_test(NAME gate_format
  COMMAND bash "${PROJECT_SOURCE_DIR}/src/formats/format_gate.sh"
          --bin $<TARGET_FILE:av2enc>)
set_tests_properties(gate_format PROPERTIES
  SKIP_RETURN_CODE 77
  TIMEOUT 600
  LABELS "gate")

# Opt-in intra tools: pinned MD5s, dav2d --strict 1, encoder-recon exactness.
add_test(NAME gate_intra
  COMMAND bash "${PROJECT_SOURCE_DIR}/tools/still_verify/intra_gate.sh"
          --bin $<TARGET_FILE:av2enc>
          --dump $<TARGET_FILE:still_recon_dump>)
set_tests_properties(gate_intra PROPERTIES
  SKIP_RETURN_CODE 77
  TIMEOUT 600
  LABELS "gate")

# Wiener / PC-Wiener / GDF / CCSO-edge / film grain loop restoration.
add_test(NAME gate_loop_restoration
  COMMAND bash "${PROJECT_SOURCE_DIR}/tools/still_verify/loop_restoration_gate.sh"
          --bin $<TARGET_FILE:av2enc>
          --dump $<TARGET_FILE:still_dump>)
set_tests_properties(gate_loop_restoration PROPERTIES
  SKIP_RETURN_CODE 77
  TIMEOUT 900
  LABELS "gate;gate-hd")

# Multi-tile stills (--tiles), including the 1080p layouts when testdata exists.
add_test(NAME gate_tiles
  COMMAND bash "${PROJECT_SOURCE_DIR}/src/pipeline/tiles/stream/gate_cli_tiles.sh"
          --bin $<TARGET_FILE:av2enc>)
set_tests_properties(gate_tiles PROPERTIES
  SKIP_RETURN_CODE 77
  TIMEOUT 600
  LABELS "gate;gate-hd")

# Per-frame rate control (--rc cqp|crf|vbr) on the --video path.
add_test(NAME gate_rc
  COMMAND bash "${PROJECT_SOURCE_DIR}/src/pipeline/video/post/gate_cli_rc.sh"
          --bin $<TARGET_FILE:av2enc>)
set_tests_properties(gate_rc PROPERTIES
  SKIP_RETURN_CODE 77
  TIMEOUT 600
  LABELS "gate;gate-video")

# Host-only multi-GPU planning contract (--gpu-plan). No decoder required.
add_test(NAME gate_host_plan
  COMMAND bash "${PROJECT_SOURCE_DIR}/src/pipeline/dispatch/gate_host_plan.sh"
          --bin $<TARGET_FILE:av2enc>)
set_tests_properties(gate_host_plan PROPERTIES
  SKIP_RETURN_CODE 77
  TIMEOUT 120
  LABELS "gate")

# Opt-in multi-frame inter path: encode -> dav2d -> sample-exact recon compare.
add_test(NAME gate_video_cli
  COMMAND bash "${PROJECT_SOURCE_DIR}/src/pipeline/video/video_cli_gate.sh"
          --bin $<TARGET_FILE:av2enc> --frames 4)
set_tests_properties(gate_video_cli PROPERTIES
  SKIP_RETURN_CODE 77
  TIMEOUT 300
  LABELS "gate;gate-video")

# Opt-in M2 loop filters on --video (--video-cdef, ...): pinned default stream,
# dav2d --strict 1, and encoder-recon == decoded output for every filter case.
add_test(NAME gate_video_filters
  COMMAND bash "${PROJECT_SOURCE_DIR}/src/pipeline/video/gate_video_filters.sh"
          --bin $<TARGET_FILE:av2enc>)
set_tests_properties(gate_video_filters PROPERTIES
  SKIP_RETURN_CODE 77
  TIMEOUT 600
  LABELS "gate;gate-video")

# Video partition tools: 128x128 / 256x256 superblocks and extended/uneven
# 4-way partitions.  Each case decodes with dav2d --strict 1 and compares the
# encoder reconstruction sample-for-sample; the default 64x64 stream MD5s are
# part of the same gate.
add_test(NAME gate_video_partitions
  COMMAND bash "${PROJECT_SOURCE_DIR}/src/pipeline/video/gate_video_partitions.sh"
          --bin $<TARGET_FILE:av2enc>)
set_tests_properties(gate_video_partitions PROPERTIES
  SKIP_RETURN_CODE 77
  TIMEOUT 900
  LABELS "gate;gate-video")

# Opt-in CUDA M2 path: CPU --video vs --video-gpu on one deterministic 360p
# source, byte-identical temporal units/recon and a dav2d --strict 1 decode,
# plus named rejections for every unported combination.  Registered only when
# the CUDA target exists; the script itself skips on a CPU-only binary.
if(TARGET av2enc_video_gpu)
  add_test(NAME gate_video_gpu
    COMMAND bash "${PROJECT_SOURCE_DIR}/src/cuda/pipeline/video/gate_video_gpu.sh"
            --bin $<TARGET_FILE:av2enc> --frames 4)
  # The reference-model cases add seven 8-frame 360p CPU+GPU encodes (the
  # NumTotalRefs 4..7 / NumRefFrames 4..8 matrix), so the budget covers a
  # loaded shared box as well as an idle one.
  set_tests_properties(gate_video_gpu PROPERTIES
    SKIP_RETURN_CODE 77
    TIMEOUT 3600
    LABELS "gate;gate-video;gate-gpu")
  # Second product case: the 8-frame long-chain subset that exercises the
  # rotating reference-slot pool.  The full matrix already runs at 4 frames,
  # and every fixed-count 8-frame case runs in the main invocation.
  add_test(NAME gate_video_gpu_long
    COMMAND bash "${PROJECT_SOURCE_DIR}/src/cuda/pipeline/video/gate_video_gpu.sh"
            --bin $<TARGET_FILE:av2enc> --frames 8 --long-only)
  set_tests_properties(gate_video_gpu_long PROPERTIES
    SKIP_RETURN_CODE 77
    TIMEOUT 1200
    LABELS "gate;gate-video;gate-gpu;long")
  # Opt-in full luma intra search (--video-intra-full): the still product's
  # 61-candidate search on CPU vs the device candidate module, byte-identical
  # units/recon and a dav2d --strict 1 decode, 8 frames at 360p.
  add_test(NAME gate_video_gpu_intra
    COMMAND bash "${PROJECT_SOURCE_DIR}/src/cuda/pipeline/video/gate_video_gpu.sh"
            --bin $<TARGET_FILE:av2enc> --frames 8 --intra-only)
  set_tests_properties(gate_video_gpu_intra PROPERTIES
    SKIP_RETURN_CODE 77
    TIMEOUT 1800
    LABELS "gate;gate-video;gate-gpu")
  # Opt-in §5.20.6/§5.20.8 luma TX size/type/partition search
  # (--video-tx-search): the device module path must be byte-identical to the
  # CPU reference and decode with dav2d --strict 1.
  add_test(NAME gate_video_tx_search
    COMMAND bash "${PROJECT_SOURCE_DIR}/src/cuda/pipeline/video/gate_video_gpu.sh"
            --bin $<TARGET_FILE:av2enc> --frames 4 --tx-only)
  set_tests_properties(gate_video_tx_search PROPERTIES
    SKIP_RETURN_CODE 77
    TIMEOUT 1800
    LABELS "gate;gate-video;gate-gpu;gate-tx")
  # Opt-in §5.20.7.6/§7.13.3.25 BAWP (--video-bawp): the fitted predictor must
  # produce byte-identical temporal units and recon on CPU and the CUDA
  # backend at 360p, 4 and 8 frames, and the stream must decode with
  # dav2d --strict 1.
  add_test(NAME gate_video_bawp
    COMMAND bash "${PROJECT_SOURCE_DIR}/src/cuda/pipeline/video/gate_video_gpu.sh"
            --bin $<TARGET_FILE:av2enc> --bawp-only)
  set_tests_properties(gate_video_bawp PROPERTIES
    SKIP_RETURN_CODE 77
    TIMEOUT 1800
    LABELS "gate;gate-video;gate-gpu;gate-bawp")
  # Opt-in §7.13.5 CfL chroma decision (--video-cfl): the CPU {DC, CfL} policy
  # vs the device `still::chroma_decide()` module, byte-identical at 4 and 8
  # frames with a dav2d --strict 1 decode.
  add_test(NAME gate_video_cfl
    COMMAND bash "${PROJECT_SOURCE_DIR}/src/cuda/pipeline/video/gate_video_gpu.sh"
            --bin $<TARGET_FILE:av2enc> --frames 4 --cfl-only)
  set_tests_properties(gate_video_cfl PROPERTIES
    SKIP_RETURN_CODE 77
    TIMEOUT 1800
    LABELS "gate;gate-video;gate-gpu;gate-cfl")
  # Opt-in §7.13.6-.8 MHCCP (--video-mhccp): enable_mhccp signalling plus the
  # three CFL_MULTI directions, CPU vs the device chroma decision module.
  add_test(NAME gate_video_mhccp
    COMMAND bash "${PROJECT_SOURCE_DIR}/src/cuda/pipeline/video/gate_video_gpu.sh"
            --bin $<TARGET_FILE:av2enc> --frames 4 --mhccp-only)
  set_tests_properties(gate_video_mhccp PROPERTIES
    SKIP_RETURN_CODE 77
    TIMEOUT 1800
    LABELS "gate;gate-video;gate-gpu;gate-mhccp")
  # Opt-in §7.20.4 luma PC-Wiener (--video-pc-wiener): its own lr_params()/
  # read_lr() dispatch plus the device pc_wiener workspace.
  add_test(NAME gate_video_pc_wiener
    COMMAND bash "${PROJECT_SOURCE_DIR}/src/cuda/pipeline/video/gate_video_gpu.sh"
            --bin $<TARGET_FILE:av2enc> --frames 4 --pcw-only)
  set_tests_properties(gate_video_pc_wiener PROPERTIES
    SKIP_RETURN_CODE 77
    TIMEOUT 1800
    LABELS "gate;gate-video;gate-gpu;gate-pcw")
  # Opt-in §5.18.6.2 default-matrix QM (--video-qm).
  add_test(NAME gate_video_qm
    COMMAND bash "${PROJECT_SOURCE_DIR}/src/cuda/pipeline/video/gate_video_gpu.sh"
            --bin $<TARGET_FILE:av2enc> --frames 4 --qm-only)
  set_tests_properties(gate_video_qm PROPERTIES
    SKIP_RETURN_CODE 77
    TIMEOUT 1800
    LABELS "gate;gate-video;gate-gpu;gate-qm")
  # Format video core: 10-bit 4:2:0/4:2:2/4:4:4/4:0:0 and 8-bit
  # 4:2:2/4:4:4/4:0:0, CPU vs CUDA byte-identical at 360p / 4 frames with a
  # dav2d --strict 1 decode and the expected picture count.
  add_test(NAME gate_video_formats
    COMMAND bash "${PROJECT_SOURCE_DIR}/src/cuda/pipeline/video/gate_video_formats.sh"
            --bin $<TARGET_FILE:av2enc> --frames 4)
  set_tests_properties(gate_video_formats PROPERTIES
    SKIP_RETURN_CODE 77
    TIMEOUT 3600
    LABELS "gate;gate-video;gate-gpu;gate-formats")
endif()
