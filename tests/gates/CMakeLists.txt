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
