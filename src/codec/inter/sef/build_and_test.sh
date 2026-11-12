#!/usr/bin/env bash
# Build and run the show-existing-frame (SEF) syntax tests and decoder gate.
#
#   ./src/codec/inter/sef/build_and_test.sh
#   ./src/codec/inter/sef/build_and_test.sh --no-run
#   ./src/codec/inter/sef/build_and_test.sh --test-only
#   ./src/codec/inter/sef/build_and_test.sh --gate-only
#   SANITIZE=1 ./src/codec/inter/sef/build_and_test.sh
#   WERROR=1   ./src/codec/inter/sef/build_and_test.sh
#   DAV2D=/path/to/dav2d ./src/codec/inter/sef/build_and_test.sh
#
# The gate builds its own 2- and 3-frame M2 streams with the existing M2
# encoder, appends SEF OBUs, and decodes with the pinned dav2d. No root
# CMakeLists.txt, src/main.cpp, M1/M2 core, or existing inter directory is
# modified.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
CXX="${CXX:-c++}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"
OUT="${OUT:-${TMPDIR:-/tmp}/sef_build}"
RUN_TEST=1
RUN_GATE=1

for arg in "$@"; do
  case "$arg" in
    --no-run)
      RUN_TEST=0
      RUN_GATE=0
      ;;
    --test-only)
      RUN_GATE=0
      ;;
    --gate-only)
      RUN_TEST=0
      ;;
    -h|--help)
      sed -n '2,18p' "$0"
      exit 0
      ;;
    *)
      echo "unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

FLAGS=(-std=c++20 -Wall -Wextra -Wpedantic -Wconversion -Wshadow
       -Wno-missing-braces -I"$ROOT/src")
if [[ "${SANITIZE}" == "1" ]]; then
  FLAGS+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
else
  FLAGS+=(-O2 -g)
fi
if [[ "${WERROR}" == "1" ]]; then
  FLAGS+=(-Werror)
fi

mkdir -p "$OUT"

echo "compiler: $("$CXX" --version | head -1)"

# --- syntax tests -----------------------------------------------------------
TEST_BIN="$OUT/sef_test"
"$CXX" "${FLAGS[@]}" \
  "$HERE/sef.cpp" "$HERE/sef_test.cpp" \
  "$ROOT/src/core/bitstream/bitwriter.cpp" \
  "$ROOT/src/core/bitstream/basic_headers.cpp" \
  -o "$TEST_BIN"
if [[ "$RUN_TEST" == "1" ]]; then
  "$TEST_BIN"
fi

# --- decoder gate -----------------------------------------------------------
if [[ "$RUN_TEST" == "0" || "$RUN_GATE" == "1" ]]; then
  GATE_BIN="$OUT/sef_gate"
  # The existing M2 core is not -Werror-clean (pre-existing sign-conversion
  # warnings), so -Werror is applied to the syntax test above, not to the gate
  # link line. This does not weaken any SEF check.
  GATE_FLAGS=()
  for flag in "${FLAGS[@]}"; do
    if [[ "$flag" != "-Werror" ]]; then GATE_FLAGS+=("$flag"); fi
  done
  "$CXX" "${GATE_FLAGS[@]}" \
    "$HERE/sef_gate.cpp" "$HERE/sef.cpp" \
    "$ROOT/src/core/io/y4m.cpp" \
    "$ROOT/src/core/io/ivf.cpp" \
    "$ROOT/src/core/bitstream/bitwriter.cpp" \
    "$ROOT/src/core/bitstream/obu.cpp" \
    "$ROOT/src/core/bitstream/basic_headers.cpp" \
    "$ROOT/src/codec/partition/tree/partition.cpp" \
    "$ROOT/src/codec/intra/modes/intra_modes.cpp" \
    "$ROOT/src/codec/transform/variants/tx_variants.cpp" \
    "$ROOT/src/codec/transform/variants/tx_select.cpp" \
    "$ROOT/src/codec/entropy/av2/av2_entropy.cpp" \
    "$ROOT/src/codec/inter/me/me.cpp" \
    "$ROOT/src/codec/inter/mc/mc.cpp" \
    "$ROOT/src/codec/inter/merge/merge.cpp" \
    "$ROOT/src/codec/inter/drl_reorder/drl_reorder.cpp" \
    "$ROOT/src/codec/inter/drl_reorder/video_stack_adapter.cpp" \
    "$ROOT/src/codec/inter/mv_precision/mv_precision.cpp" \
    "$ROOT/src/pipeline/video/video_headers.cpp" \
    "$ROOT/src/pipeline/video/video_syntax.cpp" \
    "$ROOT/src/pipeline/video/video_frame.cpp" \
    -o "$GATE_BIN"

  if [[ -z "${DAV2D:-}" ]]; then
    for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
      if [[ -x "$candidate" ]]; then
        export DAV2D="$candidate"
        break
      fi
    done
  fi
  if [[ -z "${DAV2D:-}" ]]; then
    echo "no dav2d binary found; set DAV2D (gate skipped)" >&2
    exit 127
  fi

  if [[ "$RUN_GATE" == "1" ]]; then
    cd "$ROOT"
    "$GATE_BIN" --dav2d "$DAV2D" --work "${OUT}/gate"
  fi
fi
