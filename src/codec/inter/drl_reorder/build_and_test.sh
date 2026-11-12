#!/usr/bin/env bash
# Build and run the DRL reorder module tests and decoder gate.
#
#   ./build_and_test.sh                 # unit tests + gate (if dav2d exists)
#   ./build_and_test.sh --no-gate       # unit tests only
#   ./build_and_test.sh --no-run        # build only
#   SANITIZE=1 ./build_and_test.sh      # ASan + UBSan
#
# Self-contained: does not touch the root CMake build, src/main.cpp, the M2 core
# or any other agent's directory.  The merge translation unit is compiled as a
# read-only dependency of the adapter/test.
set -euo pipefail

cd "$(dirname "$0")"

CXX="${CXX:-c++}"
BUILD_DIR="${BUILD_DIR:-build-cpu}"
SANITIZE="${SANITIZE:-0}"
RUN=1
GATE=1

for arg in "$@"; do
  case "$arg" in
    --no-run) RUN=0 ;;
    --no-gate) GATE=0 ;;
    -h|--help)
      sed -n '2,14p' "$0"
      exit 0
      ;;
    *)
      echo "unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
mkdir -p "$BUILD_DIR"

FLAGS=(-std=c++20 -Wall -Wextra -Wpedantic -Wconversion -Wshadow
       -Wno-missing-braces -I "$ROOT/src")
if [[ "$SANITIZE" == "1" ]]; then
  FLAGS+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
else
  FLAGS+=(-O2 -g)
fi

DEPS=(
  drl_reorder.cpp
  drl_syntax.cpp
  video_stack_adapter.cpp
  "$ROOT/src/codec/inter/merge/merge.cpp"
  "$ROOT/src/codec/partition/tree/partition.cpp"
  "$ROOT/src/codec/inter/mc/mc.cpp"
  "$ROOT/src/codec/entropy/av2/av2_entropy.cpp"
)

echo "CXX: $($CXX --version | head -1)"
echo "build: $BUILD_DIR/drl_reorder_test"
"$CXX" "${FLAGS[@]}" drl_reorder_test.cpp "${DEPS[@]}" \
  -o "$BUILD_DIR/drl_reorder_test"

if [[ "$RUN" -eq 1 ]]; then
  "$BUILD_DIR/drl_reorder_test" --goldens goldens/drl_reorder_goldens.txt
fi

if [[ "$GATE" -eq 0 ]]; then
  exit 0
fi

# Decoder gate: key + P1 from the M2 core, then a reordering-enabled P2 with
# 16x16 leaves and two reference slots.  dav2d is the pinned black-box decoder.
if [[ -z "${DAV2D:-}" ]]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [[ -x "$candidate" ]]; then
      export DAV2D="$candidate"
      break
    fi
  done
fi

GATE_DEPS=(
  "$ROOT/src/core/io/y4m.cpp"
  "$ROOT/src/core/io/ivf.cpp"
  "$ROOT/src/core/bitstream/bitwriter.cpp"
  "$ROOT/src/core/bitstream/obu.cpp"
  "$ROOT/src/core/bitstream/basic_headers.cpp"
  "$ROOT/src/codec/partition/tree/partition.cpp"
  "$ROOT/src/codec/intra/modes/intra_modes.cpp"
  "$ROOT/src/codec/transform/variants/tx_variants.cpp"
  "$ROOT/src/codec/transform/variants/tx_select.cpp"
  "$ROOT/src/codec/entropy/av2/av2_entropy.cpp"
  "$ROOT/src/codec/inter/me/me.cpp"
  "$ROOT/src/codec/inter/mc/mc.cpp"
  "$ROOT/src/codec/inter/merge/merge.cpp"
  "$ROOT/src/pipeline/video/video_headers.cpp"
  "$ROOT/src/pipeline/video/video_syntax.cpp"
  "$ROOT/src/pipeline/video/video_frame.cpp"
  drl_reorder.cpp
  drl_syntax.cpp
  video_stack_adapter.cpp
)

echo "build: $BUILD_DIR/drl_gate"
"$CXX" "${FLAGS[@]}" gate/drl_gate.cpp gate/drl_stream.cpp gate/drl_tile.cpp \
  "${GATE_DEPS[@]}" -o "$BUILD_DIR/drl_gate"

if [[ -z "${DAV2D:-}" ]]; then
  echo "gate: SKIPPED (no dav2d; set DAV2D=/path/to/dav2d)"
  exit 0
fi

if [[ "$RUN" -eq 1 ]]; then
  echo "gate: spec reorder (ALWAYS) vs pinned dav2d"
  "$BUILD_DIR/drl_gate" --mode spec --out "$BUILD_DIR/drl_gate.ivf"
  SRC="$ROOT/testdata/shine_1280x720_60f.y4m"
  if [[ -f "$SRC" ]]; then
    echo "gate: real 720p clip (cropped to 1280x704)"
    "$BUILD_DIR/drl_gate" --input "$SRC" --mode spec \
      --out "$BUILD_DIR/drl_gate_720p.ivf"
  fi
  # The 8x8-leaf corpus is the smallest inter block AV2 permits in a
  # MIXED_REGION frame.  On that corpus the historical full selection sort and
  # the normative single swap happen to agree, so the decoder-side negative
  # control corrupts one DRL index; it must produce a non-zero recon diff.  The
  # exact old-logic failure is pinned by drl_reorder_test (weights {1,2,3,4}).
  echo "gate: negative control (corrupted DRL index) must desynchronise"
  if "$BUILD_DIR/drl_gate" --corrupt-index --out "$BUILD_DIR/drl_gate_bad.ivf" \
       --expect-desync; then
    echo "gate: desync detection reproduced"
  else
    echo "gate: FAIL: corrupted index was not detected" >&2
    exit 1
  fi
fi
