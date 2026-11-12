#!/usr/bin/env bash
# Build and run the TIP frame-level signalling / build-TIP component.
#
#   ./build_and_test.sh                # unit tests (goldens) + gate if dav2d
#   ./build_and_test.sh --no-gate      # unit tests only
#   ./build_and_test.sh --no-run       # build only
#   SANITIZE=1 ./build_and_test.sh     # ASan + UBSan
#   CXX=clang++ ./build_and_test.sh
#
# The gate needs tools/bin/dav2d (Linux) or tools/bin/dav2d-macos (macOS), or
# DAV2D=/path/to/dav2d.  Without a decoder the unit tests still run and the
# gate is skipped (exit 0).
#
# Self-contained: this script compiles existing M2/temporal sources read-only.
# It never edits CMakeLists.txt, src/main.cpp, the M2 core or another
# component's directory.
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
      sed -n '2,16p' "$0"
      exit 0
      ;;
    *)
      echo "unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

ROOT="$(cd ../../../.. && pwd)"
mkdir -p "$BUILD_DIR"

FLAGS=(-std=c++20 -Wall -Wextra -Wpedantic -Wconversion -Wshadow
       -Wno-sign-conversion -Wno-missing-braces -ffp-contract=off
       -I "$ROOT/src")
if [[ "$SANITIZE" == "1" ]]; then
  FLAGS+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
else
  FLAGS+=(-O2 -g)
fi

TEMPORAL=(
  "$ROOT/src/encoder/inter/temporal/motion_field.cpp"
  "$ROOT/src/encoder/inter/temporal/temporal_common.cpp"
  "$ROOT/src/encoder/inter/temporal/tip.cpp"
  "$ROOT/src/encoder/inter/temporal/temporal_syntax.cpp"
  "$ROOT/src/encoder/inter/temporal/refinemv.cpp"
)

echo "CXX: $($CXX --version | head -1)"
echo "build: $BUILD_DIR/tip_stream_test"
"$CXX" "${FLAGS[@]}" tip_signalling.cpp tip_candidates.cpp tip_build.cpp \
  tip_output.cpp tip_stream_test.cpp "${TEMPORAL[@]}" \
  "$ROOT/src/bitstream/bitwriter.cpp" \
  "$ROOT/src/bitstream/m0_headers.cpp" \
  -o "$BUILD_DIR/tip_stream_test"

if [[ "$RUN" -eq 1 ]]; then
  "$BUILD_DIR/tip_stream_test" --goldens goldens/tip_goldens.txt
fi

if [[ "$GATE" -eq 0 ]]; then
  exit 0
fi

GATE_DEPS=(
  "$ROOT/src/io/y4m.cpp"
  "$ROOT/src/io/ivf.cpp"
  "$ROOT/src/bitstream/bitwriter.cpp"
  "$ROOT/src/bitstream/obu.cpp"
  "$ROOT/src/bitstream/m0_headers.cpp"
  "$ROOT/src/encoder/partition/partition.cpp"
  "$ROOT/src/encoder/intra_modes/intra_modes.cpp"
  "$ROOT/src/encoder/tx_variants/tx_variants.cpp"
  "$ROOT/src/encoder/tx_variants/tx_select.cpp"
  "$ROOT/src/entropy/av2/av2_entropy.cpp"
  "$ROOT/src/encoder/inter/me/me.cpp"
  "$ROOT/src/encoder/inter/mc/mc.cpp"
  "$ROOT/src/encoder/m2/merge/merge.cpp"
  "$ROOT/src/encoder/m2/m2_headers.cpp"
  "$ROOT/src/encoder/m2/m2_syntax.cpp"
  "$ROOT/src/encoder/m2/m2_frame.cpp"
)

echo "build: $BUILD_DIR/tip_gate"
"$CXX" "${FLAGS[@]}" gate/tip_gate.cpp tip_signalling.cpp tip_candidates.cpp \
  tip_build.cpp tip_output.cpp "${TEMPORAL[@]}" "${GATE_DEPS[@]}" \
  -o "$BUILD_DIR/tip_gate"

if [[ -z "${DAV2D:-}" ]]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [[ -x "$candidate" ]]; then
      DAV2D="$candidate"
      break
    fi
  done
fi
if [[ -z "${DAV2D:-}" ]]; then
  echo "gate: SKIPPED (no dav2d; set DAV2D=/path/to/dav2d)"
  exit 0
fi
export DAV2D

if [[ "$RUN" -eq 1 ]]; then
  echo "gate: 64x64 TIP-as-output + enable_tip=0 control"
  "$BUILD_DIR/tip_gate" --width 64 --height 64 \
    --out "$BUILD_DIR/tip_gate_64.ivf" \
    --decoded "$BUILD_DIR/tip_gate_64.y4m" \
    --control "$BUILD_DIR/tip_gate_64_control.ivf"
  echo "gate: 640x360 TIP-as-output + enable_tip=0 control"
  "$BUILD_DIR/tip_gate" --width 640 --height 360 \
    --out "$BUILD_DIR/tip_gate_360.ivf" \
    --decoded "$BUILD_DIR/tip_gate_360.y4m" \
    --control "$BUILD_DIR/tip_gate_360_control.ivf"
  if [[ -f "$ROOT/testdata/shine_1280x720_60f.y4m" ]]; then
    echo "gate: 640x360 real content (720p clip cropped)"
    "$BUILD_DIR/tip_gate" --width 640 --height 360 \
      --input "$ROOT/testdata/shine_1280x720_60f.y4m" \
      --out "$BUILD_DIR/tip_gate_real.ivf" \
      --decoded "$BUILD_DIR/tip_gate_real.y4m" \
      --control "$BUILD_DIR/tip_gate_real_control.ivf"
  fi
fi
