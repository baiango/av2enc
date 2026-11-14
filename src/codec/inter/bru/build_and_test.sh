#!/usr/bin/env bash
#
# Build and run the backward-reference-update (BRU) component.
#
#   ./build_and_test.sh                 # unit tests + dav2d/avmdec gate
#   ./build_and_test.sh --no-gate       # unit tests only
#   ./build_and_test.sh --no-run        # build only
#   SANITIZE=1 ./build_and_test.sh
#   WERROR=1 ./build_and_test.sh
#
# The layer is self-contained under src/codec/inter/bru/. It links the
# untouched M2 header/tile writer for the reference frames and for the
# enable_bru=0 byte-identity control. No CMake or core edits are required.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
CXX="${CXX:-c++}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-bru}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"
RUN=1
GATE=1

for arg in "$@"; do
  case "$arg" in
    --no-gate) GATE=0 ;;
    --no-run) RUN=0 ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

mkdir -p "$BUILD_DIR"
FLAGS=(-std=c++20 -Wall -Wextra -Wpedantic -Wconversion -Wshadow
       -Wno-missing-braces -Wno-sign-conversion -ffp-contract=off -I"$ROOT/src")
if [[ "$SANITIZE" == "1" ]]; then
  FLAGS+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
else
  FLAGS+=(-O2 -g)
fi
if [[ "$WERROR" == "1" ]]; then
  FLAGS+=(-Werror)
fi
# src/pipeline/video/video_frame.cpp relies on a transitive include of
# codec/rate/rc/rc_tables.h; force it here instead of editing another agent's
# file (same workaround as the M2 and long-term gates).
VIDEO_FLAGS=("${FLAGS[@]}" -include "$ROOT/src/codec/rate/rc/rc_tables.h")

BRU_CORE=(
  "$HERE/bru_writer.cpp"
  "$HERE/bru_reader.cpp"
  "$HERE/bru_region.cpp"
  "$ROOT/src/core/bitstream/bitwriter.cpp"
  "$ROOT/src/core/bitstream/obu.cpp"
  "$ROOT/src/core/bitstream/basic_headers.cpp"
  "$ROOT/src/codec/entropy/av2/av2_entropy.cpp"
)
VIDEO_HEADERS=("$ROOT/src/pipeline/video/video_headers.cpp")
VIDEO_CORE=(
  "$ROOT/src/core/io/y4m.cpp"
  "$ROOT/src/core/io/ivf.cpp"
  "$ROOT/src/codec/partition/tree/partition.cpp"
  "$ROOT/src/codec/intra/modes/intra_modes.cpp"
  "$ROOT/src/codec/transform/variants/tx_variants.cpp"
  "$ROOT/src/codec/transform/variants/tx_select.cpp"
  "$ROOT/src/codec/inter/me/me.cpp"
  "$ROOT/src/codec/inter/mc/mc.cpp"
  "$ROOT/src/codec/inter/merge/merge.cpp"
  "$ROOT/src/pipeline/video/video_headers.cpp"
  "$ROOT/src/pipeline/video/video_syntax.cpp"
  "$ROOT/src/pipeline/video/video_frame.cpp"
  "$ROOT/src/codec/inter/temporal/ref_mv_bank.cpp"
)

echo "CXX: $($CXX --version | head -1)"
echo "build: $BUILD_DIR"

"$CXX" "${FLAGS[@]}" \
  "$HERE/bru_test.cpp" "${BRU_CORE[@]}" "${VIDEO_HEADERS[@]}" \
  -o "$BUILD_DIR/bru_test"

if [[ "$GATE" -eq 1 ]]; then
  "$CXX" "${VIDEO_FLAGS[@]}" \
    "$HERE/bru_gate.cpp" "${BRU_CORE[@]}" "${VIDEO_CORE[@]}" \
    -o "$BUILD_DIR/bru_gate"
fi

if [[ "$RUN" -eq 1 ]]; then
  "$BUILD_DIR/bru_test"
  if [[ "$GATE" -eq 1 ]]; then
    if [[ -z "${DAV2D:-}" ]]; then
      for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
        if [[ -x "$candidate" ]]; then
          export DAV2D="$candidate"
          break
        fi
      done
    fi
    if [[ -z "${DAV2D:-}" ]]; then
      echo "no dav2d found; set DAV2D=/path/to/dav2d" >&2
      exit 77
    fi
    if [[ -z "${AVMDEC:-}" ]]; then
      for candidate in "$ROOT/tools/bin/avm-macos/avmdec" \
                       "$ROOT/tools/bin/avmdec"; do
        if [[ -x "$candidate" ]]; then
          AVMDEC="$candidate"
          break
        fi
      done
    fi
    AVM_ARGS=()
    if [[ -n "${AVMDEC:-}" ]]; then
      AVM_ARGS+=(--avmdec "$AVMDEC")
    fi
    "$BUILD_DIR/bru_gate" --control --out "$BUILD_DIR/gate" \
      --dav2d "$DAV2D" ${AVM_ARGS[@]+"${AVM_ARGS[@]}"}
    "$BUILD_DIR/bru_gate" --out "$BUILD_DIR/gate" \
      --dav2d "$DAV2D" ${AVM_ARGS[@]+"${AVM_ARGS[@]}"}
  fi
fi
