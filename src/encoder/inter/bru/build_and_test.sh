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
# The layer is self-contained under src/encoder/inter/bru/. It links the
# untouched M2 header/tile writer for the reference frames and for the
# enable_bru=0 byte-identity control. No CMake or core edits are required.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$HERE/../../../.." && pwd)"
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
# src/encoder/m2/m2_frame.cpp relies on a transitive include of
# encoder/rc/rc_tables.h; force it here instead of editing another agent's
# file (same workaround as the M2 and long-term gates).
M2_FLAGS=("${FLAGS[@]}" -include "$ROOT/src/encoder/rc/rc_tables.h")

BRU_CORE=(
  "$HERE/bru_writer.cpp"
  "$HERE/bru_reader.cpp"
  "$HERE/bru_region.cpp"
  "$ROOT/src/bitstream/bitwriter.cpp"
  "$ROOT/src/bitstream/obu.cpp"
  "$ROOT/src/bitstream/m0_headers.cpp"
  "$ROOT/src/entropy/av2/av2_entropy.cpp"
)
M2_HEADERS=("$ROOT/src/encoder/m2/m2_headers.cpp")
M2_CORE=(
  "$ROOT/src/io/y4m.cpp"
  "$ROOT/src/io/ivf.cpp"
  "$ROOT/src/encoder/partition/partition.cpp"
  "$ROOT/src/encoder/intra_modes/intra_modes.cpp"
  "$ROOT/src/encoder/tx_variants/tx_variants.cpp"
  "$ROOT/src/encoder/tx_variants/tx_select.cpp"
  "$ROOT/src/encoder/inter/me/me.cpp"
  "$ROOT/src/encoder/inter/mc/mc.cpp"
  "$ROOT/src/encoder/m2/merge/merge.cpp"
  "$ROOT/src/encoder/m2/m2_headers.cpp"
  "$ROOT/src/encoder/m2/m2_syntax.cpp"
  "$ROOT/src/encoder/m2/m2_frame.cpp"
)

echo "CXX: $($CXX --version | head -1)"
echo "build: $BUILD_DIR"

"$CXX" "${FLAGS[@]}" \
  "$HERE/bru_test.cpp" "${BRU_CORE[@]}" "${M2_HEADERS[@]}" \
  -o "$BUILD_DIR/bru_test"

if [[ "$GATE" -eq 1 ]]; then
  "$CXX" "${M2_FLAGS[@]}" \
    "$HERE/bru_gate.cpp" "${BRU_CORE[@]}" "${M2_CORE[@]}" \
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
