#!/usr/bin/env bash
# Build and run the transform-partition dav2d gate.
#
# Proves the §5.20.6 TX_MODE_SELECT / transform-partition syntax against the
# pinned decoder without editing any repository source: m2_syntax.cpp and
# m2_headers.cpp are copied into the build directory and patched there (see
# patch_m2_syntax.py / patch_m2_headers.py).
#
# Usage:
#   ./build_gate.sh                 # build + run all cases
#   ./build_gate.sh --case B        # one case
#   ./build_gate.sh --no-run
#   SANITIZE=1 ./build_gate.sh
#   WERROR=1 ./build_gate.sh
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TX_DIR="$(cd -- "$HERE/.." && pwd)"
ROOT="$(cd -- "$TX_DIR/../../../.." && pwd)"
CXX="${CXX:-c++}"
BUILD_DIR="${BUILD_DIR:-${TMPDIR:-/tmp}/av2enc_tx_partition_gate}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"
RUN=1
ARGS=()

for arg in "$@"; do
  case "$arg" in
    --no-run) RUN=0 ;;
    --case) ARGS+=("$arg") ;;
    A|B|C|D|E|F) ARGS+=("$arg") ;;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

mkdir -p "$BUILD_DIR"
cp "$ROOT/src/encoder/m2/m2_syntax.cpp" "$BUILD_DIR/m2_syntax.cpp"
cp "$ROOT/src/encoder/m2/m2_headers.cpp" "$BUILD_DIR/m2_headers.cpp"
python3 "$HERE/patch_m2_syntax.py" "$BUILD_DIR/m2_syntax.cpp"
python3 "$HERE/patch_m2_headers.py" "$BUILD_DIR/m2_headers.cpp"

FLAGS=(-std=c++20 -Wall -Wextra -Wpedantic -Wconversion -Wshadow
       -Wno-missing-braces -I"$ROOT/src")
if [ "$WERROR" = "1" ]; then
  FLAGS+=(-Werror)
fi
if [ "$SANITIZE" = "1" ]; then
  FLAGS+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
else
  FLAGS+=(-O2)
fi

echo "compiler: $($CXX --version | head -1)"
echo "build:    $BUILD_DIR/tx_partition_gate"

CORE=(
  "$ROOT/src/io/y4m.cpp"
  "$ROOT/src/io/ivf.cpp"
  "$ROOT/src/bitstream/bitwriter.cpp"
  "$ROOT/src/bitstream/obu.cpp"
  "$ROOT/src/bitstream/m0_headers.cpp"
  "$ROOT/src/encoder/partition/partition.cpp"
  "$ROOT/src/encoder/tx_variants/tx_variants.cpp"
  "$ROOT/src/encoder/tx_variants/tx_select.cpp"
  "$ROOT/src/entropy/av2/av2_entropy.cpp"
  "$ROOT/src/encoder/inter/me/me.cpp"
  "$ROOT/src/encoder/inter/mc/mc.cpp"
  "$ROOT/src/encoder/m2/merge/merge.cpp"
  "$BUILD_DIR/m2_headers.cpp"
  "$BUILD_DIR/m2_syntax.cpp"
)

"$CXX" "${FLAGS[@]}" \
  "$HERE/tx_partition_gate.cpp" \
  "$HERE/tx_partition_gate_hook.cpp" \
  "$TX_DIR/tx_partition.cpp" \
  "$TX_DIR/tx_scan.cpp" \
  "${CORE[@]}" \
  -o "$BUILD_DIR/tx_partition_gate"

if [ -z "${DAV2D:-}" ]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [ -x "$candidate" ]; then
      export DAV2D="$candidate"
      break
    fi
  done
fi

cd "$ROOT"
if [ "$RUN" -eq 1 ]; then
  "${BUILD_DIR}/tx_partition_gate" --out-dir "$BUILD_DIR" \
    ${ARGS[@]+"${ARGS[@]}"}
fi
