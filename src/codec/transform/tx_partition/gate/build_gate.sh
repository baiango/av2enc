#!/usr/bin/env bash
# Build and run the transform-partition dav2d gate.
#
# Proves the §5.20.6 TX_MODE_SELECT / transform-partition syntax against the
# pinned decoder without editing any repository source: video_syntax.cpp and
# video_headers.cpp are copied into the build directory and patched there (see
# patch_video_syntax.py / patch_video_headers.py).
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
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
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
cp "$ROOT/src/pipeline/video/video_syntax.cpp" "$BUILD_DIR/video_syntax.cpp"
cp "$ROOT/src/pipeline/video/video_headers.cpp" "$BUILD_DIR/video_headers.cpp"
python3 "$HERE/patch_video_syntax.py" "$BUILD_DIR/video_syntax.cpp"
python3 "$HERE/patch_video_headers.py" "$BUILD_DIR/video_headers.cpp"

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
  "$ROOT/src/core/io/y4m.cpp"
  "$ROOT/src/core/io/ivf.cpp"
  "$ROOT/src/core/bitstream/bitwriter.cpp"
  "$ROOT/src/core/bitstream/obu.cpp"
  "$ROOT/src/core/bitstream/basic_headers.cpp"
  "$ROOT/src/codec/partition/tree/partition.cpp"
  "$ROOT/src/codec/transform/variants/tx_variants.cpp"
  "$ROOT/src/codec/transform/variants/tx_select.cpp"
  "$ROOT/src/codec/entropy/av2/av2_entropy.cpp"
  "$ROOT/src/codec/inter/me/me.cpp"
  "$ROOT/src/codec/inter/mc/mc.cpp"
  "$ROOT/src/codec/inter/merge/merge.cpp"
  "$BUILD_DIR/video_headers.cpp"
  "$BUILD_DIR/video_syntax.cpp"
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
