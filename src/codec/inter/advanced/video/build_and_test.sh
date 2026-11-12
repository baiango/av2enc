#!/usr/bin/env bash
# Build and run the M2 advanced-inter syntax/decision gate.
#
# This is a host-only CPU reference build.  It links the real normative entropy
# coder (src/codec/entropy/av2) and the reused advanced-inter components
# (src/codec/inter/advanced/{warp,compound,bawp}.cpp).  Every symbol test is a
# round trip through av2::Decoder, so the gate proves exact bit-level
# reversibility without touching the M2 core writer.
#
# Usage:
#   ./src/codec/inter/advanced/video/build_and_test.sh
#   SANITIZE=1 ./src/codec/inter/advanced/video/build_and_test.sh
#   WERROR=1   ./src/codec/inter/advanced/video/build_and_test.sh
#   CXX=g++-13 ./src/codec/inter/advanced/video/build_and_test.sh
#   OUT=/tmp/video_advanced ./src/codec/inter/advanced/video/build_and_test.sh
#
# Extra modes:
#   --check-tables   verify advanced_cdf_tables.h matches spec/tables/
#   --no-run         build only
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
CXX="${CXX:-c++}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"
OUT="${OUT:-${TMPDIR:-/tmp}/video_advanced_test}"
BUILD="${BUILD:-${TMPDIR:-/tmp}/video_advanced_build}"

FLAGS=(-std=c++20 -Wall -Wextra -Wpedantic -Wconversion -Wshadow
       -Wno-sign-conversion -I"$ROOT/src")
if [[ "${SANITIZE}" == "1" ]]; then
  FLAGS+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
else
  FLAGS+=(-O2)
fi
if [[ "${WERROR}" == "1" ]]; then
  FLAGS+=(-Werror)
fi

SOURCES=(
  "$ROOT/src/codec/inter/advanced/video/advanced_syntax.cpp"
  "$ROOT/src/codec/inter/advanced/video/advanced_decision.cpp"
  "$ROOT/src/codec/inter/advanced/warp.cpp"
  "$ROOT/src/codec/inter/advanced/compound.cpp"
  "$ROOT/src/codec/inter/advanced/bawp.cpp"
  "$ROOT/src/codec/partition/tree/partition.cpp"
  "$ROOT/src/codec/entropy/av2/av2_entropy.cpp"
  "$ROOT/src/core/bitstream/bitwriter.cpp"
)

RUN=1
for arg in "$@"; do
  case "$arg" in
    --no-run)
      RUN=0
      ;;
    --check-tables)
      python3 "$HERE/generate_advanced_tables.py" --check
      exit 0
      ;;
    -h|--help)
      sed -n '2,24p' "$0"
      exit 0
      ;;
    *)
      echo "unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

mkdir -p "$BUILD"
echo "compiler: $("$CXX" --version | head -1)"
OBJECTS=()
for src in "${SOURCES[@]}"; do
  obj="$BUILD/$(basename "${src%.cpp}").o"
  "$CXX" "${FLAGS[@]}" -c "$src" -o "$obj"
  OBJECTS+=("$obj")
done

"$CXX" "${FLAGS[@]}" -c "$HERE/advanced_test.cpp" -o "$BUILD/advanced_test.o"
"$CXX" "${FLAGS[@]}" "${OBJECTS[@]}" "$BUILD/advanced_test.o" -o "$OUT"

if [[ "$RUN" == "1" ]]; then
  "$OUT"
fi
