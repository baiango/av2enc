#!/usr/bin/env bash
# Build/run the §5.13 / §6.12 quantizer-matrix OBU state-machine tests.
#
#   ./build_test.sh              # build + run
#   ./build_test.sh --no-run
#   SANITIZE=1 ./build_test.sh   # ASan + UBSan
set -euo pipefail
cd "$(dirname "$0")"

ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
BUILD_DIR="${BUILD_DIR:-$ROOT/build-qm-obu}"
RUN=1
for arg in "$@"; do
  case "$arg" in
    --no-run) RUN=0 ;;
    -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

mkdir -p "$BUILD_DIR"
CXX="${CXX:-c++}"
FLAGS=(-std=c++20 -O2 -Wall -Wextra -Wpedantic -Wconversion -Wshadow
       -I"$ROOT/src")
if [ "${SANITIZE:-0}" -ne 0 ]; then
  FLAGS+=(-fsanitize=address,undefined -fno-omit-frame-pointer -g)
fi

"$CXX" "${FLAGS[@]}" \
  qm_obu_test.cpp qm_obu.cpp \
  "$ROOT/src/core/bitstream/bitwriter.cpp" \
  -o "$BUILD_DIR/qm_obu_test"

if [ "$RUN" -eq 1 ]; then
  "$BUILD_DIR/qm_obu_test"
fi
