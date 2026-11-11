#!/usr/bin/env bash
# Build/run the AV2 quantizer-matrix CPU reference tests.
#
#   ./build_test.sh              # regenerate check + build + run
#   ./build_test.sh --no-run
#   SANITIZE=1 ./build_test.sh   # ASan + UBSan
set -euo pipefail
cd "$(dirname "$0")"

ROOT="$(cd ../../.. && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-qm}"
RUN=1
for arg in "$@"; do
  case "$arg" in
    --no-run) RUN=0 ;;
    -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

python3 generate_qm_tables.py --check

mkdir -p "$BUILD_DIR"
CXX="${CXX:-c++}"
FLAGS=(-std=c++20 -O2 -Wall -Wextra -Wpedantic -Wconversion -Wshadow
       -I"$ROOT/src")
if [ "${SANITIZE:-0}" -ne 0 ]; then
  FLAGS+=(-fsanitize=address,undefined -fno-omit-frame-pointer -g)
fi

"$CXX" "${FLAGS[@]}" \
  qm_test.cpp qm.cpp \
  "$ROOT/src/bitstream/bitwriter.cpp" \
  "$ROOT/src/bitstream/obu.cpp" \
  "$ROOT/src/bitstream/m0_headers.cpp" \
  -o "$BUILD_DIR/qm_test"

if [ "$RUN" -eq 1 ]; then
  "$BUILD_DIR/qm_test"
fi
