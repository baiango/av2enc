#!/usr/bin/env bash
# Build/run the AV2 DIP CPU reference tests (§5.20.7.9, §7.13.2.3-.6).
#
#   ./build_and_test.sh              # build + run (uses shine_256x256_1f.y4m)
#   ./build_and_test.sh --no-run
#   SANITIZE=1 ./build_and_test.sh   # ASan + UBSan
#   WERROR=1 ./build_and_test.sh
set -euo pipefail
cd "$(dirname "$0")"

HERE="$(pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
BUILD_DIR="${BUILD_DIR:-$ROOT/build-dip}"
CXX="${CXX:-c++}"
RUN=1
for arg in "$@"; do
  case "$arg" in
    --no-run) RUN=0 ;;
    -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

mkdir -p "$BUILD_DIR"
FLAGS=(-std=c++20 -O2 -Wall -Wextra -Wpedantic -Wconversion -Wshadow
       -I"$ROOT/src")
if [ "${SANITIZE:-0}" -ne 0 ]; then
  FLAGS+=(-fsanitize=address,undefined -fno-omit-frame-pointer -g)
fi
if [ "${WERROR:-0}" -ne 0 ]; then
  FLAGS+=(-Werror)
fi

"$CXX" "${FLAGS[@]}" \
  "$HERE/dip.cpp" "$HERE/dip_syntax.cpp" "$HERE/dip_test.cpp" \
  "$ROOT/src/core/io/y4m.cpp" \
  -o "$BUILD_DIR/dip_test"

if [ "$RUN" -eq 1 ]; then
  "$BUILD_DIR/dip_test" "$ROOT/testdata/shine_256x256_1f.y4m"
fi
