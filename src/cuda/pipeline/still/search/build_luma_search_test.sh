#!/usr/bin/env bash
# Build (and by default run) the M1 luma candidate-search CUDA component test.
#
#   ./build_luma_search_test.sh                    # sm_86, build + run
#   ./build_luma_search_test.sh --no-run           # build only
#   ./build_luma_search_test.sh --quick            # reduced timing loop
#   ./build_luma_search_test.sh --no-timing        # parity only
#   ARCH=sm_120 ./build_luma_search_test.sh        # secondary target
#
# The test links the real CPU intra module (src/codec/intra/modes) as the
# reference. Self-contained: does not touch the root CMake build or any other
# component. Output goes to <repo>/build-m1-search by default.
set -euo pipefail

cd "$(dirname "$0")"

ARCH="${ARCH:-sm_86}"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
SRC="$ROOT/src"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-m1-search}"
RUN_ARGS=()

for arg in "$@"; do
  case "$arg" in
    --no-run)
      RUN_ARGS=("${RUN_ARGS[@]}" --build-only)
      ;;
    --quick|--no-timing|--timing-only|--allow-missing-media)
      RUN_ARGS+=("$arg")
      ;;
    --y4m64=*)
      RUN_ARGS+=(--y4m64 "${arg#--y4m64=}")
      ;;
    --y4m360=*)
      RUN_ARGS+=(--y4m360 "${arg#--y4m360=}")
      ;;
    --arch=*)
      ARCH="${arg#--arch=}"
      ;;
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

RUN=1
for arg in "${RUN_ARGS[@]}"; do
  if [ "$arg" = "--build-only" ]; then
    RUN=0
  fi
done

mkdir -p "$BUILD_DIR"

echo "nvcc: $(nvcc --version | tail -1)"
echo "arch: $ARCH"
echo "build: $BUILD_DIR/luma_search_test"

nvcc -std=c++20 -O3 -lineinfo -arch="$ARCH" \
  -I"$SRC" \
  -Xcompiler -Wall,-Wextra,-Wconversion,-Wshadow \
  --ptxas-options=-v \
  luma_search.cu luma_search_test.cu \
  "$SRC/cuda/codec/intra/modes/intra_modes.cu" \
  "$SRC/codec/intra/modes/intra_modes.cpp" \
  -o "$BUILD_DIR/luma_search_test"

if [ "$RUN" -eq 1 ]; then
  cd "$ROOT"
  exec "$BUILD_DIR/luma_search_test" "${RUN_ARGS[@]}"
fi
