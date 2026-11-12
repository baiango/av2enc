#!/usr/bin/env bash
# Build/run the real-GPU AV2 temporal motion-field parity test.
#
#   ./build_gpu_test.sh                 # sm_86, build + run
#   ./build_gpu_test.sh --no-run        # build only
#   ./build_gpu_test.sh --quick         # skip the 1080p case
#   ./build_gpu_test.sh --sanitize      # run under compute-sanitizer memcheck
#   ARCH=sm_120 ./build_gpu_test.sh     # secondary target
#
# Needs nvcc and a CUDA device.  On hosts without nvcc use ./build_and_test.sh,
# which runs the same shared device header against the CPU reference on the
# host.
#
# All device buffers in the test come from one av2enc::cuda::Arena; the module
# and the test call no cudaMalloc/cudaFree/cudaMallocHost.
set -euo pipefail

cd "$(dirname "$0")"

HERE="$(pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
ARCH="${ARCH:-sm_86}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-motion-field-gpu}"
CXX="${CXX:-c++}"
RUN=1
SANITIZE=0
RUN_ARGS=()

for arg in "$@"; do
  case "$arg" in
    --no-run) RUN=0 ;;
    --quick) RUN_ARGS+=(--quick) ;;
    --sanitize) SANITIZE=1 ;;
    --arch=*) ARCH="${arg#--arch=}" ;;
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

if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc not found; use ./build_and_test.sh for host-emulation parity" >&2
  exit 2
fi

mkdir -p "$BUILD_DIR"

echo "nvcc: $(nvcc --version | tail -1)"
echo "host: $($CXX --version | head -1)"
echo "arch: $ARCH"
echo "build: $BUILD_DIR/motion_field_test"

# CPU reference objects (the oracle), compiled by the host compiler.
for src in motion_field ref_mv_bank temporal_common; do
  "$CXX" -std=c++20 -O2 -g \
    -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Wno-sign-conversion \
    -I "$ROOT/src" \
    -c "$ROOT/src/codec/inter/temporal/$src.cpp" \
    -o "$BUILD_DIR/${src}_cpu.o"
done

# Kernel + test + the persistent arena runtime.  --expt-relaxed-constexpr lets
# device code read the namespace-scope constexpr scalar constants shared with
# the CPU headers; the two runtime-indexed tables (Div_Mult, Tip_Weighting_
# Factor) are switch functions in motion_field_device.cuh so no device symbol
# is required for them.
nvcc -std=c++20 -O3 -g -lineinfo -arch="$ARCH" --expt-relaxed-constexpr \
  -Xcompiler -Wall,-Wextra,-Wconversion,-Wshadow \
  --ptxas-options=-v \
  -I "$ROOT/src" -I "$HERE" \
  "$HERE/motion_field.cu" "$HERE/motion_field_test.cu" \
  "$ROOT/src/cuda/arena.cpp" \
  "$BUILD_DIR/motion_field_cpu.o" \
  "$BUILD_DIR/ref_mv_bank_cpu.o" \
  "$BUILD_DIR/temporal_common_cpu.o" \
  -o "$BUILD_DIR/motion_field_test"

if [ "$RUN" -eq 1 ]; then
  if [ "$SANITIZE" -eq 1 ]; then
    if ! command -v compute-sanitizer >/dev/null 2>&1; then
      echo "compute-sanitizer not found" >&2
      exit 2
    fi
    CUDA_DISABLE_PTX_JIT=1 compute-sanitizer --tool memcheck \
      --error-exitcode 9 "$BUILD_DIR/motion_field_test" "${RUN_ARGS[@]}"
  elif [ "${#RUN_ARGS[@]}" -gt 0 ]; then
    CUDA_DISABLE_PTX_JIT=1 "$BUILD_DIR/motion_field_test" "${RUN_ARGS[@]}"
  else
    CUDA_DISABLE_PTX_JIT=1 "$BUILD_DIR/motion_field_test"
  fi
fi
