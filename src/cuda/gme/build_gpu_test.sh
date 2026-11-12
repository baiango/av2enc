#!/usr/bin/env bash
# Build/run the real-GPU bit-exact parity test for the CUDA global-motion
# estimator.
#
#   ./build_gpu_test.sh                 # sm_86, build + run
#   ./build_gpu_test.sh --no-run        # build only
#   ./build_gpu_test.sh --quick         # skip the real 720p pair 0/10
#   ./build_gpu_test.sh --sanitize      # run under compute-sanitizer memcheck
#   ARCH=sm_120 ./build_gpu_test.sh     # secondary target
#
# Needs nvcc and a CUDA device.  On hosts without nvcc use ./build_and_test.sh,
# which runs the same shared device header against the CPU reference on the
# host.
#
# All device buffers in the test come from one av2enc::cuda::Arena (the
# one-byte-short test uses a separate temporary arena); the module and the test
# call no cudaMalloc/cudaFree/cudaMallocHost.
set -euo pipefail

cd "$(dirname "$0")"

HERE="$(pwd)"
ROOT="$(cd ../../.. && pwd)"
ARCH="${ARCH:-sm_86}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-gme-gpu}"
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
      sed -n '2,17p' "$0"
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

# Mandatory single-arena rule.  Call-shaped so prose in comments does not trip
# it; the test's only allocation calls are Arena::allocate_checked.
if grep -nE '\b(cudaMalloc|cudaFree|cudaMallocHost|cudaHostAlloc|cudaHostRegister)[[:space:]]*\(' \
    gme.h gme_device.cuh gme_plan.cpp gme.cu gme_test.cu; then
  echo "ERROR: direct device allocation in the component sources" >&2
  exit 1
fi

mkdir -p "$BUILD_DIR"

echo "nvcc: $(nvcc --version | tail -1)"
echo "host: $($CXX --version | head -1)"
echo "arch: $ARCH"
echo "build: $BUILD_DIR/gme_test"

# CPU oracle, compiled by the host compiler.
"$CXX" -std=c++20 -O2 -g -ffp-contract=off \
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Wno-sign-conversion \
  -I "$ROOT/src" \
  -c "$ROOT/src/encoder/inter/gme/gme.cpp" \
  -o "$BUILD_DIR/gme_cpu.o"

# Kernels + host plan + test + the persistent arena runtime.  No
# --expt-relaxed-constexpr: the device core uses only enumerators and
# constexpr functions, never a namespace-scope object.
nvcc -std=c++20 -O3 -g -lineinfo -arch="$ARCH" \
  -Xcompiler -Wall,-Wextra,-Wconversion,-Wshadow \
  --ptxas-options=-v \
  -I "$ROOT/src" -I "$HERE" \
  "$HERE/gme.cu" "$HERE/gme_plan.cpp" "$HERE/gme_test.cu" \
  "$ROOT/src/cuda/arena.cpp" "$BUILD_DIR/gme_cpu.o" \
  -o "$BUILD_DIR/gme_test"

if [ "$RUN" -eq 1 ]; then
  if [ "$SANITIZE" -eq 1 ]; then
    if ! command -v compute-sanitizer >/dev/null 2>&1; then
      echo "compute-sanitizer not found" >&2
      exit 2
    fi
    if [ "${#RUN_ARGS[@]}" -gt 0 ]; then
      CUDA_DISABLE_PTX_JIT=1 compute-sanitizer --tool memcheck \
        --error-exitcode 9 "$BUILD_DIR/gme_test" "$ROOT" "${RUN_ARGS[@]}"
    else
      CUDA_DISABLE_PTX_JIT=1 compute-sanitizer --tool memcheck \
        --error-exitcode 9 "$BUILD_DIR/gme_test" "$ROOT"
    fi
  elif [ "${#RUN_ARGS[@]}" -gt 0 ]; then
    CUDA_DISABLE_PTX_JIT=1 "$BUILD_DIR/gme_test" "$ROOT" "${RUN_ARGS[@]}"
  else
    CUDA_DISABLE_PTX_JIT=1 "$BUILD_DIR/gme_test" "$ROOT"
  fi
fi
