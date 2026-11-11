#!/usr/bin/env bash
#
# Build and run the multi-GPU runtime tests.
#
#   ./build_and_test.sh                 # host mock tests; real gate if nvcc + GPU
#   SANITIZE=1 ./build_and_test.sh      # ASan + UBSan host tests
#   WERROR=1 ./build_and_test.sh
#   REQUIRE_GPU=1 ./build_and_test.sh   # fail if the real CUDA gate cannot run
#   CUDA_ARCH=sm_120 ./build_and_test.sh
#
# The host build needs only a C++20 compiler: no CUDA, no devices, no CMake.
# The CUDA gate compiles only when nvcc is present and exits 77 when no device
# is visible. A skip is reported with an explicit hardware disclaimer.
set -euo pipefail
cd "$(dirname "$0")"

CXX="${CXX:-c++}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"
REQUIRE_GPU="${REQUIRE_GPU:-0}"
CUDA_ARCH="${CUDA_ARCH:-sm_86}"
NVCC="${NVCC:-nvcc}"

FLAGS=(-std=c++20 -Wall -Wextra -Wpedantic -Wconversion -Wshadow -ffp-contract=off)
if [[ "${SANITIZE}" == "1" ]]; then
  FLAGS+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
else
  FLAGS+=(-O2)
fi
if [[ "${WERROR}" == "1" ]]; then
  FLAGS+=(-Werror)
fi

build_dir="$(mktemp -d "${TMPDIR:-/tmp}/av2enc-multigpu-runtime.XXXXXX")"
trap 'rm -rf "${build_dir}"' EXIT

host_binary="${build_dir}/runtime_test"
gpu_binary="${build_dir}/gpu_gate"

echo "compiler: ${CXX}"
"${CXX}" --version 2>/dev/null | head -n 1 || true
echo "flags:    ${FLAGS[*]}"
echo "building: runtime.cpp dispatch.cpp reference.cpp tests/runtime_test.cpp"
echo

"${CXX}" "${FLAGS[@]}" -I. -I.. -I../.. -I../encoder/tiles \
  runtime.cpp dispatch.cpp reference.cpp tests/runtime_test.cpp \
  ../planner.cpp ../../encoder/tiles/tile_layout.cpp \
  -o "${host_binary}"

echo "running: mock-backend runtime tests"
echo
"${host_binary}"

echo
if ! command -v "${NVCC}" >/dev/null 2>&1; then
  echo "SKIP: ${NVCC} not found; real CUDA gate not built."
  echo "DISCLAIMER: the mock dispatch gate above is the only runtime evidence."
  echo "            No multi-GPU hardware ran. No encoder, bitstream, or scaling"
  echo "            claim is made. Run this script on the Linux + CUDA host to"
  echo "            execute the single/2-GPU plumbing gate."
  if [[ "${REQUIRE_GPU}" == "1" ]]; then
    echo "REQUIRE_GPU=1: failing because the real gate could not run."
    exit 1
  fi
  exit 0
fi

# Build the CUDA backend and real gate. The host runtime .cpp files are compiled
# by nvcc as host code; no device kernels are present.
echo "building: cuda_backend.cpp gpu_gate.cpp for ${CUDA_ARCH}"
cuda_flags=(-std=c++20 -O2 -arch="${CUDA_ARCH}" -lineinfo
  -Xcompiler=-Wall,-Wextra,-Wconversion,-Wshadow)
if [[ "${WERROR}" == "1" ]]; then
  cuda_flags+=(-Xcompiler=-Werror)
fi
"${NVCC}" "${cuda_flags[@]}" -I. \
  cuda_backend.cpp runtime.cpp dispatch.cpp reference.cpp gpu_gate.cpp \
  ../../cuda/arena.cpp ../planner.cpp ../../encoder/tiles/tile_layout.cpp \
  -o "${gpu_binary}"

echo
echo "running: real GPU gate"
set +e
"${gpu_binary}" "$@"
gate_status=$?
set -e
if [[ ${gate_status} -eq 77 ]]; then
  echo
  echo "SKIP: no usable CUDA device for the real gate."
  echo "DISCLAIMER: the mock dispatch gate above is the only runtime evidence;"
  echo "            no multi-GPU hardware or scaling claim is made."
  if [[ "${REQUIRE_GPU}" == "1" ]]; then
    echo "REQUIRE_GPU=1: failing because the real gate could not run."
    exit 1
  fi
  exit 0
fi
exit "${gate_status}"
