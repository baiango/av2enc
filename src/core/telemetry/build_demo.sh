#!/usr/bin/env bash
# Build the standalone profiler demo without CMake.
#
#   src/core/telemetry/build_demo.sh                 # g++ CPU demo + nvcc demo if present
#   WITH_CUDA=0 src/core/telemetry/build_demo.sh     # CPU demo only
#   CXX=clang++ src/core/telemetry/build_demo.sh
#
# The CPU demo always builds.  The CUDA demo is built with nvcc when nvcc is
# available and targets sm_86 (override with CUDA_ARCH=...).  NVTX is enabled
# automatically when the CUDA toolkit's nvtx3 header and libnvToolsExt are
# found; NVTX remains inactive until AV2ENC_PROFILE_NVTX=1 is set at runtime.

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${repo}/CMakeLists.txt" ] && [ "${repo}" != "/" ]; do
    repo="$(dirname -- "${repo}")"
done
out_dir="${OUT_DIR:-${here}/build}"
mkdir -p "${out_dir}"

cxx="${CXX:-g++}"
cuda_home="${CUDA_HOME:-/usr/local/cuda}"
cuda_arch="${CUDA_ARCH:-sm_86}"

common_flags=(
  -std=c++20
  -O2
  -g
  -Wall
  -Wextra
  -Wpedantic
  -Wconversion
  -Wshadow
  -pthread
  -I"${repo}/src"
)

nvtx_available=0
if [[ -f "${cuda_home}/include/nvtx3/nvToolsExt.h" ]] &&
   [[ -e "${cuda_home}/lib64/libnvToolsExt.so" ||
      -e "${cuda_home}/lib/libnvToolsExt.so" ]]; then
  nvtx_available=1
fi

cpu_flags=("${common_flags[@]}")
cpu_ldflags=()
if [[ "${nvtx_available}" == "1" ]]; then
  cpu_flags+=(-DAV2ENC_HAVE_NVTX=1 -I"${cuda_home}/include")
  if [[ -d "${cuda_home}/lib64" ]]; then
    cpu_ldflags+=(-L"${cuda_home}/lib64")
  else
    cpu_ldflags+=(-L"${cuda_home}/lib")
  fi
  cpu_ldflags+=(-lnvToolsExt)
fi

echo "building CPU profiler demo with ${cxx}"
"${cxx}" "${cpu_flags[@]}" \
  "${here}/profiler.cpp" "${here}/profile_demo.cpp" \
  "${cpu_ldflags[@]}" -o "${out_dir}/profile_demo"
echo "  -> ${out_dir}/profile_demo"

if [[ "${WITH_CUDA:-1}" == "0" ]]; then
  echo "WITH_CUDA=0: skipping CUDA demo"
  exit 0
fi

nvcc="${NVCC:-}"
if [[ -z "${nvcc}" ]]; then
  nvcc="$(command -v nvcc || true)"
fi
if [[ -z "${nvcc}" ]]; then
  if [[ -x "${cuda_home}/bin/nvcc" ]]; then
    nvcc="${cuda_home}/bin/nvcc"
  fi
fi
if [[ -z "${nvcc}" ]]; then
  echo "nvcc not found: CUDA demo skipped (CPU demo is usable)"
  exit 0
fi

# -Wpedantic is intentionally omitted from nvcc: its generated host stubs use
# GNU line-directive extensions that trigger it.
nvcc_flags=(
  -std=c++20
  -O2
  -g
  -lineinfo
  "-arch=${cuda_arch}"
  -x cu
  -I"${repo}/src"
  -Xcompiler=-pthread
  -Xcompiler=-Wall
  -Xcompiler=-Wextra
  -Xcompiler=-Wconversion
  -Xcompiler=-Wshadow
)
nvcc_ldflags=(-DAV2ENC_HAVE_CUDA=1)
if [[ "${nvtx_available}" == "1" ]]; then
  nvcc_flags+=(-DAV2ENC_HAVE_NVTX=1)
  if [[ -d "${cuda_home}/lib64" ]]; then
    nvcc_ldflags+=(-L"${cuda_home}/lib64")
  else
    nvcc_ldflags+=(-L"${cuda_home}/lib")
  fi
  nvcc_ldflags+=(-lnvToolsExt)
fi

echo "building CUDA profiler demo with ${nvcc} (${cuda_arch})"
"${nvcc}" "${nvcc_flags[@]}" \
  "${here}/profiler.cpp" "${here}/profile_demo.cpp" \
  "${nvcc_ldflags[@]}" -o "${out_dir}/profile_demo_cuda"
echo "  -> ${out_dir}/profile_demo_cuda"
echo
echo "run: AV2ENC_PROFILE=${out_dir}/demo.jsonl ${out_dir}/profile_demo"
echo "run: AV2ENC_PROFILE=${out_dir}/demo_cuda.jsonl ${out_dir}/profile_demo_cuda"
