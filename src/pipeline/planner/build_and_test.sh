#!/usr/bin/env bash
#
# Build and run the standalone pipeline planner tests.
#
#   ./build_and_test.sh
#   CXX=g++-13 ./build_and_test.sh
#   SANITIZE=1 ./build_and_test.sh
#   WERROR=1 ./build_and_test.sh
#
# No CUDA, no devices, no CMake, no bitstream. The host-only GPU dispatch /
# arena policy (src/cuda/pipeline/planner/) and the three planning owners are compiled
# directly:
#   src/codec/rate/rc/rc.cpp
#   src/pipeline/tiles/tile_layout.cpp
#   src/pipeline/dispatch/planner.cpp
#   src/cuda/pipeline/planner/pipeline_gpu.cpp
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done

CXX="${CXX:-c++}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"

FLAGS=(-std=c++20 -Wall -Wextra -Wpedantic -Wconversion -Wshadow
       -ffp-contract=off -I"$ROOT/src")
if [[ "${SANITIZE}" == "1" ]]; then
  FLAGS+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
else
  FLAGS+=(-O2)
fi
if [[ "${WERROR}" == "1" ]]; then
  FLAGS+=(-Werror)
fi

build_dir="$(mktemp -d "${TMPDIR:-/tmp}/av2enc-pipeline.XXXXXX")"
trap 'rm -rf "${build_dir}"' EXIT
binary="${build_dir}/pipeline_test"

echo "compiler: ${CXX}"
"${CXX}" --version 2>/dev/null | head -n 1 || true
echo "flags:    ${FLAGS[*]}"
echo "building: pipeline.cpp + rc + tiles + multi_gpu + cuda/pipeline"
echo

"${CXX}" "${FLAGS[@]}" \
  "$HERE/pipeline.cpp" \
  "$HERE/pipeline_test.cpp" \
  "$ROOT/src/codec/rate/rc/rc.cpp" \
  "$ROOT/src/pipeline/tiles/tile_layout.cpp" \
  "$ROOT/src/pipeline/dispatch/planner.cpp" \
  "$ROOT/src/cuda/pipeline/planner/pipeline_gpu.cpp" \
  -o "${binary}"

echo "running: ${binary}"
echo
"${binary}"
