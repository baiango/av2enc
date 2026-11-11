#!/usr/bin/env bash
#
# Build and run the standalone SVC layer pipeline tests.
#
#   ./build_and_test.sh
#   CXX=g++-13 ./build_and_test.sh
#   SANITIZE=1 ./build_and_test.sh
#   WERROR=1 ./build_and_test.sh
#
# No CMake, no CUDA, no encoder core. The script links only the pure bitwriter
# (src/bitstream/bitwriter.cpp) and the SVC module. It does not write a stream;
# gate_dav2d.sh does the black-box decoder check with an existing encoder build.
set -euo pipefail
cd "$(dirname "$0")"

CXX="${CXX:-c++}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"

FLAGS=(-std=c++20 -Wall -Wextra -Wpedantic -Wconversion -Wshadow -ffp-contract=off)
if [[ "${SANITIZE}" == "1" ]]; then
  FLAGS+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
else
  FLAGS+=(-O2)
fi
if [[ "${WERROR}" == "1" ]]; then
  FLAGS+=(-Werror)
fi

build_dir="$(mktemp -d "${TMPDIR:-/tmp}/av2enc-svc.XXXXXX")"
trap 'rm -rf "${build_dir}"' EXIT
binary="${build_dir}/svc_test"

echo "compiler: ${CXX}"
"${CXX}" --version 2>/dev/null | head -n 1 || true
echo "flags:    ${FLAGS[*]}"
echo "building: svc_config.cpp svc_pipeline.cpp svc_framing.cpp svc_test.cpp"
echo

"${CXX}" "${FLAGS[@]}" -I. -I.. -I../.. \
  svc_config.cpp svc_pipeline.cpp svc_framing.cpp svc_test.cpp \
  ../tiles/svc_layers.cpp ../../bitstream/bitwriter.cpp \
  -o "${binary}"

echo "building: svc_mux (compile-only)"
"${CXX}" "${FLAGS[@]}" -I. -I.. -I../.. \
  svc_mux.cpp svc_config.cpp svc_pipeline.cpp svc_framing.cpp \
  ../tiles/svc_layers.cpp ../../bitstream/bitwriter.cpp ../../io/ivf.cpp \
  -o "${build_dir}/svc_mux"

echo "running: ${binary}"
echo
"${binary}"
