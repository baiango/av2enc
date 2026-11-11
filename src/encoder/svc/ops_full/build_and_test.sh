#!/usr/bin/env bash
#
# Build and run the field-complete LCR/OPS unit tests.
#
#   ./build_and_test.sh
#   CXX=g++-13 ./build_and_test.sh
#   SANITIZE=1 ./build_and_test.sh
#   WERROR=1   ./build_and_test.sh
#
# No CMake, no CUDA, no encoder core, no decoder. Links the bit writer, the
# SVC plan/framing module (for the byte-for-byte cross-check) and the layer
# validators. `gate_avmdec.sh` adds the black-box decoder check.
set -euo pipefail
cd "$(dirname "$0")"

CXX="${CXX:-c++}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"

FLAGS=(-std=c++20 -Wall -Wextra -Wpedantic -Wconversion -Wshadow
       -ffp-contract=off -I. -I.. -I../.. -I../../..)
if [[ "${SANITIZE}" == "1" ]]; then
  FLAGS+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
else
  FLAGS+=(-O2)
fi
if [[ "${WERROR}" == "1" ]]; then
  FLAGS+=(-Werror)
fi

SVC=(../svc_config.cpp ../svc_pipeline.cpp ../svc_framing.cpp
     ../../tiles/svc_layers.cpp ../../../bitstream/bitwriter.cpp)

build_dir="$(mktemp -d "${TMPDIR:-/tmp}/av2enc-opsfull.XXXXXX")"
trap 'rm -rf "${build_dir}"' EXIT

echo "compiler: ${CXX}"
"${CXX}" --version 2>/dev/null | head -n 1 || true
echo "flags:    ${FLAGS[*]}"
echo

echo "building: lcr_ops_full_test"
"${CXX}" "${FLAGS[@]}" \
  lcr_ops_full_test.cpp lcr_ops_full.cpp "${SVC[@]}" \
  -o "${build_dir}/lcr_ops_full_test"

echo "building: lcr_ops_mux (compile-only)"
"${CXX}" "${FLAGS[@]}" \
  lcr_ops_mux.cpp lcr_ops_full.cpp "${SVC[@]}" ../../../io/ivf.cpp \
  -o "${build_dir}/lcr_ops_mux"

echo "running: lcr_ops_full_test"
echo
"${build_dir}/lcr_ops_full_test"
