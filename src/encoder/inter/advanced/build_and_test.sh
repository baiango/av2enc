#!/usr/bin/env bash
# Build and run the CPU correctness reference for the advanced inter tools.
#
# This is intentionally a plain C++20 build: the product target is CUDA on Linux
# and the CPU binary only has to prove correctness. The CUDA kernels in
# src/cuda/inter/advanced/ include the same shared headers, so the CPU-vs-GPU
# test there is a launch/thread-mapping check, not a second spec transcription.
#
# Usage:
#   ./src/encoder/inter/advanced/build_and_test.sh
#   CXX=g++-13 ./src/encoder/inter/advanced/build_and_test.sh
#   SANITIZE=1 ./src/encoder/inter/advanced/build_and_test.sh
#   WERROR=1 ./src/encoder/inter/advanced/build_and_test.sh
#   OUT=/tmp/advanced_test ./src/encoder/inter/advanced/build_and_test.sh
#
# Extra modes:
#   --check-tables   verify tables/*.h matches the spec PDF (needs pymupdf)
#   --no-run         build only

set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CXX="${CXX:-g++}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"
OUT="${OUT:-${TMPDIR:-/tmp}/advanced_test}"
BUILD="${BUILD:-${TMPDIR:-/tmp}/advanced_test_build}"

SOURCES=(compound.cpp warp.cpp bawp.cpp b_frame.cpp obmc.cpp)

FLAGS=(-std=c++20 -Wall -Wextra -Wpedantic -Wconversion -Wshadow)

if [[ "${SANITIZE}" == "1" ]]; then
  FLAGS+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
else
  FLAGS+=(-O2)
fi

if [[ "${WERROR}" == "1" ]]; then
  FLAGS+=(-Werror)
fi

RUN=1
for arg in "$@"; do
  case "$arg" in
    --no-run)
      RUN=0
      ;;
    --check-tables)
      python3 "${HERE}/tables/extract_spec_tables.py" --check
      exit 0
      ;;
    -h|--help)
      sed -n '2,22p' "$0"
      exit 0
      ;;
    *)
      echo "unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

mkdir -p "${BUILD}"

echo "compiler: $("${CXX}" --version | head -1)"
OBJECTS=()
for src in "${SOURCES[@]}"; do
  obj="${BUILD}/${src%.cpp}.o"
  "${CXX}" "${FLAGS[@]}" -I"${HERE}" -c "${HERE}/${src}" -o "${obj}"
  OBJECTS+=("${obj}")
done

# The test binary indexes buffers with int in a handful of places; the library
# itself builds clean under -Wconversion. -Wno-sign-conversion keeps clang's
# -Wconversion extra strictness from turning test-index arithmetic into noise.
"${CXX}" "${FLAGS[@]}" -Wno-sign-conversion -I"${HERE}" \
  -c "${HERE}/advanced_test.cpp" -o "${BUILD}/advanced_test.o"

"${CXX}" "${FLAGS[@]}" "${OBJECTS[@]}" "${BUILD}/advanced_test.o" -o "${OUT}"

if [[ "${RUN}" -eq 1 ]]; then
  "${OUT}"
fi
