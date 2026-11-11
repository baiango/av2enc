#!/usr/bin/env bash
# Build and run the CPU correctness reference for the B-frame hierarchy layer.
#
# This is a plain C++20 build (the product target is CUDA on Linux; the CPU
# binary is a correctness oracle). It compiles the shared decoder-exact
# bookkeeping from src/encoder/inter/advanced/b_frame.cpp and links it with this
# directory's scheduler/simulator. No CMake edits are required.
#
# Usage:
#   ./src/encoder/inter/bframes/build_and_test.sh
#   CXX=g++-13 ./src/encoder/inter/bframes/build_and_test.sh
#   SANITIZE=1 ./src/encoder/inter/bframes/build_and_test.sh
#   WERROR=1 ./src/encoder/inter/bframes/build_and_test.sh
#   OUT=/tmp/bframes_test ./src/encoder/inter/bframes/build_and_test.sh
#   ./src/encoder/inter/bframes/build_and_test.sh --no-run

set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${HERE}/../../../.." && pwd)"
ADV="${ROOT}/src/encoder/inter/advanced"
CXX="${CXX:-g++}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"
OUT="${OUT:-${TMPDIR:-/tmp}/bframes_test}"
BUILD="${BUILD:-${TMPDIR:-/tmp}/bframes_test_build}"

SOURCES=("${ADV}/b_frame.cpp" "${HERE}/bframes.cpp")
FLAGS=(-std=c++20 -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
INCLUDES=(-I"${ROOT}/src" -I"${ADV}")

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
  obj="${BUILD}/$(basename "${src%.cpp}").o"
  "${CXX}" "${FLAGS[@]}" "${INCLUDES[@]}" -c "${src}" -o "${obj}"
  OBJECTS+=("${obj}")
done

# The test binary uses vector indexing and a handful of int/long long casts.
# -Wno-sign-conversion keeps clang's stricter conversion diagnostics from
# turning test bookkeeping into noise; the library itself builds clean.
"${CXX}" "${FLAGS[@]}" -Wno-sign-conversion "${INCLUDES[@]}" \
  -c "${HERE}/bframes_test.cpp" -o "${BUILD}/bframes_test.o"

"${CXX}" "${FLAGS[@]}" "${OBJECTS[@]}" "${BUILD}/bframes_test.o" -o "${OUT}"

if [[ "${RUN}" -eq 1 ]]; then
  "${OUT}"
fi
