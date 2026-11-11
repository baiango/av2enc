#!/usr/bin/env bash
#
# Build and run the HRD/timing component tests and the M2 decoder gate.
#
#   ./build_and_test.sh
#   CXX=g++-16 ./build_and_test.sh
#   HRD_SANITIZE=1 ./build_and_test.sh
#   HRD_WERROR=1   ./build_and_test.sh
#   HRD_GATE=0     ./build_and_test.sh   # unit tests only
#
# No CMake and no external dependencies. The decoder gate links the unmodified
# M2 CPU reference frame loop and uses the pinned dav2d if it is present.
set -euo pipefail
cd "$(dirname "$0")"

HERE="$(pwd)"
ROOT="$(cd -- "$HERE/../../../.." && pwd)"
CXX="${CXX:-c++}"

FLAGS=(-std=c++20 -O2 -Wall -Wextra -Wpedantic -Wconversion -Wshadow
       -ffp-contract=off -I"$ROOT/src")
if [[ "${HRD_SANITIZE:-0}" == "1" ]]; then
  FLAGS+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
fi
if [[ "${HRD_WERROR:-0}" == "1" ]]; then
  FLAGS+=(-Werror)
fi

build_dir="$(mktemp -d "${TMPDIR:-/tmp}/av2enc-hrd.XXXXXX")"
trap 'rm -rf "${build_dir}"' EXIT

echo "compiler: ${CXX}"
"${CXX}" --version 2>/dev/null | head -n 1 || true
echo "flags:    ${FLAGS[*]}"
echo

echo "building: unit tests"
"${CXX}" "${FLAGS[@]}" \
  "$HERE/hrd.cpp" \
  "$HERE/hrd_buffer.cpp" \
  "$ROOT/src/bitstream/bitwriter.cpp" \
  "$ROOT/src/bitstream/m0_headers.cpp" \
  "$HERE/hrd_test.cpp" \
  -o "${build_dir}/hrd_test"

echo "running: unit tests"
"${build_dir}/hrd_test"
echo

if [[ "${HRD_GATE:-1}" != "1" ]]; then
  echo "decoder gate: skipped (HRD_GATE=0)"
  exit 0
fi

echo "building: M2 decoder gate"
"${CXX}" "${FLAGS[@]}" -Wno-sign-conversion -Wno-missing-braces \
  "$HERE/hrd.cpp" \
  "$HERE/hrd_buffer.cpp" \
  "$HERE/hrd_stream_gate.cpp" \
  "$ROOT/src/io/y4m.cpp" \
  "$ROOT/src/io/ivf.cpp" \
  "$ROOT/src/bitstream/bitwriter.cpp" \
  "$ROOT/src/bitstream/obu.cpp" \
  "$ROOT/src/bitstream/m0_headers.cpp" \
  "$ROOT/src/encoder/partition/partition.cpp" \
  "$ROOT/src/encoder/intra_modes/intra_modes.cpp" \
  "$ROOT/src/encoder/tx_variants/tx_variants.cpp" \
  "$ROOT/src/encoder/tx_variants/tx_select.cpp" \
  "$ROOT/src/entropy/av2/av2_entropy.cpp" \
  "$ROOT/src/encoder/inter/me/me.cpp" \
  "$ROOT/src/encoder/inter/mc/mc.cpp" \
  "$ROOT/src/encoder/m2/merge/merge.cpp" \
  "$ROOT/src/encoder/m2/m2_headers.cpp" \
  "$ROOT/src/encoder/m2/m2_syntax.cpp" \
  "$ROOT/src/encoder/m2/m2_frame.cpp" \
  -o "${build_dir}/hrd_stream_gate"

if [[ -z "${DAV2D:-}" ]]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [[ -x "$candidate" ]]; then
      export DAV2D="$candidate"
      break
    fi
  done
fi

echo
cd "$ROOT"
"${build_dir}/hrd_stream_gate" "$@"
