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
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
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
  "$ROOT/src/core/bitstream/bitwriter.cpp" \
  "$ROOT/src/core/bitstream/basic_headers.cpp" \
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
  "$ROOT/src/core/io/y4m.cpp" \
  "$ROOT/src/core/io/ivf.cpp" \
  "$ROOT/src/core/bitstream/bitwriter.cpp" \
  "$ROOT/src/core/bitstream/obu.cpp" \
  "$ROOT/src/core/bitstream/basic_headers.cpp" \
  "$ROOT/src/codec/partition/tree/partition.cpp" \
  "$ROOT/src/codec/intra/modes/intra_modes.cpp" \
  "$ROOT/src/codec/transform/variants/tx_variants.cpp" \
  "$ROOT/src/codec/transform/variants/tx_select.cpp" \
  "$ROOT/src/codec/entropy/av2/av2_entropy.cpp" \
  "$ROOT/src/codec/inter/me/me.cpp" \
  "$ROOT/src/codec/inter/mc/mc.cpp" \
  "$ROOT/src/codec/inter/merge/merge.cpp" \
  "$ROOT/src/pipeline/video/video_headers.cpp" \
  "$ROOT/src/pipeline/video/video_syntax.cpp" \
  "$ROOT/src/pipeline/video/video_frame.cpp" \
  "$ROOT/src/codec/inter/temporal/ref_mv_bank.cpp"
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
