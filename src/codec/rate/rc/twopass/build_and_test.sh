#!/usr/bin/env bash
#
# Build and run the two-pass / VBV / lookahead rate-control tests.
#
#   ./build_and_test.sh
#   CXX=g++-16 ./build_and_test.sh
#   TWOPASS_SANITIZE=1 ./build_and_test.sh   # ASan + UBSan
#   TWOPASS_WERROR=1   ./build_and_test.sh   # warnings are errors
#   TWOPASS_REAL=0     ./build_and_test.sh   # skip the 720p M2 encode test
#
# The synthetic suite has no dependencies. The real-encode suite links the M2
# CPU reference frame loop (no edits to it) and reads
# testdata/shine_1280x720_60f.y4m when present.
set -euo pipefail
cd "$(dirname "$0")"

HERE="$(pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
CXX="${CXX:-c++}"
RC_DIR="$ROOT/src/codec/rate/rc"

CXXFLAGS=(
  -std=c++20
  -O2
  -Wall
  -Wextra
  -Wpedantic
  -Wconversion
  -Wshadow
  -ffp-contract=off
  -I"$ROOT/src"
)

if [[ "${TWOPASS_SANITIZE:-0}" == "1" ]]; then
  CXXFLAGS+=(-fsanitize=address,undefined -fno-omit-frame-pointer)
fi
if [[ "${TWOPASS_WERROR:-0}" == "1" ]]; then
  CXXFLAGS+=(-Werror)
fi

build_dir="$(mktemp -d "${TMPDIR:-/tmp}/av2enc-twopass.XXXXXX")"
trap 'rm -rf "${build_dir}"' EXIT

echo "compiler: ${CXX}"
"${CXX}" --version 2>/dev/null | head -n 1 || true
echo "flags:    ${CXXFLAGS[*]}"
echo

synthetic="${build_dir}/twopass_test"
echo "building: synthetic tests"
"${CXX}" "${CXXFLAGS[@]}" \
  "$HERE/twopass.cpp" \
  "$RC_DIR/rc.cpp" \
  "$HERE/twopass_test.cpp" \
  -o "${synthetic}"
"${synthetic}"

if [[ "${TWOPASS_REAL:-1}" != "1" ]]; then
  echo "real-encode tests: skipped (TWOPASS_REAL=0)"
  exit 0
fi

INPUT="$ROOT/testdata/shine_1280x720_60f.y4m"
if [[ ! -f "${INPUT}" ]]; then
  echo "real-encode tests: skipped (missing ${INPUT})"
  exit 0
fi

if [[ -z "${DAV2D:-}" ]]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [[ -x "$candidate" ]]; then
      export DAV2D="$candidate"
      break
    fi
  done
fi

REAL_FLAGS=("${CXXFLAGS[@]}" -Wno-sign-conversion -Wno-missing-braces)
real="${build_dir}/twopass_real_test"
echo
echo "building: real 720p M2 encode tests"
"${CXX}" "${REAL_FLAGS[@]}" \
  "$HERE/twopass.cpp" \
  "$RC_DIR/rc.cpp" \
  "$HERE/twopass_real_test.cpp" \
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
  -o "${real}"

echo
"${real}" "${INPUT}"
