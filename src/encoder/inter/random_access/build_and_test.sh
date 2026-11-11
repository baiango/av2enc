#!/usr/bin/env bash
# Build and run the random-access switch / RAS component tests and the pinned
# dav2d decoder gate.
#
# Host-only CPU reference build (no CUDA). The gate encodes three frames, wraps
# frame 1 in OBU_SWITCH and frame 2 in OBU_RAS_FRAME, then decodes with
#
#   dav2d --strict 1 --threads 1
#
# and compares the decoder output with the encoder reconstruction.
#
# Usage:
#   ./src/encoder/inter/random_access/build_and_test.sh
#   ./src/encoder/inter/random_access/build_and_test.sh --input clip.y4m
#   SANITIZE=1 ./src/encoder/inter/random_access/build_and_test.sh
#   WERROR=1   ./src/encoder/inter/random_access/build_and_test.sh
#   OUT=/tmp/ra_gate ./src/encoder/inter/random_access/build_and_test.sh
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$HERE/../../../.." && pwd)"
CXX="${CXX:-c++}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"
OUT="${OUT:-${TMPDIR:-/tmp}/random_access_gate}"

mkdir -p "$OUT"

FLAGS=(-std=c++20 -Wall -Wextra -Wpedantic -Wconversion -Wshadow
       -Wno-missing-braces -I"$ROOT/src")
STRICT=()
if [[ "${SANITIZE}" == "1" ]]; then
  FLAGS+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
else
  FLAGS+=(-O2)
fi
if [[ "${WERROR}" == "1" ]]; then
  STRICT+=(-Werror)
fi

CORE=(
  "$ROOT/src/io/y4m.cpp"
  "$ROOT/src/io/ivf.cpp"
  "$ROOT/src/bitstream/bitwriter.cpp"
  "$ROOT/src/bitstream/obu.cpp"
  "$ROOT/src/bitstream/m0_headers.cpp"
  "$ROOT/src/encoder/partition/partition.cpp"
  "$ROOT/src/encoder/intra_modes/intra_modes.cpp"
  "$ROOT/src/encoder/tx_variants/tx_variants.cpp"
  "$ROOT/src/encoder/tx_variants/tx_select.cpp"
  "$ROOT/src/entropy/av2/av2_entropy.cpp"
  "$ROOT/src/encoder/inter/me/me.cpp"
  "$ROOT/src/encoder/inter/mc/mc.cpp"
  "$ROOT/src/encoder/m2/merge/merge.cpp"
  "$ROOT/src/encoder/m2/m2_headers.cpp"
  "$ROOT/src/encoder/m2/m2_syntax.cpp"
  "$ROOT/src/encoder/m2/m2_frame.cpp"
)

# Unit tests: only our component plus the small bitstream/M2 header helpers.
# These files are warning-clean, so WERROR applies to the whole test build.
"${CXX}" "${FLAGS[@]}" ${STRICT[@]+"${STRICT[@]}"} \
  "$HERE/random_access_test.cpp" \
  "$HERE/random_access.cpp" \
  "$ROOT/src/bitstream/bitwriter.cpp" \
  "$ROOT/src/bitstream/m0_headers.cpp" \
  "$ROOT/src/encoder/m2/m2_headers.cpp" \
  -o "$OUT/random_access_test"

# Gate: compile our sources with WERROR, then link the M2 core (which has
# pre-existing sign-conversion warnings) without -Werror.
"${CXX}" "${FLAGS[@]}" ${STRICT[@]+"${STRICT[@]}"} -c \
  "$HERE/random_access.cpp" -o "$OUT/random_access.o"
"${CXX}" "${FLAGS[@]}" ${STRICT[@]+"${STRICT[@]}"} -c \
  "$HERE/ras_gate.cpp" -o "$OUT/ras_gate.o"
"${CXX}" "${FLAGS[@]}" \
  "$OUT/ras_gate.o" "$OUT/random_access.o" "${CORE[@]}" \
  -o "$OUT/ras_gate"

if [[ -z "${DAV2D:-}" ]]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [[ -x "$candidate" ]]; then
      export DAV2D="$candidate"
      break
    fi
  done
fi

cd "$ROOT"

echo "== unit tests =="
"$OUT/random_access_test"

if [[ $# -gt 0 ]]; then
  echo "== gate (arguments: $*) =="
  "$OUT/ras_gate" "$@"
  exit $?
fi

STATUS=0
echo "== gate: synthetic 128x128, 3 frames =="
"$OUT/ras_gate" --out "$OUT/ra_gate.ivf" || STATUS=1

if [[ -f "$ROOT/testdata/shine_1280x720_60f.y4m" ]]; then
  echo "== gate: real 720p, 3 frames =="
  "$OUT/ras_gate" --input "$ROOT/testdata/shine_1280x720_60f.y4m" \
    --out "$OUT/ra_gate_720p.ivf" || STATUS=1
fi

if [[ "$STATUS" == "0" ]]; then
  echo "RANDOM ACCESS GATE PASS"
else
  echo "RANDOM ACCESS GATE FAIL"
fi
exit "$STATUS"
