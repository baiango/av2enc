#!/usr/bin/env bash
# Build and run the film grain stream tests and the M2 decoder gate.
#
#   ./src/encoder/filters/film_grain_stream/build_and_test.sh
#   ./src/encoder/filters/film_grain_stream/build_and_test.sh --test-only
#   ./src/encoder/filters/film_grain_stream/build_and_test.sh --gate-only
#   ./src/encoder/filters/film_grain_stream/build_and_test.sh --no-run
#   SANITIZE=1 ...   WERROR=1 ...   DAV2D=/path/to/dav2d ...
#
# Nothing outside this directory is modified: the unit test links the existing
# bit writer and the spec_tables synthesis reference, and the gate links the
# existing M2 encoder to build its own grain-bearing streams.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${HERE}/../../../.." && pwd)"
CXX="${CXX:-c++}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"
OUT_DIR="${OUT_DIR:-${TMPDIR:-/tmp}/film_grain_stream_build}"
RUN_TEST=1
RUN_GATE=1

for arg in "$@"; do
  case "$arg" in
    --no-run) RUN_TEST=0; RUN_GATE=0 ;;
    --test-only) RUN_GATE=0 ;;
    --gate-only) RUN_TEST=0 ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

FLAGS=(-std=c++20 -Wall -Wextra -Wpedantic -Wconversion -Wshadow
       -Wno-missing-braces -I"$ROOT/src")
if [[ "${SANITIZE}" == "1" ]]; then
  FLAGS+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
else
  FLAGS+=(-O2)
fi
if [[ "${WERROR}" == "1" ]]; then
  FLAGS+=(-Werror)
fi

MODULE=(
  "$HERE/film_grain_syntax.cpp"
  "$HERE/grain_stream.cpp"
)

SPEC=(
  "$ROOT/src/encoder/filters/spec_tables/film_grain.cpp"
)

CORE=(
  "$ROOT/src/bitstream/bitwriter.cpp"
  "$ROOT/src/bitstream/obu.cpp"
  "$ROOT/src/bitstream/m0_headers.cpp"
)

mkdir -p "${OUT_DIR}"

if [[ "${RUN_TEST}" == "1" ]]; then
  "${CXX}" "${FLAGS[@]}" \
    "$HERE/film_grain_stream_test.cpp" \
    "${MODULE[@]}" "${SPEC[@]}" "${CORE[@]}" \
    -o "${OUT_DIR}/film_grain_stream_test"
  "${OUT_DIR}/film_grain_stream_test"
fi

if [[ "${RUN_GATE}" == "1" ]]; then
  # The M2 core has pre-existing -Wsign-conversion warnings, so the gate build
  # drops -Werror (this module itself is warning-clean under WERROR=1 for the
  # unit-test build).
  GATE_FLAGS=()
  for flag in "${FLAGS[@]}"; do
    if [[ "$flag" != "-Werror" ]]; then GATE_FLAGS+=("$flag"); fi
  done
  M2_CORE=(
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
  "${CXX}" "${GATE_FLAGS[@]}" \
    "$HERE/film_grain_stream_gate.cpp" \
    "${MODULE[@]}" "${SPEC[@]}" "${M2_CORE[@]}" \
    -o "${OUT_DIR}/film_grain_stream_gate"

  if [[ -z "${DAV2D:-}" ]]; then
    for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
      if [[ -x "$candidate" ]]; then
        export DAV2D="$candidate"
        break
      fi
    done
  fi
  "${OUT_DIR}/film_grain_stream_gate" --work "${OUT_DIR}/work"
fi
