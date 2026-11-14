#!/usr/bin/env bash
# Build and run the leading-picture unit tests and the pinned dav2d gate.
#
# The library is CPU-only framing code; picture payloads come from the existing
# M2 intra/inter encoder.  No CMake edits are required.
#
# Usage:
#   ./src/codec/inter/leading/build_and_test.sh
#   SANITIZE=1 ./src/codec/inter/leading/build_and_test.sh
#   WERROR=1   ./src/codec/inter/leading/build_and_test.sh
#   ./src/codec/inter/leading/build_and_test.sh --no-decode
#   ./src/codec/inter/leading/build_and_test.sh --no-run
#   DAV2D=/path/to/dav2d ./src/codec/inter/leading/build_and_test.sh
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
CXX="${CXX:-c++}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"
OUT="${OUT:-${TMPDIR:-/tmp}/leading_test}"
OUT_DIR="${OUT_DIR:-${TMPDIR:-/tmp}/leading_gate_out}"
RUN=1
DECODE=1

for arg in "$@"; do
  case "$arg" in
    --no-run) RUN=0 ;;
    --no-decode) DECODE=0 ;;
    -h|--help)
      sed -n '2,18p' "$0"
      exit 0
      ;;
    *)
      echo "unknown argument: $arg" >&2
      exit 2
      ;;
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

LIB=(
  "$ROOT/src/codec/inter/leading/leading.cpp"
  "$ROOT/src/codec/inter/leading/leading_reader.cpp"
)
UNIT_DEPS=(
  "${LIB[@]}"
  "$ROOT/src/pipeline/video/video_headers.cpp"
  "$ROOT/src/core/bitstream/basic_headers.cpp"
  "$ROOT/src/core/bitstream/bitwriter.cpp"
  "$ROOT/src/pipeline/grain/film_grain_syntax.cpp"
)
printf 'compiler: %s\n' "$("${CXX}" --version | head -1)"
"${CXX}" "${FLAGS[@]}" \
  "$ROOT/src/codec/inter/leading/leading_test.cpp" \
  "${UNIT_DEPS[@]}" -o "${OUT}"

if [[ "${RUN}" -eq 1 ]]; then
  "${OUT}"
fi

if [[ -z "${DAV2D:-}" ]]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [[ -x "$candidate" ]]; then
      export DAV2D="$candidate"
      break
    fi
  done
fi

GATE="${OUT}.gate"
CORE=(
  "$ROOT/src/core/io/y4m.cpp"
  "$ROOT/src/core/io/ivf.cpp"
  "$ROOT/src/core/bitstream/bitwriter.cpp"
  "$ROOT/src/core/bitstream/obu.cpp"
  "$ROOT/src/core/bitstream/basic_headers.cpp"
  "$ROOT/src/pipeline/grain/film_grain_syntax.cpp"
  "$ROOT/src/codec/partition/tree/partition.cpp"
  "$ROOT/src/codec/intra/modes/intra_modes.cpp"
  "$ROOT/src/codec/transform/variants/tx_variants.cpp"
  "$ROOT/src/codec/transform/variants/tx_select.cpp"
  "$ROOT/src/codec/entropy/av2/av2_entropy.cpp"
  "$ROOT/src/codec/inter/me/me.cpp"
  "$ROOT/src/codec/inter/mc/mc.cpp"
  "$ROOT/src/codec/inter/merge/merge.cpp"
  "$ROOT/src/codec/inter/drl_reorder/drl_reorder.cpp"
  "$ROOT/src/codec/inter/drl_reorder/video_stack_adapter.cpp"
  "$ROOT/src/codec/inter/mv_precision/mv_precision.cpp"
  "$ROOT/src/pipeline/video/video_headers.cpp"
  "$ROOT/src/pipeline/video/video_syntax.cpp"
  "$ROOT/src/pipeline/video/video_frame.cpp"
  "$ROOT/src/codec/inter/temporal/ref_mv_bank.cpp"
  "${LIB[@]}"
  "$ROOT/src/codec/inter/leading/leading_encode.cpp"
)
# The M2 sources carry pre-existing -Wsign-conversion diagnostics; -Werror is
# only applied to the unit test above.
GATE_FLAGS=("${FLAGS[@]}" -Wno-sign-conversion)
"${CXX}" "${GATE_FLAGS[@]}" \
  "$ROOT/src/codec/inter/leading/leading_gate.cpp" \
  "${CORE[@]}" -o "${GATE}"

mkdir -p "${OUT_DIR}"
GATE_ARGS=(--out-dir "${OUT_DIR}")
if [[ "${DECODE}" -eq 0 ]]; then
  GATE_ARGS+=(--no-decode)
fi
cd "$ROOT"
"${GATE}" "${GATE_ARGS[@]}"
