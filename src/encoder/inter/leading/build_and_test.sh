#!/usr/bin/env bash
# Build and run the leading-picture unit tests and the pinned dav2d gate.
#
# The library is CPU-only framing code; picture payloads come from the existing
# M2 intra/inter encoder.  No CMake edits are required.
#
# Usage:
#   ./src/encoder/inter/leading/build_and_test.sh
#   SANITIZE=1 ./src/encoder/inter/leading/build_and_test.sh
#   WERROR=1   ./src/encoder/inter/leading/build_and_test.sh
#   ./src/encoder/inter/leading/build_and_test.sh --no-decode
#   ./src/encoder/inter/leading/build_and_test.sh --no-run
#   DAV2D=/path/to/dav2d ./src/encoder/inter/leading/build_and_test.sh
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$HERE/../../../.." && pwd)"
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
  "$ROOT/src/encoder/inter/leading/leading.cpp"
  "$ROOT/src/encoder/inter/leading/leading_reader.cpp"
)
UNIT_DEPS=(
  "${LIB[@]}"
  "$ROOT/src/encoder/m2/m2_headers.cpp"
  "$ROOT/src/bitstream/m0_headers.cpp"
  "$ROOT/src/bitstream/bitwriter.cpp"
)
printf 'compiler: %s\n' "$("${CXX}" --version | head -1)"
"${CXX}" "${FLAGS[@]}" \
  "$ROOT/src/encoder/inter/leading/leading_test.cpp" \
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
  "${LIB[@]}"
  "$ROOT/src/encoder/inter/leading/leading_encode.cpp"
)
# The M2 sources carry pre-existing -Wsign-conversion diagnostics; -Werror is
# only applied to the unit test above.
GATE_FLAGS=("${FLAGS[@]}" -Wno-sign-conversion)
"${CXX}" "${GATE_FLAGS[@]}" \
  "$ROOT/src/encoder/inter/leading/leading_gate.cpp" \
  "${CORE[@]}" -o "${GATE}"

mkdir -p "${OUT_DIR}"
GATE_ARGS=(--out-dir "${OUT_DIR}")
if [[ "${DECODE}" -eq 0 ]]; then
  GATE_ARGS+=(--no-decode)
fi
cd "$ROOT"
"${GATE}" "${GATE_ARGS[@]}"
