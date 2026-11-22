#!/usr/bin/env bash
# Build and run the temporal-layer planner tests and decoder gate.
#
# The unit test is standalone (C++20, no CUDA). The decoder gate links the
# host M2 encoder as a single-reference carrier and then re-wraps its tile
# payloads with tlayers' sequence/frame headers, so it needs the M2 source
# list (still no CMake changes).
#
# Usage:
#   ./src/codec/inter/tlayers/build_and_test.sh
#   ./src/codec/inter/tlayers/build_and_test.sh --no-gate
#   SANITIZE=1 ./src/codec/inter/tlayers/build_and_test.sh --no-gate
#   WERROR=1   ./src/codec/inter/tlayers/build_and_test.sh --no-gate
#   OUT=/tmp/tlayers ./src/codec/inter/tlayers/build_and_test.sh
#   DAV2D=/path/to/dav2d ./src/codec/inter/tlayers/build_and_test.sh
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
CXX="${CXX:-c++}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"
OUT="${OUT:-${TMPDIR:-/tmp}/tlayers}"
BUILD="${BUILD:-${TMPDIR:-/tmp}/tlayers_build}"

FLAGS=(-std=c++20 -Wall -Wextra -Wpedantic -Wconversion -Wshadow
       -Wno-missing-braces -I"${ROOT}/src")
if [[ "${SANITIZE}" == "1" ]]; then
  FLAGS+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
else
  FLAGS+=(-O2)
fi
if [[ "${WERROR}" == "1" ]]; then
  FLAGS+=(-Werror)
fi

RUN_GATE=1
for arg in "$@"; do
  case "$arg" in
    --no-gate) RUN_GATE=0 ;;
    -h|--help)
      sed -n '2,20p' "$0"
      exit 0
      ;;
    *)
      echo "unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

mkdir -p "${BUILD}" "${OUT}"

TEST_BIN="${BUILD}/tlayers_test"

# The unit test needs the whole video-header dependency closure, which is what
# CMake already models as av2enc_core.  This script used to carry its own hand
# written source list; it went stale the moment video_headers.cpp grew (CCSO,
# Wiener, GDF, film grain, QM, TIP, delta-Q writers) and the test stopped
# linking on main.  Prefer the CMake-built archive so there is exactly one list
# of dependencies; fall back to the historical explicit list only when no build
# tree is present, and say so.
# The suite links libav2enc_core.a so the dependency list lives in CMake and
# cannot drift from the tree.  But linking whatever archive happens to exist
# tests whatever was last built, not the sources in front of you: a stale
# archive made this script report "undefined symbol: minimum_slot_requirement"
# for a function that was plainly in the tree, and it would just as happily
# PASS against code you had already changed.
#
# So rebuild the library first.  CORE_LIB may preset the archive path, in which
# case its build directory is rebuilt; otherwise the newest archive is chosen and
# rebuilt.  If no build tree exists the explicit source lists below are used, so
# the script still works from a bare checkout.
CORE_LIB="${CORE_LIB:-}"
if [[ -z "${CORE_LIB}" ]]; then
  CORE_LIB="$(ls -t "${ROOT}"/build*/libav2enc_core.a 2>/dev/null | head -1 || true)"
fi
if [[ -n "${CORE_LIB}" ]]; then
  CORE_BUILD_DIR="$(dirname "${CORE_LIB}")"
  if [[ ! -f "${CORE_BUILD_DIR}/build.ninja" && ! -f "${CORE_BUILD_DIR}/Makefile" ]]; then
    echo "tlayers: ${CORE_BUILD_DIR#"${ROOT}/"} has an archive but no build files;" >&2
    echo "  re-run cmake in that directory, or set CORE_LIB to a real build tree" >&2
    exit 2
  fi
  echo "tlayers: rebuilding av2enc_core in ${CORE_BUILD_DIR#"${ROOT}/"}"
  if ! cmake --build "${CORE_BUILD_DIR}" --target av2enc_core >/dev/null; then
    echo "tlayers: av2enc_core rebuild failed; refusing to test a stale archive" >&2
    exit 2
  fi
  if [[ ! -f "${CORE_LIB}" ]]; then
    echo "tlayers: rebuild produced no ${CORE_LIB#"${ROOT}/"}" >&2
    exit 2
  fi
fi

if [[ -n "${CORE_LIB}" ]]; then
  # Only the test source is compiled here: tlayers.cpp / tlayers_syntax.cpp come
  # from the archive.  Compiling them again would shadow the archive's copies
  # (one-definition-rule violations, and a symbol defined twice).
  echo "tlayers_test: linking against ${CORE_LIB#"${ROOT}/"}"
  "${CXX}" "${FLAGS[@]}" "${HERE}/tlayers_test.cpp" "${CORE_LIB}" \
    -o "${TEST_BIN}"
else
  echo "tlayers_test: no libav2enc_core.a found (configure with cmake);" >&2
  echo "  falling back to the explicit source list, which may be stale" >&2
  "${CXX}" "${FLAGS[@]}" \
    "${HERE}/tlayers_test.cpp" \
    "${HERE}/tlayers.cpp" \
    "${HERE}/tlayers_syntax.cpp" \
    "${ROOT}/src/pipeline/video/video_headers.cpp" \
    "${ROOT}/src/core/bitstream/bitwriter.cpp" \
    "${ROOT}/src/core/bitstream/basic_headers.cpp" \
    -o "${TEST_BIN}"
fi
"${TEST_BIN}"

if [[ "${RUN_GATE}" == "0" ]]; then
  exit 0
fi

GATE_BIN="${BUILD}/tlayers_gate"
# Same reasoning as the unit test above: the gate's source list is the whole M2
# video core, which is exactly what CMake models as av2enc_core.  Linking the
# archive keeps one list of dependencies; the explicit VIDEO_CORE list is kept
# only as a fallback for a checkout with no CMake build tree, and it is known to
# drift (it missed the CSSO/Wiener/GDF/QM/TIP writers that video_headers.cpp now
# needs).  The M2 core is not -Werror clean, so the fallback build keeps the
# warnings visible.
if [[ -n "${CORE_LIB}" ]]; then
  echo "tlayers_gate: linking against ${CORE_LIB#"${ROOT}/"}"
  "${CXX}" "${FLAGS[@]}" "${HERE}/tlayers_gate.cpp" "${CORE_LIB}" \
    -o "${GATE_BIN}"
else
  VIDEO_CORE=(
    "${ROOT}/src/core/io/y4m.cpp"
    "${ROOT}/src/core/io/ivf.cpp"
    "${ROOT}/src/core/bitstream/bitwriter.cpp"
    "${ROOT}/src/core/bitstream/obu.cpp"
    "${ROOT}/src/core/bitstream/basic_headers.cpp"
    "${ROOT}/src/codec/partition/tree/partition.cpp"
    "${ROOT}/src/codec/intra/modes/intra_modes.cpp"
    "${ROOT}/src/codec/transform/variants/tx_variants.cpp"
    "${ROOT}/src/codec/transform/variants/tx_select.cpp"
    "${ROOT}/src/codec/entropy/av2/av2_entropy.cpp"
    "${ROOT}/src/codec/inter/me/me.cpp"
    "${ROOT}/src/codec/inter/mc/mc.cpp"
    "${ROOT}/src/codec/inter/merge/merge.cpp"
    "${ROOT}/src/pipeline/video/video_headers.cpp"
    "${ROOT}/src/pipeline/video/video_syntax.cpp"
    "${ROOT}/src/pipeline/video/video_syntax_block.cpp"
    "${ROOT}/src/pipeline/video/video_syntax_inter.cpp"
    "${ROOT}/src/pipeline/video/video_syntax_coeff.cpp"
    "$ROOT/src/codec/transform/delta_q/syntax.cpp"
    "${ROOT}/src/pipeline/video/video_frame.cpp"
    "${ROOT}/src/codec/inter/temporal/ref_mv_bank.cpp"
  )
  "${CXX}" "${FLAGS[@]}" \
    "${HERE}/tlayers_gate.cpp" \
    "${HERE}/tlayers.cpp" \
    "${HERE}/tlayers_syntax.cpp" \
    "${VIDEO_CORE[@]}" \
    -o "${GATE_BIN}"
fi

if [[ -z "${DAV2D:-}" ]]; then
  for candidate in "${ROOT}/tools/bin/dav2d" "${ROOT}/tools/bin/dav2d-macos"; do
    if [[ -x "${candidate}" ]]; then
      export DAV2D="${candidate}"
      break
    fi
  done
fi
if [[ -z "${DAV2D:-}" ]]; then
  echo "tlayers gate skipped: pinned dav2d not found (set DAV2D)" >&2
  exit 0
fi

cd "${ROOT}"
"${GATE_BIN}" --width 64 --height 64 --frames 8 --out "${OUT}/64x64"
"${GATE_BIN}" --width 640 --height 360 --frames 8 --out "${OUT}/640x360"
