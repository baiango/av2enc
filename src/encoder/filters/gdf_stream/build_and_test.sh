#!/usr/bin/env bash
# Build and run the AV2 §7.20.5 GDF stream tests and the dav2d gate.
#
#   ./src/encoder/filters/gdf_stream/build_and_test.sh
#   ./src/encoder/filters/gdf_stream/build_and_test.sh --test-only
#   ./src/encoder/filters/gdf_stream/build_and_test.sh --gate-only
#   ./src/encoder/filters/gdf_stream/build_and_test.sh --no-run
#   ./src/encoder/filters/gdf_stream/build_and_test.sh --asan
#   GDF_GATE_INPUT=clip.y4m   GDF_GATE_QP=90   DAV2D=/path/to/dav2d ...
#
# Nothing outside this directory is modified.  The gate links the existing M1
# CPU encoder (the same source list as CMake's av2enc_core) and the recovered
# spec_tables GDF reference; it writes only into the work directory.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${HERE}/../../../.." && pwd)"
CXX="${CXX:-c++}"
SANITIZE=0
OUT_DIR="${OUT_DIR:-${TMPDIR:-/tmp}/gdf_stream_build}"
RUN_TEST=1
RUN_GATE=1

for arg in "$@"; do
  case "$arg" in
    --no-run) RUN_TEST=0; RUN_GATE=0 ;;
    --test-only) RUN_GATE=0 ;;
    --gate-only) RUN_TEST=0 ;;
    --asan) SANITIZE=1 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

FLAGS=(-std=c++20 -Wall -Wextra -Wpedantic -Wconversion -Wshadow
       -Wno-missing-braces -Wno-sign-conversion -ffp-contract=off -I"${ROOT}/src")
if [[ "${SANITIZE}" == "1" ]]; then
  FLAGS+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
else
  FLAGS+=(-O2)
fi
if [[ "${WERROR:-0}" == "1" ]]; then
  FLAGS+=(-Werror)
fi

MODULE=(
  "${HERE}/gdf_syntax.cpp"
  "${HERE}/gdf_decision.cpp"
)
SPEC=(
  "${ROOT}/src/encoder/filters/spec_tables/gdf.cpp"
  "${ROOT}/src/encoder/filters/spec_tables/luma_source.cpp"
)
ENTROPY=(
  "${ROOT}/src/entropy/av2/av2_entropy.cpp"
  "${ROOT}/src/bitstream/bitwriter.cpp"
)

# Same source set as CMake's av2enc_core (CPU reference M1 encoder).
CORE=(
  "${ROOT}/src/io/y4m.cpp"
  "${ROOT}/src/io/ivf.cpp"
  "${ROOT}/src/bitstream/bitwriter.cpp"
  "${ROOT}/src/bitstream/obu.cpp"
  "${ROOT}/src/bitstream/m0_headers.cpp"
  "${ROOT}/src/encoder/m0_still.cpp"
  "${ROOT}/src/encoder/partition/partition.cpp"
  "${ROOT}/src/encoder/intra_modes/intra_modes.cpp"
  "${ROOT}/src/encoder/rdoq/rdoq.cpp"
  "${ROOT}/src/encoder/rdoq/rdoq_av2_model.cpp"
  "${ROOT}/src/encoder/tx_type/tx_type.cpp"
  "${ROOT}/src/encoder/tx_variants/tx_variants.cpp"
  "${ROOT}/src/encoder/tx_variants/tx_select.cpp"
  "${ROOT}/src/encoder/cfl/cfl.cpp"
  "${ROOT}/src/encoder/filters/deblock/deblock.cpp"
  "${ROOT}/src/encoder/filters/cdef/cdef.cpp"
  "${ROOT}/src/encoder/filters/restoration/restoration.cpp"
  "${ROOT}/src/encoder/rdo/rdo.cpp"
  "${ROOT}/src/encoder/rdo/intra_decision.cpp"
  "${ROOT}/src/encoder/m1/m1_headers.cpp"
  "${ROOT}/src/encoder/m1/m1_syntax.cpp"
  "${ROOT}/src/encoder/m1/m1_still.cpp"
  "${ROOT}/src/encoder/m1/stream/m1_stream_bridge.cpp"
  "${ROOT}/src/entropy/av2/av2_entropy.cpp"
  "${ROOT}/src/profile/profiler.cpp"
  "${ROOT}/src/cuda/m1/cdef/m1_cdef.cpp"
)

mkdir -p "${OUT_DIR}"

if [[ "${RUN_TEST}" == "1" ]]; then
  "${CXX}" "${FLAGS[@]}" \
    "${HERE}/gdf_stream_test.cpp" "${MODULE[@]}" "${SPEC[@]}" "${ENTROPY[@]}" \
    -o "${OUT_DIR}/gdf_stream_test"
  "${OUT_DIR}/gdf_stream_test"
fi

if [[ "${RUN_GATE}" == "1" ]]; then
  "${CXX}" "${FLAGS[@]}" \
    "${HERE}/gdf_stream_gate.cpp" "${MODULE[@]}" "${SPEC[@]}" "${CORE[@]}" \
    -o "${OUT_DIR}/gdf_stream_gate"

  INPUT="${GDF_GATE_INPUT:-${ROOT}/testdata/m0_64x64_1f.y4m}"
  QP="${GDF_GATE_QP:-90}"
  if [[ -z "${DAV2D:-}" ]]; then
    for candidate in "${ROOT}/tools/bin/dav2d" "${ROOT}/tools/bin/dav2d-macos"; do
      if [[ -x "${candidate}" ]]; then
        export DAV2D="${candidate}"
        break
      fi
    done
  fi
  if [[ -z "${DAV2D:-}" ]]; then
    echo "dav2d not found; gate skipped (set DAV2D)" >&2
    exit 77
  fi
  "${OUT_DIR}/gdf_stream_gate" --input "${INPUT}" --qp "${QP}" \
    --work "${OUT_DIR}/work" --dav2d "${DAV2D}"
fi
