#!/usr/bin/env bash
# Real deblock -> CDEF -> CCSO -> Wiener CPU-vs-GPU parity gate.
#
# Encodes the pinned M1 QP90 stills twice (no CDEF, with CDEF), decodes both
# with dav2d, and runs still_restoration_test in --ivf-pair mode.  The no-CDEF
# decode is post-deblock (CurrFrame); the CDEF decode is the CCSO/§7.20 input.
#
# Run on a CUDA host:
#   BIN=build-cuda/av2enc DAV2D=tools/bin/dav2d \
#     src/cuda/pipeline/still/restoration/run_real_chain_gate.sh
#   QP=1 .../run_real_chain_gate.sh   # near-lossless fixture
#
# Requires a CUDA build of av2enc (no CPU fallback for the GPU half).
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
BIN="${BIN:-$ROOT/build-cuda/av2enc}"
DAV2D="${DAV2D:-$ROOT/tools/bin/dav2d}"
QP="${QP:-90}"
TMP="${TMPDIR:-/tmp}/av2enc_lr_chain_qp${QP}"
mkdir -p "$TMP"

if [[ ! -x "$BIN" ]]; then
  echo "missing encoder: $BIN (build the CUDA target first)" >&2
  exit 2
fi
if [[ ! -x "$DAV2D" ]]; then
  echo "missing decoder: $DAV2D" >&2
  exit 2
fi
# The parity test changes cwd; make both tools absolute.
BIN="$(cd "$(dirname "$BIN")" && pwd)/$(basename "$BIN")"
DAV2D="$(cd "$(dirname "$DAV2D")" && pwd)/$(basename "$DAV2D")"

"$BIN" "$ROOT/tests/data/basic_64x64_1f.y4m" -o "$TMP/pre64.ivf" --still \
  --qp "$QP"
"$BIN" "$ROOT/tests/data/basic_64x64_1f.y4m" -o "$TMP/post64.ivf" --still \
  --still-cdef --qp "$QP"
"$BIN" "$ROOT/tests/data/basic_640x360_1f.y4m" -o "$TMP/pre360.ivf" --still \
  --qp "$QP"
"$BIN" "$ROOT/tests/data/basic_640x360_1f.y4m" -o "$TMP/post360.ivf" --still \
  --still-cdef --qp "$QP"

"$DAV2D" -q -i "$TMP/pre64.ivf" -o "$TMP/pre64.y4m" --threads 1
"$DAV2D" -q -i "$TMP/post64.ivf" -o "$TMP/post64.y4m" --threads 1
"$DAV2D" -q -i "$TMP/pre360.ivf" -o "$TMP/pre360.y4m" --threads 1
"$DAV2D" -q -i "$TMP/post360.ivf" -o "$TMP/post360.y4m" --threads 1

cd "$HERE"
RUN_ARGS="--ivf-pair $TMP/pre64.ivf $TMP/post64.ivf \
          --ivf-pair $TMP/pre360.ivf $TMP/post360.ivf --dav2d $DAV2D" \
  ./build_and_test.sh
