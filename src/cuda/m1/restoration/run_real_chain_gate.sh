#!/usr/bin/env bash
# Real deblock -> CDEF -> CCSO -> Wiener CPU-vs-GPU parity gate.
#
# Encodes the pinned M1 QP90 stills twice (no CDEF, with CDEF), decodes both
# with dav2d, and runs m1_restoration_test in --ivf-pair mode.  The no-CDEF
# decode is post-deblock (CurrFrame); the CDEF decode is the CCSO/§7.20 input.
#
# Run on a CUDA host:
#   BIN=build-cuda/av2enc DAV2D=tools/bin/dav2d \
#     src/cuda/m1/restoration/run_real_chain_gate.sh
#
# Requires a CUDA build of av2enc (no CPU fallback for the GPU half).
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$HERE/../../../.." && pwd)"
BIN="${BIN:-$ROOT/build-cuda/av2enc}"
DAV2D="${DAV2D:-$ROOT/tools/bin/dav2d}"
TMP="${TMPDIR:-/tmp}/av2enc_lr_chain"
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

"$BIN" "$ROOT/tests/data/m0_64x64_1f.y4m" -o "$TMP/pre64.ivf" --m1
"$BIN" "$ROOT/tests/data/m0_64x64_1f.y4m" -o "$TMP/post64.ivf" --m1 --m1-cdef
"$BIN" "$ROOT/tests/data/m0_640x360_1f.y4m" -o "$TMP/pre360.ivf" --m1
"$BIN" "$ROOT/tests/data/m0_640x360_1f.y4m" -o "$TMP/post360.ivf" --m1 --m1-cdef

"$DAV2D" -q -i "$TMP/pre64.ivf" -o "$TMP/pre64.y4m" --threads 1
"$DAV2D" -q -i "$TMP/post64.ivf" -o "$TMP/post64.y4m" --threads 1
"$DAV2D" -q -i "$TMP/pre360.ivf" -o "$TMP/pre360.y4m" --threads 1
"$DAV2D" -q -i "$TMP/post360.ivf" -o "$TMP/post360.y4m" --threads 1

cd "$HERE"
RUN_ARGS="--ivf-pair $TMP/pre64.ivf $TMP/post64.ivf \
          --ivf-pair $TMP/pre360.ivf $TMP/post360.ivf --dav2d $DAV2D" \
  ./build_and_test.sh
