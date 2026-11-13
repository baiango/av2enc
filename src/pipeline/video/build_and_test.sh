#!/usr/bin/env bash
# Build and run the M2 P-frame correctness gate.
#
# Host-only CPU reference build (no CUDA). The gate encodes a 2-frame stream
# and a short clip, runs the pinned dav2d, and compares the decoded pictures
# against the encoder reconstruction sample by sample.
#
# Usage:
#   ./src/pipeline/video/build_and_test.sh
#   ./src/pipeline/video/build_and_test.sh --input testdata/shine_256x256_1f.y4m
#   SANITIZE=1 ./src/pipeline/video/build_and_test.sh
#   WERROR=1   ./src/pipeline/video/build_and_test.sh
#   OUT=/tmp/video_gate ./src/pipeline/video/build_and_test.sh
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
CXX="${CXX:-c++}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"
OUT="${OUT:-${TMPDIR:-/tmp}/video_gate}"

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

CORE=(
  "$ROOT/src/core/io/y4m.cpp"
  "$ROOT/src/core/io/ivf.cpp"
  "$ROOT/src/core/bitstream/bitwriter.cpp"
  "$ROOT/src/core/bitstream/obu.cpp"
  "$ROOT/src/core/bitstream/basic_headers.cpp"
  "$ROOT/src/codec/partition/tree/partition.cpp"
  "$ROOT/src/codec/intra/modes/intra_modes.cpp"
  "$ROOT/src/codec/transform/variants/tx_variants.cpp"
  "$ROOT/src/codec/transform/variants/tx_select.cpp"
  "$ROOT/src/codec/entropy/av2/av2_entropy.cpp"
  "$ROOT/src/codec/inter/me/me.cpp"
  "$ROOT/src/codec/inter/mc/mc.cpp"
  "$ROOT/src/codec/inter/merge/merge.cpp"
  "$ROOT/src/pipeline/video/video_headers.cpp"
  "$ROOT/src/pipeline/video/video_syntax.cpp"
  "$ROOT/src/pipeline/video/video_frame.cpp"
)

# The M2 writer keeps growing optional modules (TIP/temporal motion field,
# TX search, filters, RC).  Take the canonical av2enc_core host source list
# instead of chasing undefined references by hand.
source "$ROOT/src/cuda/pipeline/cpu_reference_sources.sh"
CORE_RELATIVE=()
for src in "${CORE[@]}"; do
  CORE_RELATIVE+=("${src#"$ROOT"/}")
done
av2enc_cpu_core_append CORE "$ROOT/" \
  "${CORE_RELATIVE[@]+"${CORE_RELATIVE[@]}"}"

"${CXX}" "${FLAGS[@]}" \
  "$ROOT/src/pipeline/video/video_gate.cpp" \
  "${CORE[@]}" \
  -o "${OUT}"

if [[ -z "${DAV2D:-}" ]]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [[ -x "$candidate" ]]; then
      export DAV2D="$candidate"
      break
    fi
  done
fi

cd "$ROOT"
if [[ $# -gt 0 ]]; then
  "${OUT}" "$@"
  exit $?
fi

# Default gate: a fast synthetic two-frame smoke test, then the real short
# clip from testdata when it is present. Both phases run the full 2-frame +
# short-clip decode/recon comparison.
STATUS=0
"${OUT}" --subpu-selftest || STATUS=1
"${OUT}" --frames 2 --out "${TMPDIR:-/tmp}/video_gate_2f.ivf" || STATUS=1
# §7.9/§7.12.2.7-.8 temporal MVs: two references so the motion field is
# non-empty, 8 frames so the saved-field chain feeds later frames.
"${OUT}" --frames 8 --refs 2 --ref-mvs \
  --out "${TMPDIR:-/tmp}/video_gate_refmvs.ivf" || STATUS=1
"${OUT}" --frames 8 --refs 3 --ref-mvs \
  --out "${TMPDIR:-/tmp}/video_gate_refmvs3.ivf" || STATUS=1
# TIP (§7.10) product case: key + future anchor P + TIP_FRAME_AS_OUTPUT
# pictures at 4 and 8 pictures.  Uses the synthetic clip unless an input is
# supplied; each phase decodes with dav2d --strict 1 and compares the recon.
"${OUT}" --frames 4 --tip --out "${TMPDIR:-/tmp}/video_gate_tip4.ivf" || STATUS=1
"${OUT}" --frames 8 --tip --out "${TMPDIR:-/tmp}/video_gate_tip8.ivf" || STATUS=1
if [[ -f "$ROOT/testdata/shine_1280x720_60f.y4m" ]]; then
  "${OUT}" --input "$ROOT/testdata/shine_1280x720_60f.y4m" --frames 8 \
    --single-block --out "${TMPDIR:-/tmp}/video_gate_720p.ivf" || STATUS=1
fi
exit $STATUS
