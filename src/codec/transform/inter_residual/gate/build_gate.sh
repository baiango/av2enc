#!/usr/bin/env bash
# Build and run the M2 inter-residual decoder gate.
#
# Two binaries are produced from the same driver:
#   * gate_core     links the stock src/pipeline/video/video_syntax.cpp (DCT-only inter
#                   residuals);
#   * gate_patched  links a build-dir copy of that file rewritten by
#                   patch_video_syntax.py so the inter TX type comes from this
#                   component.
# The DCT-only runs must produce byte-identical IVF files; the "all" run is the
# feature gate (non-DCT inter TX types) and is decoded with the pinned dav2d.
#
# Usage: ./gate/build_gate.sh [--out DIR] [--input clip.y4m] [--frames N]
#                             [--depth N] [--quick] [driver args...]
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
CXX="${CXX:-c++}"
OUT="${TMPDIR:-/tmp}/video_residual_gate"
DRIVER_ARGS=()
QUICK=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --quick) QUICK=1; shift ;;
    *) DRIVER_ARGS+=("$1"); shift ;;
  esac
done
mkdir -p "$OUT"

FLAGS=(-std=c++20 -O2 -Wall -Wextra -Wpedantic -Wconversion -Wshadow
       -Wno-missing-braces -I"$ROOT/src")

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
  "$ROOT/src/codec/transform/tx_type/tx_type.cpp"
  "$ROOT/src/codec/rate/rdo/rdo.cpp"
  "$ROOT/src/codec/rate/rdoq/rdoq.cpp"
  "$ROOT/src/codec/rate/rdoq/rdoq_av2_model.cpp"
  "$ROOT/src/codec/entropy/av2/av2_entropy.cpp"
  "$ROOT/src/codec/inter/me/me.cpp"
  "$ROOT/src/codec/inter/mc/mc.cpp"
  "$ROOT/src/codec/inter/merge/merge.cpp"
  "$ROOT/src/pipeline/video/video_headers.cpp"
  "$ROOT/src/pipeline/video/video_frame.cpp"
  "$ROOT/src/codec/transform/inter_residual/inter_tx_type.cpp"
  "$ROOT/src/codec/transform/inter_residual/inter_coeff_syntax.cpp"
  "$ROOT/src/codec/transform/inter_residual/inter_residual.cpp"
)

PATCHED="$OUT/video_syntax_patched.cpp"
python3 "$HERE/patch_video_syntax.py" "$ROOT/src/pipeline/video/video_syntax.cpp" "$PATCHED"
diff -u --label a/src/pipeline/video/video_syntax.cpp \
  --label b/src/pipeline/video/video_syntax.cpp \
  "$ROOT/src/pipeline/video/video_syntax.cpp" "$PATCHED" \
  >"$OUT/video_syntax_inter.patch" || true

echo "== build gate_core =="
"${CXX}" "${FLAGS[@]}" -DGATE_USE_PATCHED=0 \
  "$HERE/video_residual_gate.cpp" "$ROOT/src/pipeline/video/video_syntax.cpp" \
  "${CORE[@]}" -o "$OUT/gate_core"
echo "== build gate_patched =="
"${CXX}" "${FLAGS[@]}" -DGATE_USE_PATCHED=1 \
  "$HERE/video_residual_gate.cpp" "$PATCHED" "${CORE[@]}" -o "$OUT/gate_patched"

if [[ -z "${DAV2D:-}" ]]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [[ -x "$candidate" ]]; then
      export DAV2D="$candidate"
      break
    fi
  done
fi
cd "$ROOT"

if [[ "$QUICK" == "1" ]]; then
  DRIVER_ARGS+=(--frames 3 --depth 0)
fi

STATUS=0
echo "== DCT-only byte parity: core vs patched writer =="
"$OUT/gate_patched" --tx-types dct --no-decode --out "$OUT/patched_dct.ivf" \
  "${DRIVER_ARGS[@]+"${DRIVER_ARGS[@]}"}" || STATUS=1
"$OUT/gate_core" --tx-types dct --no-decode --out "$OUT/core_dct.ivf" \
  "${DRIVER_ARGS[@]+"${DRIVER_ARGS[@]}"}" || STATUS=1
if cmp -s "$OUT/patched_dct.ivf" "$OUT/core_dct.ivf"; then
  echo "  PASS: DCT-only streams are byte-identical"
else
  echo "  FAIL: DCT-only streams differ"
  STATUS=1
fi

echo "== feature gate: non-DCT inter TX types, dav2d recon check =="
"$OUT/gate_patched" --tx-types all --out "$OUT/patched_all.ivf" \
  "${DRIVER_ARGS[@]+"${DRIVER_ARGS[@]}"}" || STATUS=1

exit $STATUS
