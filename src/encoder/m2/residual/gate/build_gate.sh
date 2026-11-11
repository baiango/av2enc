#!/usr/bin/env bash
# Build and run the M2 inter-residual decoder gate.
#
# Two binaries are produced from the same driver:
#   * gate_core     links the stock src/encoder/m2/m2_syntax.cpp (DCT-only inter
#                   residuals);
#   * gate_patched  links a build-dir copy of that file rewritten by
#                   patch_m2_syntax.py so the inter TX type comes from this
#                   component.
# The DCT-only runs must produce byte-identical IVF files; the "all" run is the
# feature gate (non-DCT inter TX types) and is decoded with the pinned dav2d.
#
# Usage: ./gate/build_gate.sh [--out DIR] [--input clip.y4m] [--frames N]
#                             [--depth N] [--quick] [driver args...]
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$HERE/../../../../../" && pwd)"
CXX="${CXX:-c++}"
OUT="${TMPDIR:-/tmp}/m2_residual_gate"
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
  "$ROOT/src/io/y4m.cpp"
  "$ROOT/src/io/ivf.cpp"
  "$ROOT/src/bitstream/bitwriter.cpp"
  "$ROOT/src/bitstream/obu.cpp"
  "$ROOT/src/bitstream/m0_headers.cpp"
  "$ROOT/src/encoder/partition/partition.cpp"
  "$ROOT/src/encoder/intra_modes/intra_modes.cpp"
  "$ROOT/src/encoder/tx_variants/tx_variants.cpp"
  "$ROOT/src/encoder/tx_variants/tx_select.cpp"
  "$ROOT/src/encoder/tx_type/tx_type.cpp"
  "$ROOT/src/encoder/rdo/rdo.cpp"
  "$ROOT/src/encoder/rdoq/rdoq.cpp"
  "$ROOT/src/encoder/rdoq/rdoq_av2_model.cpp"
  "$ROOT/src/entropy/av2/av2_entropy.cpp"
  "$ROOT/src/encoder/inter/me/me.cpp"
  "$ROOT/src/encoder/inter/mc/mc.cpp"
  "$ROOT/src/encoder/m2/merge/merge.cpp"
  "$ROOT/src/encoder/m2/m2_headers.cpp"
  "$ROOT/src/encoder/m2/m2_frame.cpp"
  "$ROOT/src/encoder/m2/residual/inter_tx_type.cpp"
  "$ROOT/src/encoder/m2/residual/inter_coeff_syntax.cpp"
  "$ROOT/src/encoder/m2/residual/inter_residual.cpp"
)

PATCHED="$OUT/m2_syntax_patched.cpp"
python3 "$HERE/patch_m2_syntax.py" "$ROOT/src/encoder/m2/m2_syntax.cpp" "$PATCHED"
diff -u --label a/src/encoder/m2/m2_syntax.cpp \
  --label b/src/encoder/m2/m2_syntax.cpp \
  "$ROOT/src/encoder/m2/m2_syntax.cpp" "$PATCHED" \
  >"$OUT/m2_syntax_inter.patch" || true

echo "== build gate_core =="
"${CXX}" "${FLAGS[@]}" -DGATE_USE_PATCHED=0 \
  "$HERE/m2_residual_gate.cpp" "$ROOT/src/encoder/m2/m2_syntax.cpp" \
  "${CORE[@]}" -o "$OUT/gate_core"
echo "== build gate_patched =="
"${CXX}" "${FLAGS[@]}" -DGATE_USE_PATCHED=1 \
  "$HERE/m2_residual_gate.cpp" "$PATCHED" "${CORE[@]}" -o "$OUT/gate_patched"

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
