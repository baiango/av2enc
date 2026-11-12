#!/usr/bin/env bash
# Build and run the M2 inter-DDT decoder gate.
#
# Construction (no core file is edited):
#   * the M2 inter-residual gate driver is compiled unchanged;
#   * video_syntax.cpp is copied and patched by gate/patch_video_syntax.py so
#     non-DCT inter TX types are signalled;
#   * inter_residual.cpp is copied and patched (one call site) so the encoder
#     reconstruction uses ddt::inverse_block();
#   * video_headers.cpp is copied and patched (one bit) so DDT_GATE_ENABLE
#     controls §5.4.8 enable_inter_ddt.
#
# The gate then, for both DDTX (forced inter TX type 1, DCT_ADST) and FDDT
# (forced type 4, DCT_FLIPADST):
#   1. encodes the same clip with flag 0 and flag 1;
#   2. asserts exactly one bit differs, inside the sequence-header OBU, and
#      splices it into the flag-0 stream;
#   3. asserts the spliced stream is byte-identical to the flag-1 stream;
#   4. decodes the flag-1 stream with the pinned dav2d and requires the
#      decoder reconstruction to match the DDT encoder reconstruction
#      sample-for-sample;
#   5. negative control: the unspliced flag-0 stream must NOT match (the
#      decoder then uses ADST/FDST while the encoder used DDT).
#
# Usage: ./gate/build_gate.sh [--quick] [driver args...]
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
CXX="${CXX:-c++}"
OUT="${OUT:-${TMPDIR:-/tmp}/video_ddt_gate}"
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
  "$ROOT/src/pipeline/video/video_frame.cpp"
  "$ROOT/src/codec/transform/inter_residual/inter_tx_type.cpp"
  "$ROOT/src/codec/transform/inter_residual/inter_coeff_syntax.cpp"
  "$ROOT/src/codec/transform/ddt/ddt_transform.cpp"
  "$ROOT/src/codec/transform/ddt/gate/ddt_gate_hook.cpp"
)

SYNTAX="$OUT/video_syntax_patched.cpp"
HEADERS="$OUT/video_headers_patched.cpp"
INTER="$OUT/inter_residual_patched.cpp"
python3 "$HERE/patch_video_syntax.py" \
  "$ROOT/src/pipeline/video/video_syntax.cpp" "$SYNTAX"
python3 "$HERE/patch_video_headers.py" \
  "$ROOT/src/pipeline/video/video_headers.cpp" "$HEADERS"
python3 "$HERE/patch_inter_residual.py" \
  "$ROOT/src/codec/transform/inter_residual/inter_residual.cpp" "$INTER"

echo "== build ddt_gate =="
"${CXX}" "${FLAGS[@]}" -DGATE_USE_PATCHED=1 \
  "$ROOT/src/codec/transform/inter_residual/gate/video_residual_gate.cpp" \
  "$SYNTAX" "$HEADERS" "$INTER" "${CORE[@]}" -o "$OUT/ddt_gate"
"${CXX}" "${FLAGS[@]}" "$HERE/ddt_splice.cpp" \
  "$ROOT/src/codec/transform/ddt/ddt_syntax.cpp" \
  "$ROOT/src/core/bitstream/bitwriter.cpp" -o "$OUT/ddt_splice"

if [[ -z "${DAV2D:-}" ]]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [[ -x "$candidate" ]]; then
      export DAV2D="$candidate"
      break
    fi
  done
fi
cd "$ROOT"

CLIP="$OUT/shine256_motion3.y4m"
python3 "$HERE/make_motion_clip.py" \
  "$ROOT/testdata/shine_256x256_1f.y4m" "$CLIP"

# The gate clip has three frames and needs 8x8/16x16 leaves for DDT; the
# driver defaults are 4 frames / depth 0.
DRIVER_ARGS+=(--frames 3 --depth 3)

STATUS=0
run_pass() {
  local label="$1"   # DDTX / FDDT
  local tx_type="$2" # 1 = DCT_ADST (DDTX), 4 = DCT_FLIPADST (FDDT)
  local expect="$3"  # ddtx / fddt
  local base="$OUT/${label}_base.ivf"
  local ddt="$OUT/${label}_ddt.ivf"
  local spliced="$OUT/${label}_spliced.ivf"

  echo "== $label: encode with enable_inter_ddt = 0 (recon still uses DDT) =="
  DDT_GATE_ENABLE=0 AV2ENC_VIDEO_RESIDUAL_TX_TYPE="$tx_type" \
    "$OUT/ddt_gate" --input "$CLIP" --tx-types all --no-decode \
    --out "$base" "${DRIVER_ARGS[@]+"${DRIVER_ARGS[@]}"}" \
    2> "$OUT/${label}_base.log" || STATUS=1

  echo "== $label: encode + decode with enable_inter_ddt = 1 =="
  DDT_GATE_ENABLE=1 AV2ENC_VIDEO_RESIDUAL_TX_TYPE="$tx_type" \
    "$OUT/ddt_gate" --input "$CLIP" --tx-types all \
    --out "$ddt" "${DRIVER_ARGS[@]+"${DRIVER_ARGS[@]}"}" \
    2> "$OUT/${label}_ddt.log" || STATUS=1
  grep -E "DDT_GATE|non_dct_leaves|PASS|MISMATCH" "$OUT/${label}_ddt.log" || true

  echo "== $label: splice the sequence-header bit =="
  "$OUT/ddt_splice" "$base" "$ddt" "$spliced" || STATUS=1
  if cmp -s "$spliced" "$ddt"; then
    echo "  PASS: spliced stream is byte-identical to the flag-1 stream"
  else
    echo "  FAIL: spliced stream differs"
    STATUS=1
  fi

  echo "== $label: DDT actually ran =="
  local blocks dirs ddtx fddt
  blocks="$(sed -n 's/.*DDT_GATE blocks=\([0-9]*\).*/\1/p' "$OUT/${label}_ddt.log" | tail -1)"
  dirs="$(sed -n 's/.*directions=\([0-9]*\).*/\1/p' "$OUT/${label}_ddt.log" | tail -1)"
  ddtx="$(sed -n 's/.*ddtx=\([0-9]*\).*/\1/p' "$OUT/${label}_ddt.log" | tail -1)"
  fddt="$(sed -n 's/.*fddt=\([0-9]*\).*/\1/p' "$OUT/${label}_ddt.log" | tail -1)"
  echo "  DDT blocks=${blocks:-0} directions=${dirs:-0} ddtx=${ddtx:-0} fddt=${fddt:-0}"
  if [[ -z "${blocks:-}" || "$blocks" == "0" || -z "${dirs:-}" || "$dirs" == "0" ]]; then
    echo "  FAIL: gate did not exercise any DDT block"
    STATUS=1
  fi
  if [[ "$expect" == "ddtx" && ( -z "${ddtx:-}" || "$ddtx" == "0" ) ]]; then
    echo "  FAIL: no DDTX direction was exercised"
    STATUS=1
  fi
  if [[ "$expect" == "fddt" && ( -z "${fddt:-}" || "$fddt" == "0" ) ]]; then
    echo "  FAIL: no FDDT direction was exercised"
    STATUS=1
  fi

  echo "== $label: negative control (flag-0 stream must mismatch) =="
  if DDT_GATE_ENABLE=0 AV2ENC_VIDEO_RESIDUAL_TX_TYPE="$tx_type" \
       "$OUT/ddt_gate" --input "$CLIP" --tx-types all \
       --out "$OUT/${label}_base_dec.ivf" \
       "${DRIVER_ARGS[@]+"${DRIVER_ARGS[@]}"}" \
       > "$OUT/${label}_base_dec.out" 2> "$OUT/${label}_base_dec.log"; then
    echo "  FAIL: flag-0 stream unexpectedly matched the DDT reconstruction"
    STATUS=1
  else
    if grep -q "RECON MISMATCH" "$OUT/${label}_base_dec.log"; then
      echo "  PASS: flag-0 decode disagrees with the DDT reconstruction"
    else
      echo "  FAIL: flag-0 run failed for the wrong reason"
      cat "$OUT/${label}_base_dec.log"
      STATUS=1
    fi
  fi
}

# Type 1/4 put the DDT component in the column (WIDE_32 leaves); types 2/5
# put it in the row (HIGH_32 leaves).  Both orientations are decoded.
run_pass DDTX 1 ddtx
run_pass DDTX_ROW 2 ddtx
run_pass FDDT 4 fddt
run_pass FDDT_ROW 5 fddt

if [[ "$STATUS" == "0" ]]; then
  echo "DDT GATE PASS"
else
  echo "DDT GATE FAIL"
fi
exit "$STATUS"
