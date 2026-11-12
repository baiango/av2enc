#!/usr/bin/env bash
#
# Black-box gate for the field-complete LCR (§5.8) and OPS (§5.10/§5.11)
# writers, using `avmdec` (which parses LCR and OPS) as the validator.
#
# What is built and checked:
#
#   * layer sources are real non-single-picture intra streams from `video_gate`
#     (single-picture headers have no seq_lcr_id field and cannot activate an
#     LCR, which is why `svc/gate_dav2d.sh` skips its LCR cases);
#   * `svc_mux --no-lcr --no-ops` composes them (OBU layer ids + leb128);
#   * `lcr_ops_mux` splices the LCR/OPS bytes produced by this module:
#       - global LCR: aggregate info, per-layer PTL, rep info, purpose,
#         colour, atlas ids, embedded-layer type/view/priority/rendering;
#       - local LCR per extended layer (must agree with the global one);
#       - global OPS: intent, PTL, colour, decoder model, initial display
#         delay, and ops_mlayer_info_idc = 2 inheritance;
#       - local OPS per extended layer (--select-local-ops);
#   * `avmdec --all-layers` and `avmdec --select-ops=0,<op>` must reproduce
#     exactly the concatenation of the independently decoded layers, which is
#     also the per-operating-point check (op i selects xlayers 0..i);
#   * a deliberately inconsistent local LCR must be rejected by the local
#     validator (negative control: the gate is not vacuous);
#   * `dav2d --alllayers 1` decodes the same streams (it ignores LCR/OPS, so
#     this proves the added OBUs do not disturb the picture).
#
# Environment:
#   AV2ENC   encoder binary       (default: build/av2enc, build-m1/av2enc)
#   AVMDEC   avmdec binary        (default: tools/bin/avmdec,
#                                  tools/bin/avm-macos/avmdec)
#   DAV2D    dav2d binary         (default: tools/bin/dav2d,
#                                  tools/bin/dav2d-macos); optional
#   KEEP=1   keep the temporary streams
#
# Exit codes: 0 pass, 1 failure, 2 usage, 77 skip (missing tool/input).
set -euo pipefail
cd "$(dirname "$0")"

ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
CXX="${CXX:-c++}"
TMP=$(mktemp -d "${TMPDIR:-/tmp}/av2enc-opsfull-gate.XXXXXX")
cleanup() {
  if [[ "${KEEP:-0}" != "1" ]]; then
    rm -rf "$TMP"
  else
    echo "kept: $TMP"
  fi
}
trap cleanup EXIT

find_tool() {
  local env_name=$1
  shift
  if [[ -n "${!env_name:-}" && -x "${!env_name}" ]]; then
    echo "${!env_name}"
    return 0
  fi
  local candidate
  for candidate in "$@"; do
    if [[ -x "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

AV2ENC=$(find_tool AV2ENC "$ROOT/build/av2enc" "$ROOT/build-m1/av2enc" || true)
AVMDEC=$(find_tool AVMDEC "$ROOT/tools/bin/avmdec" \
         "$ROOT/tools/bin/avm-macos/avmdec" || true)
DAV2D=$(find_tool DAV2D "$ROOT/tools/bin/dav2d" \
        "$ROOT/tools/bin/dav2d-macos" || true)

if [[ -z "$AVMDEC" ]]; then
  echo "SKIP: no avmdec binary (set AVMDEC=/path/to/avmdec)" >&2
  exit 77
fi

Y4M64="$ROOT/tests/data/basic_64x64_1f.y4m"
Y4M360="$ROOT/tests/data/basic_640x360_1f.y4m"
[[ -f "$Y4M64" ]] || { echo "SKIP: missing $Y4M64" >&2; exit 77; }
[[ -f "$Y4M360" ]] || { echo "SKIP: missing $Y4M360" >&2; exit 77; }

echo "avmdec:  $AVMDEC"
if [[ -n "$DAV2D" ]]; then
  echo "dav2d:   $DAV2D ($("$DAV2D" --version 2>&1 | tail -1))"
else
  echo "dav2d:   not found (cross-check skipped)"
fi
if [[ -n "$AV2ENC" ]]; then
  echo "encoder: $AV2ENC (unused; layer sources come from video_gate)"
fi

echo
echo "building video_gate (non-single-picture layer sources)"
VIDEO_CORE=(
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
if ! "$CXX" -std=c++20 -O2 -Wall -Wextra -Wpedantic -Wconversion -Wshadow \
    -Wno-missing-braces -ffp-contract=off -I"$ROOT/src" \
    "$ROOT/src/pipeline/video/video_gate.cpp" "${VIDEO_CORE[@]}" -o "$TMP/video_gate" \
    >"$TMP/build-m2.log" 2>&1; then
  cat "$TMP/build-m2.log" >&2
  fail "video_gate did not build"
fi

echo "building svc_mux (composition only)"
SVC_CORE=(
  "$ROOT/src/pipeline/svc/svc_mux.cpp"
  "$ROOT/src/pipeline/svc/svc_config.cpp"
  "$ROOT/src/pipeline/svc/svc_pipeline.cpp"
  "$ROOT/src/pipeline/svc/svc_framing.cpp"
  "$ROOT/src/pipeline/tiles/svc_layers.cpp"
  "$ROOT/src/core/bitstream/bitwriter.cpp"
  "$ROOT/src/core/io/ivf.cpp"
)
if ! "$CXX" -std=c++20 -O2 -Wall -Wextra -Wpedantic -Wconversion -Wshadow \
    -ffp-contract=off -I"$ROOT/src" -I"$ROOT/src/encoder" "${SVC_CORE[@]}" \
    -o "$TMP/svc_mux" >"$TMP/build-svc.log" 2>&1; then
  cat "$TMP/build-svc.log" >&2
  fail "svc_mux did not build"
fi

echo "building lcr_ops_mux (LCR/OPS splice)"
if ! "$CXX" -std=c++20 -O2 -Wall -Wextra -Wpedantic -Wconversion -Wshadow \
    -ffp-contract=off -I. -I.. -I../.. -I../../.. \
    lcr_ops_mux.cpp lcr_ops_full.cpp \
    "$ROOT/src/pipeline/svc/svc_config.cpp" \
    "$ROOT/src/pipeline/svc/svc_pipeline.cpp" \
    "$ROOT/src/pipeline/svc/svc_framing.cpp" \
    "$ROOT/src/pipeline/tiles/svc_layers.cpp" \
    "$ROOT/src/core/bitstream/bitwriter.cpp" \
    "$ROOT/src/core/io/ivf.cpp" -o "$TMP/lcr_ops_mux" \
    >"$TMP/build-opsfull.log" 2>&1; then
  cat "$TMP/build-opsfull.log" >&2
  fail "lcr_ops_mux did not build"
fi

echo
echo "encoding layer sources (intra, non-single-picture headers)"
encode_layer() {
  local input=$1
  local qp=$2
  local out=$3
  local log=$4
  DAV2D="${DAV2D:-}" "$TMP/video_gate" --input "$input" --frames 2 --force-intra \
    --qp "$qp" --out "$out" >"$log" 2>&1 || true
  # video_gate exits non-zero when it cannot run its own dav2d cross-check; the
  # stream is what this gate needs.
  [[ -s "$out" ]] || { cat "$log" >&2; fail "video_gate did not produce $out"; }
}
encode_layer "$Y4M64" 90 "$TMP/L0.ivf" "$TMP/m2-L0.log"
encode_layer "$Y4M360" 90 "$TMP/L1.ivf" "$TMP/m2-L1.log"
encode_layer "$Y4M360" 120 "$TMP/L2.ivf" "$TMP/m2-L2.log"

decode_avmdec() {
  local input=$1
  local output=$2
  local log=$3
  shift 3
  "$AVMDEC" --i420 "$@" -o "$output" "$input" >"$log" 2>&1
}

for layer in L0 L1 L2; do
  decode_avmdec "$TMP/$layer.ivf" "$TMP/$layer.avm.yuv" "$TMP/$layer.avm.log" \
    || fail "avmdec single-layer $layer"
done
cat "$TMP/L0.avm.yuv" "$TMP/L1.avm.yuv" > "$TMP/expected2.avm.yuv"
cat "$TMP/L0.avm.yuv" "$TMP/L1.avm.yuv" "$TMP/L2.avm.yuv" \
  > "$TMP/expected3.avm.yuv"

compare() {
  local expected=$1
  local actual=$2
  local label=$3
  if ! cmp -s "$expected" "$actual"; then
    fail "$label: decoded output differs from independently decoded layers"
  fi
  echo "ok: $label ($(wc -c < "$actual" | tr -d ' ') bytes)"
}

run_case() {
  local layers=$1
  local source=$2
  local expected_ops=$3
  shift 3
  local tag="x${layers}"

  "$TMP/svc_mux" -o "$TMP/composed$tag.ivf" "$@" --no-lcr --no-ops \
    >/dev/null 2>&1 || fail "svc_mux $tag"
  "$TMP/lcr_ops_mux" -o "$TMP/full$tag.ivf" --in "$TMP/composed$tag.ivf" \
    --features full --inherit-ops --atlas --local-ops --local-lcr \
    "${LAYER_ARGS[@]}" >"$TMP/mux$tag.log" 2>&1 \
    || { cat "$TMP/mux$tag.log" >&2; fail "lcr_ops_mux $tag"; }

  decode_avmdec "$TMP/full$tag.ivf" "$TMP/full$tag.avm.yuv" \
    "$TMP/full$tag.avm.log" --all-layers \
    || { cat "$TMP/full$tag.avm.log" >&2; fail "avmdec --all-layers $tag"; }
  compare "$source" "$TMP/full$tag.avm.yuv" "avmdec ${layers}-layer --all-layers"

  # --select-ops=0,<i> must reproduce exactly the cumulative layer set.
  local op
  for ((op = 0; op < expected_ops; ++op)); do
    decode_avmdec "$TMP/full$tag.ivf" "$TMP/full$tag.op$op.yuv" \
      "$TMP/full$tag.op$op.log" "--select-ops=0,$op" \
      || { cat "$TMP/full$tag.op$op.log" >&2; \
           fail "avmdec --select-ops=0,$op $tag"; }
    if [[ $op -eq 0 ]]; then
      compare "$TMP/L0.avm.yuv" "$TMP/full$tag.op$op.yuv" \
        "avmdec ${layers}-layer --select-ops=0,0"
    elif [[ $op -eq 1 ]]; then
      compare "$TMP/expected2.avm.yuv" "$TMP/full$tag.op$op.yuv" \
        "avmdec ${layers}-layer --select-ops=0,1"
    else
      compare "$TMP/expected3.avm.yuv" "$TMP/full$tag.op$op.yuv" \
        "avmdec ${layers}-layer --select-ops=0,$op"
    fi
  done

  # Local OPS per extended layer: avmdec must accept the selection.
  for ((op = 0; op < layers; ++op)); do
    decode_avmdec "$TMP/full$tag.ivf" "$TMP/full$tag.local$op.yuv" \
      "$TMP/full$tag.local$op.log" "--select-local-ops=$op,1,0" \
      || { cat "$TMP/full$tag.local$op.log" >&2; \
           fail "avmdec --select-local-ops=$op,1,0 $tag"; }
    compare "$source" "$TMP/full$tag.local$op.yuv" \
      "avmdec ${layers}-layer --select-local-ops=$op,1,0"
  done

  if [[ -n "$DAV2D" ]]; then
    "$DAV2D" -q -i "$TMP/full$tag.ivf" -o "$TMP/full$tag.dav.yuv" \
      --muxer yuv --alllayers 1 --strict 1 >"$TMP/full$tag.dav.log" 2>&1 \
      || { cat "$TMP/full$tag.dav.log" >&2; fail "dav2d $tag"; }
    compare "$source" "$TMP/full$tag.dav.yuv" \
      "dav2d ${layers}-layer --alllayers 1"
  fi
}

echo
echo "2-layer: x0 = 64x64, x1 = 640x360"
LAYER_ARGS=(--layer 64x64 --layer 640x360)
run_case 2 "$TMP/expected2.avm.yuv" 2 \
  --layer "$TMP/L0.ivf" --layer "$TMP/L1.ivf"

echo
echo "3-layer: x0 = 64x64, x1/x2 = 640x360"
LAYER_ARGS=(--layer 64x64 --layer 640x360 --layer 640x360)
run_case 3 "$TMP/expected3.avm.yuv" 3 \
  --layer "$TMP/L0.ivf" --layer "$TMP/L1.ivf" --layer "$TMP/L2.ivf"

echo
echo "negative control: inconsistent local LCR must be rejected"
if "$TMP/lcr_ops_mux" -o "$TMP/bad_local.ivf" --in "$TMP/composedx2.ivf" \
    --layer 64x64 --layer 640x360 --features full --local-lcr-mismatch \
    >"$TMP/bad_local.log" 2>&1; then
  fail "the local/global LCR disagreement was not caught"
fi
grep -q "differs from the global LCR" "$TMP/bad_local.log" \
  || { cat "$TMP/bad_local.log" >&2; fail "unexpected negative-control error"; }
echo "ok: local LCR mismatch rejected before the decoder saw it"

echo
echo "skeleton form (byte-identical to svc_framing) still decodes"
"$TMP/lcr_ops_mux" -o "$TMP/basic2.ivf" --in "$TMP/composedx2.ivf" \
  --layer 64x64 --layer 640x360 --features basic \
  >"$TMP/basic2.log" 2>&1 \
  || { cat "$TMP/basic2.log" >&2; fail "basic splice"; }
decode_avmdec "$TMP/basic2.ivf" "$TMP/basic2.avm.yuv" "$TMP/basic2.avm.log" \
  --all-layers || { cat "$TMP/basic2.avm.log" >&2; fail "basic decode"; }
compare "$TMP/expected2.avm.yuv" "$TMP/basic2.avm.yuv" \
  "avmdec 2-layer basic LCR/OPS"

echo
echo "PASS: LCR/OPS full writers gated by avmdec at 2 and 3 layers"
