#!/usr/bin/env bash
#
# Black-box decoder gate for the SVC layer pipeline.
#
# Composes real 2- and 3-layer streams from independent single-layer encodes,
# then checks that the decoders reproduce exactly the concatenation of the
# independently decoded layers:
#
#   * avmdec --select-ops=0,N   (global OPS selection)
#   * avmdec --all-layers
#   * dav2d  --alllayers 1
#   * svc_mux --extract N (Annex F.3.2 extraction) -> avmdec
#
# Note: dav2d f9dafbc decodes multiple xlayers and honours --alllayers but does
# not implement LCR (type 16) / OPS (type 18); its --oppoint is a no-op on these
# streams. OPS selection is gated with avmdec --select-ops.
#
# Environment:
#   AV2ENC   encoder binary (default: build/av2enc, build-m1/av2enc)
#   AVMDEC   AVM decoder     (default: tools/bin/avmdec, tools/bin/avm-macos/avmdec)
#   DAV2D    dav2d binary    (default: tools/bin/dav2d, tools/bin/dav2d-macos)
#   KEEP=1   keep the temporary streams
#
# Exit codes: 0 pass, 1 failure, 2 usage, 77 skip (missing encoder/decoder).
set -euo pipefail
cd "$(dirname "$0")"
ROOT=$(cd ../../.. && pwd)
CXX="${CXX:-c++}"

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

AV2ENC=$(find_tool AV2ENC "$ROOT/build/av2enc" "$ROOT/build-m1/av2enc" \
         "$ROOT/build-m1-cdef/av2enc" || true)
if [[ -z "$AV2ENC" ]]; then
  echo "SKIP: no encoder binary (set AV2ENC=/path/to/av2enc)" >&2
  exit 77
fi
DAV2D=$(find_tool DAV2D "$ROOT/tools/bin/dav2d" \
        "$ROOT/tools/bin/dav2d-macos" || true)
if [[ -z "$DAV2D" ]]; then
  echo "SKIP: no dav2d binary (set DAV2D=/path/to/dav2d)" >&2
  exit 77
fi
AVMDEC=$(find_tool AVMDEC "$ROOT/tools/bin/avmdec" \
         "$ROOT/tools/bin/avm-macos/avmdec" || true)

TMP=$(mktemp -d "${TMPDIR:-/tmp}/av2enc-svc-gate.XXXXXX")
cleanup() {
  if [[ "${KEEP:-0}" != "1" ]]; then
    rm -rf "$TMP"
  else
    echo "kept: $TMP"
  fi
}
trap cleanup EXIT

echo "encoder: $AV2ENC"
echo "dav2d:   $DAV2D ($("$DAV2D" --version 2>&1 | tail -1))"
if [[ -n "$AVMDEC" ]]; then
  echo "avmdec:  $AVMDEC"
else
  echo "avmdec:  not found; OPS selection checks will be skipped"
fi

echo "building svc_mux"
"$CXX" -std=c++20 -O2 -Wall -Wextra -Wpedantic -Wconversion -Wshadow \
  -ffp-contract=off -I. -I.. -I../.. \
  svc_mux.cpp svc_config.cpp svc_pipeline.cpp svc_framing.cpp \
  ../tiles/svc_layers.cpp ../../bitstream/bitwriter.cpp ../../io/ivf.cpp \
  -o "$TMP/svc_mux"

Y4M64="$ROOT/tests/data/m0_64x64_1f.y4m"
Y4M360="$ROOT/tests/data/m0_640x360_1f.y4m"
[[ -f "$Y4M64" ]] || { echo "missing $Y4M64" >&2; exit 1; }
[[ -f "$Y4M360" ]] || { echo "missing $Y4M360" >&2; exit 1; }

encode() {
  local source=$1 qp=$2 out=$3
  if ! "$AV2ENC" "$source" -o "$out" --m1 --qp "$qp" >/dev/null 2>&1; then
    "$AV2ENC" "$source" -o "$out" --qp "$qp" >/dev/null
  fi
}

encode "$Y4M64" 90 "$TMP/L0.ivf"
encode "$Y4M64" 120 "$TMP/L1.ivf"
encode "$Y4M360" 90 "$TMP/L2.ivf"

decode_dav2d() {
  "$DAV2D" -q -i "$1" -o "$2" --muxer yuv --alllayers 1 --strict 1 \
    >"$3" 2>&1
}

decode_avmdec() {
  local input=$1 output=$2 log=$3
  shift 3
  "$AVMDEC" --i420 "$@" -o "$output" "$input" >"$log" 2>&1
}

for layer in L0 L1 L2; do
  decode_dav2d "$TMP/$layer.ivf" "$TMP/$layer.dav2d.yuv" "$TMP/$layer.dav2d.log"
  if [[ -n "$AVMDEC" ]]; then
    decode_avmdec "$TMP/$layer.ivf" "$TMP/$layer.avm.yuv" "$TMP/$layer.avm.log"
  fi
done

cat "$TMP/L0.dav2d.yuv" "$TMP/L1.dav2d.yuv" "$TMP/L2.dav2d.yuv" \
  > "$TMP/expected3.dav2d.yuv"
cat "$TMP/L0.dav2d.yuv" "$TMP/L1.dav2d.yuv" > "$TMP/expected2.dav2d.yuv"
if [[ -n "$AVMDEC" ]]; then
  cat "$TMP/L0.avm.yuv" "$TMP/L1.avm.yuv" "$TMP/L2.avm.yuv" \
    > "$TMP/expected3.avm.yuv"
  cat "$TMP/L0.avm.yuv" "$TMP/L1.avm.yuv" > "$TMP/expected2.avm.yuv"
fi

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

compare() {
  local expected=$1 actual=$2 label=$3
  if ! cmp -s "$expected" "$actual"; then
    fail "$label: decoded output differs from independently decoded layers"
  fi
  echo "ok: $label ($(wc -c < "$actual" | tr -d ' ') bytes)"
}

echo
echo "composing 2-layer stream"
"$TMP/svc_mux" -o "$TMP/mux2.ivf" --layer "$TMP/L0.ivf" --layer "$TMP/L1.ivf"

decode_dav2d "$TMP/mux2.ivf" "$TMP/mux2.dav2d.yuv" "$TMP/mux2.dav2d.log"
compare "$TMP/expected2.dav2d.yuv" "$TMP/mux2.dav2d.yuv" "dav2d 2-layer --alllayers"

if [[ -n "$AVMDEC" ]]; then
  decode_avmdec "$TMP/mux2.ivf" "$TMP/mux2.avm.yuv" "$TMP/mux2.avm.log" \
    --all-layers
  compare "$TMP/expected2.avm.yuv" "$TMP/mux2.avm.yuv" "avmdec 2-layer --all-layers"
  for op in 0 1; do
    decode_avmdec "$TMP/mux2.ivf" "$TMP/mux2.op$op.yuv" \
      "$TMP/mux2.op$op.log" "--select-ops=0,$op"
    if [[ $op -eq 0 ]]; then
      compare "$TMP/L0.avm.yuv" "$TMP/mux2.op$op.yuv" \
        "avmdec 2-layer --select-ops=0,0"
    else
      compare "$TMP/expected2.avm.yuv" "$TMP/mux2.op$op.yuv" \
        "avmdec 2-layer --select-ops=0,1"
    fi
  done
fi

echo
echo "composing 3-layer stream"
"$TMP/svc_mux" -o "$TMP/mux3.ivf" --layer "$TMP/L0.ivf" --layer "$TMP/L1.ivf" \
  --layer "$TMP/L2.ivf"

decode_dav2d "$TMP/mux3.ivf" "$TMP/mux3.dav2d.yuv" "$TMP/mux3.dav2d.log"
compare "$TMP/expected3.dav2d.yuv" "$TMP/mux3.dav2d.yuv" "dav2d 3-layer --alllayers"

if [[ -n "$AVMDEC" ]]; then
  decode_avmdec "$TMP/mux3.ivf" "$TMP/mux3.avm.yuv" "$TMP/mux3.avm.log" \
    --all-layers
  compare "$TMP/expected3.avm.yuv" "$TMP/mux3.avm.yuv" "avmdec 3-layer --all-layers"
  for op in 0 1 2; do
    decode_avmdec "$TMP/mux3.ivf" "$TMP/mux3.op$op.yuv" \
      "$TMP/mux3.op$op.log" "--select-ops=0,$op"
    case $op in
      0) cp "$TMP/L0.avm.yuv" "$TMP/expected.op0.yuv" ;;
      1) cat "$TMP/L0.avm.yuv" "$TMP/L1.avm.yuv" > "$TMP/expected.op1.yuv" ;;
      2) cp "$TMP/expected3.avm.yuv" "$TMP/expected.op2.yuv" ;;
    esac
    compare "$TMP/expected.op$op.yuv" "$TMP/mux3.op$op.yuv" \
      "avmdec 3-layer --select-ops=0,$op"
  done
fi

echo
echo "Annex F.3.2 extraction (svc_mux --extract)"
for op in 0 1 2; do
  "$TMP/svc_mux" -o "$TMP/extract$op.ivf" --layer "$TMP/L0.ivf" \
    --layer "$TMP/L1.ivf" --layer "$TMP/L2.ivf" --extract "$op" >/dev/null
  decode_dav2d "$TMP/extract$op.ivf" "$TMP/extract$op.dav2d.yuv" \
    "$TMP/extract$op.dav2d.log"
  case $op in
    0) cp "$TMP/L0.dav2d.yuv" "$TMP/expected.extract0.yuv" ;;
    1) cat "$TMP/L0.dav2d.yuv" "$TMP/L1.dav2d.yuv" > "$TMP/expected.extract1.yuv" ;;
    2) cp "$TMP/expected3.dav2d.yuv" "$TMP/expected.extract2.yuv" ;;
  esac
  compare "$TMP/expected.extract$op.yuv" "$TMP/extract$op.dav2d.yuv" \
    "dav2d extracted OP$op"
  if [[ -n "$AVMDEC" ]]; then
    decode_avmdec "$TMP/extract$op.ivf" "$TMP/extract$op.avm.yuv" \
      "$TMP/extract$op.avm.log"
    if [[ $op -eq 0 ]]; then
      cp "$TMP/L0.avm.yuv" "$TMP/expected.extract$op.avm.yuv"
    elif [[ $op -eq 1 ]]; then
      cat "$TMP/L0.avm.yuv" "$TMP/L1.avm.yuv" > "$TMP/expected.extract$op.avm.yuv"
    else
      cp "$TMP/expected3.avm.yuv" "$TMP/expected.extract$op.avm.yuv"
    fi
    compare "$TMP/expected.extract$op.avm.yuv" "$TMP/extract$op.avm.yuv" \
      "avmdec extracted OP$op"
  fi
done

echo
echo "PASS: SVC 2/3-layer composition, OPS selection, and extraction gate"
