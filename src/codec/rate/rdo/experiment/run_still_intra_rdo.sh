#!/usr/bin/env bash
#
# CPU experiment: SSE-only vs rate-aware M1 luma intra selection at QP 90.
#
# Builds an experiment-only patched copy of src/pipeline/still/still_encoder.cpp (the
# repository file is never edited), then for each input compares:
#
#   dc    --still                         (DC luma, the current default)
#   sse   --still --still-full-intra         (current SSE-only full search)
#   rdo   --still --still-full-intra + AV2ENC_RDO_INTRA=1
#
# It checks that dc/sse are byte-identical to the unpatched CLI, decodes every
# stream with quality.py, and verifies the rdo stream reconstruction bit-exactly
# with tools/still_verify.
#
# Usage:
#   ./run_still_intra_rdo.sh
#   ./run_still_intra_rdo.sh --scales "0.02,0.06,0.125"
#   ./run_still_intra_rdo.sh --input path.y4m --work /tmp/rdo-exp
#
set -euo pipefail

SELF=$(cd "$(dirname "$0")" && pwd)
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
export ROOT

CXX=${CXX:-c++}
CORE_LIB=${RDO_CORE_LIB:-"$ROOT/build/libav2enc_core.a"}
REF_BIN=${RDO_REF_BIN:-"$ROOT/build/av2enc"}
BUILD=${RDO_EXPERIMENT_BUILD:-"$ROOT/build-rdo-exp"}
WORK=${RDO_WORK:-"$BUILD/results"}
SCALES=${RDO_SCALES:-"0.125"}
REBUILD=0
VERIFY=1
INPUTS=()

usage() {
  sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
}

while [ $# -gt 0 ]; do
  case "$1" in
    --input|-i) INPUTS+=("$2"); shift ;;
    --scales) SCALES=$2; shift ;;
    --work) WORK=$2; shift ;;
    --rebuild) REBUILD=1 ;;
    --no-verify) VERIFY=0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "run_still_intra_rdo: unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

if [ ! -f "$CORE_LIB" ]; then
  echo "run_still_intra_rdo: missing $CORE_LIB; build the CPU core first" >&2
  exit 2
fi

if [ ${#INPUTS[@]} -eq 0 ]; then
  INPUTS=(
    "$ROOT/testdata/basic_64x64_1f.y4m"
    "$ROOT/tests/data/basic_640x360_1f.y4m"
  )
fi
for input in "${INPUTS[@]}"; do
  [ -f "$input" ] || { echo "run_still_intra_rdo: missing input $input" >&2; exit 2; }
done

mkdir -p "$BUILD" "$WORK"
PATCHED="$BUILD/still_encoder_rdo.cpp"
BIN="$BUILD/av2enc-rdo"
DUMP_BIN="$BUILD/still_dump_rdo"

if [ "$REBUILD" = 1 ] || [ ! -x "$BIN" ] || [ ! -x "$DUMP_BIN" ] ||
   [ "$PATCHED" -ot "$ROOT/src/pipeline/still/still_encoder.cpp" ] ||
   [ "$PATCHED" -ot "$ROOT/src/codec/rate/rdo/intra_decision.cpp" ]; then
  python3 "$SELF/patch_still_intra_rdo.py" -o "$PATCHED"

  COMMON=(-std=c++20 -O2 -I"$ROOT/src" -ffp-contract=off)
  STRICT=(-Wall -Wextra -Wpedantic -Wconversion -Wshadow)
  # The patched M1 file has pre-existing sign-conversion warnings; keep the
  # component build strict (build_and_test.sh) and silence them here.
  STILL_FLAGS=(-Wall -Wextra -Wpedantic -Wno-sign-conversion)

  "$CXX" "${COMMON[@]}" "${STRICT[@]}" -c "$ROOT/src/main.cpp" -o "$BUILD/main.o"
  "$CXX" "${COMMON[@]}" "${STRICT[@]}" -c \
    "$ROOT/src/codec/rate/rdo/intra_decision.cpp" -o "$BUILD/intra_decision.o"
  "$CXX" "${COMMON[@]}" "${STILL_FLAGS[@]}" -c "$PATCHED" -o "$BUILD/still_encoder_rdo.o"

  "$CXX" "$BUILD/main.o" "$BUILD/still_encoder_rdo.o" "$BUILD/intra_decision.o" \
    "$CORE_LIB" -o "$BIN"

  "$CXX" "${COMMON[@]}" "${STRICT[@]}" -c \
    "$ROOT/tools/still_verify/still_dump.cpp" -o "$BUILD/still_dump.o"
  "$CXX" "$BUILD/still_dump.o" "$BUILD/still_encoder_rdo.o" \
    "$BUILD/intra_decision.o" "$CORE_LIB" -o "$DUMP_BIN"
fi

echo "experiment binary: $BIN"
echo "core library:      $CORE_LIB"
echo "scales:            $SCALES"
echo

encode() {
  local input=$1 out=$2
  shift 2
  "$BIN" "$input" -o "$out" "$@" >/dev/null
}

quality() {
  local input=$1 ivf=$2 json=$3 log=$4
  python3 "$ROOT/tools/quality.py" --ref "$input" --ivf "$ivf" \
    --json "$json" >"$log" 2>&1
}

FAIL=0
for input in "${INPUTS[@]}"; do
  name=$(basename "$input" .y4m)
  dir="$WORK/$name"
  mkdir -p "$dir"
  echo "=== $input ==="

  encode "$input" "$dir/dc.ivf" --still
  encode "$input" "$dir/sse.ivf" --still --still-full-intra

  # Patch-neutrality gate: dc/sse must match the unpatched CLI byte-for-byte.
  if [ -x "$REF_BIN" ]; then
    "$REF_BIN" "$input" -o "$dir/ref_dc.ivf" --still >/dev/null
    "$REF_BIN" "$input" -o "$dir/ref_sse.ivf" --still --still-full-intra >/dev/null
    if ! cmp -s "$dir/dc.ivf" "$dir/ref_dc.ivf" ||
       ! cmp -s "$dir/sse.ivf" "$dir/ref_sse.ivf"; then
      echo "  PATCH NEUTRALITY FAILED (dc/sse differ from $REF_BIN)" >&2
      FAIL=1
    else
      echo "  patch neutrality: dc/sse byte-identical to reference CLI"
    fi
  else
    echo "  patch neutrality: skipped (no $REF_BIN)"
  fi

  quality "$input" "$dir/dc.ivf" "$dir/dc.json" "$dir/dc.quality.log"
  quality "$input" "$dir/sse.ivf" "$dir/sse.json" "$dir/sse.quality.log"

  for scale in ${SCALES//,/ }; do
    tag="rdo@$scale"
    out="$dir/rdo_${scale}.ivf"
    env AV2ENC_RDO_INTRA=1 AV2ENC_RDO_SCALE="$scale" \
      "$BIN" "$input" -o "$out" --still --still-full-intra >/dev/null
    quality "$input" "$out" "$dir/rdo_${scale}.json" \
      "$dir/rdo_${scale}.quality.log"

    if [ "$VERIFY" = 1 ] && [ -x "$DUMP_BIN" ]; then
      env AV2ENC_RDO_INTRA=1 AV2ENC_RDO_SCALE="$scale" \
        "$DUMP_BIN" -i "$input" -o "$dir/rdo_${scale}.dump" --full-intra \
        >/dev/null
      if python3 "$ROOT/tools/still_verify/verify.py" \
           --ivf "$out" --dump "$dir/rdo_${scale}.dump" --tsv \
           >"$dir/rdo_${scale}.verify.tsv"; then
        echo "  $tag: bit-exact"
      else
        echo "  $tag: RECONSTRUCTION MISMATCH" >&2
        FAIL=1
      fi
    fi
    echo "  $tag: $(wc -c <"$out" | tr -d ' ') bytes"
  done

  python3 - "$dir" "$name" "$SCALES" <<'PY'
import json, os, sys
directory, name, scales = sys.argv[1], sys.argv[2], sys.argv[3]
labels = [("dc", "DC"), ("sse", "SSE-only full")]
for scale in scales.replace(",", " ").split():
    labels.append((f"rdo_{scale}", f"RDO full @{scale}"))
print(f"  {'case':<18} {'bytes':>7} {'PSNR-Y':>8} {'PSNR-avg':>9} {'CVVDP':>8}")
dc = None
for key, label in labels:
    path = os.path.join(directory, key + ".json")
    if not os.path.exists(path):
        continue
    d = json.load(open(path))
    if dc is None:
        dc = d
    print(f"  {label:<18} {d['ivf_bytes']:7d} {d['psnr']['psnr_y']:8.4f} "
          f"{d['psnr']['psnr_avg']:9.4f} {d['cvvdp_jod']:8.4f}")
for key, label in labels[1:]:
    path = os.path.join(directory, key + ".json")
    if not os.path.exists(path):
        continue
    d = json.load(open(path))
    print(f"  {label:<18} vs DC: {d['ivf_bytes'] - dc['ivf_bytes']:+d} B, "
          f"PSNR-Y {d['psnr']['psnr_y'] - dc['psnr']['psnr_y']:+.4f} dB, "
          f"CVVDP {d['cvvdp_jod'] - dc['cvvdp_jod']:+.4f} JOD")
PY
  echo
done

if [ "$FAIL" != 0 ]; then
  echo "run_still_intra_rdo: FAILED" >&2
  exit 1
fi
echo "run_still_intra_rdo: PASS"
