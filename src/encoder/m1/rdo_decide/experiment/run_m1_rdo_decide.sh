#!/usr/bin/env bash
#
# CPU experiment: SSE-only vs rate-aware M1 decisions at QP 90.
#
# Builds an experiment-only patched copy of src/encoder/m1/m1_still.cpp and
# compares, per input:
#
#   dc          --m1                              (current default: DC luma)
#   sse         --m1 --m1-full-intra              (SSE-only full intra)
#   rdo         --m1 --m1-full-intra + RDO        (layer on: intra/part/chroma)
#   rdo_part    --m1 --m1-full-intra + RDO, legacy intra
#   rdo_nodc    --m1 + RDO                        (partition/chroma only)
#   rdo_single  --m1 --m1-single --m1-full-intra + RDO (no partition splits)
#
# Every stream is decoded by tools/quality.py (dav2d + CVVDP).  Each RDO stream
# is checked bit-exactly with tools/m1_verify.  The dc/sse streams must be
# byte-identical to the unpatched CLI.
#
# Usage:
#   ./run_m1_rdo_decide.sh
#   ./run_m1_rdo_decide.sh --scales "0.5,1.0,2.0"
#   ./run_m1_rdo_decide.sh --input path.y4m --work /tmp/rdo-decide
#
set -euo pipefail

SELF=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$SELF/../../../../.." && pwd)
CXX=${CXX:-c++}
CORE_LIB=${RDO_DECIDE_CORE_LIB:-"$ROOT/build/libav2enc_core.a"}
REF_BIN=${RDO_DECIDE_REF_BIN:-"$ROOT/build/av2enc"}
BUILD=${RDO_DECIDE_BUILD:-"$ROOT/build-rdo-decide"}
WORK=${RDO_DECIDE_WORK:-"$BUILD/results"}
SCALES=${RDO_DECIDE_SCALES:-"0.5"}
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
    *) echo "run_m1_rdo_decide: unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

if [ ! -f "$CORE_LIB" ]; then
  echo "run_m1_rdo_decide: missing $CORE_LIB; build the CPU core first" >&2
  exit 2
fi
if [ ${#INPUTS[@]} -eq 0 ]; then
  INPUTS=(
    "$ROOT/testdata/m0_64x64_1f.y4m"
    "$ROOT/tests/data/m0_640x360_1f.y4m"
  )
fi
for input in "${INPUTS[@]}"; do
  [ -f "$input" ] || { echo "run_m1_rdo_decide: missing input $input" >&2; exit 2; }
done

mkdir -p "$BUILD" "$WORK"
PATCHED="$BUILD/m1_still_rdo_decide.cpp"
BIN="$BUILD/av2enc-rdo-decide"
DUMP_BIN="$BUILD/m1_dump_rdo_decide"

if [ "$REBUILD" = 1 ] || [ ! -x "$BIN" ] || [ ! -x "$DUMP_BIN" ] ||
   [ "$PATCHED" -ot "$ROOT/src/encoder/m1/m1_still.cpp" ] ||
   [ "$PATCHED" -ot "$SELF/patch_m1_rdo_decide.py" ]; then
  python3 "$SELF/patch_m1_rdo_decide.py" -o "$PATCHED"

  COMMON=(-std=c++20 -O2 -I"$ROOT/src" -ffp-contract=off)
  STRICT=(-Wall -Wextra -Wpedantic -Wconversion -Wshadow)
  M1_FLAGS=(-Wall -Wextra -Wpedantic -Wno-sign-conversion)

  "$CXX" "${COMMON[@]}" "${STRICT[@]}" -c "$ROOT/src/main.cpp" \
    -o "$BUILD/main.o"

  LAYER_OBJECTS=()
  for source in \
      "$ROOT/src/encoder/m1/rdo_decide/config.cpp" \
      "$ROOT/src/encoder/m1/rdo_decide/transform_stream.cpp" \
      "$ROOT/src/encoder/m1/rdo_decide/intra_rdo.cpp" \
      "$ROOT/src/encoder/m1/rdo_decide/chroma_rdo.cpp" \
      "$ROOT/src/encoder/m1/rdo_decide/partition_rdo.cpp" \
      "$ROOT/src/encoder/m1/rdo_decide/tx_rdo.cpp" \
      "$ROOT/src/encoder/tx_type/tx_type.cpp" \
      "$ROOT/src/encoder/rdo/rdo.cpp" \
      "$ROOT/src/encoder/rdo/intra_decision.cpp" \
      "$ROOT/src/encoder/rdoq/rdoq.cpp" \
      "$ROOT/src/encoder/rdoq/rdoq_av2_model.cpp"; do
    object="$BUILD/$(basename "${source%.cpp}").o"
    "$CXX" "${COMMON[@]}" "${STRICT[@]}" -c "$source" -o "$object"
    LAYER_OBJECTS+=("$object")
  done

  "$CXX" "${COMMON[@]}" "${M1_FLAGS[@]}" -c "$PATCHED" \
    -o "$BUILD/m1_still_rdo_decide.o"

  "$CXX" "$BUILD/main.o" "$BUILD/m1_still_rdo_decide.o" \
    "${LAYER_OBJECTS[@]}" "$CORE_LIB" -o "$BIN"

  "$CXX" "${COMMON[@]}" "${STRICT[@]}" -c \
    "$ROOT/tools/m1_verify/m1_dump.cpp" -o "$BUILD/m1_dump.o"
  "$CXX" "$BUILD/m1_dump.o" "$BUILD/m1_still_rdo_decide.o" \
    "${LAYER_OBJECTS[@]}" "$CORE_LIB" -o "$DUMP_BIN"
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

  encode "$input" "$dir/dc.ivf" --m1
  encode "$input" "$dir/sse.ivf" --m1 --m1-full-intra

  if [ -x "$REF_BIN" ]; then
    "$REF_BIN" "$input" -o "$dir/ref_dc.ivf" --m1 >/dev/null
    "$REF_BIN" "$input" -o "$dir/ref_sse.ivf" --m1 --m1-full-intra >/dev/null
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

  # The rdo_part case needs the environment; the hook stays off otherwise.
  env AV2ENC_RDO_DECIDE=1 AV2ENC_RDO_NO_INTRA=1 AV2ENC_RDO_SCALE=0.5 \
    "$BIN" "$input" -o "$dir/rdo_part.ivf" --m1 --m1-full-intra >/dev/null
  quality "$input" "$dir/rdo_part.ivf" "$dir/rdo_part.json" \
    "$dir/rdo_part.quality.log"

  env AV2ENC_RDO_DECIDE=1 AV2ENC_RDO_NO_INTRA=1 AV2ENC_RDO_NO_PART=1 \
    AV2ENC_RDO_SCALE=0.5 \
    "$BIN" "$input" -o "$dir/rdo_chroma.ivf" --m1 --m1-full-intra >/dev/null
  quality "$input" "$dir/rdo_chroma.ivf" "$dir/rdo_chroma.json" \
    "$dir/rdo_chroma.quality.log"

  env AV2ENC_RDO_DECIDE=1 AV2ENC_RDO_NO_PART=1 AV2ENC_RDO_SCALE=0.5 \
    "$BIN" "$input" -o "$dir/rdo_nopart.ivf" --m1 --m1-full-intra >/dev/null
  quality "$input" "$dir/rdo_nopart.ivf" "$dir/rdo_nopart.json" \
    "$dir/rdo_nopart.quality.log"

  env AV2ENC_RDO_DECIDE=1 AV2ENC_RDO_SCALE=0.5 \
    "$BIN" "$input" -o "$dir/rdo_single.ivf" --m1 --m1-single \
    --m1-full-intra >/dev/null
  quality "$input" "$dir/rdo_single.ivf" "$dir/rdo_single.json" \
    "$dir/rdo_single.quality.log"

  encode "$input" "$dir/tx_builtin.ivf" --m1 --m1-full-intra --m1-tx-type
  quality "$input" "$dir/tx_builtin.ivf" "$dir/tx_builtin.json" \
    "$dir/tx_builtin.quality.log"
  env AV2ENC_RDO_DECIDE=1 AV2ENC_RDO_SCALE=0.5 AV2ENC_RDO_TXTYPE=1 \
    AV2ENC_RDO_NO_INTRA=1 AV2ENC_RDO_NO_PART=1 AV2ENC_RDO_NO_CHROMA=1 \
    "$BIN" "$input" -o "$dir/tx_rdo_decide.ivf" --m1 --m1-full-intra \
    --m1-tx-type >/dev/null
  quality "$input" "$dir/tx_rdo_decide.ivf" "$dir/tx_rdo_decide.json" \
    "$dir/tx_rdo_decide.quality.log"

  encode "$input" "$dir/rdoq_builtin.ivf" --m1 --m1-full-intra --m1-rdoq
  quality "$input" "$dir/rdoq_builtin.ivf" "$dir/rdoq_builtin.json" \
    "$dir/rdoq_builtin.quality.log"
  env AV2ENC_RDO_DECIDE=1 AV2ENC_RDO_SCALE=0.5 AV2ENC_RDO_RDOQ=1 \
    AV2ENC_RDO_NO_INTRA=1 AV2ENC_RDO_NO_PART=1 AV2ENC_RDO_NO_CHROMA=1 \
    "$BIN" "$input" -o "$dir/rdoq_rdo_decide.ivf" --m1 --m1-full-intra \
    --m1-rdoq >/dev/null
  quality "$input" "$dir/rdoq_rdo_decide.ivf" "$dir/rdoq_rdo_decide.json" \
    "$dir/rdoq_rdo_decide.quality.log"

  if [ "$VERIFY" = 1 ] && [ -x "$DUMP_BIN" ]; then
    env AV2ENC_RDO_DECIDE=1 AV2ENC_RDO_SCALE=0.5 AV2ENC_RDO_TXTYPE=1 \
      AV2ENC_RDO_NO_INTRA=1 AV2ENC_RDO_NO_PART=1 AV2ENC_RDO_NO_CHROMA=1 \
      "$DUMP_BIN" -i "$input" -o "$dir/tx_rdo_decide.dump" --full-intra \
      --tx-type >/dev/null
    if python3 "$ROOT/tools/m1_verify/verify.py" \
         --ivf "$dir/tx_rdo_decide.ivf" --dump "$dir/tx_rdo_decide.dump" \
         --tsv >"$dir/tx_rdo_decide.verify.tsv"; then
      echo "  tx rdo-decide: bit-exact"
    else
      echo "  tx rdo-decide: RECONSTRUCTION MISMATCH" >&2
      FAIL=1
    fi
    env AV2ENC_RDO_DECIDE=1 AV2ENC_RDO_SCALE=0.5 AV2ENC_RDO_RDOQ=1 \
      AV2ENC_RDO_NO_INTRA=1 AV2ENC_RDO_NO_PART=1 AV2ENC_RDO_NO_CHROMA=1 \
      "$DUMP_BIN" -i "$input" -o "$dir/rdoq_rdo_decide.dump" --full-intra \
      --rdoq >/dev/null
    if python3 "$ROOT/tools/m1_verify/verify.py" \
         --ivf "$dir/rdoq_rdo_decide.ivf" --dump "$dir/rdoq_rdo_decide.dump" \
         --tsv >"$dir/rdoq_rdo_decide.verify.tsv"; then
      echo "  rdoq rdo-decide: bit-exact"
    else
      echo "  rdoq rdo-decide: RECONSTRUCTION MISMATCH" >&2
      FAIL=1
    fi
  fi

  for scale in ${SCALES//,/ }; do
    tag="rdo@$scale"
    out="$dir/rdo_${scale}.ivf"
    env AV2ENC_RDO_DECIDE=1 AV2ENC_RDO_SCALE="$scale" \
      "$BIN" "$input" -o "$out" --m1 --m1-full-intra >/dev/null
    quality "$input" "$out" "$dir/rdo_${scale}.json" \
      "$dir/rdo_${scale}.quality.log"

    if [ "$VERIFY" = 1 ] && [ -x "$DUMP_BIN" ]; then
      env AV2ENC_RDO_DECIDE=1 AV2ENC_RDO_SCALE="$scale" \
        "$DUMP_BIN" -i "$input" -o "$dir/rdo_${scale}.dump" --full-intra \
        >/dev/null
      if python3 "$ROOT/tools/m1_verify/verify.py" \
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
labels = [
    ("dc", "DC (default)"),
    ("sse", "SSE-only full"),
    ("rdo_part", "RDO partition"),
    ("rdo_chroma", "RDO chroma"),
    ("rdo_nopart", "RDO intra+chroma"),
    ("rdo_single", "RDO single-block"),
    ("tx_builtin", "TX built-in"),
    ("tx_rdo_decide", "TX rdo-decide"),
    ("rdoq_builtin", "RDOQ built-in"),
    ("rdoq_rdo_decide", "RDOQ rdo-decide"),
]
for scale in scales.replace(",", " ").split():
    labels.append((f"rdo_{scale}", f"RDO full @{scale}"))
print(f"  {'case':<20} {'bytes':>7} {'PSNR-Y':>8} {'PSNR-avg':>9} {'CVVDP':>8}")
base = None
for key, label in labels:
    path = os.path.join(directory, key + ".json")
    if not os.path.exists(path):
        continue
    d = json.load(open(path))
    if base is None:
        base = d
    print(f"  {label:<20} {d['ivf_bytes']:7d} {d['psnr']['psnr_y']:8.4f} "
          f"{d['psnr']['psnr_avg']:9.4f} {d['cvvdp_jod']:8.4f}")
if base is not None:
    for key, label in labels[1:]:
        path = os.path.join(directory, key + ".json")
        if not os.path.exists(path):
            continue
        d = json.load(open(path))
        print(f"  {label:<20} vs DC: {d['ivf_bytes'] - base['ivf_bytes']:+d} B, "
              f"PSNR-Y {d['psnr']['psnr_y'] - base['psnr']['psnr_y']:+.4f} dB, "
              f"CVVDP {d['cvvdp_jod'] - base['cvvdp_jod']:+.4f} JOD")
PY
  echo
done

if [ "$FAIL" != 0 ]; then
  echo "run_m1_rdo_decide: FAILED" >&2
  exit 1
fi
echo "run_m1_rdo_decide: PASS"
