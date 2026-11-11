#!/usr/bin/env bash
#
# CPU experiment: current M1 heuristic vs the unified RDO loop at QP 90.
#
# Builds, under build-rdo-loop/:
#   av2enc-ref             unpatched current src/encoder/m1/m1_still.cpp
#   av2enc-rdo-loop        patched experiment copy with the loop hooks
#   m1_dump_rdo_loop       reconstruction dump for bit-exact verification
#
# Cases per input:
#   heuristic      av2enc-ref --m1
#   heuristic_full av2enc-ref --m1 --m1-full-intra --m1-tx-type --m1-rdoq
#                  --m1-uv-full
#   loop_part      patched, AV2ENC_RDO_LOOP=1, --m1
#   loop_full      patched, AV2ENC_RDO_LOOP=1, --m1
#                  --m1-tx-type --m1-rdoq --m1-uv-full
#   loop@S         loop_full at each --scales entry (default 0.0005)
#
# Every stream is decoded by tools/quality.py (dav2d + PSNR + CVVDP).  Every
# loop stream is checked bit-exactly with tools/m1_verify.  The patched binary
# with AV2ENC_RDO_LOOP unset must be byte-identical to av2enc-ref for --m1 and
# --m1-full-intra.
#
# Usage:
#   ./run_m1_rdo_loop.sh
#   ./run_m1_rdo_loop.sh --scales 0.0002,0.0005,0.001,0.002
#   ./run_m1_rdo_loop.sh --no-1080p
#   ./run_m1_rdo_loop.sh --input path.y4m --work /tmp/rdo-loop
#
set -euo pipefail

SELF=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$SELF/../../../../.." && pwd)
CXX=${CXX:-c++}
BUILD=${RDO_LOOP_BUILD:-"$ROOT/build-rdo-loop"}
WORK=${RDO_LOOP_WORK:-"$BUILD/results"}
CORE_LIB=${RDO_LOOP_CORE_LIB:-"$ROOT/build/libav2enc_core.a"}
SCALES=${RDO_LOOP_SCALES:-"0.0005"}
DO_1080P=1
INPUTS=()

usage() {
  sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'
}

while [ $# -gt 0 ]; do
  case "$1" in
    --input|-i) INPUTS+=("$2"); shift ;;
    --scales) SCALES=$2; shift ;;
    --work) WORK=$2; shift ;;
    --no-1080p) DO_1080P=0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "run_m1_rdo_loop: unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

if [ ! -f "$CORE_LIB" ]; then
  echo "run_m1_rdo_loop: missing $CORE_LIB; build the CPU core first" >&2
  exit 2
fi
if [ ${#INPUTS[@]} -eq 0 ]; then
  INPUTS=(
    "$ROOT/testdata/m0_64x64_1f.y4m"
    "$ROOT/tests/data/m0_640x360_1f.y4m"
  )
  if [ "$DO_1080P" = 1 ]; then
    INPUTS+=("$ROOT/testdata/shine_1920x1080_10f.y4m")
  fi
fi
for input in "${INPUTS[@]}"; do
  [ -f "$input" ] || { echo "run_m1_rdo_loop: missing input $input" >&2; exit 2; }
done

mkdir -p "$BUILD" "$WORK"

COMMON=(-std=c++20 -O2 -I"$ROOT/src" -ffp-contract=off)
STRICT=(-Wall -Wextra -Wpedantic -Wconversion -Wshadow)
M1_FLAGS=(-Wall -Wextra -Wpedantic -Wno-sign-conversion)

LAYER_SOURCES=(
  "$ROOT/src/encoder/m1/rdo_loop/config.cpp"
  "$ROOT/src/encoder/m1/rdo_loop/cost.cpp"
  "$ROOT/src/encoder/m1/rdo_loop/search.cpp"
  "$ROOT/src/encoder/m1/rdo_loop/batch.cpp"
  "$ROOT/src/encoder/m1/rdo_loop/activity.cpp"
  "$ROOT/src/encoder/partition_ext/partition_ext.cpp"
  "$ROOT/src/encoder/m1/rdo_decide/config.cpp"
  "$ROOT/src/encoder/m1/rdo_decide/transform_stream.cpp"
  "$ROOT/src/encoder/m1/rdo_decide/intra_rdo.cpp"
  "$ROOT/src/encoder/m1/rdo_decide/chroma_rdo.cpp"
  "$ROOT/src/encoder/m1/rdo_decide/partition_rdo.cpp"
  "$ROOT/src/encoder/m1/rdo_decide/tx_rdo.cpp"
  "$ROOT/src/encoder/tx_type/tx_type.cpp"
  "$ROOT/src/encoder/rdo/rdo.cpp"
  "$ROOT/src/encoder/rdo/intra_decision.cpp"
  "$ROOT/src/encoder/rdoq/rdoq.cpp"
  "$ROOT/src/encoder/rdoq/rdoq_av2_model.cpp"
)

echo "building experiment binaries under $BUILD"
"$CXX" "${COMMON[@]}" "${STRICT[@]}" -c "$ROOT/src/main.cpp" \
  -o "$BUILD/main.o"
"$CXX" "${COMMON[@]}" "${STRICT[@]}" -c "$ROOT/tools/m1_verify/m1_dump.cpp" \
  -o "$BUILD/m1_dump.o"

OBJECTS=()
for source in "${LAYER_SOURCES[@]}"; do
  object="$BUILD/$(echo "$source" | sed 's#^'"$ROOT"'/##; s#/#_#g; s/\.cpp$/.o/')"
  "$CXX" "${COMMON[@]}" "${STRICT[@]}" -c "$source" -o "$object"
  OBJECTS+=("$object")
done

# Unpatched reference M1.
"$CXX" "${COMMON[@]}" "${M1_FLAGS[@]}" -c \
  "$ROOT/src/encoder/m1/m1_still.cpp" -o "$BUILD/m1_still_ref.o"
"$CXX" "$BUILD/main.o" "$BUILD/m1_still_ref.o" "${OBJECTS[@]}" \
  "$CORE_LIB" -o "$BUILD/av2enc-ref"

# Patched experiment M1.
python3 "$SELF/patch_m1_rdo_loop.py" -o "$BUILD/m1_still_rdo_loop.cpp"
"$CXX" "${COMMON[@]}" "${M1_FLAGS[@]}" -c \
  "$BUILD/m1_still_rdo_loop.cpp" -o "$BUILD/m1_still_rdo_loop.o"
"$CXX" "$BUILD/main.o" "$BUILD/m1_still_rdo_loop.o" "${OBJECTS[@]}" \
  "$CORE_LIB" -o "$BUILD/av2enc-rdo-loop"
"$CXX" "$BUILD/m1_dump.o" "$BUILD/m1_still_rdo_loop.o" "${OBJECTS[@]}" \
  "$CORE_LIB" -o "$BUILD/m1_dump_rdo_loop"

REF="$BUILD/av2enc-ref"
BIN="$BUILD/av2enc-rdo-loop"
DUMP="$BUILD/m1_dump_rdo_loop"

echo "reference: $REF"
echo "loop:      $BIN"
echo "scales:    $SCALES"
echo

FAIL=0
encode() {
  local input=$1 out=$2 frames=$3 bin=$4
  shift 4
  if [ "$frames" = 1 ]; then
    "$bin" "$input" -o "$out" "$@" --frames 1 >/dev/null 2>&1
  else
    "$bin" "$input" -o "$out" "$@" >/dev/null 2>&1
  fi
}
quality() {
  local input=$1 ivf=$2 json=$3 log=$4
  python3 "$ROOT/tools/quality.py" --ref "$input" --ivf "$ivf" \
    --json "$json" >"$log" 2>&1
}
verify() {
  local input=$1 ivf=$2 dump_opts=$3 dump=$4 tsv=$5 label=$6
  # shellcheck disable=SC2086
  if ! env AV2ENC_RDO_LOOP=1 $DUMP -i "$input" -o "$dump" $dump_opts \
       >/dev/null 2>&1; then
    echo "  $label: dump failed" >&2
    FAIL=1
    return
  fi
  if python3 "$ROOT/tools/m1_verify/verify.py" --ivf "$ivf" --dump "$dump" \
       --tsv >"$tsv"; then
    echo "  $label: bit-exact"
  else
    echo "  $label: RECONSTRUCTION MISMATCH" >&2
    FAIL=1
  fi
}

for input in "${INPUTS[@]}"; do
  name=$(basename "$input" .y4m)
  dir="$WORK/$name"
  mkdir -p "$dir"
  frames=0
  case "$name" in
    *1080*) frames=1 ;;
  esac
  echo "=== $input ==="

  # Heuristic baseline.
  encode "$input" "$dir/heuristic.ivf" "$frames" "$REF" --m1
  quality "$input" "$dir/heuristic.ivf" "$dir/heuristic.json" \
    "$dir/heuristic.quality.log"

  # Patch neutrality on the loop-off path.
  encode "$input" "$dir/neutral_dc.ivf" "$frames" "$BIN" --m1
  encode "$input" "$dir/neutral_sse.ivf" "$frames" "$BIN" --m1 --m1-full-intra
  encode "$input" "$dir/ref_sse.ivf" "$frames" "$REF" --m1 --m1-full-intra
  if cmp -s "$dir/neutral_dc.ivf" "$dir/heuristic.ivf" &&
     cmp -s "$dir/neutral_sse.ivf" "$dir/ref_sse.ivf"; then
    echo "  patch neutrality: dc/sse byte-identical to reference"
  else
    echo "  PATCH NEUTRALITY FAILED" >&2
    FAIL=1
  fi

  # Heuristic with the same opt-in tools but no unified loop.
  encode "$input" "$dir/heuristic_full.ivf" "$frames" "$REF" --m1 \
    --m1-full-intra --m1-tx-type --m1-rdoq --m1-uv-full
  quality "$input" "$dir/heuristic_full.ivf" "$dir/heuristic_full.json" \
    "$dir/heuristic_full.quality.log"

  # Unified loop, partition + rate-aware modes/chroma.
  env AV2ENC_RDO_LOOP=1 AV2ENC_RDO_LOOP_SCALE=0.0005 \
    AV2ENC_RDO_LOOP_TRACE="$dir/loop_part.trace.csv" \
    AV2ENC_RDO_LOOP_STATS=1 \
    "$BIN" "$input" -o "$dir/loop_part.ivf" --m1 \
    $([ "$frames" = 1 ] && echo --frames 1) >/dev/null 2>&1
  quality "$input" "$dir/loop_part.ivf" "$dir/loop_part.json" \
    "$dir/loop_part.quality.log"
  verify "$input" "$dir/loop_part.ivf" "" "$dir/loop_part.dump" \
    "$dir/loop_part.verify.tsv" "loop_part"

  # Unified loop with TX type, RDOQ and full UV.
  env AV2ENC_RDO_LOOP=1 AV2ENC_RDO_LOOP_SCALE=0.0005 \
    AV2ENC_RDO_LOOP_TRACE="$dir/loop_full.trace.csv" \
    AV2ENC_RDO_LOOP_STATS=1 \
    "$BIN" "$input" -o "$dir/loop_full.ivf" --m1 \
    --m1-tx-type --m1-rdoq --m1-uv-full \
    $([ "$frames" = 1 ] && echo --frames 1) >/dev/null 2>&1
  quality "$input" "$dir/loop_full.ivf" "$dir/loop_full.json" \
    "$dir/loop_full.quality.log"
  verify "$input" "$dir/loop_full.ivf" \
    "--full-intra --uv-full --tx-type --rdoq" "$dir/loop_full.dump" \
    "$dir/loop_full.verify.tsv" "loop_full"

  # Lambda sweep (the default is one point; pass --scales for the full curve).
  for scale in ${SCALES//,/ }; do
    if [ "$scale" = "0.0005" ]; then
      cp "$dir/loop_full.ivf" "$dir/loop_${scale}.ivf"
      cp "$dir/loop_full.json" "$dir/loop_${scale}.json"
      continue
    fi
    env AV2ENC_RDO_LOOP=1 AV2ENC_RDO_LOOP_SCALE="$scale" \
      "$BIN" "$input" -o "$dir/loop_${scale}.ivf" --m1 \
      --m1-tx-type --m1-rdoq --m1-uv-full \
      $([ "$frames" = 1 ] && echo --frames 1) >/dev/null 2>&1
    quality "$input" "$dir/loop_${scale}.ivf" "$dir/loop_${scale}.json" \
      "$dir/loop_${scale}.quality.log"
  done

  python3 - "$dir" "$name" "$SCALES" <<'PY'
import csv
import json
import os
import sys

directory, name, scales = sys.argv[1], sys.argv[2], sys.argv[3]
labels = [
    ("heuristic", "heuristic (current)"),
    ("heuristic_full", "heuristic + tools"),
    ("loop_part", "loop part+modes"),
    ("loop_full", "loop full @0.0005"),
]
for scale in scales.replace(",", " ").split():
    if scale != "0.0005":
        labels.append((f"loop_{scale}", f"loop full @{scale}"))

print(f"  {'case':<22} {'bytes':>7} {'PSNR-Y':>8} {'PSNR-avg':>9} "
      f"{'CVVDP':>8}")
base = None
rows = {}
for key, label in labels:
    path = os.path.join(directory, key + ".json")
    if not os.path.exists(path):
        continue
    with open(path) as handle:
        data = json.load(handle)
    rows[key] = data
    if base is None:
        base = data
    print(f"  {label:<22} {data['ivf_bytes']:7d} "
          f"{data['psnr']['psnr_y']:8.4f} {data['psnr']['psnr_avg']:9.4f} "
          f"{data['cvvdp_jod']:8.4f}")
if base is not None:
    for key, label in labels[1:]:
        if key not in rows:
            continue
        data = rows[key]
        print(f"  {label:<22} vs heuristic: "
              f"{data['ivf_bytes'] - base['ivf_bytes']:+d} B, "
              f"PSNR-Y {data['psnr']['psnr_y'] - base['psnr']['psnr_y']:+.4f} "
              f"dB, CVVDP {data['cvvdp_jod'] - base['cvvdp_jod']:+.4f} JOD")

trace = os.path.join(directory, "loop_full.trace.csv")
if os.path.exists(trace):
    nodes = leaves = 0
    sse = 0.0
    mode_bits = coeff_bits = tx_bits = signal_bits = 0.0
    with open(trace, newline="") as handle:
        for row in csv.DictReader(handle):
            nodes += 1
            if row["partition"] == "0":
                leaves += 1
                sse += float(row["sse"])
                mode_bits += float(row["mode_bits"])
                coeff_bits += float(row["coeff_bits"])
                tx_bits += float(row["tx_bits"])
            signal_bits += float(row["signal_bits"])
    print(f"  cost breakdown: nodes={nodes} leaves={leaves} "
          f"sse={sse:.0f} mode_bits={mode_bits:.1f} "
          f"coeff_bits={coeff_bits:.1f} tx_bits={tx_bits:.1f} "
          f"signal_bits={signal_bits:.1f}")
PY
  echo
done

if [ "$FAIL" != 0 ]; then
  echo "run_m1_rdo_loop: FAILED" >&2
  exit 1
fi
echo "run_m1_rdo_loop: PASS"
