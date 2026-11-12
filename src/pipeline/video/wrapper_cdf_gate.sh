#!/usr/bin/env bash
# Product gate for the frame-type wrappers (BUG-CDF-001).
#
# Every wrapper that rebuilds the sequence/frame headers (B frames, temporal
# layers, long-term refs, OLK, leading pictures, SEF, SWITCH, RAS) reuses tile
# payloads coded by the plain path.  The headers must therefore carry the same
# disable_cdf_update / enable_avg_cdf state as those payloads, and the tiled
# writer must code the timing block and the referenced MFH.
#
# Each wrapper case is encoded with `--avg-cdf --adaptive-cdf`, decoded by the
# pinned dav2d (`--strict 1`) through video_cli_gate.sh, and compared sample for
# sample against the encoder reconstruction.  In addition every wrapper is
# encoded with `--avg-cdf` alone and must differ from the same command without
# it (the old code silently dropped the flag).  Two tiled cases check the
# structure of the sequence header / first frame header.
#
# Usage:
#   src/pipeline/video/wrapper_cdf_gate.sh --bin build/av2enc [--dav2d PATH]
#       [--input testdata/shine_1280x720_60f.y4m]
#
# Exit codes: 0 pass, 1 failure, 2 usage, 77 missing fixture/decoder.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "$0")/../../.." && pwd)
BIN=""
DAV2D="${DAV2D:-}"
INPUT="${AV2ENC_TESTDATA:-$ROOT/testdata}/shine_1280x720_60f.y4m"
GATE="$ROOT/src/pipeline/video/video_cli_gate.sh"

while [ $# -gt 0 ]; do
  case "$1" in
    --bin) BIN=$2; shift 2 ;;
    --dav2d) DAV2D=$2; shift 2 ;;
    --input) INPUT=$2; shift 2 ;;
    -h|--help)
      sed -n '2,23p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "wrapper_cdf_gate: unknown argument $1" >&2; exit 2 ;;
  esac
done

[ -n "$BIN" ] && [ -x "$BIN" ] || {
  echo "wrapper_cdf_gate: --bin must point at an av2enc executable" >&2
  exit 2
}
[ -f "$INPUT" ] || { echo "SKIP: missing input $INPUT" >&2; exit 77; }
if [ -z "$DAV2D" ]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [ -x "$candidate" ]; then DAV2D=$candidate; break; fi
  done
fi
[ -n "$DAV2D" ] && [ -x "$DAV2D" ] || {
  echo "wrapper_cdf_gate: no dav2d binary (set DAV2D=...)" >&2
  exit 77
}

OUTDIR=$(mktemp -d "${TMPDIR:-/tmp}/av2enc-wrapper-cdf.XXXXXX")
cleanup() {
  if [ -n "${KEEP:-}" ]; then
    echo "wrapper_cdf_gate: streams kept in $OUTDIR"
  else
    rm -rf "$OUTDIR"
  fi
}
trap cleanup EXIT

# name|frames|wrapper args
CASES=(
  "plain|4|"
  "bframes|4|--b-frames 2"
  "tlayers|4|--video-tlayers 4"
  "longterm|3|--video-long-term"
  "olk|4|--video-olk"
  "leading|4|--video-leading 1"
  "sef|4|--video-sef"
  "switch|2|--video-switch"
  "ras|3|--video-ras"
)

echo "== wrapper CDF gate: $INPUT =="
for row in "${CASES[@]}"; do
  IFS='|' read -r name frames args <<<"$row"
  read -r -a extra <<<"$args"
  # Leading pictures and SEF code pictures that are decoded but never output,
  # so the decoded Y4M has fewer frames than the IVF temporal-unit count.
  # Declare that to tools/validate_dav2d.sh; every other case must decode one
  # frame per temporal unit.
  allow_nonoutput=0
  case "$name" in
    leading|sef) allow_nonoutput=1 ;;
  esac
  echo "-- $name: --avg-cdf --adaptive-cdf"
  OUT="$OUTDIR/$name" ALLOW_NONOUTPUT_FRAMES="$allow_nonoutput" \
    bash "$GATE" --bin "$BIN" --dav2d "$DAV2D" --input "$INPUT" \
    --frames "$frames" -- --avg-cdf --adaptive-cdf \
    ${extra[@]+"${extra[@]}"} >"$OUTDIR/$name.log" 2>&1 || {
      cat "$OUTDIR/$name.log" >&2
      echo "FAIL: $name wrapper does not decode recon-exact" >&2
      exit 1
    }
  tail -n 1 "$OUTDIR/$name.log"

  # --avg-cdf alone must reach the sequence header instead of being dropped.
  "$BIN" "$INPUT" --video --frames "$frames" ${extra[@]+"${extra[@]}"} \
    -o "$OUTDIR/$name.base.ivf" >/dev/null 2>&1
  "$BIN" "$INPUT" --video --frames "$frames" ${extra[@]+"${extra[@]}"} \
    --avg-cdf -o "$OUTDIR/$name.avg.ivf" >/dev/null 2>&1
  if cmp -s "$OUTDIR/$name.base.ivf" "$OUTDIR/$name.avg.ivf"; then
    echo "FAIL: $name silently drops --avg-cdf (stream is byte-identical)" >&2
    exit 1
  fi
done

# Tiled header structure: --video-timing must reach the sequence header and
# --video-mfh must be referenced by the frame headers (not just emitted).
TILED=(--tiles 2x2 --no-deblock)
mkdir -p "$OUTDIR/tiled_base"
"$BIN" "$INPUT" --video --frames 4 --tiles 2x2 --no-deblock \
  -o "$OUTDIR/tiled_base/video_cli.ivf" >/dev/null 2>&1
for name in timing mfh; do
  read -r -a extra <<<"${TILED[*]}"
  echo "-- tiled_$name: --video-$name"
  OUT="$OUTDIR/tiled_$name" \
    bash "$GATE" --bin "$BIN" --dav2d "$DAV2D" --input "$INPUT" \
    --frames 4 -- --video-"$name" "${extra[@]}" \
    >"$OUTDIR/tiled_$name.log" 2>&1 || {
      cat "$OUTDIR/tiled_$name.log" >&2
      echo "FAIL: tiled --video-$name does not decode recon-exact" >&2
      exit 1
    }
  tail -n 1 "$OUTDIR/tiled_$name.log"
done

python3 - "$OUTDIR/tiled_base/video_cli.ivf" "$OUTDIR/tiled_timing/video_cli.ivf" \
    "$OUTDIR/tiled_mfh/video_cli.ivf" <<'PY'
import struct
import sys


def first_tu_obus(path):
    data = open(path, 'rb').read()
    offset = 32
    length, _index = struct.unpack_from('<IQ', data, offset)
    unit = data[offset + 12:offset + 12 + length]
    out = []
    pos = 0
    while pos < len(unit):
        size = 0
        shift = 0
        while True:
            byte = unit[pos]
            pos += 1
            size |= (byte & 0x7F) << shift
            shift += 7
            if not byte & 0x80:
                break
        header = unit[pos]
        out.append(((header >> 2) & 0x1F, unit[pos + 1:pos + size]))
        pos += size
    return out


def sequence_header(path):
    for obu_type, payload in first_tu_obus(path):
        if obu_type == 1:  # OBU_SEQUENCE_HEADER
            return payload
    return None


def first_frame(path):
    for obu_type, payload in first_tu_obus(path):
        if obu_type in (4, 7):  # CLOSED_LOOP_KEY / REGULAR_TILE_GROUP
            return payload
    return None


base_seq = sequence_header(sys.argv[1])
timing_seq = sequence_header(sys.argv[2])
if timing_seq is None or base_seq is None:
    raise SystemExit('tiled sequence header missing')
if timing_seq == base_seq:
    raise SystemExit('tiled --video-timing does not change the sequence header')

base_frame = first_frame(sys.argv[1])
mfh_frame = first_frame(sys.argv[3])
if mfh_frame is None or base_frame is None:
    raise SystemExit('tiled frame OBU missing')
if not any(t == 3 for t, _ in first_tu_obus(sys.argv[3])):
    raise SystemExit('tiled --video-mfh emitted no MFH OBU')
if mfh_frame == base_frame:
    raise SystemExit('tiled --video-mfh frame header does not reference the MFH')
print('  tiled header checks: timing block signalled, MFH referenced')
PY

echo "PASS: wrapper average-CDF and tiled timing/MFH gate"
