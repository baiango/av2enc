#!/usr/bin/env bash
# Decoder gate for video film-grain signalling (--video-grain).
#
#   ./video_grain_gate.sh
#   AV2ENC_BIN=build/av2enc DAV2D=/path/to/dav2d ./video_grain_gate.sh
#   KEEP=1 ./video_grain_gate.sh
#
# What it proves:
#   1. `--video-grain` writes sequence-header film_grain_params_present = 1,
#      at most one §5.14 film grain OBU (type 23) immediately before the
#      coded-frame OBU of every temporal unit, and the §5.18.10.1 per-frame
#      apply_grain/fgm_id/seed.
#   2. Pinned dav2d --strict 1 decodes the stream, and `--filmgrain 0` output
#      equals the encoder's --video-recon dump exactly (grain is an output-side
#      synthesis, never part of the reference reconstruction).
#   3. `--filmgrain 1` decodes the same picture count and differs from the
#      pre-grain reconstruction, so the model is actually applied.
#   4. `--video` streams stay byte-identical when the flag is off.
#
# Covered compositions (all decoder-gated at 640x360, plus 64x64):
#   * the historical flat key+P chain (8-bit 4:2:0),
#   * B frames (--b-frames 2), hierarchical temporal layers (--video-tlayers 4),
#     TIP as output (--video-tip),
#   * a 2x1 / 2x2 tile grid (--tiles) at 640x360 (a 64x64 picture is one
#     superblock and cannot form a 2x1 grid),
#   * the format video cores: 10-bit 4:2:0/4:2:2/4:4:4/4:0:0 and 8-bit
#     4:2:2/4:4:4/4:0:0 (--bit-depth/--chroma),
#   * the §7.17-§7.20.5 loop filters and the §5.4.8 average-CDF chain
#     (--avg-cdf/--adaptive-cdf) on the 8-bit 4:2:0 core.
#
# The frame-type wrappers (SEF/SWITCH/RAS/OLK/leading/long-term) stay rejected
# by name: their component writers code no film_grain_params_present or
# film_grain_config(), so a model OBU alone would be inert.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "$0")/../../.." && pwd)"
TMP="${TMPDIR:-/tmp}/av2enc_video_grain_gate.$$"
mkdir -p "$TMP"
cleanup() {
  if [ "${KEEP:-0}" != "1" ]; then rm -rf "$TMP"; fi
}
trap cleanup EXIT

AV2ENC_BIN="${AV2ENC_BIN:-}"
if [ -z "$AV2ENC_BIN" ]; then
  for candidate in "$ROOT/build-int/av2enc" "$ROOT/build/av2enc" \
                   "$ROOT/build-m1/av2enc"; do
    if [ -x "$candidate" ]; then AV2ENC_BIN="$candidate"; break; fi
  done
fi
if [ -z "$AV2ENC_BIN" ] || [ ! -x "$AV2ENC_BIN" ]; then
  echo "video_grain_gate: no av2enc binary; set AV2ENC_BIN=..." >&2
  exit 127
fi

DAV2D="${DAV2D:-}"
if [ -z "$DAV2D" ]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [ -x "$candidate" ]; then DAV2D="$candidate"; break; fi
  done
fi
if [ -z "$DAV2D" ] || [ ! -x "$DAV2D" ]; then
  echo "video_grain_gate: no dav2d binary (set DAV2D=...)" >&2
  exit 77
fi

INPUT="${AV2ENC_INPUT:-$ROOT/testdata/shine_640x360_4f.y4m}"
if [ ! -f "$INPUT" ]; then
  echo "video_grain_gate: missing input $INPUT" >&2
  exit 77
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "video_grain_gate: SKIP (python3 not available)" >&2
  exit 77
fi

md5_of() {
  md5 -q "$1" 2>/dev/null || md5sum "$1" | awk '{print $1}'
}

# List every OBU per IVF frame as "frame:type:length".
list_obus() {
  python3 - "$1" <<'PY'
import struct, sys
data = open(sys.argv[1], "rb").read()
off = 32
frame = 0
while off + 12 <= len(data):
    size = struct.unpack_from("<I", data, off)[0]
    off += 12
    payload = data[off:off + size]
    off += size
    i = 0
    while i < len(payload):
        value = 0
        shift = 0
        while True:
            byte = payload[i]
            i += 1
            value |= (byte & 0x7F) << shift
            if not (byte & 0x80):
                break
            shift += 7
        obu = payload[i:i + value]
        i += value
        print(f"{frame}:{(obu[0] >> 2) & 0x1F}:{value}")
    frame += 1
PY
}

# Compare a decoded Y4M payload against the encoder's recon dump and check that
# --filmgrain 1 differs from --filmgrain 0 with the same picture count.
compare_case() {
  python3 - "$@" <<'PY'
import sys

def payload(path):
    data = open(path, "rb").read()
    out = bytearray()
    i = data.find(b"FRAME\n")
    while i >= 0:
        j = data.find(b"FRAME\n", i + 6)
        out += data[i + 6:j if j >= 0 else len(data)]
        i = j
    return bytes(out)

recon = payload(sys.argv[1])
pre = payload(sys.argv[2])
post = payload(sys.argv[3])
name = sys.argv[4]
if len(recon) != len(pre):
    raise SystemExit(f"{name}: recon {len(recon)} B != decoded {len(pre)} B "
                     "(picture count/geometry)")
bad = sum(1 for x, y in zip(recon, pre) if x != y)
if bad:
    raise SystemExit(f"{name}: {bad}/{len(recon)} samples differ")
if len(post) != len(pre):
    raise SystemExit(f"{name}: --filmgrain 1 changed the picture count")
if pre == post:
    raise SystemExit(f"{name}: film grain changed no samples")
print(f"exact {name}: {len(recon)} B, grain applied")
PY
}

failures=0
passes=0
pass_case() { echo "PASS $1"; passes=$((passes + 1)); }
fail_case() { echo "FAIL $1: $2" >&2; failures=$((failures + 1)); }

# One grain composition: encode, pin (optional), OBU placement, exact recon
# with --filmgrain 0, observable synthesis with --filmgrain 1.
run_case() {
  local name=$1 expected=$2 frames=$3 input=$4
  shift 4
  local out="$TMP/$name.ivf"
  local recon="$TMP/$name.recon.y4m"
  local pre="$TMP/$name.pre.y4m"
  local post="$TMP/$name.post.y4m"
  if ! "$AV2ENC_BIN" "$input" -o "$out" --frames "$frames" --video \
        --video-grain --video-recon "$recon" "$@" >"$TMP/$name.log" 2>&1; then
    fail_case "$name" "encode failed"
    cat "$TMP/$name.log" >&2
    return
  fi
  if [ -n "$expected" ]; then
    local got; got=$(md5_of "$out")
    if [ "$got" != "$expected" ]; then
      fail_case "${name}_pin" "IVF md5 $got, expected $expected"
    else
      pass_case "${name}_pin (md5 $got)"
    fi
  fi
  # At most one film grain OBU per temporal unit, right before the frame OBU.
  if ! list_obus "$out" | python3 -c '
import collections, sys
pos = collections.defaultdict(list)
for line in sys.stdin:
    frame, kind, _ = line.strip().split(":")
    pos[int(frame)].append(int(kind))
bad = []
for frame, kinds in pos.items():
    grain = [i for i, k in enumerate(kinds) if k == 23]
    frame_obus = [i for i, k in enumerate(kinds)
                  if k in (4, 6, 7, 8, 10, 11, 12, 13, 14, 19, 21)]
    if len(grain) > 1 or len(frame_obus) != 1:
        bad.append((frame, kinds))
        continue
    if grain and grain[0] != frame_obus[0] - 1:
        bad.append((frame, kinds))
if bad:
    print(bad, file=sys.stderr)
    sys.exit(1)
frames = len(pos)
if frames == 0:
    sys.exit(1)
print(frames)
'; then
    fail_case "${name}_order" "film grain OBU placement/model count"
  else
    pass_case "${name}_order (<=1 type-23 OBU per unit, before the frame OBU)"
  fi
  if ! "$DAV2D" -q -i "$out" -o "$pre" --threads 1 --strict 1 --filmgrain 0 \
        >"$TMP/$name.dav2d0.log" 2>&1; then
    fail_case "${name}_dav2d" "dav2d --filmgrain 0 rejected the stream"
    cat "$TMP/$name.dav2d0.log" >&2
    return
  fi
  if ! "$DAV2D" -q -i "$out" -o "$post" --threads 1 --strict 1 --filmgrain 1 \
        >"$TMP/$name.dav2d1.log" 2>&1; then
    fail_case "${name}_apply" "dav2d --filmgrain 1 rejected the stream"
    cat "$TMP/$name.dav2d1.log" >&2
    return
  fi
  if ! compare_case "$recon" "$pre" "$post" "$name"; then
    fail_case "${name}_recon" "recon/synthesis mismatch"
    return
  fi
  pass_case "${name}_recon (dav2d --filmgrain 0 == --video-recon)"
  pass_case "${name}_apply (--filmgrain 1 differs, same picture count)"
}

# Deterministic noise clip at an arbitrary bit depth / chroma format.
generate_clip() {
  local tag=$1 bitdepth=$2 chroma=$3 path=$4 frames=$5 width=$6 height=$7
  python3 - "$tag" "$bitdepth" "$chroma" "$path" "$frames" "$width" \
    "$height" <<'PY'
import math
import sys

tag, bitdepth, chroma, path, frames, width, height = sys.argv[1:8]
bitdepth = int(bitdepth)
frames = int(frames)
w, h = int(width), int(height)
sub_x = 1 if chroma in ("420", "422") else 0
sub_y = 1 if chroma == "420" else 0
cw, ch = w >> sub_x, h >> sub_y
peak = (1 << bitdepth) - 1
planes = [(w, h)]
if chroma != "400":
    planes += [(cw, ch), (cw, ch)]

def noise(x, y, f, plane):
    v = (x * 374761393 + y * 668265263 + f * 2246822519 +
         plane * 3266489917) & 0xFFFFFFFF
    v = ((v ^ (v >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((v ^ (v >> 16)) & 0x7FF) - 1024

def sample(x, y, f, plane):
    v = (1 << (bitdepth - 1)) + \
        (1 << (bitdepth - 3)) * math.sin((x + 4 * f + 7 * plane) * 0.05) + \
        (1 << (bitdepth - 4)) * math.cos((y + f + 3 * plane) * 0.07) + \
        (1 << (bitdepth - 5)) * math.sin((x + y + 11 * f) * 0.013)
    v += noise(x, y, f, plane) * ((1 << (bitdepth - 6)) / 1024.0)
    return max(0, min(peak, int(v + 0.5)))

with open(path, "wb") as out:
    out.write(("YUV4MPEG2 W%d H%d F30:1 Ip A1:1 C%s\n" % (w, h, tag)).encode())
    for f in range(frames):
        out.write(b"FRAME\n")
        for plane, (pw, ph) in enumerate(planes):
            if bitdepth == 8:
                for _ in range(ph):
                    out.write(bytes(sample(x, _, f, plane) for x in range(pw)))
            else:
                for _ in range(ph):
                    row = bytearray()
                    for x in range(pw):
                        v = sample(x, _, f, plane)
                        row += bytes((v & 0xFF, (v >> 8) & 0xFF))
                    out.write(row)
PY
}

# ------------------------------------------------------------------ base -----
BASE_INPUT="$ROOT/testdata/shine_1280x720_60f.y4m"
BASE="$TMP/base.ivf"
if ! "$AV2ENC_BIN" "$BASE_INPUT" -o "$BASE" --frames 4 --video \
      >"$TMP/base.log" 2>&1; then
  fail_case base "default encode failed"
else
  BASE_MD5=$(md5_of "$BASE")
  if [ "$BASE_MD5" != "e4cf72dd13ddc2c32c2f8683153b3c81" ]; then
    fail_case base "IVF md5 $BASE_MD5, expected e4cf72dd13ddc2c32c2f8683153b3c81"
  else
    pass_case "base (flag off, md5 unchanged)"
  fi
fi

# ---------------------------------------------------- historical plain chain -
run_case plain4 e60ded0a500a856984cd1b575c89103b 4 \
  "$ROOT/testdata/shine_1280x720_60f.y4m"
run_case plain8 e4c9dbd7e105160aea90c66947b69bbb 8 \
  "$ROOT/testdata/shine_1280x720_60f.y4m"

# ------------------------------------------------- B / temporal / TIP / tiles -
# 640x360 (4 frames) and 64x64 (4 frames).
CLIP64="$TMP/clip64.y4m"
python3 - "$ROOT/tests/data/basic_64x64_1f.y4m" "$CLIP64" <<'PY'
import sys
src = open(sys.argv[1], "rb").read()
i = src.find(b"FRAME\n")
body = src[i + 6:]
with open(sys.argv[2], "wb") as out:
    out.write(b"YUV4MPEG2 W64 H64 F30:1 Ip A1:1 C420jpeg\n")
    for _ in range(4):
        out.write(b"FRAME\n")
        out.write(body)
PY

for size in 360 64; do
  if [ "$size" = "360" ]; then input="$INPUT"; frames=4; else input="$CLIP64"; frames=4; fi
  run_case "bframes${size}" "" $frames "$input" --b-frames 2
  run_case "tlayers${size}" "" $frames "$input" --video-tlayers 4
  run_case "tip${size}" "" $frames "$input" --video-tip
  run_case "bframes_implicit${size}" "" $frames "$input" --b-frames 2 \
    --video-implicit-ref-map
done
run_case tiles2x1 "" 4 "$INPUT" --tiles 2x1
run_case tiles2x2 "" 4 "$INPUT" --tiles 2x2

# ------------------------------------------------------- format video cores ---
# 640x360: every signalable format.  64x64: 10-bit 4:2:0 and 8-bit 4:4:4.
fmt_case() {
  local tag=$1 depth=$2 chroma=$3 frames=$4 width=$5 height=$6
  local clip="$TMP/${tag}_${width}x${height}.y4m"
  generate_clip "$tag" "$depth" "$chroma" "$clip" "$frames" "$width" "$height"
  run_case "fmt_${tag}_${width}x${height}" "" "$frames" "$clip" \
    --bit-depth "$depth" --chroma "$chroma"
}
for spec in "420p10 10 420" "422p10 10 422" "444p10 10 444" \
            "mono10 10 400" "422 8 422" "444 8 444" "mono 8 400"; do
  # shellcheck disable=SC2086
  set -- $spec
  fmt_case "$1" "$2" "$3" 4 640 360
done
fmt_case 420p10 10 420 4 64 64
fmt_case 444 8 444 4 64 64

# --------------------------------------- grain + loop filters / average CDF ---
# Film grain is output-side synthesis, so the encoder reconstruction is
# unchanged by the filters; these cases prove the model OBU/config compose
# with each filter plan and with the §5.4.8 average-CDF chain.
FILTERS="--video-deblock --video-cdef --video-ccso --video-wiener --video-gdf"
run_case grain_deblock "" 4 "$INPUT" --video-deblock
run_case grain_cdef "" 4 "$INPUT" --video-cdef
run_case grain_ccso "" 4 "$INPUT" --video-ccso
run_case grain_wiener "" 4 "$INPUT" --video-wiener
run_case grain_chroma_lr "" 4 "$INPUT" --video-chroma-lr
run_case grain_gdf "" 4 "$INPUT" --video-gdf
run_case grain_avgcdf "" 4 "$INPUT" --avg-cdf
run_case grain_avgcdf_adaptive "" 4 "$INPUT" --avg-cdf --adaptive-cdf
# shellcheck disable=SC2086
run_case grain_filters "" 4 "$INPUT" $FILTERS
# shellcheck disable=SC2086
run_case grain_filters_avgcdf "" 4 "$INPUT" $FILTERS --avg-cdf --adaptive-cdf
# shellcheck disable=SC2086
run_case grain_filters_avgcdf_64 "" 4 "$CLIP64" $FILTERS --avg-cdf --adaptive-cdf
# shellcheck disable=SC2086
run_case grain_bframes_filters "" 4 "$INPUT" --b-frames 2 $FILTERS \
  --avg-cdf --adaptive-cdf
# shellcheck disable=SC2086
run_case grain_tlayers_filters "" 4 "$INPUT" --video-tlayers 4 $FILTERS \
  --avg-cdf --adaptive-cdf
# Tiles cannot use deblock/Wiener (the pinned decoder filters interior tile
# edges); CDEF/CCSO/GDF and the average-CDF chain compose.
run_case grain_tiles_filters "" 4 "$INPUT" --tiles 2x1 --video-cdef \
  --video-ccso --video-gdf --avg-cdf --adaptive-cdf
# Format cores: filters + grain (no --avg-cdf on the format writer).
fmt_case_filters() {
  local tag=$1 depth=$2 chroma=$3 frames=$4 width=$5 height=$6
  local clip="$TMP/filt_${tag}_${width}x${height}.y4m"
  generate_clip "$tag" "$depth" "$chroma" "$clip" "$frames" "$width" "$height"
  # shellcheck disable=SC2086
  run_case "fmt_filters_${tag}_${width}x${height}" "" "$frames" "$clip" \
    --bit-depth "$depth" --chroma "$chroma" $FILTERS
}
fmt_case_filters 420p10 10 420 4 640 360
fmt_case_filters 444 8 444 4 640 360
fmt_case_filters 420p10 10 420 4 64 64

# ------------------------------------------------------------ rejections -----
if "$AV2ENC_BIN" "$INPUT" -o "$TMP/reject.ivf" --frames 4 --video \
      --video-grain --video-sef >/dev/null 2>"$TMP/reject.err"; then
  fail_case sef_reject "expected --video-grain --video-sef to be rejected"
elif grep -q "frame-type wrappers" "$TMP/reject.err"; then
  pass_case "sef_reject (named rejection)"
else
  fail_case sef_reject "rejection message changed"
  cat "$TMP/reject.err" >&2
fi

if [ "$failures" -ne 0 ]; then
  echo "video_grain_gate: $failures failed, $passes passed" >&2
  exit 1
fi
echo "video_grain_gate: $passes passed, 0 failed"
if [ "${KEEP:-0}" = "1" ]; then echo "streams: $TMP"; fi
