#!/usr/bin/env bash
# Decoder gate for the lossless video path (--video-lossless).
#
#   ./video_lossless_gate.sh
#   AV2ENC_BIN=build/av2enc DAV2D=/path/to/dav2d ./video_lossless_gate.sh
#   KEEP=1 ./video_lossless_gate.sh
#
# What it proves:
#   1. `--video-lossless` routes the plain multi-frame sequence to the §5.18.2
#      lossless inter writer: its own sequence/frame headers (base_q_idx = 0,
#      CodedLossless = 1, TxMode = ONLY_4X4), DPCM, reversible WHT/IDTX and
#      FSC coefficient syntax.
#   2. Pinned dav2d --strict 1 --threads 1 decodes every picture and the
#      decoded samples equal `--video-recon` exactly, for the all-inter
#      stream, the mixed intra/inter stream (--video-lossless-intra-period 4)
#      and the forced-FSC path (--lossless-fsc).
#   3. Four and eight frame streams are IVF-pinned.
#   4. A `--video` stream is unchanged (this is an independent writer) and the
#      video tools the lossless writer does not implement are rejected by name.
#   5. The format cores (10-bit 4:0:0/4:2:0/4:2:2/4:4:4 and 8-bit
#      4:0:0/4:2:2/4:4:4) decode exactly at 640x360 and 64x64, and the
#      hierarchical-P schedule (--video-tlayers 4/8) decodes exactly, including
#      10-bit.
#   6. The hierarchical-B schedule (--video-lossless-b --video-tlayers 4/8)
#      codes two references and COMPOUND_AVERAGE interior frames, and decodes
#      exactly at 640x360 in 8/10-bit 4:2:0, 4:4:4 and 4:0:0.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "$0")/../../.." && pwd)"
TMP="${TMPDIR:-/tmp}/av2enc_video_lossless_gate.$$"
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
  echo "video_lossless_gate: no av2enc binary; set AV2ENC_BIN=..." >&2
  exit 127
fi

DAV2D="${DAV2D:-}"
if [ -z "$DAV2D" ]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [ -x "$candidate" ]; then DAV2D="$candidate"; break; fi
  done
fi
if [ -z "$DAV2D" ] || [ ! -x "$DAV2D" ]; then
  echo "video_lossless_gate: no dav2d binary (set DAV2D=...)" >&2
  exit 77
fi

INPUT="${AV2ENC_INPUT:-$ROOT/testdata/shine_1280x720_60f.y4m}"
if [ ! -f "$INPUT" ]; then
  echo "video_lossless_gate: missing input $INPUT" >&2
  exit 77
fi

md5_of() {
  md5 -q "$1" 2>/dev/null || md5sum "$1" | awk '{print $1}'
}

y4m_payload() {
  python3 - "$1" <<'PY'
import sys
data = open(sys.argv[1], "rb").read()
out = bytearray()
i = data.find(b"FRAME\n")
while i >= 0:
    j = data.find(b"FRAME\n", i + 6)
    out += data[i + 6:j if j >= 0 else len(data)]
    i = j
sys.stdout.buffer.write(out)
PY
}

compare_exact() {
  python3 - "$1" "$2" "$3" <<'PY'
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

a = payload(sys.argv[1])
b = payload(sys.argv[2])
if len(a) != len(b):
    raise SystemExit(f"{sys.argv[3]}: {len(a)} != {len(b)} bytes")
bad = sum(1 for x, y in zip(a, b) if x != y)
if bad:
    raise SystemExit(f"{sys.argv[3]}: {bad}/{len(a)} samples differ")
print(f"exact {sys.argv[3]}: {len(a)} samples")
PY
}

failures=0
passes=0
pass_case() { echo "PASS $1"; passes=$((passes + 1)); }
fail_case() { echo "FAIL $1: $2" >&2; failures=$((failures + 1)); }

# run_input_case <name> <input> <frames> <expected md5|""> [encoder args...]
run_input_case() {
  local name=$1 input=$2 frames=$3 expected=$4; shift 4
  local out="$TMP/$name.ivf" recon="$TMP/$name.recon.y4m" dec="$TMP/$name.dav.y4m"
  if ! "$AV2ENC_BIN" "$input" -o "$out" --frames "$frames" \
        --video-lossless --video-recon "$recon" "$@" >"$TMP/$name.log" 2>&1; then
    fail_case "$name" "encode failed"; cat "$TMP/$name.log" >&2; return
  fi
  local got; got=$(md5_of "$out")
  if [ -n "$expected" ] && [ "$got" != "$expected" ]; then
    fail_case "$name" "IVF md5 $got, expected $expected"; return
  fi
  if ! "$DAV2D" -q -i "$out" -o "$dec" --threads 1 --strict 1 \
        >"$TMP/$name.dec.log" 2>&1; then
    fail_case "$name" "dav2d --strict 1 rejected the stream"
    cat "$TMP/$name.dec.log" >&2; return
  fi
  if ! compare_exact "$recon" "$dec" "$name" >>"$TMP/$name.cmp.log" 2>&1; then
    fail_case "$name" "encoder recon differs from dav2d"
    cat "$TMP/$name.cmp.log" >&2; return
  fi
  pass_case "$name ($(wc -c <"$out" | tr -d ' ') B, md5 $got)"
}

# run_case <name> <frames> <expected md5> [encoder args...]
run_case() {
  local name=$1 frames=$2 expected=$3; shift 3
  run_input_case "$name" "$INPUT" "$frames" "$expected" "$@"
}

# Deterministic noise clip at an arbitrary bit depth / chroma format.
generate_clip() {
  local tag=$1 bitdepth=$2 chroma=$3 path=$4 frames=$5 width=$6 height=$7
  python3 - "$tag" "$bitdepth" "$chroma" "$path" "$frames" "$width" \
    "$height" <<'PYGEN'
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
PYGEN
}

# All-inter (IDTX/FSC on every P block), mixed DPCM/WHT, forced FSC, 8 frames.
run_case all_inter_4 4 3569437af68c2ff640e6deda5be2b5a5
run_case mixed_4 4 59cf90c17e97639bfd9ed8d06670a964 \
  --video-lossless-intra-period 4
run_case forced_fsc_4 4 31e50ec0d113cc02d1383664ff371124 \
  --video-lossless-intra-period 4 --lossless-fsc
run_case all_inter_8 8 19c4b81811692127c108fad1c6ec7f47
run_case tlayers_4 4 f0a2ce304b609a2312822838ab4f7125 --video-tlayers 4
run_case tlayers_8 8 1b02c94e0c0b044157e86e973261ac36 --video-tlayers 8

# ------------------------------------------------- hierarchical B schedule --
# Two references per interior frame with COMPOUND_AVERAGE prediction; the
# 8-frame GOP includes the single-reference B anchor (display index 7).
b8_case() {
  local tag=$1 depth=$2 chroma=$3 frames=$4 pin=$5
  local bw=${6:-640} bh=${7:-360}
  local cs
  if [ "$chroma" = "400" ]; then
    cs="400"
  elif [ "$depth" = "10" ]; then
    cs="${chroma}p10"
  else
    cs="$chroma"
  fi
  local clip="$TMP/b_${tag}.y4m"
  generate_clip "$cs" "$depth" "$chroma" "$clip" "$frames" "$bw" "$bh"
  run_input_case "b_${tag}_${bw}x${bh}" "$clip" "$frames" "$pin" \
    --bit-depth "$depth" --chroma "$chroma" --video-lossless-b \
    --video-tlayers "$frames"
}
b8_case gop4_420 8 420 4 "b9b324b0624372c67d2c4e73bda36785"
b8_case gop8_420 8 420 8 "c61cd6d197e7c3d68edbe87e09e16dd3"
b8_case gop8_420_64 8 420 8 "6a24a8f4c88c3737f29e224f2d6433ff" 64 64
b8_case gop8_420p10 10 420 8 "659dd57b40c6f67f6f4dca591ba4ef64"
b8_case gop8_420p10_64 10 420 8 "6670c5a4489eb2a8c50a87f288dab57c" 64 64
b8_case gop8_422p10 10 422 8 "97e6eba8db7cc88e13fe8e04a520475f"
b8_case gop8_400 8 400 8 "fbf6d94f2a26bafce98f614bb6dbad2d"
b8_case gop8_444 8 444 8 "a23334f1b623d65f0ec9109f9df3b9f8"

# ------------------------------------------------------- format video cores --
# Pinned MD5s for the format cases (646x360 / 64x64 4-frame noise clips).
fmt_pin() {
  case "$1:$2" in
    420p10:640) echo 3b8293a818b771cdab190a53f5635813 ;;
    420p10:64) echo 631592b06357a3554bbacf298609e9cd ;;
    422p10:640) echo b5835150471b2d3f163df693cc6e4061 ;;
    444p10:640) echo 37921674d1ef264eec9194af26cf38ce ;;
    422:640) echo 34304fc9f5dd82347bf60dd548fd670e ;;
    444:640) echo ea09a31d9544d4eaf476a6c1461b0df1 ;;
    444:64) echo b8bdea194e12ea3eac2d23c11f8073bc ;;
    400:640) echo 2cac09117589e436fc1d77204a119aa3 ;;
    400:64) echo 7b94105c4d17f653a55dfa6e49fd9d37 ;;
    400p10:640) echo c6dba17c4b9e882278205cb7f88c7ca2 ;;
    400p10:64) echo 376b32cb3676e6cf6344eb4b97cc273b ;;
    *) echo "" ;;
  esac
}
for spec in "420p10 10 420" "422p10 10 422" "444p10 10 444" \
            "422 8 422" "444 8 444" "400p10 10 400" "400 8 400"; do
  # shellcheck disable=SC2086
  set -- $spec
  tag=$1 depth=$2 chroma=$3
  clip="$TMP/f_${tag}_640.y4m"
  generate_clip "$tag" "$depth" "$chroma" "$clip" 4 640 360
  run_input_case "fmt_${tag}_640x360" "$clip" 4 "$(fmt_pin "$tag" 640)" \
    --bit-depth "$depth" --chroma "$chroma"
  if [ "$tag" = "420p10" ] || [ "$tag" = "444" ] || \
     [ "$tag" = "400p10" ] || [ "$tag" = "400" ]; then
    clip="$TMP/f_${tag}_64.y4m"
    generate_clip "$tag" "$depth" "$chroma" "$clip" 4 64 64
    run_input_case "fmt_${tag}_64x64" "$clip" 4 "$(fmt_pin "$tag" 64)" \
      --bit-depth "$depth" --chroma "$chroma"
    if [ "$tag" = "420p10" ]; then
      run_input_case "fmt_${tag}_tlayers4_640" \
        "$TMP/f_420p10_640.y4m" 4 2cf0113f2a27957a86e836b0344e9335 \
        --bit-depth 10 --chroma 420 --video-tlayers 4
    fi
  fi
done

# Independent writer: a normal --video stream must not move.
BASE="$TMP/base.ivf"
if ! "$AV2ENC_BIN" "$INPUT" -o "$BASE" --frames 4 --video \
      >"$TMP/base.log" 2>&1; then
  fail_case base "default --video encode failed"
else
  BASE_MD5=$(md5_of "$BASE")
  if [ "$BASE_MD5" != "e4cf72dd13ddc2c32c2f8683153b3c81" ]; then
    fail_case base "IVF md5 $BASE_MD5, expected e4cf72dd13ddc2c32c2f8683153b3c81"
  else
    pass_case "base (--video unchanged)"
  fi
fi

# Named rejections for tools the lossless writer does not apply.
reject_flag() {
  local label=$1 needle=$2; shift 2
  if "$AV2ENC_BIN" "$INPUT" -o "$TMP/reject.ivf" --frames 4 \
        --video-lossless "$@" >/dev/null 2>"$TMP/reject.err"; then
    fail_case "$label" "expected a rejection"
  elif grep -q "$needle" "$TMP/reject.err"; then
    pass_case "$label (named rejection)"
  else
    fail_case "$label" "rejection message changed"
    cat "$TMP/reject.err" >&2
  fi
}
reject_flag reject_qm "not applied by the --video-lossless writer" --video-qm
reject_flag reject_inter_residual \
  "not applied by the --video-lossless writer" --video-inter-residual
reject_flag reject_tiles "not applied by the --video-lossless writer" --tiles 2x1
reject_flag reject_qp "is Q=0" --qp 90

if [ "$failures" -ne 0 ]; then
  echo "video_lossless_gate: $failures failed, $passes passed" >&2
  exit 1
fi
echo "video_lossless_gate: $passes passed, 0 failed"
if [ "${KEEP:-0}" = "1" ]; then echo "streams: $TMP"; fi
