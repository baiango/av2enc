#!/usr/bin/env bash
# Decoder gate for the two CPU `--video` prediction tools added on
# integration/cpu-video-morph:
#
#   --video-morph                §5.20.5.4/§7.13.3.26 morphological prediction
#                                on the intra keyframe IBC mode-info path
#   --video-compound-interintra  §5.20.7.14/.15/§7.13.3.29/.30 single-reference
#                                inter-intra (mask blend)
#
# The fixtures are deterministic and generated here, so the pinned IVF MD5s
# are reproducible on any CPU build:
#
#   vgrad     a vertical linear gradient.  An IBC copy two 64x64 superblock
#             rows up differs by a constant, so morph_pred = 1 reproduces the
#             source exactly (the plain copy SSE is huge).
#   repeat    a repeated 64-row texture.  The copy is already good, so the
#             encoder signals morph_pred = 0 (the symbol is still coded).
#   move      a translating gradient/texture clip.  Inter and intra errors are
#             complementary, so the inter-intra blend wins on 68 blocks, both
#             with and without a wedge mask.
#
# Every case runs the pinned `dav2d --strict 1 --threads 1`, checks the picture
# count and compares the decoded Y4M byte-for-byte against `--video-recon`.
# The trace counts guard against a PASS on a stream that never signals the
# feature.
#
# Usage:
#   src/pipeline/video/gate_video_morph_interintra.sh [--bin build/av2enc]
#       [--dav2d PATH]
#
# Exit codes: 0 pass, 1 failure, 2 usage, 77 missing tool/decoder.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "$0")/../../.." && pwd)
BIN="$ROOT/build/av2enc"
DAV2D="${DAV2D:-}"
while [ $# -gt 0 ]; do
  case "$1" in
    --bin) BIN=$2; shift 2 ;;
    --dav2d) DAV2D=$2; shift 2 ;;
    -h|--help) sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "gate_video_morph_interintra: unknown argument $1" >&2; exit 2 ;;
  esac
done
if [ ! -x "$BIN" ]; then
  echo "gate_video_morph_interintra: missing encoder $BIN" >&2
  exit 77
fi
if [ -z "$DAV2D" ]; then
  for c in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [ -x "$c" ]; then DAV2D=$c; break; fi
  done
fi
if [ -z "$DAV2D" ] || [ ! -x "$DAV2D" ]; then
  echo "gate_video_morph_interintra: pinned dav2d not found (set DAV2D)" >&2
  exit 77
fi
command -v python3 >/dev/null || {
  echo "gate_video_morph_interintra: python3 is required for the fixtures" >&2
  exit 77
}

OUT="${OUT:-${TMPDIR:-/tmp}/video_morph_gate}"
mkdir -p "$OUT"
FAIL=0
pass() { echo "PASS: $*"; }
fail() { echo "FAIL: $*" >&2; FAIL=1; }

# ---------------------------------------------------------------------------
# Deterministic fixtures.
# ---------------------------------------------------------------------------
python3 - "$OUT" <<'PY'
import sys
out = sys.argv[1]

def write(name, w, h, frames):
    hdr = ("YUV4MPEG2 W%d H%d F30:1 Ip A1:1 C420jpeg\n" % (w, h)).encode()
    blob = bytearray(hdr)
    for y, u, v in frames:
        blob += b"FRAME\n" + bytes(y) + bytes(u) + bytes(v)
    open(out + "/" + name, "wb").write(blob)

def vgrad(w, h, t):
    y = bytearray()
    for j in range(h):
        val = 16 + ((j + t * 2) * 200) // (h - 1)
        y.extend([min(255, max(0, val))] * w)
    u = bytes([128]) * ((w // 2) * (h // 2))
    return y, u, u

w, h = 256, 192
write("morph_vgrad.y4m", w, h, [vgrad(w, h, 0)])
write("morph_vgrad4.y4m", w, h, [vgrad(w, h, t) for t in range(4)])

def repeat_frame(w, h):
    y = bytearray()
    for j in range(h):
        for i in range(w):
            if j < 64:
                val = 16 + ((i * 13 + j * 7) ^ ((i * i + j * j) >> 3)) % 220
            elif j < 128:
                val = 128
            else:
                jj = j - 128
                val = 16 + ((i * 13 + jj * 7) ^ ((i * i + jj * jj) >> 3)) % 220
            y.append(max(0, min(255, val)))
    u = bytes([128]) * ((w // 2) * (h // 2))
    return y, u, u
write("morph_repeat.y4m", 256, 192, [repeat_frame(256, 192)])

def move_frame(w, h, t):
    y = bytearray()
    for j in range(h):
        for i in range(w):
            val = 16 + (((i + t * 3) * 200) // (w - 1) + ((j * 5) // 3)) % 220
            y.append(min(255, max(0, val)))
    u = bytearray()
    for j in range(h // 2):
        for i in range(w // 2):
            u.append(96 + ((i + t * 2) % 64))
    v = bytearray()
    for j in range(h // 2):
        for i in range(w // 2):
            v.append(128 + ((j * 3) // 2) % 48)
    return y, u, v
w, h = 320, 192
write("interintra.y4m", w, h, [move_frame(w, h, t) for t in range(4)])
PY

DEFAULT_PIN=ef85e55bd85357f965169051dd691612
MORPH_VGRAD_PIN=1100a99b1c8f2e20d17ec2c55fe09ebe
MORPH_VGRAD4_PIN=596e2af5db198939b5e8fa20df1a21d5
MORPH_REPEAT_PIN=0ddbee1a49758c3c64e15a1389687184
INTERINTRA_PIN=7d52d066c7437bc3bc9c66f2fc1e52ca

# run_case <name> <expect_pin> <trace_regex> <frames> <args...>
run_case() {
  local name=$1 pin=$2 regex=$3 frames=$4
  shift 4
  local base="$OUT/$name"
  if ! "$BIN" "$base.y4m" -o "$base.ivf" "$@" \
        --video-recon "$base.enc.y4m" --video-trace >"$base.log" 2>&1; then
    fail "$name: encode failed"; tail -3 "$base.log" >&2; return
  fi
  local md5
  md5=$(md5sum "$base.ivf" | cut -d' ' -f1)
  [ "$md5" = "$pin" ] && pass "$name: IVF MD5 $md5" ||
    fail "$name: IVF MD5 $md5 != $pin"
  if ! "$DAV2D" --strict 1 --threads 1 -i "$base.ivf" \
        -o "$base.dec.y4m" >"$base.dec.log" 2>&1; then
    fail "$name: dav2d --strict 1 rejected the stream"
    tail -3 "$base.dec.log" >&2
    return
  fi
  local n
  # LC_ALL=C: BSD grep otherwise stops at the first non-UTF-8 sample byte.
  n=$(LC_ALL=C grep -a -o 'FRAME' "$base.dec.y4m" | wc -l | tr -d ' ')
  [ "$n" = "$frames" ] && pass "$name: $n decoded picture(s)" ||
    fail "$name: decoded $n pictures, expected $frames"
  cmp -s "$base.enc.y4m" "$base.dec.y4m" &&
    pass "$name: decoder recon == --video-recon" ||
    fail "$name: decoded recon differs from --video-recon"
  local count
  count=$(grep -Ec "$regex" "$base.log" || true)
  [ "$count" -gt 0 ] && pass "$name: trace matched $count '$regex'" ||
    fail "$name: trace never matched '$regex'"
}

# Default stream must stay byte-identical.
"$BIN" "$ROOT/testdata/shine_640x360_1f.y4m" -o "$OUT/default.ivf" \
  --video --frames 1 >"$OUT/default.log" 2>&1 ||
  { echo "gate_video_morph_interintra: default encode failed" >&2; exit 1; }
md5=$(md5sum "$OUT/default.ivf" | cut -d' ' -f1)
[ "$md5" = "$DEFAULT_PIN" ] && pass "default --video stream unchanged ($md5)" ||
  fail "default --video MD5 $md5 != $DEFAULT_PIN"

run_case "morph_vgrad" "$MORPH_VGRAD_PIN" 'm2 ibc .*morph=1' 1 \
  --video --frames 1 --video-morph
run_case "morph_vgrad4" "$MORPH_VGRAD4_PIN" 'm2 ibc .*morph=1' 4 \
  --video --frames 4 --video-morph
run_case "morph_repeat" "$MORPH_REPEAT_PIN" 'm2 ibc .*morph=0' 1 \
  --video --frames 1 --video-morph
run_case "interintra" "$INTERINTRA_PIN" 'cand interintra applied=1' 4 \
  --video --frames 4 --video-compound-interintra

# Both §7.13.3.29 masks must appear: COMPOUND_INTRA (wedge=0) and the
# §7.13.3.27 wedge-inter-intra form (wedge=1).
grep -q 'interintra applied=1.*wedge=0' "$OUT/interintra.log" &&
  pass "inter-intra: COMPOUND_INTRA (wedge=0) exercised" ||
  fail "inter-intra: no wedge=0 block"
grep -q 'interintra applied=1.*wedge=1' "$OUT/interintra.log" &&
  pass "inter-intra: wedge_interintra=1 exercised" ||
  fail "inter-intra: no wedge=1 block"

if [ "$FAIL" -ne 0 ]; then
  echo "VIDEO MORPH/INTER-INTRA GATE FAIL" >&2
  exit 1
fi
echo "VIDEO MORPH/INTER-INTRA GATE PASS"
