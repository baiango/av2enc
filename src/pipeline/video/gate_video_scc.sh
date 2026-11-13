#!/usr/bin/env bash
# Decoder gate for the two CPU `--video` screen-content tools:
#
#   --video-palette   §5.20.8 luma palette selection on the intra keyframe
#   --video-ibc       §5.20.5.3/.4 general intra block copy on the keyframe
#
# Both carriers signal the §5.4.7 per-frame screen-content form and code the
# §5.18.3.3/.4 frame bits; palette codes palette_mode_info()/tokens, IBC codes
# use_intrabc + the §5.20.5.4 BV syntax.  The fixtures are deterministic and
# generated here, so the pinned IVF MD5s reproduce on any CPU build:
#
#   scc64    64x64, 32-pixel horizontal period, few colours.  Palette is
#            exercised on every leaf.  A 64x64 frame is one superblock, so the
#            §6.19.5.4 global-IBC delay window leaves no legal BV (the flag is
#            still accepted and must stay decoder-exact).
#   scc360   640x360, the same 32-pixel period.  The keyframe is split by the
#            ordinary partition tree, so palette and IBC leaves coexist.
#
# Every case runs the pinned `dav2d --strict 1 --threads 1`, checks the picture
# count and compares the decoded Y4M byte-for-byte against `--video-recon`.
# Trace counts guard against a PASS on a stream that never signals the feature.
#
# Usage:
#   src/pipeline/video/gate_video_scc.sh [--bin build/av2enc] [--dav2d PATH]
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
    -h|--help) sed -n '2,34p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "gate_video_scc: unknown argument $1" >&2; exit 2 ;;
  esac
done
if [ ! -x "$BIN" ]; then
  echo "gate_video_scc: missing encoder $BIN" >&2
  exit 77
fi
if [ -z "$DAV2D" ]; then
  for c in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [ -x "$c" ]; then DAV2D=$c; break; fi
  done
fi
if [ -z "$DAV2D" ] || [ ! -x "$DAV2D" ]; then
  echo "gate_video_scc: pinned dav2d not found (set DAV2D)" >&2
  exit 77
fi
command -v python3 >/dev/null || {
  echo "gate_video_scc: python3 is required for the fixtures" >&2
  exit 77
}

OUT="${OUT:-${TMPDIR:-/tmp}/video_scc_gate}"
mkdir -p "$OUT"
FAIL=0
pass() { echo "PASS: $*"; }
fail() { echo "FAIL: $*" >&2; FAIL=1; }
md5_of() { md5sum "$1" | cut -d' ' -f1; }

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

def scc_frame(w, h, shift=0):
    # 32-pixel horizontal period and 64-pixel vertical period -> identical
    # 16x16/32x32 regions that IBC can copy and only 6 luma values for the
    # palette.  `shift` translates the frame for the P-frame cases.
    cols = [16, 48, 96, 144, 200, 240]
    y = bytearray(w * h)
    for j in range(h):
        for i in range(w):
            x = (i + shift) % 32
            bx = (x // 16) % 2
            by = (j // 16) % 4
            kind = (bx + by * 2) % 3
            if kind == 0:
                val = cols[(bx + by * 2) % len(cols)]
            elif kind == 1:
                val = cols[0] if (((x // 4) + (j // 4)) & 1) == 0 else cols[2]
            else:
                val = cols[1] if (j % 16) < 8 else cols[3]
            y[j * w + i] = val
    u = bytes([96]) * ((w // 2) * (h // 2))
    v = bytes([160]) * ((w // 2) * (h // 2))
    return y, u, v

write("scc64.y4m", 64, 64, [scc_frame(64, 64)])
write("scc64_4f.y4m", 64, 64, [scc_frame(64, 64, t * 2) for t in range(4)])
write("scc360.y4m", 640, 360, [scc_frame(640, 360)])
write("scc360_4f.y4m", 640, 360,
      [scc_frame(640, 360, t * 2) for t in range(4)])
PY

DEFAULT64_1_PIN=0bd528e1e7896cce1e1e1802f1155a15
DEFAULT360_1_PIN=93655a2d5bbf07ed13ee6fe3e3a25c20
PAL64_PIN=f172516a575a626a7c36d8a1f76c468f
IBC64_PIN=23b8f93c1e061e6d9289074c30664ae8
BOTH64_PIN=305312a8dbb3487b336677a1d18a0ffb
BOTH64_FULL_PIN=305312a8dbb3487b336677a1d18a0ffb
PAL360_PIN=5a79c9d8076f04c35c4e90b8a5f1b6db
IBC360_PIN=bbee14f71d1e453ef8d5247962674c88
BOTH360_PIN=cce44dd8f011babbd5d7f777a5ec81b5
BOTH360_FULL_PIN=cce44dd8f011babbd5d7f777a5ec81b5
BOTH360_4F_PIN=a465b846e201521368e9c0e602910f12
PAL_MORPH360_PIN=0c3de5aa02e135028ef9e2cdd33e0ddc
BOTH_MORPH360_PIN=0c3de5aa02e135028ef9e2cdd33e0ddc
BOTH_FILTERS360_4F_PIN=6546d6802b8beb2652cfd0ca4d21cb3e

# run_case <name> <fixture> <expect_pin> <frames> <min_palette> <min_ibc>
#          <args...>
run_case() {
  local name=$1 fixture=$2 pin=$3 frames=$4 min_pal=$5 min_ibc=$6
  shift 6
  local base="$OUT/$name"
  if ! "$BIN" "$OUT/$fixture" -o "$base.ivf" \
        --video --video-recon "$base.enc.y4m" --video-trace "$@" \
        >"$base.log" 2>&1; then
    fail "$name: encode failed"; tail -3 "$base.log" >&2; return
  fi
  local md5
  md5=$(md5_of "$base.ivf")
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
  local pal ibc
  pal=$(grep -c 'm2 palette' "$base.log" || true)
  ibc=$(grep -c 'm2 ibc' "$base.log" || true)
  [ "$pal" -ge "$min_pal" ] && pass "$name: $pal palette leaf/leaves" ||
    fail "$name: $pal palette leaves, expected >= $min_pal"
  [ "$ibc" -ge "$min_ibc" ] && pass "$name: $ibc IBC leaf/leaves" ||
    fail "$name: $ibc IBC leaves, expected >= $min_ibc"
}

# Default streams must stay byte-identical.
run_case default64 scc64.y4m "$DEFAULT64_1_PIN" 1 0 0 --frames 1
run_case default360 scc360.y4m "$DEFAULT360_1_PIN" 1 0 0 --frames 1

# 64x64: palette on/off, IBC on/off, both, both + full intra search.  A 64x64
# frame is one superblock, so global IBC has no legal BV (0 leaves, asserted).
run_case pal64 scc64.y4m "$PAL64_PIN" 1 1 0 --frames 1 --video-palette
run_case ibc64 scc64.y4m "$IBC64_PIN" 1 0 0 --frames 1 --video-ibc
run_case both64 scc64.y4m "$BOTH64_PIN" 1 1 0 --frames 1 \
  --video-palette --video-ibc
run_case both64_full scc64.y4m "$BOTH64_FULL_PIN" 1 1 0 --frames 1 \
  --video-palette --video-ibc --video-intra-full

# 640x360: palette on/off, IBC on/off, both, both + full intra search.  IBC is
# exercised on the keyframe (>= 600 leaves).
run_case pal360 scc360.y4m "$PAL360_PIN" 1 1 0 --frames 1 --video-palette
run_case ibc360 scc360.y4m "$IBC360_PIN" 1 0 600 --frames 1 --video-ibc
run_case both360 scc360.y4m "$BOTH360_PIN" 1 1 600 --frames 1 \
  --video-palette --video-ibc
run_case both360_full scc360.y4m "$BOTH360_FULL_PIN" 1 1 600 --frames 1 \
  --video-palette --video-ibc --video-intra-full

# Four-frame key+P chain: the tools are keyframe-only, so the P frames must
# carry the same screen-content header bits without palette/IBC leaves.
run_case both360_4f scc360_4f.y4m "$BOTH360_4F_PIN" 4 1 600 --frames 4 \
  --video-palette --video-ibc

# Composition with the morph carrier (§5.20.5.4 morph_pred) and with every
# opt-in loop filter.  The IBC skip leaf must stay out of CDEF/LR exactly as
# the decoder derives Skips[]/LrTxSkip, and the CCSO unit symbol is coded
# independently of the CDEF skip early return.
run_case pal_morph360 scc360.y4m "$PAL_MORPH360_PIN" 1 1 0 --frames 1 \
  --video-palette --video-morph
run_case both_morph360 scc360.y4m "$BOTH_MORPH360_PIN" 1 1 600 --frames 1 \
  --video-palette --video-ibc --video-morph
run_case both_filters360_4f scc360_4f.y4m "$BOTH_FILTERS360_4F_PIN" 4 1 600 \
  --frames 4 --video-palette --video-ibc --video-deblock --video-cdef \
  --video-ccso --video-wiener --video-chroma-lr --video-gdf

# ---------------------------------------------------------------------------
# Named rejections: a composition that is not decoder-exact must be refused,
# never silently ignored.
# ---------------------------------------------------------------------------
reject_case() {
  local name=$1 expect=$2
  shift 2
  local log="$OUT/reject_$name.log"
  if "$BIN" "$OUT/scc64.y4m" -o "$OUT/reject_$name.ivf" "$@" \
        >"$log" 2>&1; then
    fail "$name: expected a named rejection"
    return
  fi
  grep -q -- "$expect" "$log" && pass "$name: rejected by name" ||
    fail "$name: rejection message did not mention '$expect'"
}

reject_case pal_txsearch "--video-tx-search" \
  --video --frames 1 --video-palette --video-tx-search
reject_case ibc_bframes "plain key+P chain" \
  --video --b-frames 2 --video-ibc
reject_case pal_tiles "single tile" \
  --video --tiles 2x1 --video-palette
reject_case ibc_10bit "8-bit 4:2:0" \
  --video --bit-depth 10 --video-ibc
reject_case ibc_gpu "CPU-only" \
  --video --video-gpu --video-ibc
reject_case pal_cdf "--cdf-continuity" \
  --video --avg-cdf --video-palette
reject_case ibc_drl_always "DRL_REORDER_ALWAYS" \
  --video --frames 4 --video-ibc --video-drl-reorder-always
reject_case ibc_segmentation "segmentation" \
  --video --frames 4 --video-ibc --video-segmentation

if [ "$FAIL" -ne 0 ]; then
  echo "VIDEO SCREEN-CONTENT GATE FAIL" >&2
  exit 1
fi
echo "VIDEO SCREEN-CONTENT GATE PASS"
