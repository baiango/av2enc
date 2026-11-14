#!/usr/bin/env bash
# Decoder gate for the remaining CPU `--video` sequence-header surface:
#
#   --video-scc-force          §5.4.7 forced screen-content form
#                              (seq_choose_screen_content_tools = 0,
#                              seq_force_screen_content_tools = 1)
#   --video-force-integer-mv   §5.4.7 seq_force_integer_mv = 1, so every frame
#                              derives FrameMvPrecision = one pel and codes no
#                              precision pair
#   --video-reduce-pb-aspect   §5.4.3 reduce_pb_aspect_ratio + the 2:1/4:1
#                              MaxPbAspectRatio partition decision
#   --video-seq-seg-info       §5.4.4 seq_seg_info_present_flag = 1 with the
#                              sequence segment table and frame reuse_seg_info
#
# Every case is decoded with the pinned `dav2d --strict 1 --threads 1`: the
# picture count must match and the decoded Y4M must equal `--video-recon`
# byte for byte.  The fixtures are deterministic and generated here, so the
# pinned IVF MD5s reproduce on any CPU build.
#
#   scc64     64x64, 32-pixel horizontal period, few colours.
#   scc360    640x360, the same period; the keyframe is split by the ordinary
#             partition tree.
#   *_4f      four pictures: key + three P frames, so the inter-frame header
#             and the one-pel MVD path are exercised.
#
# Trace checks guard against a PASS on a stream that never signals the
# feature: the force-integer case must choose at least one NEWMV block and
# every traced MVD component must be a multiple of 8 (one-pel units).
#
# Usage:
#   src/pipeline/video/gate_video_seq_surface.sh [--bin build/av2enc] [--dav2d PATH]
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
    *) echo "gate_video_seq_surface: unknown argument $1" >&2; exit 2 ;;
  esac
done
if [ ! -x "$BIN" ]; then
  echo "gate_video_seq_surface: missing encoder $BIN" >&2
  exit 77
fi
if [ -z "$DAV2D" ]; then
  for c in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [ -x "$c" ]; then DAV2D=$c; break; fi
  done
fi
if [ -z "$DAV2D" ] || [ ! -x "$DAV2D" ]; then
  echo "gate_video_seq_surface: pinned dav2d not found (set DAV2D)" >&2
  exit 77
fi
command -v python3 >/dev/null || {
  echo "gate_video_seq_surface: python3 is required for the fixtures" >&2
  exit 77
}

OUT="${OUT:-${TMPDIR:-/tmp}/video_seq_surface_gate}"
mkdir -p "$OUT"
FAIL=0
pass() { echo "PASS: $*"; }
fail() { echo "FAIL: $*" >&2; FAIL=1; }
md5_of() { md5sum "$1" | cut -d' ' -f1; }

# ---------------------------------------------------------------------------
# Deterministic fixtures (same generator as gate_video_scc.sh).
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
write("scc360.y4m", 640, 360, [scc_frame(640, 360)])
write("scc360_4f.y4m", 640, 360,
      [scc_frame(640, 360, t * 2) for t in range(4)])
PY

# Default streams must stay byte-identical (same pins as gate_video_scc.sh).
DEFAULT64_1_PIN=0bd528e1e7896cce1e1e1802f1155a15
DEFAULT360_1_PIN=93655a2d5bbf07ed13ee6fe3e3a25c20
# SELECT-form palette/IBC pins from gate_video_scc.sh: the forced form must
# produce a different stream.
SELECT_PAL64_PIN=f172516a575a626a7c36d8a1f76c468f
SELECT_IBC64_PIN=23b8f93c1e061e6d9289074c30664ae8

# Placeholders: filled in below with the pinned values.
SCC_PAL64_PIN=87b3aea9ec57a888e0ebbe1186f3cc0b
SCC_IBC64_PIN=a064f3d53a57555a095e624fcf980ee6
SCC_BOTH64_PIN=d8baf0ef201424b827473f11d1d76bea
SCC_MORPH64_PIN=de5d3d4ac7709a4d8f6b01377982bee4
SCC_PAL360_PIN=6f345a2a565e37a40b3aaf7000634628
SCC_IBC360_PIN=084f9774a0afe3388cf226b03d8fd502
SCC_BOTH360_PIN=62a11db6114a69598ede7f54432bbb46
SCC_MORPH360_PIN=3fb73216fccb6ac52cdb6ead103cc234
SCC_BOTH360_4F_PIN=0e32d11a4d9b584c77fefa778140dffd
FI_PAL64_PIN=26ec742ee71473f171bccf30be7c5f73
FI_IBC64_PIN=05146f8219356342c0de848e06e20028
FI_BOTH64_PIN=e9cb05d40af3d8e11ccb764eda4f0746
FI_MORPH64_PIN=c99ae94c6a573eddc98099dd19e43c25
FI_PAL360_PIN=7c7f47a6b70d235a953d5e3a77c0477e
FI_IBC360_PIN=740eb09eae04ff1bbed5cedcc1d90c40
FI_BOTH360_PIN=ac726991e597eff3d0c59021b252cc75
FI_MORPH360_PIN=9fdb668b5b818c8d36bf65a46a4262e9
FI_BOTH360_4F_PIN=92f46721f11ae211f40fd4efae237bab
RPB64_PIN=e80e10f6f9255730d6ab73fac88c6636
RPB2_64_PIN=f4c2d731cc83b8643ee09de39fd17594
RPB360_PIN=90989e42d332095285445ae9e9e71e6d
RPB2_360_PIN=bdd74d894a7a67860ff9dca2c29e400a
RPB360_4F_PIN=508e62ee7803253edeed5fac6dce3b34
RPB_PAL360_PIN=fad365c42cf1499d1fe875520c227e90
RPB_IBC360_PIN=d943157f802feb555e1aacab74551737
RPB_BOTH360_4F_PIN=ae7a540ed0e74ebd1c9f90a3353e71f3
SEGSI360_4F_PIN=dbf416afb2dbdd5ab9f6af613e55467b
SEGSIF360_4F_PIN=d0084702354632b9db5a553410416cf0
SEGSI_SKIP360_4F_PIN=366568faef7b3d1e97d4fb35032a95d8
SEGSI_EXT360_4F_PIN=b67ba30b50b6d9298b8a7be5e8ea8de6
SEGSI_PAL360_4F_PIN=20f23911d7aa55b2d5cd7713c74e57d8

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
  if [ "${PRINT_PINS:-0}" = "1" ]; then
    echo "$name=$md5"
    return
  fi
  [ "$md5" = "$pin" ] && pass "$name: IVF MD5 $md5" ||
    fail "$name: IVF MD5 $md5 != $pin"
  if ! "$DAV2D" --strict 1 --threads 1 -i "$base.ivf" \
        -o "$base.dec.y4m" >"$base.dec.log" 2>&1; then
    fail "$name: dav2d --strict 1 rejected the stream"
    tail -3 "$base.dec.log" >&2
    return
  fi
  local n
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

# ---------------------------------------------------------------------------
# Defaults: byte-identical.
# ---------------------------------------------------------------------------
run_case default64 scc64.y4m "$DEFAULT64_1_PIN" 1 0 0 --frames 1
run_case default360 scc360.y4m "$DEFAULT360_1_PIN" 1 0 0 --frames 1

# ---------------------------------------------------------------------------
# §5.4.7 forced screen-content form.
# ---------------------------------------------------------------------------
run_case scc_pal64 scc64.y4m "$SCC_PAL64_PIN" 1 1 0 --frames 1 \
  --video-palette --video-scc-force
run_case scc_ibc64 scc64.y4m "$SCC_IBC64_PIN" 1 0 0 --frames 1 \
  --video-ibc --video-scc-force
run_case scc_both64 scc64.y4m "$SCC_BOTH64_PIN" 1 1 0 --frames 1 \
  --video-palette --video-ibc --video-scc-force
run_case scc_morph64 scc64.y4m "$SCC_MORPH64_PIN" 1 0 0 --frames 1 \
  --video-morph --video-scc-force
run_case scc_pal360 scc360.y4m "$SCC_PAL360_PIN" 1 1 0 --frames 1 \
  --video-palette --video-scc-force
run_case scc_ibc360 scc360.y4m "$SCC_IBC360_PIN" 1 0 600 --frames 1 \
  --video-ibc --video-scc-force
run_case scc_both360 scc360.y4m "$SCC_BOTH360_PIN" 1 1 600 --frames 1 \
  --video-palette --video-ibc --video-scc-force
run_case scc_morph360 scc360.y4m "$SCC_MORPH360_PIN" 1 0 600 --frames 1 \
  --video-morph --video-scc-force
run_case scc_both360_4f scc360_4f.y4m "$SCC_BOTH360_4F_PIN" 4 1 600 \
  --frames 4 --video-palette --video-ibc --video-scc-force

# The forced form is a different sequence/frame header, so it must not
# reproduce the SELECT-form stream.
for case_pin in "scc_pal64:$SCC_PAL64_PIN:$SELECT_PAL64_PIN" \
                "scc_ibc64:$SCC_IBC64_PIN:$SELECT_IBC64_PIN"; do
  IFS=: read -r name want avoid <<<"$case_pin"
  [ "$want" != "$avoid" ] && pass "$name: forced form differs from SELECT" ||
    fail "$name: forced form reused the SELECT stream"
done

# ---------------------------------------------------------------------------
# §5.4.7 seq_force_integer_mv: one-pel inter MVs, no precision pair.
# ---------------------------------------------------------------------------
run_case fi_pal64 scc64.y4m "$FI_PAL64_PIN" 1 1 0 --frames 1 \
  --video-palette --video-scc-force --video-force-integer-mv
run_case fi_ibc64 scc64.y4m "$FI_IBC64_PIN" 1 0 0 --frames 1 \
  --video-ibc --video-scc-force --video-force-integer-mv
run_case fi_both64 scc64.y4m "$FI_BOTH64_PIN" 1 1 0 --frames 1 \
  --video-palette --video-ibc --video-scc-force --video-force-integer-mv
run_case fi_morph64 scc64.y4m "$FI_MORPH64_PIN" 1 0 0 --frames 1 \
  --video-morph --video-scc-force --video-force-integer-mv
run_case fi_pal360 scc360.y4m "$FI_PAL360_PIN" 1 1 0 --frames 1 \
  --video-palette --video-scc-force --video-force-integer-mv
run_case fi_ibc360 scc360.y4m "$FI_IBC360_PIN" 1 0 600 --frames 1 \
  --video-ibc --video-scc-force --video-force-integer-mv
run_case fi_both360 scc360.y4m "$FI_BOTH360_PIN" 1 1 600 --frames 1 \
  --video-palette --video-ibc --video-scc-force --video-force-integer-mv
run_case fi_morph360 scc360.y4m "$FI_MORPH360_PIN" 1 0 600 --frames 1 \
  --video-morph --video-scc-force --video-force-integer-mv
run_case fi_both360_4f scc360_4f.y4m "$FI_BOTH360_4F_PIN" 4 1 600 \
  --frames 4 --video-palette --video-ibc --video-scc-force \
  --video-force-integer-mv

# Trace proof: at least one NEWMV block is chosen (mode=2) and every coded
# MVD component is a multiple of 8 (one-pel units).
fi_trace="$OUT/fi_both360_4f.log"
newmv=$(grep -c 'm2 inter .*mode=2 ' "$fi_trace" || true)
[ "$newmv" -ge 1 ] && pass "force-integer: $newmv NEWMV block(s) chosen" ||
  fail "force-integer: no NEWMV block exercised the one-pel MVD path"
bad_mvd=$(grep -o 'new me=([^)]*) res=([^)]*) mvd=([^)]*)' "$fi_trace" |
  sed -e 's/.*mvd=(//' -e 's/).*//' |
  awk -F, '{ if (($1 % 8) != 0 || ($2 % 8) != 0) print $0 }' | wc -l |
  tr -d ' ')
[ "$bad_mvd" = "0" ] && pass "force-integer: every traced MVD is one-pel" ||
  fail "force-integer: $bad_mvd MVD(s) are not one-pel aligned"

# ---------------------------------------------------------------------------
# §5.4.3 reduce_pb_aspect_ratio and the MaxPbAspectRatio search.
# ---------------------------------------------------------------------------
run_case rpb64 scc64.y4m "$RPB64_PIN" 1 0 0 --frames 1 \
  --video-reduce-pb-aspect
run_case rpb2_64 scc64.y4m "$RPB2_64_PIN" 1 0 0 --frames 1 \
  --video-reduce-pb-aspect --video-pb-aspect-2
run_case rpb360 scc360.y4m "$RPB360_PIN" 1 0 0 --frames 1 \
  --video-reduce-pb-aspect
run_case rpb2_360 scc360.y4m "$RPB2_360_PIN" 1 0 0 --frames 1 \
  --video-reduce-pb-aspect --video-pb-aspect-2
run_case rpb360_4f scc360_4f.y4m "$RPB360_4F_PIN" 4 0 0 --frames 4 \
  --video-reduce-pb-aspect
run_case rpb_pal360 scc360.y4m "$RPB_PAL360_PIN" 1 1 0 --frames 1 \
  --video-reduce-pb-aspect --video-palette
run_case rpb_ibc360 scc360.y4m "$RPB_IBC360_PIN" 1 0 600 --frames 1 \
  --video-reduce-pb-aspect --video-ibc
run_case rpb_both360_4f scc360_4f.y4m "$RPB_BOTH360_4F_PIN" 4 1 600 \
  --frames 4 --video-reduce-pb-aspect --video-palette --video-ibc \
  --video-scc-force --video-force-integer-mv

# The reduced limit must change the coded tree: 4:1, 2:1 and the default must
# be three distinct streams at 640x360.
rpb_default=$(md5_of "$OUT/default360.ivf")
[ "$RPB360_PIN" != "$rpb_default" ] &&
  pass "reduce-pb-aspect: 4:1 stream differs from the default tree" ||
  fail "reduce-pb-aspect: 4:1 stream equals the default tree"
[ "$RPB2_360_PIN" != "$rpb_default" ] &&
  [ "$RPB2_360_PIN" != "$RPB360_PIN" ] &&
  pass "reduce-pb-aspect: 2:1 differs from 4:1 and the default" ||
  fail "reduce-pb-aspect: the 2:1 limit did not change the tree"

# ---------------------------------------------------------------------------
# §5.4.4 seq_seg_info_present_flag with the sequence segment table.
# ---------------------------------------------------------------------------
run_case segsi360_4f scc360_4f.y4m "$SEGSI360_4F_PIN" 4 0 0 --frames 4 \
  --video-segmentation --video-seq-seg-info
run_case segsif360_4f scc360_4f.y4m "$SEGSIF360_4F_PIN" 4 0 0 --frames 4 \
  --video-segmentation --video-seq-seg-info-frozen
run_case segsi_skip360_4f scc360_4f.y4m "$SEGSI_SKIP360_4F_PIN" 4 0 0 \
  --frames 4 --video-segmentation --video-seg-skip --video-seq-seg-info
run_case segsi_ext360_4f scc360_4f.y4m "$SEGSI_EXT360_4F_PIN" 4 0 0 \
  --frames 4 --video-segmentation --video-segments 12 --video-seq-seg-info
run_case segsi_pal360_4f scc360_4f.y4m "$SEGSI_PAL360_4F_PIN" 4 1 0 \
  --frames 4 --video-segmentation --video-seq-seg-info --video-palette \
  --video-scc-force

# ---------------------------------------------------------------------------
# Named rejections.
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

reject_case scc_force_alone "--video-scc-force needs one of" \
  --video --frames 1 --video-scc-force
reject_case fi_no_scc "--video-force-integer-mv needs --video-scc-force" \
  --video --frames 1 --video-palette --video-force-integer-mv
reject_case fi_mv_precision "drop --video-mv-precision" \
  --video --frames 1 --video-palette --video-scc-force \
  --video-force-integer-mv --video-mv-precision quarter
reject_case fi_flex "--video-flex-mvres" \
  --video --frames 1 --video-palette --video-scc-force \
  --video-force-integer-mv --video-flex-mvres
reject_case fi_gme "--video-gme/--video-warp" \
  --video --frames 1 --video-palette --video-scc-force \
  --video-force-integer-mv --video-gme
reject_case rpb_tiles "gated on a single tile" \
  --video --frames 1 --video-reduce-pb-aspect --tiles 2x1
reject_case rpb_bframes "gated on the plain key+P chain" \
  --video --frames 4 --b-frames 2 --video-reduce-pb-aspect
reject_case rpb_tip "gated on the plain key+P chain" \
  --video --frames 4 --video-tip --video-reduce-pb-aspect
reject_case seqsi_no_seg "needs --video-segmentation" \
  --video --frames 1 --video-seq-seg-info
reject_case seqsi_frozen_skip "frozen is not wired" \
  --video --frames 4 --video-segmentation --video-seg-skip \
  --video-seq-seg-info-frozen
reject_case seqsi_tiles "gated on a single tile" \
  --video --frames 1 --video-segmentation --video-seq-seg-info --tiles 2x1

if [ "$FAIL" -ne 0 ]; then
  echo "VIDEO SEQUENCE-SURFACE GATE FAIL" >&2
  exit 1
fi
echo "VIDEO SEQUENCE-SURFACE GATE PASS"
