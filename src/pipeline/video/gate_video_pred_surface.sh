#!/usr/bin/env bash
# Product gate: the compound prediction-surface overlays and DF sub-PU.
#
#   --video-cwp-weighted  §5.4.6 enable_cwp + §5.20.7.6 cwp_idx and the
#                         non-equal Cwp_Weighting_Factor search on a
#                         NEAR_NEARMV COMPOUND_AVERAGE block.
#   --video-imp-mask      §5.4.6 enable_imp_msk_bld + §7.13.3.14 get_mask()
#                         on the eligible CWP_EQUAL average.
#   --video-df-sub-pu     §5.4.6 enable_df_sub_pu + §5.18.7.9
#                         allow_df_sub_pu and the §7.13.3.15 sub-PU grid in
#                         §7.17 deblocking (exercised with --video-opfl).
#
# The carrier is the two-reference B-frame NEAR_NEARMV compound path
# (--video-compound --video-compound-near-near --b-frames) for CWP, and the
# ordinary key+P/B tile chain for DF sub-PU.  Every case decodes with the pinned
# dav2d --strict 1, checks the picture count and compares --video-recon
# byte-for-byte with the decoded Y4M.
#
# Usage:
#   src/pipeline/video/gate_video_pred_surface.sh --bin build/av2enc \
#       [--dav2d /path/to/dav2d]
#
# Exit codes: 0 pass, 1 failure, 2 usage, 77 missing fixture/decoder.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "$0")/../../.." && pwd)
BIN=""
DAV2D="${DAV2D:-}"

while [ $# -gt 0 ]; do
  case "$1" in
    --bin) BIN=$2; shift 2 ;;
    --dav2d) DAV2D=$2; shift 2 ;;
    -h|--help) sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "gate_video_pred_surface: unknown argument $1" >&2; exit 2 ;;
  esac
done

BIN=${BIN:-"$ROOT/build/av2enc"}
[ -x "$BIN" ] || { echo "gate_video_pred_surface: no encoder at $BIN" >&2; exit 2; }
if [ -z "$DAV2D" ]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [ -x "$candidate" ]; then DAV2D=$candidate; break; fi
  done
fi
[ -n "$DAV2D" ] && [ -x "$DAV2D" ] || {
  echo "gate_video_pred_surface: SKIP (no pinned dav2d; set DAV2D)" >&2; exit 77; }
command -v python3 >/dev/null 2>&1 || {
  echo "gate_video_pred_surface: SKIP (python3 required)" >&2; exit 77; }

TMP="$(mktemp -d "${TMPDIR:-/tmp}/gate_pred_surface.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT

# Deterministic fixtures: a smooth sinusoid panned by (3,1) per frame (the
# gate_video_near_near.sh generator) at 64x64 (3 frames) and 640x360 (5).
python3 - "$TMP" <<'PY'
import math
import sys
root = sys.argv[1]

def sample(x, y, f, plane):
    xx, yy = x + 3 * f, y + f
    if plane == 0:
        v = 128 + 60 * math.sin(xx * 0.05) + 40 * math.cos(yy * 0.08) + \
            20 * math.sin((xx + yy) * 0.03)
    elif plane == 1:
        v = 128 + 40 * math.sin((x + 2 * f) * 0.04) + \
            20 * math.cos((y + f) * 0.05)
    else:
        v = 128 + 35 * math.cos((x + 2 * f) * 0.06) + \
            25 * math.sin((y + f) * 0.04)
    return max(0, min(255, int(v + 0.5)))

for w, h, frames, name in ((64, 64, 3, "small.y4m"),
                           (640, 360, 5, "pan5.y4m")):
    cw, ch = w // 2, h // 2
    with open("%s/%s" % (root, name), "wb") as out:
        out.write(b"YUV4MPEG2 W%d H%d F30:1 Ip A1:1 C420\n" % (w, h))
        for f in range(frames):
            out.write(b"FRAME\n")
            for y in range(h):
                out.write(bytes(sample(x, y, f, 0) for x in range(w)))
            for plane in (1, 2):
                for y in range(ch):
                    out.write(bytes(sample(2 * x, 2 * y, f, plane)
                                    for x in range(cw)))
PY

fail() { echo "gate_video_pred_surface: FAIL: $*" >&2; exit 1; }

decode_check() {
  local name=$1 ivf=$2 recon=$3 want=$4
  local dec="$TMP/$name.dec.y4m"
  "$DAV2D" --strict 1 --threads 1 -i "$ivf" -o "$dec" \
      >"$TMP/$name.dec.log" 2>&1 || fail "$name: dav2d --strict 1 failed"
  local got
  got=$(sed -n 's/.*Decoded \([0-9]*\)\/\([0-9]*\) frames.*/\1\/\2/p' \
      "$TMP/$name.dec.log" | tail -1)
  [ "$got" = "$want/$want" ] || \
      fail "$name: decoder emitted $got, expected $want/$want"
  cmp -s "$dec" "$recon" || fail "$name: recon is not byte-exact"
}

encode() {
  local name=$1 input=$2 md5=$3 frames=$4; shift 4
  local ivf="$TMP/$name.ivf"
  local recon="$TMP/$name.recon.y4m"
  "$BIN" "$TMP/$input" -o "$ivf" --video --video-recon "$recon" "$@" \
      >"$TMP/$name.enc.log" 2>&1 || fail "$name: encode failed"
  local got
  got=$(md5sum "$ivf" | cut -d' ' -f1)
  [ "$got" = "$md5" ] || fail "$name: IVF MD5 $got != $md5"
  decode_check "$name" "$ivf" "$recon" "$frames"
  echo "PASS $name: $frames pictures, recon exact, IVF $md5"
}

BASE_B=("--b-frames" "3" "--video-compound" "--video-compound-near-near"
        "--video-max-depth" "3")
BASE_S=("--b-frames" "1" "--video-compound" "--video-compound-near-near")

# Defaults: the plain NEAR_NEARMV carrier without any new flag must stay on the
# pinned pre-change stream (this is gate_video_near_near.sh's p5 case).
encode default360 pan5.y4m 8baef2a99f13a16b3df21440c678ff41 5 "${BASE_B[@]}"

encode cwp64 small.y4m ad3ad386e5ff7875f7a660dd22b2198b 3 "${BASE_S[@]}" \
    --video-cwp-weighted
encode cwp360 pan5.y4m 9e7e40db6c2eea0f01b60dffda5ef437 5 "${BASE_B[@]}" \
    --video-cwp-weighted
encode imp64 small.y4m edfff08748ad148d7d7c1eb2cfc4beeb 3 "${BASE_S[@]}" \
    --video-imp-mask
encode imp360 pan5.y4m 3477961530b105969b7b7dac65fc4311 5 "${BASE_B[@]}" \
    --video-imp-mask
encode both360 pan5.y4m cc238f857a86d653b31b8ac15ee8735f 5 "${BASE_B[@]}" \
    --video-cwp-weighted --video-imp-mask

# Non-inert proof for the implicit mask: the derived get_mask() blend changes
# the reconstruction against the equal-weight carrier.
if cmp -s "$TMP/imp360.recon.y4m" "$TMP/default360.recon.y4m"; then
  fail "imp360: recon equals the equal-weight carrier (get_mask not exercised)"
fi
echo "PASS imp360: get_mask changes the reconstruction vs equal-weight CWP"

# DF sub-PU: the plain deblock carrier and an OPFL carrier whose §7.13.3.15
# subblocks are smaller than Max_Tx_Size_Rect, so allow_df_sub_pu actually
# changes §7.17 filtering.
encode dfp360 pan5.y4m 8d3841bb1c5f2ab1e944925a7011300c 5 \
    --video-deblock --video-df-sub-pu --video-max-depth 3
encode dfop64 small.y4m f217c8d400da70e3c2c7628d75174b83 3 "${BASE_S[@]}" \
    --video-opfl --video-deblock --video-df-sub-pu
encode dfop360 pan5.y4m ca9635e8cf4206373595d4337db548ec 5 "${BASE_B[@]}" \
    --video-opfl --video-deblock --video-df-sub-pu
"$BIN" "$TMP/pan5.y4m" -o "$TMP/opdb.ivf" --video --video-deblock \
    --video-recon "$TMP/opdb.recon.y4m" \
    --video-max-depth 3 "${BASE_B[@]}" >/dev/null 2>&1 || \
    fail "opdb: encode failed"
if cmp -s "$TMP/opdb.recon.y4m" "$TMP/dfop360.recon.y4m"; then
  fail "dfop360: recomputed recon equals the allow_df_sub_pu=0 stream"
fi
echo "PASS dfop360: sub-PU edges change the deblocked reconstruction"

echo "PASS gate_video_pred_surface"
