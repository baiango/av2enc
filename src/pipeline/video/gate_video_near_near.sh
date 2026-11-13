#!/usr/bin/env bash
# Product gate: §5.20.7.6 NEAR_NEARMV compound on a two-reference B frame.
#
# This is the decoder-gated carrier for the temporal refinement tools
# (§7.13.3.3-.12): a B picture has one past and one future reference, so
# opfl_allowed_for_refs() is true and a NEAR_NEARMV block can request
# use_optflow / derive use_refinemv.  The carrier itself is gated here; the
# refinement overlays (--video-opfl, --video-refine-mv) have their own gate in
# gate_video_opfl_refine.sh.
#
# Usage:
#   src/pipeline/video/gate_video_near_near.sh --bin build/av2enc \
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
    -h|--help) sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "gate_video_near_near: unknown argument $1" >&2; exit 2 ;;
  esac
done

BIN=${BIN:-"$ROOT/build/av2enc"}
[ -x "$BIN" ] || { echo "gate_video_near_near: no encoder at $BIN" >&2; exit 2; }
if [ -z "$DAV2D" ]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [ -x "$candidate" ]; then DAV2D=$candidate; break; fi
  done
fi
[ -n "$DAV2D" ] && [ -x "$DAV2D" ] || {
  echo "gate_video_near_near: SKIP (no pinned dav2d; set DAV2D)" >&2; exit 77; }
command -v python3 >/dev/null 2>&1 || {
  echo "gate_video_near_near: SKIP (python3 required)" >&2; exit 77; }

TMP="$(mktemp -d "${TMPDIR:-/tmp}/gate_near_near.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT

# Deterministic fixture: a smooth sinusoid panned by (3,1) per frame.  The
# B picture sits halfway between the key and the P anchor, so the compound
# candidate and its Merge neighbours carry real (non-zero) MVs.
python3 - "$TMP" <<'PY'
import math
import sys
root = sys.argv[1]
w, h = 640, 360
cw, ch = w // 2, h // 2

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

for frames, name in ((3, "pan3.y4m"), (5, "pan5.y4m")):
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

fail() { echo "gate_video_near_near: FAIL: $*" >&2; exit 1; }

# name | input | expected frames | depth | expected MD5 | min near-near blocks
CASES=(
  "p3|pan3.y4m|1|3|7c9b6213ee3eb5c16a25fcbe9e799466|1"
  "p5|pan5.y4m|3|3|8baef2a99f13a16b3df21440c678ff41|1"
  "p3d4|pan3.y4m|1|4|e961f4e3af862c7e8cde83e6c83b2f67|1"
)

for spec in "${CASES[@]}"; do
  IFS='|' read -r name input bframes depth md5 min_nn <<<"$spec"
  ivf="$TMP/$name.ivf"
  recon="$TMP/$name.recon.y4m"
  dec="$TMP/$name.dec.y4m"
  trace="$TMP/$name.trace"
  "$BIN" "$TMP/$input" -o "$ivf" --video --b-frames "$bframes" \
      --video-compound --video-compound-near-near \
      --video-max-depth "$depth" --video-recon "$recon" --video-trace \
      >/dev/null 2>"$trace" || fail "$name: encode failed"

  got=$(md5sum "$ivf" | cut -d' ' -f1)
  [ "$got" = "$md5" ] || fail "$name: IVF MD5 $got != $md5"

  nn=$(grep -c "m2 inter.*cm=0" "$trace" || true)
  [ "$nn" -ge "$min_nn" ] || \
      fail "$name: no NEAR_NEARMV block selected (GLOBAL-only stream)"

  "$DAV2D" --strict 1 --threads 1 -i "$ivf" -o "$dec" \
      >"$TMP/$name.dec.log" 2>&1 || fail "$name: dav2d --strict 1 failed"
  decoded=$(sed -n 's/.*Decoded \([0-9]*\)\/\([0-9]*\) frames.*/\1\/\2/p' \
      "$TMP/$name.dec.log" | tail -1)
  want="$((bframes + 2))"
  [ "$decoded" = "$want/$want" ] || \
      fail "$name: decoder emitted $decoded, expected $want/$want"
  cmp -s "$dec" "$recon" || fail "$name: recon is not byte-exact"
  echo "PASS $name: $nn NEAR_NEARMV block(s), $decoded pictures, recon exact"
done

echo "PASS gate_video_near_near"
