#!/usr/bin/env bash
# Product gate: §5.4.6 / §7.12.2.19 DRL reordering on the CPU video path.
#
# The component gate (src/codec/inter/drl_reorder/gate/) pins the transform in
# isolation.  This gate pins the *product* carrier: the sequence header codes
# disable_drl_reorder = 0 and every P picture builds the same RefStackMv order
# the pinned decoder does, so the coded RefMvIdx names the same MV.
#
# Two facts this gate must prove, because the previous attempt failed both:
#
#   1. The transform actually moves candidates.  `--video-trace` prints one
#      `drl reorder mi=...` line per block whose [0, numNearest) prefix the
#      move-to-front changed.  Each ALWAYS case requires at least one.
#   2. recon == decoded, sample for sample, with `dav2d --strict 1`.
#
# DRL_REORDER_CONSTRAINT (--video-drl-reorder) is bit-exact but inert under the
# pinned decoder: dav2d f9dafbc derives its §7.12.2 step-16 high-priority-TMVP
# term without the spec's use_ref_frame_mvs guard, so a single reference within
# two POCs never sorts.  The encoder mirrors that (video_frame.cpp
# drl_constraint_sort_suppressed) and this gate pins the resulting 0-swap
# stream.  DRL_REORDER_ALWAYS (--video-drl-reorder-always) exercises the sort.
#
# The fixture is generated here (a deterministic 640x360 pan) so the pinned
# MD5s do not depend on ignored testdata.
#
# Usage:
#   src/pipeline/video/gate_video_drl_reorder.sh --bin build/av2enc \
#       [--dav2d /path/to/dav2d]
#
# Exit codes: 0 pass, 1 failure, 2 usage, 77 missing prerequisite.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "$0")/../../.." && pwd)
BIN=""
DAV2D="${DAV2D:-}"

while [ $# -gt 0 ]; do
  case "$1" in
    --bin) BIN=$2; shift 2 ;;
    --dav2d) DAV2D=$2; shift 2 ;;
    -h|--help) sed -n '2,35p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "gate_video_drl_reorder: unknown argument $1" >&2; exit 2 ;;
  esac
done

BIN=${BIN:-"$ROOT/build/av2enc"}
[ -x "$BIN" ] || {
  echo "gate_video_drl_reorder: no encoder at $BIN" >&2; exit 2; }
if [ -z "$DAV2D" ]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [ -x "$candidate" ]; then DAV2D=$candidate; break; fi
  done
fi
[ -n "$DAV2D" ] && [ -x "$DAV2D" ] || {
  echo "gate_video_drl_reorder: SKIP (no pinned dav2d; set DAV2D)" >&2
  exit 77; }
command -v python3 >/dev/null 2>&1 || {
  echo "gate_video_drl_reorder: SKIP (python3 required)" >&2; exit 77; }

TMP="$(mktemp -d "${TMPDIR:-/tmp}/gate_drl_reorder.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT

# Deterministic 640x360 fixture: a smooth sinusoid panned by (3,1) per frame,
# so neighbouring blocks carry distinct non-zero MVs and the weight stack has
# ties the move-to-front has to resolve.
python3 - "$TMP/pan6.y4m" <<'PY'
import math
import sys

w, h = 640, 360
cw, ch = w // 2, h // 2


def sample(x, y, f, plane):
    xx, yy = x + 3 * f, y + f
    if plane == 0:
        v = (128 + 60 * math.sin(xx * 0.05) + 40 * math.cos(yy * 0.08) +
             20 * math.sin((xx + yy) * 0.03))
    elif plane == 1:
        v = (128 + 40 * math.sin((x + 2 * f) * 0.04) +
             20 * math.cos((y + f) * 0.05))
    else:
        v = (128 + 35 * math.cos((x + 2 * f) * 0.06) +
             25 * math.sin((y + f) * 0.04))
    return max(0, min(255, int(v + 0.5)))


with open(sys.argv[1], "wb") as out:
    out.write(b"YUV4MPEG2 W%d H%d F30:1 Ip A1:1 C420\n" % (w, h))
    for f in range(6):
        out.write(b"FRAME\n")
        for y in range(h):
            out.write(bytes(sample(x, y, f, 0) for x in range(w)))
        for plane in (1, 2):
            for y in range(ch):
                out.write(bytes(sample(2 * x, 2 * y, f, plane)
                                for x in range(cw)))
PY

fail() { echo "gate_video_drl_reorder: FAIL: $*" >&2; exit 1; }

md5_of() {
  if command -v md5sum >/dev/null 2>&1; then
    md5sum "$1" | cut -d' ' -f1
  else
    md5 -q "$1"
  fi
}

# name | mode | extra encoder args | expected MD5 | expected swaps | min swaps
CASES=(
  "d3-constraint|--video-drl-reorder|--video-max-depth 3|7c70eb490e8ab43191ea3b1d7567959b|0|0"
  "d3-always|--video-drl-reorder-always|--video-max-depth 3|a17a1c05c32f9181f85a39cd93ee2d6a|18|1"
  "d4-always|--video-drl-reorder-always|--video-max-depth 4 --video-min-block 8|41f2ae883cf0a0147570e51142a0d3ff|74|1"
)

for spec in "${CASES[@]}"; do
  IFS='|' read -r name flag depth md5 want_swaps min_swaps <<<"$spec"
  ivf="$TMP/$name.ivf"
  recon="$TMP/$name.recon.y4m"
  dec="$TMP/$name.dec.y4m"
  trace="$TMP/$name.trace"
  # shellcheck disable=SC2086
  "$BIN" "$TMP/pan6.y4m" -o "$ivf" --video --frames 6 $flag $depth \
      --video-recon "$recon" --video-trace >/dev/null 2>"$trace" || \
      fail "$name: encode failed"

  got=$(md5_of "$ivf")
  [ "$got" = "$md5" ] || fail "$name: IVF MD5 $got != $md5"

  swaps=$(grep -c 'drl reorder mi=' "$trace" || true)
  [ "$swaps" -ge "$min_swaps" ] || \
      fail "$name: reorder moved $swaps candidate(s), want >= $min_swaps" \
           "(the transform is not being exercised)"
  if [ "$want_swaps" -ge 0 ] && [ "$swaps" -ne "$want_swaps" ]; then
    fail "$name: reorder moved $swaps candidate(s), pinned $want_swaps;" \
         "if the decoder was re-pinned, re-check the step-16 gate"
  fi

  "$DAV2D" --strict 1 --threads 1 -i "$ivf" -o "$dec" \
      >"$TMP/$name.dec.log" 2>&1 || fail "$name: dav2d --strict 1 failed"
  decoded=$(sed -n 's/.*Decoded \([0-9]*\)\/\([0-9]*\) frames.*/\1\/\2/p' \
      "$TMP/$name.dec.log" | tail -1)
  [ "$decoded" = "6/6" ] || \
      fail "$name: decoder emitted $decoded, expected 6/6 pictures"
  cmp -s "$dec" "$recon" || fail "$name: recon is not byte-exact"
  echo "PASS $name: $swaps reorder(s), $decoded pictures, recon exact"
done

echo "PASS gate_video_drl_reorder"
