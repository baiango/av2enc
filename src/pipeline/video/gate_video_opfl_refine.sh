#!/usr/bin/env bash
# Product gate: the two temporal refinement tools on the NEAR_NEARMV compound
# carrier.
#
#   --video-opfl       §7.13.3.8-.12 optical-flow refinement
#   --video-refine-mv  §7.13.3.3-.6 refine-MV
#
# Both run on a B picture (--b-frames) with a past and a future reference, so
# §5.20.7.6 `use_optflow` is codable and §5.20.7.17 derives `use_refinemv` for
# a NEAR_NEARMV block.  The gate pins the deterministic pan fixture (a smooth
# sinusoid panned by (3,1) per frame, so the coded MVs are integer in luma but
# fractional in chroma), a pinned IVF MD5 per case, the decoded picture count,
# and `--video-recon` byte-for-byte against `dav2d --strict 1 --threads 1`.
#
# Coverage:
#   * 16x32/32x16 leaves: several 16x16 refine-MV sub-blocks (§7.13.3.1) and
#     several 8x8/4x4 OPFL sub-blocks (§7.13.3.8) per coding block.
#   * `--video-max-depth 5 --video-min-block 8`: 8x16/16x8 leaves at a non-16
#     ("non-strided") origin, i.e. tw/th = 8 in §7.13.3.1 and 4x4 chroma
#     sub-blocks in §7.13.3.8.
#
# Usage:
#   src/pipeline/video/gate_video_opfl_refine.sh --bin build/av2enc \
#       [--dav2d /path/to/dav2d]
#
# Exit codes: 0 pass, 1 failure, 2 usage, 77 missing fixture/decoder or the
# flags are still rejected by name (the gate then has nothing to verify).
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "$0")/../../.." && pwd)
BIN=""
DAV2D="${DAV2D:-}"

while [ $# -gt 0 ]; do
  case "$1" in
    --bin) BIN=$2; shift 2 ;;
    --dav2d) DAV2D=$2; shift 2 ;;
    -h|--help) sed -n '2,28p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "gate_video_opfl_refine: unknown argument $1" >&2; exit 2 ;;
  esac
done

BIN=${BIN:-"$ROOT/build/av2enc"}
[ -x "$BIN" ] || { echo "gate_video_opfl_refine: no encoder at $BIN" >&2; exit 2; }
if [ -z "$DAV2D" ]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [ -x "$candidate" ]; then DAV2D=$candidate; break; fi
  done
fi
[ -n "$DAV2D" ] && [ -x "$DAV2D" ] || {
  echo "gate_video_opfl_refine: SKIP (no pinned dav2d; set DAV2D)" >&2; exit 77; }
command -v python3 >/dev/null 2>&1 || {
  echo "gate_video_opfl_refine: SKIP (python3 required)" >&2; exit 77; }

TMP="$(mktemp -d "${TMPDIR:-/tmp}/gate_opfl_refine.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT

# Deterministic fixture: the same generator as gate_video_near_near.sh.
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

fail() { echo "gate_video_opfl_refine: FAIL: $*" >&2; exit 1; }

fixture_md5() {
  if command -v md5sum >/dev/null 2>&1; then
    md5sum "$1" | cut -d' ' -f1
  else
    md5 -q "$1"
  fi
}

# The fixture itself is pinned: a fixture drift would silently move every
# stream MD5 below.
[ "$(fixture_md5 "$TMP/pan3.y4m")" = "7c1f3b97efa556307f1458bc31fb0a47" ] || \
  fail "pan3.y4m fixture MD5 drifted"
[ "$(fixture_md5 "$TMP/pan5.y4m")" = "b51c8933199775353eb40db9a6e89ea3" ] || \
  fail "pan5.y4m fixture MD5 drifted"

# The gate only has something to verify once the flags are enabled; while a
# mode is still rejected by name, report a skip instead of a spurious failure.
"$BIN" "$TMP/pan3.y4m" -o "$TMP/probe.ivf" --video --b-frames 1 \
    --video-compound --video-compound-near-near --video-opfl \
    >/dev/null 2>"$TMP/probe.err" || {
  if grep -q "not decoder-exact" "$TMP/probe.err"; then
    echo "gate_video_opfl_refine: SKIP (--video-opfl still rejected by name)" >&2
    exit 77
  fi
  cat "$TMP/probe.err" >&2
  fail "probe encode failed"
}

# name | input | bframes | depth | min-block | flag | expected IVF MD5
#   o*: --video-opfl,  r*: --video-refine-mv
#   *s: --video-min-block 8 (8x16/16x8 leaves at a non-16-aligned origin)
CASES=(
  "o3|pan3.y4m|1|3|16|--video-opfl|5620b7d2f86735a47463385e0595f5db"
  "r3|pan3.y4m|1|3|16|--video-refine-mv|3aecd0f5b58b59ad56a9c1b8d2352272"
  "o5|pan5.y4m|3|3|16|--video-opfl|520800817c0c6c520385467433a571b7"
  "r5|pan5.y4m|3|3|16|--video-refine-mv|2b9264866e6a9c02e01319448af292fe"
  "o3s|pan3.y4m|1|5|8|--video-opfl|4838386129aa8df393a65a5d9f4f7919"
  "r3s|pan3.y4m|1|5|8|--video-refine-mv|ab0bd1c4625a648525b299f84a8a34cc"
)

for spec in "${CASES[@]}"; do
  IFS='|' read -r name input bframes depth minblock flag md5 <<<"$spec"
  ivf="$TMP/$name.ivf"
  recon="$TMP/$name.recon.y4m"
  dec="$TMP/$name.dec.y4m"
  trace="$TMP/$name.trace"
  "$BIN" "$TMP/$input" -o "$ivf" --video --b-frames "$bframes" \
      --video-compound --video-compound-near-near "$flag" \
      --video-max-depth "$depth" --video-min-block "$minblock" \
      --video-recon "$recon" --video-trace \
      >/dev/null 2>"$trace" || fail "$name: encode failed"

  got=$(md5sum "$ivf" | cut -d' ' -f1)
  [ "$got" = "$md5" ] || fail "$name: IVF MD5 $got != $md5"

  # At least one *selected* leaf must use the refinement.
  case "$flag" in
    --video-opfl) used=$(grep -c '^m2 inter.*optflow=1' "$trace" || true) ;;
    *) used=$(grep -c '^m2 inter.*refinemv=1' "$trace" || true) ;;
  esac
  [ "$used" -ge 1 ] || fail "$name: no block selected the $flag mode"

  # The split cases must exercise a leaf whose origin is not a multiple of 16
  # (tw/th = 8 in the §7.13.3.1 sub-block split).
  if [ "$minblock" -eq 8 ]; then
    unaligned=$(awk '/^m2 inter/ {
        r = substr($3, 3); c = substr($4, 3);
        if (((r * 4) % 16) != 0 || ((c * 4) % 16) != 0) n++
      } END { print n + 0 }' "$trace")
    [ "$unaligned" -ge 1 ] || \
      fail "$name: no non-16-aligned leaf (non-strided case not exercised)"
  fi

  "$DAV2D" --strict 1 --threads 1 -i "$ivf" -o "$dec" \
      >"$TMP/$name.dec.log" 2>&1 || fail "$name: dav2d --strict 1 failed"
  decoded=$(sed -n 's/.*Decoded \([0-9]*\)\/\([0-9]*\) frames.*/\1\/\2/p' \
      "$TMP/$name.dec.log" | tail -1)
  want="$((bframes + 2))"
  [ "$decoded" = "$want/$want" ] || \
    fail "$name: decoder emitted $decoded, expected $want/$want"
  cmp -s "$dec" "$recon" || fail "$name: recon is not byte-exact"
  echo "PASS $name: $used $flag block(s), $decoded pictures, recon exact"
done

# The device path has no CPU refinement and must keep rejecting both flags.
# A CPU-only build stops at "--video-gpu requires a CUDA build" before the
# per-tool rejection, so only the CUDA message is matched by name.
for flag in --video-opfl --video-refine-mv; do
  if "$BIN" "$TMP/pan3.y4m" -o "$TMP/gpu.ivf" --video-gpu --b-frames 1 \
      --video-compound --video-compound-near-near "$flag" \
      >/dev/null 2>"$TMP/gpu.err"; then
    fail "$flag: --video-gpu accepted a CPU-only refinement"
  fi
  if grep -q "requires a CUDA build" "$TMP/gpu.err"; then
    continue  # CPU-only build: no device path to reject it.
  fi
  grep -q -- "$flag" "$TMP/gpu.err" || \
    fail "$flag: --video-gpu rejection does not name the flag"
done

echo "PASS gate_video_opfl_refine"
