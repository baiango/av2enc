#!/usr/bin/env bash
# Product gate: §5.4.3 enable_sdp and §5.4.8 enable_inter_ddt on the CPU
# `--video` writer.
#
# SDP (--video-sdp) codes the intra keyframe as a luma partition tree followed
# by a chroma tree for every 64x64 superblock.  The luma tree uses the normal
# --video-max-depth / --video-min-block search limits; the chroma tree is a
# single PARTITION_NONE block (see src/pipeline/video/README.md for why the
# pinned dav2d cannot gate a decoded chroma tree).  The gate proves:
#   * the stream decodes with `dav2d --strict 1` and the picture count is exact;
#   * the encoder reconstruction is sample-exact against the decoded Y4M;
#   * the tree search is live: depth 0/64 and depth 4/16 produce different
#     streams on the same fixture;
#   * SDP composes with --video-inter-residual and --video-tx-search.
#
# DDT (--video-ddt) is a sequence bit plus §7.15.4.1 kernel substitution, so
# the gate pins:
#   * the flag alone (bit only) decodes exactly;
#   * with --video-inter-residual (and at QP 105, where the residual wins) the
#     encoder reports coded DDT leaves via --video-trace and the reconstruction
#     is still sample-exact;
#   * the same for --video-tx-search and both tools together.
#
# Usage:
#   src/pipeline/video/gate_video_sdp_ddt.sh --bin build-int/av2enc \
#       [--dav2d /path/to/dav2d-macos]
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
    -h|--help) sed -n '2,34p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "gate_video_sdp_ddt: unknown argument $1" >&2; exit 2 ;;
  esac
done

BIN=${BIN:-"$ROOT/build/av2enc"}
[ -x "$BIN" ] || {
  echo "gate_video_sdp_ddt: no encoder at $BIN" >&2; exit 2; }
if [ -z "$DAV2D" ]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [ -x "$candidate" ]; then DAV2D=$candidate; break; fi
  done
fi
[ -n "$DAV2D" ] && [ -x "$DAV2D" ] || {
  echo "gate_video_sdp_ddt: SKIP (no pinned dav2d; set DAV2D)" >&2
  exit 77; }
command -v python3 >/dev/null 2>&1 || {
  echo "gate_video_sdp_ddt: SKIP (python3 required)" >&2; exit 77; }

TMP="$(mktemp -d "${TMPDIR:-/tmp}/gate_sdp_ddt.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT

# Deterministic textured two-frame fixtures.  SDP needs non-uniform luma to
# exercise the partition search, and DDT needs real residual coefficients.
python3 - "$TMP" <<'PY'
import math
import os
import sys

out = sys.argv[1]


def write(name, w, h, frames=2):
    cw, ch = w // 2, h // 2
    with open(os.path.join(out, name), "wb") as f:
        f.write(b"YUV4MPEG2 W%d H%d F30:1 Ip A1:1 C420\n" % (w, h))
        for k in range(frames):
            f.write(b"FRAME\n")
            y = bytearray(w * h)
            for r in range(h):
                for c in range(w):
                    v = (128 + 70 * math.sin((c + 3 * k) * 0.11) +
                         30 * math.cos((r + k) * 0.17) +
                         20 * math.sin((r + c) * 0.07))
                    y[r * w + c] = max(0, min(255, int(v + 0.5)))
            f.write(y)
            for plane in (0, 1):
                p = bytearray(cw * ch)
                for r in range(ch):
                    for c in range(cw):
                        if plane == 0:
                            v = 128 + 40 * math.sin((c + 2 * k) * 0.09) + \
                                20 * math.cos(r * 0.13)
                        else:
                            v = 128 + 35 * math.cos((c + 2 * k) * 0.08) + \
                                25 * math.sin(r * 0.11)
                        p[r * cw + c] = max(0, min(255, int(v + 0.5)))
                f.write(p)


write("t64.y4m", 64, 64)
write("t360.y4m", 640, 360)
PY

fail() { echo "gate_video_sdp_ddt: FAIL: $*" >&2; exit 1; }

md5_of() {
  if command -v md5sum >/dev/null 2>&1; then
    md5sum "$1" | cut -d' ' -f1
  else
    md5 -q "$1"
  fi
}

# encode NAME SOURCE FRAMES "ARGS..." -> sets IVF/RECON/TRACE globals
encode_case() {
  local name=$1 src=$2 frames=$3
  shift 3
  IVF="$TMP/$name.ivf"
  RECON="$TMP/$name.recon.y4m"
  TRACE="$TMP/$name.trace"
  # shellcheck disable=SC2086
  "$BIN" "$src" -o "$IVF" --video --frames "$frames" --video-recon "$RECON" \
      --video-trace "$@" >/dev/null 2>"$TRACE" ||
      fail "$name: encode failed"
}

# decode NAME -> requires exact picture count and recon
decode_case() {
  local name=$1 want=$2
  local dec="$TMP/$name.dec.y4m" log="$TMP/$name.dec.log"
  "$DAV2D" --strict 1 --threads 1 -i "$IVF" -o "$dec" >"$log" 2>&1 ||
      fail "$name: dav2d --strict 1 failed: $(tail -1 "$log")"
  local count
  count=$(sed -n 's/.*Decoded \([0-9]*\)\/\([0-9]*\) frames.*/\1\/\2/p' \
      "$log" | tail -1)
  [ "$count" = "$want" ] || \
      fail "$name: decoded $count pictures, expected $want"
  cmp -s "$RECON" "$dec" || fail "$name: recon is not sample-exact"
  echo "PASS $name: $count pictures, recon exact"
}

# --- SDP ------------------------------------------------------------------
encode_case sdp64 "$TMP/t64.y4m" 2 --video-sdp
decode_case sdp64 2/2
sdp64_md5=$(md5_of "$IVF")

encode_case sdp64_none "$TMP/t64.y4m" 2 --video-sdp \
    --video-max-depth 0 --video-min-block 64
decode_case sdp64_none 2/2
sdp64_none_md5=$(md5_of "$IVF")
[ "$sdp64_md5" != "$sdp64_none_md5" ] || \
    fail "sdp64: luma tree search did not move the stream (depth 0 == depth 4)"

encode_case sdp360 "$TMP/t360.y4m" 2 --video-sdp
decode_case sdp360 2/2

encode_case sdp360_d3 "$TMP/t360.y4m" 2 --video-sdp \
    --video-max-depth 3 --video-min-block 8
decode_case sdp360_d3 2/2

encode_case sdp360_res "$TMP/t360.y4m" 2 --video-sdp \
    --video-inter-residual
decode_case sdp360_res 2/2

encode_case sdp360_tx "$TMP/t360.y4m" 2 --video-sdp --video-tx-search
decode_case sdp360_tx 2/2

# --- DDT ------------------------------------------------------------------
encode_case ddt64 "$TMP/t64.y4m" 2 --video-ddt
decode_case ddt64 2/2

encode_case ddt360 "$TMP/t360.y4m" 2 --video-ddt
decode_case ddt360 2/2

# QP 60/90 make the inter residual win, so the DDT kernels are actually used.
encode_case ddt64_res "$TMP/t64.y4m" 2 --qp 60 --video-ddt \
    --video-inter-residual
decode_case ddt64_res 2/2
coded=$(grep -c 'm2 inter ddt residual' "$TRACE" || true)
[ "$coded" -ge 1 ] || \
    fail "ddt64_res: no coded DDT leaf (the gate would not exercise DDT)"
echo "PASS ddt64_res: $coded coded DDT leaf/leaves"

# --video-tx-search alone has no inter residual path to search, so it only
# proves the sequence bit and intra TX search stay decodable.
encode_case ddt64_tx "$TMP/t64.y4m" 2 --qp 60 --video-ddt \
    --video-tx-search
decode_case ddt64_tx 2/2

encode_case ddt64_both "$TMP/t64.y4m" 2 --qp 60 --video-ddt \
    --video-inter-residual --video-tx-search
decode_case ddt64_both 2/2
coded=$(grep -c 'm2 inter ddt residual' "$TRACE" || true)
[ "$coded" -ge 1 ] || \
    fail "ddt64_both: no coded DDT leaf (the gate would not exercise DDT)"
echo "PASS ddt64_both: $coded coded DDT leaf/leaves"

encode_case ddt360_res "$TMP/t360.y4m" 2 --qp 105 --video-ddt \
    --video-inter-residual
decode_case ddt360_res 2/2
coded=$(grep -c 'm2 inter ddt residual' "$TRACE" || true)
[ "$coded" -ge 1 ] || \
    fail "ddt360_res: no coded DDT leaf (the gate would not exercise DDT)"
echo "PASS ddt360_res: $coded coded DDT leaf/leaves"

encode_case ddt360_tx "$TMP/t360.y4m" 2 --qp 90 --video-ddt \
    --video-inter-residual --video-tx-search
decode_case ddt360_tx 2/2
coded=$(grep -c 'm2 inter ddt residual' "$TRACE" || true)
[ "$coded" -ge 1 ] || \
    fail "ddt360_tx: no coded DDT leaf (the gate would not exercise DDT)"
echo "PASS ddt360_tx: $coded coded DDT leaf/leaves"

# Both tools on one stream: SDP keyframe + DDT inter residual.
encode_case sdp_ddt "$TMP/t360.y4m" 2 --qp 105 --video-sdp --video-ddt \
    --video-inter-residual
decode_case sdp_ddt 2/2

echo "PASS gate_video_sdp_ddt"
