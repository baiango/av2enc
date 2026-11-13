#!/usr/bin/env bash
# Product gate for the opt-in M2 loop filters on --video (CPU reference).
#
# For every wired filter this encodes a deterministic 360p clip, requires
#   * a `dav2d --strict 1` decode,
#   * the encoder reconstruction dump to equal the decoded output exactly
#     (the filtered frame is the decoder-visible picture: encoder and decoder
#     must agree on the reconstruction, which is also the next reference), and
#   * the default (filters-off) stream to keep its pinned MD5.
# It also checks that the flag changes the stream (a silently ignored option
# would otherwise pass), and that pinned rejections stay explicit.
#
# Exits 77 (CTest skip) when the pinned decoder or python3 is unavailable.
#
# Usage: gate_video_filters.sh --bin <av2enc> [--dav2d <path>] [--frames N]
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${HERE}/../../.." && pwd)"
BIN=""
FRAMES=6
DAV2D="${DAV2D:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bin) BIN="$2"; shift 2 ;;
    --frames) FRAMES="$2"; shift 2 ;;
    --dav2d) DAV2D="$2"; shift 2 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$BIN" || ! -x "$BIN" ]]; then
  echo "gate_video_filters: missing --bin" >&2
  exit 2
fi

if [[ -z "$DAV2D" ]]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [[ -x "$candidate" ]]; then DAV2D="$candidate"; break; fi
  done
fi
if [[ -z "$DAV2D" || ! -x "$DAV2D" ]]; then
  echo "gate_video_filters: SKIP (no pinned dav2d; set DAV2D)"
  exit 77
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "gate_video_filters: SKIP (python3 not available)"
  exit 77
fi

TMP="$(mktemp -d "${TMPDIR:-/tmp}/gate_video_filters.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT
CLIP="$TMP/clip360.y4m"

# Deterministic textured 360p pan: enough high-frequency detail for CDEF's
# strength search to pick a non-zero set, and a per-frame perturbation so the
# P frames code real inter leaves (some of them skipped, which exercises the
# §7.18.1 Skips[] grid).
python3 - "$CLIP" "$FRAMES" <<'PY'
import sys

path = sys.argv[1]
frames = int(sys.argv[2])
w, h = 640, 360
cw, ch = w // 2, h // 2

def sample(x, y, f, plane):
    sx = 8 + 6 * f
    sy = 12 + 3 * f
    inside = sx <= x < sx + 96 and sy <= y < sy + 80
    if plane == 0:
        if inside:
            return 32 + ((x * 29 + y * 17 + f * 41) & 0xBF)
        return 88 + (((x + 3 * f) * 5 + (y + 2 * f) * 3 +
                      ((x * y + f * 37) & 15)) & 0x7F)
    if plane == 1:
        return 104 + (32 if inside else 0) + ((x + 2 * f + y) & 15)
    return 120 + (24 if inside else 0) + ((y + 3 * f + x) & 15)

with open(path, "wb") as out:
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

fail() {
  echo "gate_video_filters: FAIL: $*" >&2
  exit 1
}

md5_of() {
  if command -v md5sum >/dev/null 2>&1; then
    md5sum "$1" | cut -d' ' -f1
  else
    md5 -q "$1"
  fi
}

# Pinned default stream for this exact synthetic clip at 6 frames.  It is the
# filters-off `--video` output, so any change here means the default path moved.
DEFAULT_PIN="c4608283abb8e0b877d0ef25af67103b"

# Compares an encoder reconstruction dump against the decoded output (luma and
# both chroma planes, every frame).
compare_recon() {
  local recon="$1" decoded="$2" label="$3"
  python3 - "$recon" "$decoded" "$label" <<'PY'
import sys

recon, decoded, label = sys.argv[1], sys.argv[2], sys.argv[3]
w, h = 640, 360
size = w * h + 2 * (w // 2) * (h // 2)

def frames(path):
    data = open(path, "rb").read()
    i = data.index(b"FRAME\n")
    out = []
    while i < len(data):
        assert data[i:i + 6] == b"FRAME\n", (path, data[i:i + 6])
        i += 6
        out.append(data[i:i + size])
        i += size
    return out

a = frames(recon)
b = frames(decoded)
if len(a) != len(b):
    sys.exit("%s: frame count differs (%d vs %d)" % (label, len(a), len(b)))
for index, (x, y) in enumerate(zip(a, b)):
    if x != y:
        diff = sum(1 for u, v in zip(x, y) if u != v)
        sys.exit("%s: frame %d differs in %d samples" % (label, index, diff))
PY
}

run_case() {
  local label="$1"
  shift
  local out="$TMP/${label}.ivf"
  local recon="$TMP/${label}.recon.y4m"
  local dec="$TMP/${label}.dec.y4m"
  "$BIN" "$CLIP" -o "$out" --video --frames "$FRAMES" \
    --video-recon "$recon" "$@" >/dev/null 2>"$TMP/${label}.err" ||
    fail "$label: encode failed ($(cat "$TMP/${label}.err"))"
  "$DAV2D" --strict 1 -q -i "$out" -o "$dec" >/dev/null 2>&1 ||
    fail "$label: dav2d --strict 1 rejected the stream"
  compare_recon "$recon" "$dec" "$label" ||
    fail "$label: encoder reconstruction differs from the decoded output"
  echo "PASS video-filters $label ($(md5_of "$out"))"
}

# 1. Default path: pinned and filters off.
run_case default
DEFAULT_MD5="$(md5_of "$TMP/default.ivf")"
if [[ "$DEFAULT_PIN" != "__DEFAULT_PIN__" && "$DEFAULT_MD5" != "$DEFAULT_PIN" ]]; then
  fail "default stream MD5 moved ($DEFAULT_MD5 != $DEFAULT_PIN)"
fi

# 2. CDEF with one strength set (header-only: no cdef_idx tile symbol).
run_case cdef --video-cdef
CDEF_MD5="$(md5_of "$TMP/cdef.ivf")"
if [[ "$CDEF_MD5" == "$DEFAULT_MD5" ]]; then
  fail "--video-cdef did not change the stream"
fi

# 3. CDEF with a per-64x64 index grid (§5.20.10.1 tile symbols).
run_case cdef-index4 --video-cdef --video-cdef-strength 4
run_case cdef-index4-damping4 --video-cdef-strength 4 --video-cdef-damping 4

# 4. CCSO (§7.19): band-offset only, then combined with CDEF so the
#    CDEF->CCSO order and the pre-CDEF CurrFrame path are both exercised.
run_case ccso --video-ccso
run_case ccso-bands6 --video-ccso --video-ccso-bands 6
run_case cdef-ccso --video-cdef --video-ccso
run_case cdef-index4-ccso2 --video-cdef-strength 4 --video-ccso \
  --video-ccso-bands 2

# 5. §7.20 loop restoration: luma Wiener NS, chroma LR, then the full
#    CDEF -> CCSO -> LR chain.
run_case wiener --video-wiener
run_case chroma-lr --video-chroma-lr
run_case wiener-chroma-lr --video-wiener --video-chroma-lr
run_case lr-chain --video-cdef --video-ccso --video-wiener --video-chroma-lr

# 6. §7.20.5 GDF (luma, header-only) and the complete §7.18 -> §7.20.5 chain.
run_case gdf --video-gdf
run_case gdf-lr --video-wiener --video-chroma-lr --video-gdf
run_case full-chain --video-cdef --video-ccso --video-wiener --video-chroma-lr \
  --video-gdf

# 7. Named rejections must stay explicit rather than silently ignored.
expect_reject() {
  local label="$1"
  shift
  set +e
  "$BIN" "$CLIP" -o "$TMP/reject.ivf" --video --frames "$FRAMES" "$@" \
    >"$TMP/reject.out" 2>"$TMP/reject.err"
  local rc=$?
  set -e
  [[ $rc -ne 0 ]] || fail "$label: expected a non-zero exit, got $rc"
  grep -q -- "video" "$TMP/reject.err" ||
    fail "$label: rejection did not name the video flag"
  echo "PASS video-filters reject $label"
}
# §5.19 tiles: one picture-level filter pass over the assembled reconstruction.
run_case tiles-cdef --video-cdef --tiles 2x1
run_case tiles-gdf --video-gdf --tiles 2x2
# B frames / temporal layers carry the per-frame plan in their headers.
run_case_bframes() {
  local label="$1"
  shift
  local out="$TMP/${label}.ivf"
  local recon="$TMP/${label}.recon.y4m"
  local dec="$TMP/${label}.dec.y4m"
  "$BIN" "$CLIP" -o "$out" --video --frames 4 --video-recon "$recon" "$@"     >/dev/null 2>"$TMP/${label}.err" ||
    fail "$label: encode failed ($(cat "$TMP/${label}.err"))"
  "$DAV2D" --strict 1 -q -i "$out" -o "$dec" >/dev/null 2>&1 ||
    fail "$label: dav2d --strict 1 rejected the stream"
  compare_recon "$recon" "$dec" "$label" ||
    fail "$label: encoder reconstruction differs from the decoded output"
  echo "PASS video-filters $label ($(md5_of "$out"))"
}
run_case_bframes bframes2-cdef --b-frames 2 --video-cdef
run_case_bframes tlayers4-cdef --video-tlayers 4 --video-cdef
# Loop restoration keeps its tile rejection; the frame-type wrapper header
# writers still hard-code the filter rows off.
expect_reject lr-tiles --video-wiener --tiles 2x1
expect_reject wrappers --video-cdef --video-long-term

echo "gate_video_filters: PASS"
