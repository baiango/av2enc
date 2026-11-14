#!/usr/bin/env bash
# Product gate for arbitrary frame dimensions on the CPU video encoder.
#
# Every dimension the format can represent is legal: §5.18.4.4 codes
# MiCols = 2*((W+7)>>3) / MiRows = 2*((H+7)>>3) 4x4 units, reconstructs that
# MI-aligned padded picture, and the output/reconstruction is cropped back to
# the signalled W x H.  For each clip the gate requires
#   * a `dav2d --strict 1` decode,
#   * the decoded picture count to equal the input count,
#   * the decoder's signalled dimensions to equal the real W x H,
#   * the encoder reconstruction to equal the decoded output byte-for-byte
#     (the filtered frame is the decoder-visible picture and the next
#     reference), and
#   * every requested filter to change the stream (a silently ignored flag
#     would otherwise pass).
#
# Matrix:
#   8-bit 4:2:0:  100x70, 130x74, 424x240, 640x362   (all W/H even)
#   8-bit 4:4:4:  100x70, 130x74, 424x240, 640x362, 101x71 (odd W and H)
# Each clip is a 4-frame key+P chain, encoded with and without
# deblock / CDEF / CCSO / Wiener NS LR / chroma LR / GDF and the full chains.
# A vertically moving 100x70 and 640x362 pair additionally exercises
# --video-edge-inter, whose reference pad must clamp at MiRows*4 (not the
# visible height) for heights that are not a multiple of 8.
# The 4:2:2/4:4:4 deblock+chroma-LR pair keeps its named rejection.
#
# Exits 77 (CTest skip) when the pinned decoder or python3 is unavailable.
#
# Usage: gate_video_any_dims.sh --bin <av2enc> [--dav2d <path>] [--frames N]
set -uo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${HERE}/../../.." && pwd)"
BIN=""
FRAMES=4
DAV2D="${DAV2D:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bin) BIN="$2"; shift 2 ;;
    --frames) FRAMES="$2"; shift 2 ;;
    --dav2d) DAV2D="$2"; shift 2 ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$BIN" || ! -x "$BIN" ]]; then
  echo "gate_video_any_dims: missing --bin" >&2
  exit 2
fi

if [[ -z "$DAV2D" ]]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [[ -x "$candidate" ]]; then DAV2D="$candidate"; break; fi
  done
fi
if [[ -z "$DAV2D" || ! -x "$DAV2D" ]]; then
  echo "gate_video_any_dims: SKIP (no pinned dav2d; set DAV2D)"
  exit 77
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "gate_video_any_dims: SKIP (python3 not available)"
  exit 77
fi

TMP="$(mktemp -d "${TMPDIR:-/tmp}/gate_video_any_dims.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT

# Deterministic textured pan: a static high-frequency background plus a small
# moving box, so the keyframe has real residuals and the P frames code inter
# leaves (including skipped ones for the CDEF/LR skip grids).
python3 - "$TMP" "$FRAMES" <<'PY'
import sys

tmp, frames = sys.argv[1], int(sys.argv[2])
cases = [
    (100, 70, "420"),
    (130, 74, "420"),
    (424, 240, "420"),
    (640, 362, "420"),
    (100, 70, "444"),
    (130, 74, "444"),
    (424, 240, "444"),
    (640, 362, "444"),
    (101, 71, "444"),
]

def sample(x, y, f, plane):
    sx, sy = 8 + f, 12
    inside = sx <= x < sx + 24 and sy <= y < sy + 20
    if plane == 0:
        if inside:
            return 32 + ((x * 29 + y * 17) & 0xBF)
        return 88 + ((x * 5 + y * 3 + ((x * y) & 15)) & 0x7F)
    if plane == 1:
        return 104 + (32 if inside else 0) + ((x + y) & 15)
    return 120 + (24 if inside else 0) + ((y + x) & 15)

for w, h, chroma in cases:
    sx, sy = (1, 1) if chroma == "420" else (0, 0)
    cw, ch = w >> sx, h >> sy
    path = "%s/%dx%d_%s.y4m" % (tmp, w, h, chroma)
    with open(path, "wb") as out:
        out.write(("YUV4MPEG2 W%d H%d F30:1 Ip A1:1 C%s\n" %
                   (w, h, chroma)).encode())
        for f in range(frames):
            out.write(b"FRAME\n")
            for y in range(h):
                out.write(bytes(sample(x, y, f, 0) for x in range(w)))
            for plane in (1, 2):
                for y in range(ch):
                    out.write(bytes(sample(x << sx, y << sy, f, plane)
                                    for x in range(cw)))
PY

fail() {
  echo "gate_video_any_dims: FAIL: $*" >&2
  exit 1
}

md5_of() {
  if command -v md5sum >/dev/null 2>&1; then
    md5sum "$1" | cut -d' ' -f1
  else
    md5 -q "$1"
  fi
}

# Validates the decoded Y4M header (real W x H) and the picture count.
check_y4m() {
  local path="$1" w="$2" h="$3" frames="$4" label="$5"
  python3 - "$path" "$w" "$h" "$frames" "$label" <<'PY'
import sys

path, w, h, frames, label = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), \
    int(sys.argv[4]), sys.argv[5]
data = open(path, "rb").read()
end = data.index(b"\n") + 1
tokens = data[:end].decode("latin1").split()
got_w = int(next(t[1:] for t in tokens if t.startswith("W")))
got_h = int(next(t[1:] for t in tokens if t.startswith("H")))
if (got_w, got_h) != (w, h):
    sys.exit("%s: signalled %dx%d, expected %dx%d" %
             (label, got_w, got_h, w, h))
count = data.count(b"FRAME\n")
if count != frames:
    sys.exit("%s: decoded %d of %d pictures" % (label, count, frames))
PY
}

TOTAL=0
LAST_MD5=""

run_case() {
  local label="$1" clip="$2" w="$3" h="$4" chroma="$5"
  shift 5
  local out="$TMP/${label}.ivf"
  local recon="$TMP/${label}.recon.y4m"
  local dec="$TMP/${label}.dec.y4m"
  local args=(--video)
  if [[ "$chroma" == "444" ]]; then args+=(--chroma 444); fi
  if ! "$BIN" "$clip" -o "$out" --frames "$FRAMES" --video-recon "$recon" \
      "${args[@]}" "$@" >"$TMP/${label}.log" 2>&1; then
    fail "$label: encode failed ($(tail -1 "$TMP/${label}.log"))"
  fi
  if ! "$DAV2D" --strict 1 -q -i "$out" -o "$dec" >/dev/null 2>&1; then
    fail "$label: dav2d --strict 1 rejected the stream"
  fi
  check_y4m "$dec" "$w" "$h" "$FRAMES" "$label" ||
    fail "$label: decoded geometry/picture count mismatch"
  if ! cmp -s "$recon" "$dec"; then
    fail "$label: encoder reconstruction differs from the decoded output"
  fi
  LAST_MD5="$(md5_of "$out")"
  TOTAL=$((TOTAL + 1))
  echo "PASS video-any-dims $label ($LAST_MD5)"
}

check_changed() {
  local label="$1" got="$2" want="$3"
  if [[ "$got" == "$want" ]]; then
    fail "$label: filter flag did not change the stream"
  fi
}

run_matrix() {
  local clip="$1" w="$2" h="$3" chroma="$4" base base_md5
  base="${w}x${h}_${chroma}"
  run_case "$base-none" "$clip" "$w" "$h" "$chroma"
  base_md5="$LAST_MD5"
  run_case "$base-deblock" "$clip" "$w" "$h" "$chroma" --video-deblock
  check_changed "$base-deblock" "$LAST_MD5" "$base_md5"
  run_case "$base-cdef" "$clip" "$w" "$h" "$chroma" --video-cdef
  check_changed "$base-cdef" "$LAST_MD5" "$base_md5"
  run_case "$base-ccso" "$clip" "$w" "$h" "$chroma" --video-ccso
  check_changed "$base-ccso" "$LAST_MD5" "$base_md5"
  run_case "$base-wiener" "$clip" "$w" "$h" "$chroma" --video-wiener
  check_changed "$base-wiener" "$LAST_MD5" "$base_md5"
  run_case "$base-gdf" "$clip" "$w" "$h" "$chroma" --video-gdf
  check_changed "$base-gdf" "$LAST_MD5" "$base_md5"
  run_case "$base-chroma-lr" "$clip" "$w" "$h" "$chroma" --video-chroma-lr
  check_changed "$base-chroma-lr" "$LAST_MD5" "$base_md5"
  if [[ "$chroma" == "420" ]]; then
    run_case "$base-lr-chain" "$clip" "$w" "$h" "$chroma" \
      --video-cdef --video-ccso --video-wiener --video-chroma-lr --video-gdf
    check_changed "$base-lr-chain" "$LAST_MD5" "$base_md5"
    run_case "$base-deblock-chain" "$clip" "$w" "$h" "$chroma" \
      --video-deblock --video-cdef --video-ccso --video-wiener \
      --video-chroma-lr --video-gdf
    check_changed "$base-deblock-chain" "$LAST_MD5" "$base_md5"
  else
    # 4:2:2/4:4:4 deblock+chroma-LR stays a named rejection (chroma
    # lossless-mask aliasing), so the two chains are exercised separately.
    run_case "$base-lr-chain" "$clip" "$w" "$h" "$chroma" \
      --video-cdef --video-ccso --video-wiener --video-chroma-lr --video-gdf
    check_changed "$base-lr-chain" "$LAST_MD5" "$base_md5"
    run_case "$base-deblock-chain" "$clip" "$w" "$h" "$chroma" \
      --video-deblock --video-cdef --video-ccso --video-wiener --video-gdf
    check_changed "$base-deblock-chain" "$LAST_MD5" "$base_md5"
  fi
}

for spec in 100x70 130x74 424x240 640x362; do
  run_matrix "$TMP/${spec}_420.y4m" "${spec%x*}" "${spec#*x}" 420
done
for spec in 100x70 130x74 424x240 640x362 101x71; do
  run_matrix "$TMP/${spec}_444.y4m" "${spec%x*}" "${spec#*x}" 444
done

# --- bottom-edge inter on non-/8 heights ---------------------------------
# The --video-edge-inter reference pad clamps at MiRows*4, not the visible
# height: 100x70 has MiRows = 18 (72 rows), so rows 70..71 are real coded
# samples and only rows >= 72 are replicated.  A vertically moving pattern
# makes the next frame read the partly-visible last MI row, which is exactly
# where a visible-height pad decodes differently.
python3 - "$TMP" <<'PY'
import math
import sys

tmp = sys.argv[1]
frames = 4
for w, h in ((100, 70), (640, 362)):
    path = "%s/edge_%dx%d.y4m" % (tmp, w, h)
    with open(path, "wb") as out:
        out.write(("YUV4MPEG2 W%d H%d F30:1 Ip A1:1 C420jpeg\n" %
                   (w, h)).encode())
        for t in range(frames):
            out.write(b"FRAME\n")
            out.write(bytes(
                max(0, min(255, int(
                    100.0 + 70.0 * math.sin((i + 2 * t) / 7.0) +
                    50.0 * math.cos((j + t) / 5.0) +
                    30.0 * math.sin((i + j + 3 * t) / 11.0))))
                for j in range(h) for i in range(w)))
            for _ in range(2):
                out.write(bytes([128] * ((w // 2) * (h // 2))))
PY
for spec in 100x70 640x362; do
  run_case "edge-${spec}" "$TMP/edge_${spec}.y4m" "${spec%x*}" "${spec#*x}" \
    420 --video-edge-inter
done

# --- format constraints --------------------------------------------------
# 4:2:0 cannot represent an odd dimension in this Y4M.
python3 - "$TMP/odd420.y4m" <<'PY'
import sys
w, h, cw, ch = 130, 75, 65, 37
with open(sys.argv[1], "wb") as out:
    out.write(("YUV4MPEG2 W%d H%d F30:1 Ip A1:1 C420\n" % (w, h)).encode())
    out.write(b"FRAME\n")
    out.write(bytes(((x * 29 + y * 17) & 255)
                    for y in range(h) for x in range(w)))
    for p in (1, 2):
        out.write(bytes((((x * 11 + y * 7 + p * 13) & 255)
                         for y in range(ch) for x in range(cw))))
PY
if "$BIN" "$TMP/odd420.y4m" -o "$TMP/reject.ivf" --video --frames 1 \
    >"$TMP/reject.out" 2>"$TMP/reject.err"; then
  fail "odd 4:2:0 dimensions were accepted"
fi
grep -q "4:2:0 video requires even dimensions" "$TMP/reject.err" ||
  fail "odd 4:2:0 rejection did not name the even-dimension constraint"

# 4:2:2 cannot represent an odd width.
python3 - "$TMP/odd422.y4m" <<'PY'
import sys
w, h, cw = 131, 74, 66
with open(sys.argv[1], "wb") as out:
    out.write(("YUV4MPEG2 W%d H%d F30:1 Ip A1:1 C422\n" % (w, h)).encode())
    out.write(b"FRAME\n")
    out.write(bytes(((x * 29 + y * 17) & 255)
                    for y in range(h) for x in range(w)))
    for p in (1, 2):
        out.write(bytes((((x * 11 + y * 7 + p * 13) & 255)
                         for y in range(h) for x in range(cw))))
PY
if "$BIN" "$TMP/odd422.y4m" -o "$TMP/reject.ivf" --video --frames 1 \
    --chroma 422 >"$TMP/reject.out" 2>"$TMP/reject.err"; then
  fail "odd-width 4:2:2 dimensions were accepted"
fi
grep -q "4:2:2 cannot represent" "$TMP/reject.err" ||
  fail "odd-width 4:2:2 rejection did not name the chroma constraint"

# GPU: a CUDA build may keep a named CPU-only rejection for non-multiple-of-8
# dimensions; the CPU-only reference build rejects --video-gpu earlier.
if "$BIN" "$TMP/100x70_420.y4m" -o "$TMP/reject.ivf" --video --frames 1 \
    --video-gpu >"$TMP/reject.out" 2>"$TMP/reject.err"; then
  echo "NOTE video-any-dims: --video-gpu accepted non-multiple-of-8 dimensions"
elif ! grep -q "CPU-only for dimensions not divisible by 8" "$TMP/reject.err"; then
  if ! grep -q "requires a CUDA build" "$TMP/reject.err"; then
    fail "--video-gpu non-/8 rejection was not named as CPU-only"
  fi
fi
TOTAL=$((TOTAL + 3))
echo "PASS video-any-dims format-constraints"

echo "gate_video_any_dims: PASS ($TOTAL checks)"
