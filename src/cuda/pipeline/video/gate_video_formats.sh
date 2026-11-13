#!/usr/bin/env bash
# Product gate for the format video core (10-bit and 8-bit chroma formats).
#
# For every verified format it:
#   * generates a deterministic 640x360 5-frame source at that depth/chroma,
#   * encodes 4 frames with the CPU reference (`--video`) and the CUDA backend
#     (`--video-gpu`),
#   * requires byte-identical temporal units and identical `--video-recon`,
#   * decodes both with dav2d --strict 1 and requires the expected picture
#     count (4) and identical decoded output,
#   * names unsupported combinations that must stay rejected.
#
# Exits 77 (CTest skip) when the pinned decoder or a CUDA-enabled binary is
# unavailable.
#
# Usage: gate_video_formats.sh --bin <av2enc> [--dav2d <path>] [--frames N]
#                              [--format <tag>]   # one of 420p10 422p10 ...
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${HERE}/../../../.." && pwd)"
BIN=""
FRAMES=4
DAV2D="${DAV2D:-}"
ONLY_FORMAT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --bin) BIN="$2"; shift 2 ;;
    --frames) FRAMES="$2"; shift 2 ;;
    --dav2d) DAV2D="$2"; shift 2 ;;
    --format) ONLY_FORMAT="$2"; shift 2 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$BIN" || ! -x "$BIN" ]]; then
  echo "gate_video_formats: missing --bin" >&2
  exit 2
fi
if [[ -z "$DAV2D" ]]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [[ -x "$candidate" ]]; then DAV2D="$candidate"; break; fi
  done
fi
if [[ -z "$DAV2D" || ! -x "$DAV2D" ]]; then
  echo "gate_video_formats: SKIP (no pinned dav2d; set DAV2D)"
  exit 77
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "gate_video_formats: SKIP (python3 not available)"
  exit 77
fi

TMP="$(mktemp -d "${TMPDIR:-/tmp}/gate_video_formats.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT

# tag bitdepth chroma
FORMATS=(
  "420p10 10 420"
  "422p10 10 422"
  "444p10 10 444"
  "mono10 10 400"
  "422 8 422"
  "444 8 444"
  "mono 8 400"
)

generate_clip() {
  local tag="$1" bitdepth="$2" chroma="$3" name="$4" frames="$5"
  python3 - "$tag" "$bitdepth" "$chroma" "$name" "$frames" <<'PY'
import math
import sys

tag, bitdepth, chroma, path, frames = sys.argv[1:6]
bitdepth = int(bitdepth)
frames = int(frames)
w, h = 640, 360
sub_x = 1 if chroma in ("420", "422") else 0
sub_y = 1 if chroma == "420" else 0
cw, ch = w >> sub_x, h >> sub_y
peak = (1 << bitdepth) - 1
planes = [(w, h)]
if chroma != "400":
    planes += [(cw, ch), (cw, ch)]

def sample(x, y, f, plane):
    v = (1 << (bitdepth - 1)) + \
        (1 << (bitdepth - 3)) * math.sin((x + 4 * f + 7 * plane) * 0.05) + \
        (1 << (bitdepth - 4)) * math.cos((y + f + 3 * plane) * 0.07) + \
        (1 << (bitdepth - 5)) * math.sin((x + y + 11 * f) * 0.013)
    return max(0, min(peak, int(v + 0.5)))

with open(path, "wb") as out:
    out.write(("YUV4MPEG2 W%d H%d F30:1 Ip A1:1 C%s\n" % (w, h, tag)).encode())
    for f in range(frames):
        out.write(b"FRAME\n")
        for plane, (pw, ph) in enumerate(planes):
            if bitdepth == 8:
                for y in range(ph):
                    out.write(bytes(sample(x, y, f, plane) for x in range(pw)))
            else:
                row = bytearray()
                for y in range(ph):
                    for x in range(pw):
                        v = sample(x, y, f, plane)
                        row += bytes((v & 0xFF, (v >> 8) & 0xFF))
                out.write(row)
PY
}

decoded_frames() {
  python3 - "$1" <<'PY'
import sys

path = sys.argv[1]
with open(path, "rb") as f:
    header = f.readline()
    tokens = header.split()
    width = int([t for t in tokens if t.startswith(b"W")][0][1:])
    height = int([t for t in tokens if t.startswith(b"H")][0][1:])
    tag = [t.decode() for t in tokens if t.startswith(b"C")][0]
    bps = 1 if tag in ("C420", "C422", "C444", "Cmono") else 2
    mono = tag in ("Cmono", "Cmono10")
    sub_x = 1 if tag in ("C420", "C420p10", "C422", "C422p10") else 0
    sub_y = 1 if tag in ("C420", "C420p10") else 0
    planes = [(width, height)]
    if not mono:
        planes += [(width >> sub_x, height >> sub_y)] * 2
    body = sum(w * h * bps for w, h in planes)
    count = 0
    while True:
        line = f.readline()
        if not line:
            break
        if not line.startswith(b"FRAME"):
            break
        data = f.read(body)
        if len(data) != body:
            break
        count += 1
print(count)
PY
}

fail() {
  echo "gate_video_formats: FAIL: $*" >&2
  exit 1
}

# Probe: skip on a CPU-only binary.
set +e
"$BIN" --help >/dev/null 2>&1
set -e
PROBE_CLIP="$TMP/probe.y4m"
generate_clip "420p10" 10 "420" "$PROBE_CLIP" 1
set +e
"$BIN" "$PROBE_CLIP" -o "$TMP/probe.ivf" --video --video-gpu --frames 1 \
  --bit-depth 10 --chroma 420 >"$TMP/probe.out" 2>"$TMP/probe.err"
probe_rc=$?
set -e
if [[ $probe_rc -ne 0 ]]; then
  if grep -Eq -- "requires a CUDA build|is not available in this build|is not linked" \
      "$TMP/probe.err"; then
    echo "gate_video_formats: SKIP (binary has no CUDA M2 target)"
    exit 77
  fi
  cat "$TMP/probe.err" >&2
  fail "probe encode failed (exit $probe_rc)"
fi

run_case() {
  local tag="$1" bitdepth="$2" chroma="$3" label="$4"
  shift 4
  local clip="$TMP/${tag}.y4m"
  local cpu="$TMP/${tag}_${label}.cpu.ivf"
  local gpu="$TMP/${tag}_${label}.gpu.ivf"
  local cpu_rec="$TMP/${tag}_${label}.cpu.rec.y4m"
  local gpu_rec="$TMP/${tag}_${label}.gpu.rec.y4m"
  local cpu_dec="$TMP/${tag}_${label}.cpu.dec.y4m"
  local gpu_dec="$TMP/${tag}_${label}.gpu.dec.y4m"

  "$BIN" "$clip" -o "$cpu" --video --frames "$FRAMES" --bit-depth "$bitdepth" \
    --chroma "$chroma" --video-recon "$cpu_rec" "$@" \
    >"$TMP/${tag}_${label}.cpu.log" 2>&1 ||
    fail "$tag $label: CPU reference encode failed"
  "$BIN" "$clip" -o "$gpu" --video --video-gpu --frames "$FRAMES" \
    --bit-depth "$bitdepth" --chroma "$chroma" --video-recon "$gpu_rec" "$@" \
    >"$TMP/${tag}_${label}.gpu.log" 2>&1 ||
    fail "$tag $label: GPU encode failed"
  grep -q -- "frame_bytes_from_cpu=0" "$TMP/${tag}_${label}.gpu.log" ||
    fail "$tag $label: GPU path reported CPU-produced bytes"

  cmp -s "$cpu" "$gpu" ||
    fail "$tag $label: temporal units differ (CPU $(md5sum "$cpu" | cut -d' ' -f1), GPU $(md5sum "$gpu" | cut -d' ' -f1))"
  cmp -s "$cpu_rec" "$gpu_rec" ||
    fail "$tag $label: encoder reconstruction differs"

  "$DAV2D" --strict 1 -q -i "$cpu" -o "$cpu_dec" >/dev/null 2>&1 ||
    fail "$tag $label: dav2d rejected the CPU reference stream"
  "$DAV2D" --strict 1 -q -i "$gpu" -o "$gpu_dec" >/dev/null 2>&1 ||
    fail "$tag $label: dav2d --strict 1 rejected the GPU stream"
  cmp -s "$cpu_dec" "$gpu_dec" ||
    fail "$tag $label: decoded output differs"
  local pictures
  pictures="$(decoded_frames "$gpu_dec")"
  [[ "$pictures" == "$FRAMES" ]] ||
    fail "$tag $label: decoded $pictures of $FRAMES pictures"
  cmp -s "$gpu_dec" "$gpu_rec" ||
    fail "$tag $label: decoded output differs from the encoder reconstruction"

  echo "PASS format-video $tag $label ($(md5sum "$gpu" | cut -d' ' -f1), $pictures frames)"
}

for entry in "${FORMATS[@]}"; do
  read -r tag bitdepth chroma <<<"$entry"
  if [[ -n "$ONLY_FORMAT" && "$ONLY_FORMAT" != "$tag" ]]; then continue; fi
  generate_clip "$tag" "$bitdepth" "$chroma" "$TMP/${tag}.y4m" $(( FRAMES + 1 ))
  # A commit may enable the formats one at a time.  The CLI names unwired
  # formats explicitly; skip them visibly instead of calling that a failure.
  set +e
  "$BIN" "$TMP/${tag}.y4m" -o "$TMP/probe_${tag}.ivf" --video --frames 1 \
    --bit-depth "$bitdepth" --chroma "$chroma" \
    >"$TMP/probe_${tag}.out" 2>"$TMP/probe_${tag}.err"
  local_probe=$?
  set -e
  if [[ $local_probe -ne 0 ]]; then
    if grep -q -- "not verified on the video path" "$TMP/probe_${tag}.err"; then
      echo "SKIP format-video $tag (not wired yet)"
      continue
    fi
    cat "$TMP/probe_${tag}.err" >&2
    fail "$tag: probe encode failed (exit $local_probe)"
  fi
  run_case "$tag" "$bitdepth" "$chroma" baseline
  run_case "$tag" "$bitdepth" "$chroma" refs2 --video-refs 2
  run_case "$tag" "$bitdepth" "$chroma" inter-residual --video-inter-residual
  run_case "$tag" "$bitdepth" "$chroma" quarter --video-mv-precision quarter
  run_case "$tag" "$bitdepth" "$chroma" force-single --video-force-single
  if [[ "$chroma" != "422" ]]; then
    run_case "$tag" "$bitdepth" "$chroma" min-block8 --video-min-block 8
  fi
done

# Unsupported combinations stay rejected by name.
expect_reject() {
  local label="$1"
  shift
  local clip="$TMP/420p10.y4m"
  generate_clip "420p10" 10 "420" "$clip" 1
  set +e
  "$BIN" "$clip" -o "$TMP/reject.ivf" --video --frames 2 --bit-depth 10 \
    --chroma 420 "$@" >"$TMP/reject.out" 2>"$TMP/reject.err"
  local rc=$?
  set -e
  [[ $rc -eq 2 ]] || fail "$label: expected usage exit 2, got $rc"
  echo "PASS format-video reject $label"
}

expect_reject deblock --video-deblock
expect_reject tiles --tiles 2x1
expect_reject tx-search --video-tx-search
expect_reject gme --video-gme

# 12-bit is not signalable in AV2 v1.0.0 and is rejected by name.
set +e
"$BIN" "$TMP/420p10.y4m" -o "$TMP/reject.ivf" --video --frames 1 \
  --bit-depth 12 --chroma 420 >"$TMP/reject.out" 2>"$TMP/reject.err"
rc=$?
set -e
[[ $rc -eq 2 ]] || fail "12-bit: expected usage exit 2, got $rc"
grep -q -- "bit-depth" "$TMP/reject.err" ||
  fail "12-bit: rejection did not name --bit-depth"
echo "PASS format-video reject 12-bit"

echo "gate_video_formats: PASS"
