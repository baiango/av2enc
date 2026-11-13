#!/usr/bin/env bash
# Product gate for the opt-in CUDA M2 path (--video-gpu).
#
# For every wired mode it encodes the same 360p source with the CPU reference
# (--video) and the CUDA backend (--video-gpu), then requires:
#   * byte-identical temporal units (stream MD5 equal, cmp clean),
#   * byte-identical encoder reconstruction dumps (--video-recon),
#   * identical dav2d --strict 1 decoded output.
# It also asserts that every unported combination is rejected by name with
# usage exit 2, never silently ignored or CPU-fallback encoded.
#
# Exits 77 (CTest skip) when the pinned decoder or a CUDA-enabled binary is
# unavailable.  The script generates its own deterministic source, so it does
# not depend on gitignored testdata.
#
# Usage: gate_video_gpu.sh --bin <av2enc> [--frames N] [--dav2d <path>]
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${HERE}/../../../.." && pwd)"
BIN=""
FRAMES=4
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
  echo "gate_video_gpu: missing --bin" >&2
  exit 2
fi

if [[ -z "$DAV2D" ]]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [[ -x "$candidate" ]]; then
      DAV2D="$candidate"
      break
    fi
  done
fi
if [[ -z "$DAV2D" || ! -x "$DAV2D" ]]; then
  echo "gate_video_gpu: SKIP (no pinned dav2d; set DAV2D)"
  exit 77
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "gate_video_gpu: SKIP (python3 not available)"
  exit 77
fi

TMP="$(mktemp -d "${TMPDIR:-/tmp}/gate_video_gpu.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT
CLIP="$TMP/clip360.y4m"
# Second deterministic clip: a smooth sinusoid panned by (3,1) per frame.  On the
# textured clip above every P leaf selects intra at QP 90, so it exercises the
# headers but not reference-indexed MC/ME; this clip selects inter leaves and is
# used for the multi-reference / motion-compensation cases.
CLIP_INTER="$TMP/clip360_inter.y4m"
# Third clip for compound prediction: key A, P anchor B, and a third picture
# that is exactly (A+B)/2.  A zero-MV CWP_EQUAL average of the two references
# is then the best predictor, so the gated compound subset is actually selected
# (the smooth clip never selects it).
CLIP_AVG="$TMP/clip360_avg.y4m"
# Generate enough frames for the longest gated sequence (8-frame temporal GOP)
# even when the caller lowers --frames for the default cases.
GEN_FRAMES=$(( FRAMES > 8 ? FRAMES : 8 ))

python3 - "$CLIP" "$GEN_FRAMES" <<'PY'
import sys

path = sys.argv[1]
frames = int(sys.argv[2])
w, h = 640, 360
cw, ch = w // 2, h // 2

def sample(x, y, f, plane):
    sx = 8 + 6 * f
    sy = 12 + 3 * f
    inside = sx <= x < sx + 96 and sy <= y < sy + 80
    # A moving textured background plus a drifting square.  The per-frame
    # perturbation keeps some inter residuals non-zero so --video-inter-residual
    # exercises the TX/quant/recon chain instead of selecting every block as
    # skip.
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

python3 - "$CLIP_INTER" "$GEN_FRAMES" <<'PY'
import math
import sys

path = sys.argv[1]
frames = int(sys.argv[2])
w, h = 640, 360
cw, ch = w // 2, h // 2

def sample(x, y, f, plane):
    xx = x + 3 * f
    yy = y + f
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

python3 - "$CLIP_AVG" <<'PY'
import math
import sys

path = sys.argv[1]
w, h = 640, 360
cw, ch = w // 2, h // 2

def a(x, y, plane):
    if plane == 0:
        return max(0, min(255, int(128 + 70 * math.sin(x * 0.07 + 1) +
                                     45 * math.cos(y * 0.05))))
    if plane == 1:
        return max(0, min(255, int(120 + 40 * math.sin(x * 0.05))))
    return max(0, min(255, int(130 + 35 * math.cos(y * 0.06))))

def b(x, y, plane):
    if plane == 0:
        return max(0, min(255, int(128 + 70 * math.sin(x * 0.07 + 3.2) +
                                     45 * math.cos(y * 0.05 + 1.7))))
    if plane == 1:
        return max(0, min(255, int(120 + 40 * math.sin(x * 0.05 + 2.1))))
    return max(0, min(255, int(130 + 35 * math.cos(y * 0.06 + 2.4))))

def mid(x, y, plane):
    return (a(x, y, plane) + b(x, y, plane)) // 2

frames = [a, b, mid]
with open(path, "wb") as out:
    out.write(b"YUV4MPEG2 W%d H%d F30:1 Ip A1:1 C420\n" % (w, h))
    for fn in frames:
        out.write(b"FRAME\n")
        for y in range(h):
            out.write(bytes(fn(x, y, 0) for x in range(w)))
        for plane in (1, 2):
            for y in range(ch):
                out.write(bytes(fn(2 * x, 2 * y, plane)
                                for x in range(cw)))
PY

fail() {
  echo "gate_video_gpu: FAIL: $*" >&2
  exit 1
}

# Probe: a CPU-only build (or a CUDA build without the M2 target) must reject
# the flag with a clear reason.  That is a skip for this gate, not a failure.
set +e
"$BIN" "$CLIP" -o "$TMP/probe.ivf" --video-gpu --frames 2 \
  >"$TMP/probe.out" 2>"$TMP/probe.err"
probe_rc=$?
set -e
if [[ $probe_rc -ne 0 ]]; then
  if grep -Eq -- "requires a CUDA build|is not available in this build|is not linked" \
      "$TMP/probe.err"; then
    echo "gate_video_gpu: SKIP (binary has no CUDA M2 target)"
    exit 77
  fi
  cat "$TMP/probe.err" >&2
  fail "probe encode failed (exit $probe_rc)"
fi
grep -q -- "backend=cuda" "$TMP/probe.err" ||
  fail "GPU path did not report backend=cuda"
grep -q -- "frame_bytes_from_cpu=0" "$TMP/probe.err" ||
  fail "GPU path did not report frame_bytes_from_cpu=0"

run_case_on() {
  local clip="$1"
  local label="$2"
  shift 2
  local cpu="$TMP/${label}.cpu.ivf"
  local gpu="$TMP/${label}.gpu.ivf"
  local cpu_recon="$TMP/${label}.cpu.recon.y4m"
  local gpu_recon="$TMP/${label}.gpu.recon.y4m"
  local cpu_dec="$TMP/${label}.cpu.dec.y4m"
  local gpu_dec="$TMP/${label}.gpu.dec.y4m"

  "$BIN" "$clip" -o "$cpu" --video --frames "$FRAMES" \
    --video-recon "$cpu_recon" "$@" >/dev/null 2>"$TMP/${label}.cpu.log" ||
    fail "$label: CPU reference encode failed"
  "$BIN" "$clip" -o "$gpu" --video --video-gpu --frames "$FRAMES" \
    --video-recon "$gpu_recon" "$@" >/dev/null 2>"$TMP/${label}.gpu.log" ||
    fail "$label: GPU encode failed"
  grep -q -- "frame_bytes_from_cpu=0" "$TMP/${label}.gpu.log" ||
    fail "$label: GPU path reported CPU-produced bytes"

  cmp -s "$cpu" "$gpu" ||
    fail "$label: temporal units differ (CPU $(md5sum "$cpu" | cut -d' ' -f1), GPU $(md5sum "$gpu" | cut -d' ' -f1))"
  cmp -s "$cpu_recon" "$gpu_recon" ||
    fail "$label: encoder reconstruction differs"

  "$DAV2D" --strict 1 -q -i "$cpu" -o "$cpu_dec" >/dev/null 2>&1 ||
    fail "$label: dav2d rejected the CPU reference stream"
  "$DAV2D" --strict 1 -q -i "$gpu" -o "$gpu_dec" >/dev/null 2>&1 ||
    fail "$label: dav2d --strict 1 rejected the GPU stream"
  cmp -s "$cpu_dec" "$gpu_dec" ||
    fail "$label: decoded output differs"

  echo "PASS video-gpu $label ($(md5sum "$gpu" | cut -d' ' -f1))"
}

run_case() {
  run_case_on "$CLIP" "$@"
}

# Same comparison with an explicit frame count (the wrapper modes require exact
# sequence lengths).
run_case_frames() {
  local count="$1"
  shift
  local saved="$FRAMES"
  FRAMES="$count"
  run_case_on "$CLIP_INTER" "$@"
  FRAMES="$saved"
}

run_case baseline
run_case quarter --video-mv-precision quarter
run_case half --video-mv-precision half
run_case subpel-eighth --video-subpel-eighth
run_case force-single --video-force-single
run_case depth2 --video-max-depth 2
run_case min-block8 --video-min-block 8
run_case me-range8 --video-me-range 8
# Inter-residual may legitimately pick no residual on this source; the CPU and
# GPU paths must still agree, and the GPU selftest exercises the TX chain.
run_case inter-residual --video-inter-residual
# Reference-indexed MC/ME: run on the smooth panned clip, where P leaves select
# inter.  The textured clip above is intra-only at QP 90 and cannot exercise the
# per-reference path.
run_case_on "$CLIP_INTER" inter-baseline
run_case_on "$CLIP_INTER" refs2 --video-refs 2
run_case_on "$CLIP_INTER" refs3 --video-refs 3
run_case_on "$CLIP_INTER" refs2-quarter --video-refs 2 --video-mv-precision quarter
run_case_on "$CLIP_INTER" refs2-half --video-refs 2 --video-mv-precision half
# §5.7 multi-frame header on the non-single-picture path.
run_case_on "$CLIP_INTER" mfh --video-mfh
# §5.4.12/.13 decoder-compatible timing + §5.15 content interpretation OBU.
run_case_on "$CLIP_INTER" timing --video-timing
# §8.2.4 average-CDF / §8.2.6 adaptation.
run_case_on "$CLIP_INTER" avg-cdf --avg-cdf
run_case_on "$CLIP_INTER" adaptive-cdf --avg-cdf --adaptive-cdf
# Frame-type wrappers: the tile payloads are the plain GPU sub-encodes; only
# header/OBU assembly changes.
run_case_frames 4 long-term --video-long-term
run_case_frames 4 olk --video-olk
run_case_frames 4 leading --video-leading 1
run_case_frames 4 sef --video-sef
run_case_frames 2 switch --video-switch
run_case_frames 3 ras --video-ras
# B frames and hierarchical-P temporal layers (plan-owned coding order).
run_case_frames 4 bframes2 --b-frames 2
run_case_frames 5 bframes3 --b-frames 3
run_case_frames 4 tlayers4 --video-tlayers 4
run_case_frames 8 tlayers8 --video-tlayers 8
run_case_frames 8 tlayers8-drop1 --video-tlayers 8 --video-tlayer-drop 1
# Compound/CWP AVERAGE is selected on the A/B/mid clip.
_saved_frames="$FRAMES"
FRAMES=3
run_case_on "$CLIP_AVG" compound --video-refs 2 --video-compound
FRAMES="$_saved_frames"

expect_reject() {
  local label="$1"
  shift
  set +e
  "$BIN" "$CLIP" -o "$TMP/reject.ivf" --video --video-gpu "$@" \
    >"$TMP/reject.out" 2>"$TMP/reject.err"
  local rc=$?
  set -e
  [[ $rc -eq 2 ]] || fail "$label: expected usage exit 2, got $rc"
  grep -q -- "--video-gpu" "$TMP/reject.err" ||
    fail "$label: rejection did not name --video-gpu"
  echo "PASS video-gpu reject $label"
}

expect_reject rc-crf --rc crf --crf 32
expect_reject tiles --tiles 2x1 --no-deblock
expect_reject cdf-continuity --avg-cdf --adaptive-cdf --cdf-continuity
expect_reject svc --svc-layers 0=640x360

echo "gate_video_gpu: PASS"
