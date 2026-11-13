#!/usr/bin/env bash
# CLI gate for the opt-in multi-frame inter path.
#
# Encodes N frames through the product CLI (`--video`), decodes the resulting
# IVF with the pinned dav2d through tools/validate_dav2d.sh, then compares every
# decoded sample against the encoder reconstruction dumped by --video-recon.
# Also checks the usage-error preflight: invalid option combinations exit 2
# before IvfWriter::open and leave a pre-existing -o file untouched.
#
# Usage:
#   src/pipeline/video/video_cli_gate.sh --bin build-int/av2enc \
#       [--input testdata/shine_1280x720_60f.y4m] [--frames 4] [--dav2d PATH]
#       [--illumination-shift] [--expect-inter-residual]
#       [-- extra encoder args...]
#
# --illumination-shift generates a deterministic 320x192 clip whose frames are
# a smooth source plus a constant per-frame brightness offset.  That is the
# fixture where the --video-inter-residual decision actually codes residual
# leaves (the 720p shine clip codes none).  --expect-inter-residual turns on
# --video-trace and fails unless at least one inter candidate reports
# residual=1, so the gate cannot pass with a skip-only stream.
#
# The legacy positional form is still accepted:
#   src/pipeline/video/video_cli_gate.sh [BIN] [INPUT] [FRAMES] [extra args...]
#
# Defaults: BIN=build-int/av2enc, INPUT=testdata/shine_1280x720_60f.y4m,
# FRAMES=4.  Set OUT=/tmp/prefix to keep the streams.
#
# Exit codes: 0 pass, 1 failure, 2 usage, 77 missing fixture/decoder.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "$0")/../../.." && pwd)
BIN=""
INPUT=""
FRAMES=""
DAV2D="${DAV2D:-}"
TESTDATA="${AV2ENC_TESTDATA:-$ROOT/testdata}"
ILLUMINATION_SHIFT=0
EXPECT_INTER_RESIDUAL=0
EXTRA=()
POSITIONAL=()

while [ $# -gt 0 ]; do
  case "$1" in
    --bin) BIN=$2; shift 2 ;;
    --input) INPUT=$2; shift 2 ;;
    --frames) FRAMES=$2; shift 2 ;;
    --dav2d) DAV2D=$2; shift 2 ;;
    --illumination-shift) ILLUMINATION_SHIFT=1; shift ;;
    --expect-inter-residual) EXPECT_INTER_RESIDUAL=1; shift ;;
    --) shift; EXTRA+=("$@"); break ;;
    -h|--help)
      sed -n '2,29p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    -*) echo "video_cli_gate: unknown argument $1" >&2; exit 2 ;;
    *) POSITIONAL+=("$1"); shift ;;
  esac
done

# Legacy positional form: BIN [INPUT] [FRAMES] [extra...].
if [ "${#POSITIONAL[@]}" -gt 0 ]; then
  [ -n "$BIN" ] || BIN=${POSITIONAL[0]}
  if [ "${#POSITIONAL[@]}" -gt 1 ] && [ -z "$INPUT" ]; then
    INPUT=${POSITIONAL[1]}
  fi
  if [ "${#POSITIONAL[@]}" -gt 2 ] && [ -z "$FRAMES" ]; then
    FRAMES=${POSITIONAL[2]}
  fi
  if [ "${#POSITIONAL[@]}" -gt 3 ]; then
    EXTRA+=("${POSITIONAL[@]:3}")
  fi
fi

BIN=${BIN:-"$ROOT/build-int/av2enc"}
INPUT=${INPUT:-"$TESTDATA/shine_1280x720_60f.y4m"}
FRAMES=${FRAMES:-4}

if [ "$ILLUMINATION_SHIFT" -eq 1 ] && ! command -v python3 >/dev/null 2>&1; then
  echo "video_cli_gate: python3 is required for --illumination-shift" >&2
  exit 77
fi

if [ ! -x "$BIN" ]; then
  echo "video_cli_gate: --bin must point at an av2enc executable" >&2
  exit 2
fi
if [ ! -f "$INPUT" ]; then
  echo "SKIP: missing input $INPUT" >&2
  exit 77
fi
if [ -z "$DAV2D" ]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [ -x "$candidate" ]; then DAV2D=$candidate; break; fi
  done
fi
if [ -z "$DAV2D" ] || [ ! -x "$DAV2D" ]; then
  echo "video_cli_gate: no dav2d binary (set DAV2D=...)" >&2
  exit 77
fi
export DAV2D
if ! command -v python3 >/dev/null 2>&1; then
  echo "video_cli_gate: python3 is required for the recon comparison" >&2
  exit 77
fi

if [ -n "${OUT:-}" ]; then
  OUT="${OUT%*/}/video_cli"
else
  OUT="$(mktemp -d "${TMPDIR:-/tmp}/av2enc-video-gate.XXXXXX")/video_cli"
fi
mkdir -p "$(dirname -- "$OUT")"

echo "== video CLI gate: $INPUT ($FRAMES frame(s)) =="

# --- usage rejections (no decode) ----------------------------------------
reject() {
  local label=$1 needle=$2
  shift 2
  if "$BIN" "$INPUT" -o "${OUT}.reject.ivf" "$@" >/dev/null 2>"${OUT}.reject.err"; then
    echo "FAIL: $label: expected a rejection" >&2
    exit 1
  fi
  if ! grep -qF -e "$needle" "${OUT}.reject.err"; then
    cat "${OUT}.reject.err" >&2
    echo "FAIL: $label: error does not mention '$needle'" >&2
    exit 1
  fi
  echo "PASS: $label rejected by name"
}
reject "sub-8 min block" "shared-chroma" \
  --video --frames 1 --video-min-block 4 --video-max-depth 5
reject "sub-8 min block (depth 4)" "shared-chroma" \
  --video --frames 1 --video-min-block 4 --video-max-depth 4

ENCODE_INPUT=$INPUT
if [ "$ILLUMINATION_SHIFT" -eq 1 ]; then
  # A smooth source plus a constant per-frame brightness offset.  The 720p
  # shine clip codes no residual at QP 90; this fixture makes the
  # --video-inter-residual rate decision pick residual leaves.  Keep $INPUT
  # untouched: the usage-error preflight below relies on the longer fixture's
  # frame count.
  ENCODE_INPUT="${OUT}.illum.y4m"
  python3 - "$ENCODE_INPUT" <<'PY'
import math
import sys

w, h, n = 320, 192, 4
cw, ch = w // 2, h // 2
with open(sys.argv[1], "wb") as f:
    f.write(f"YUV4MPEG2 W{w} H{h} F30:1 Ip A1:1 C420jpeg\n".encode())
    for i in range(n):
        f.write(b"FRAME\n")
        offset = 15 * i
        y = bytearray(w * h)
        for r in range(h):
            for c in range(w):
                v = 100 + 30 * math.sin(c / 9.0) + 20 * math.cos(r / 11.0) + offset
                y[r * w + c] = max(1, min(255, int(v)))
        f.write(y)
        f.write(bytes([128]) * (cw * ch))
        f.write(bytes([128]) * (cw * ch))
PY
fi

echo "== video CLI gate: $ENCODE_INPUT ($FRAMES frame(s)) =="
if [ "$EXPECT_INTER_RESIDUAL" -eq 1 ]; then
  # --video-trace prints one "cand inter final ... residual=N" line per inter
  # candidate.  Capture it so the gate can assert the feature actually coded a
  # residual instead of passing on a skip-only stream.
  "$BIN" "$ENCODE_INPUT" --video --frames "$FRAMES" --video-trace \
    --video-recon "${OUT}.recon.y4m" -o "${OUT}.ivf" \
    "${EXTRA[@]+"${EXTRA[@]}"}" 2>"${OUT}.trace.log"
  RESIDUAL_LEAVES=$(grep -c 'residual=1' "${OUT}.trace.log" || true)
  INTER_CANDIDATES=$(grep -c 'cand inter final' "${OUT}.trace.log" || true)
  if [ "${RESIDUAL_LEAVES:-0}" -lt 1 ]; then
    echo "FAIL: --video-inter-residual coded 0 residual leaves" \
         "(${INTER_CANDIDATES:-0} inter candidates); the gate would be" \
         "vacuous" >&2
    tail -20 "${OUT}.trace.log" >&2
    exit 1
  fi
  echo "  inter residual: $RESIDUAL_LEAVES of $INTER_CANDIDATES candidates" \
       "coded residual (skip_flag = 0)"
else
  "$BIN" "$ENCODE_INPUT" --video --frames "$FRAMES" \
    --video-recon "${OUT}.recon.y4m" -o "${OUT}.ivf" "${EXTRA[@]+"${EXTRA[@]}"}"
fi

"$ROOT/tools/validate_dav2d.sh" "${OUT}.ivf"

"$DAV2D" -q -i "${OUT}.ivf" -o "${OUT}.decoded.y4m" --strict 1 --threads 1

python3 - "${OUT}.recon.y4m" "${OUT}.decoded.y4m" <<'PY'
import sys


def read_y4m(path):
    frames = []
    with open(path, "rb") as f:
        header = f.readline()
        if not header.startswith(b"YUV4MPEG2 "):
            raise SystemExit(f"{path}: not a Y4M file")
        fields = header.split()
        width = height = None
        for field in fields[1:]:
            if field.startswith(b"W"):
                width = int(field[1:])
            elif field.startswith(b"H"):
                height = int(field[1:])
        if width is None or height is None:
            raise SystemExit(f"{path}: missing dimensions")
        cw = (width + 1) // 2
        ch = (height + 1) // 2
        size = width * height + 2 * cw * ch
        while True:
            line = f.readline()
            if not line:
                break
            if not line.startswith(b"FRAME"):
                raise SystemExit(f"{path}: bad frame marker")
            data = f.read(size)
            if len(data) != size:
                raise SystemExit(f"{path}: truncated frame")
            frames.append(data)
    return frames


expected = read_y4m(sys.argv[1])
decoded = read_y4m(sys.argv[2])
if len(expected) != len(decoded):
    raise SystemExit(
        f"frame count mismatch: encoder recon {len(expected)} vs decoder {len(decoded)}")
diffs = 0
first = None
for index, (a, b) in enumerate(zip(expected, decoded)):
    if a == b:
        continue
    for pos, (x, y) in enumerate(zip(a, b)):
        if x != y:
            diffs += 1
            if first is None:
                first = (index, pos, x, y)
print(f"  recon compare: {len(expected)} frame(s), {diffs} sample diff(s)")
if diffs:
    print(f"  first diff: frame={first[0]} offset={first[1]} "
          f"dec={first[2]} enc={first[3]}")
    raise SystemExit(1)
print("PASS: decoder output matches encoder reconstruction sample for sample")
PY

# --- usage-error preflight ------------------------------------------------
#
# Regression for BUG-CLI-001 (SCOUT-RC-02, SCOUT-RC-03, SCOUT-QUANT-01,
# SCOUT-INTER-ADV-02): every option-combination error must exit 2 before
# IvfWriter::open, so a pre-existing -o stream is never truncated to a
# header-only IVF.
SENTINEL="${OUT}.sentinel.ivf"
usage_cases=0
usage_failures=0

expect_usage_error() {
  local label=$1
  shift
  usage_cases=$((usage_cases + 1))
  printf 'SENTINEL\n' > "$SENTINEL"
  set +e
  "$BIN" "$@" >"${OUT}.usage.log" 2>&1
  local rc=$?
  set -e
  if [ "$rc" -ne 2 ]; then
    echo "FAIL: $label: expected exit 2, got $rc" >&2
    sed -n '1,3p' "${OUT}.usage.log" >&2
    usage_failures=$((usage_failures + 1))
    return 0
  fi
  if [ "$(cat "$SENTINEL")" != "SENTINEL" ]; then
    echo "FAIL: $label: exit 2 modified the -o file" >&2
    usage_failures=$((usage_failures + 1))
    return 0
  fi
  echo "PASS usage: $label"
}

# Rate-control combinations and per-frame-QP conflicts (SCOUT-RC-02/RC-03).
expect_usage_error "rc vbr without bitrate" \
  "$INPUT" -o "$SENTINEL" --video --rc vbr
expect_usage_error "crf without rc crf" \
  "$INPUT" -o "$SENTINEL" --video --rc cqp --crf 20
expect_usage_error "bitrate without rc vbr" \
  "$INPUT" -o "$SENTINEL" --video --bitrate 500
expect_usage_error "rc crf + olk" \
  "$INPUT" -o "$SENTINEL" --video --frames 4 --video-olk --rc crf --crf 32
expect_usage_error "rc crf + leading" \
  "$INPUT" -o "$SENTINEL" --video --frames 6 --video-leading 2 --rc crf --crf 32
expect_usage_error "rc crf + ras" \
  "$INPUT" -o "$SENTINEL" --video --frames 3 --video-ras --rc crf --crf 32
expect_usage_error "rc crf + avg-cdf chain" \
  "$INPUT" -o "$SENTINEL" --video --frames 3 --avg-cdf --adaptive-cdf \
  --rc crf --crf 32

# Frame-count / order-hint preflight (SCOUT-INTER-ADV-02).
expect_usage_error "olk frame count" \
  "$INPUT" -o "$SENTINEL" --video --frames 3 --video-olk
expect_usage_error "switch frame count" \
  "$INPUT" -o "$SENTINEL" --video --frames 3 --video-switch
expect_usage_error "ras frame count" \
  "$INPUT" -o "$SENTINEL" --video --frames 2 --video-ras
expect_usage_error "sef frame count" \
  "$INPUT" -o "$SENTINEL" --video --frames 1 --video-sef
expect_usage_error "leading frame count" \
  "$INPUT" -o "$SENTINEL" --video --frames 3 --video-leading 2
expect_usage_error "leading slot cap" \
  "$INPUT" -o "$SENTINEL" --video --frames 10 --video-leading 4
expect_usage_error "tlayers frame count" \
  "$INPUT" -o "$SENTINEL" --video --frames 6 --video-tlayers 4
expect_usage_error "tlayer-drop above plan maximum" \
  "$INPUT" -o "$SENTINEL" --video --frames 4 --video-tlayers 4 \
  --video-tlayer-drop 2
expect_usage_error "b-frames frame count" \
  "$INPUT" -o "$SENTINEL" --video --frames 3 --b-frames 2
expect_usage_error "long-term order-hint cap" \
  "$INPUT" -o "$SENTINEL" --video --frames 17 --video-long-term

# Reference-model bounds (§5.18.2 REFS_PER_FRAME = 7, §5.4.6 NumRefFrames 4..8)
# and the §7.8 implicit-map carrier.  None of these may be clamped silently.
expect_usage_error "refs above REFS_PER_FRAME" \
  "$INPUT" -o "$SENTINEL" --video --frames 4 --video-refs 8
expect_usage_error "num-ref-frames below 4" \
  "$INPUT" -o "$SENTINEL" --video --frames 4 --video-num-ref-frames 3
expect_usage_error "refs above NumRefFrames" \
  "$INPUT" -o "$SENTINEL" --video --frames 4 --video-refs 5 \
  --video-num-ref-frames 4
expect_usage_error "implicit reference map" \
  "$INPUT" -o "$SENTINEL" --video --frames 4 --video-implicit-ref-map

# Unsupported CPU Still tool pairs (SCOUT-QUANT-01).
STILL_INPUT="$ROOT/tests/data/basic_640x360_1f.y4m"
if [ -f "$STILL_INPUT" ]; then
  expect_usage_error "still qm + tcq" \
    "$STILL_INPUT" -o "$SENTINEL" --still --qp 90 --still-qm --still-tcq
  expect_usage_error "still qm + cctx" \
    "$STILL_INPUT" -o "$SENTINEL" --still --qp 90 --still-qm --still-cctx
  expect_usage_error "still qm + tx-partition" \
    "$STILL_INPUT" -o "$SENTINEL" --still --qp 90 --still-qm \
    --still-tx-partition
  expect_usage_error "still qm + per-segment-q" \
    "$STILL_INPUT" -o "$SENTINEL" --still --qp 90 --still-qm \
    --still-per-segment-q
  expect_usage_error "still tcq + ist" \
    "$STILL_INPUT" -o "$SENTINEL" --still --qp 90 --still-tcq --still-ist
  expect_usage_error "still dc-ac-delta-q + cctx" \
    "$STILL_INPUT" -o "$SENTINEL" --still --qp 90 --still-dc-ac-delta-q \
    --still-cctx
  expect_usage_error "still sdp + cdef" \
    "$STILL_INPUT" -o "$SENTINEL" --still --qp 90 --still-sdp --still-cdef
  expect_usage_error "still sdp + segmentation" \
    "$STILL_INPUT" -o "$SENTINEL" --still --qp 90 --still-sdp \
    --still-segmentation
  expect_usage_error "still tx-partition + dip" \
    "$STILL_INPUT" -o "$SENTINEL" --still --qp 90 --still-tx-partition \
    --still-dip
  expect_usage_error "still ddt (inter-only)" \
    "$STILL_INPUT" -o "$SENTINEL" --still --qp 90 --still-ddt
else
  echo "SKIP still tool-pair preflight: missing $STILL_INPUT"
fi

if [ "$usage_failures" -ne 0 ]; then
  echo "FAIL: $usage_failures/$usage_cases usage-preflight case(s)" >&2
  exit 1
fi
echo "PASS: $usage_cases usage-preflight case(s), exit 2 and -o untouched"

# Accepted rate-control/frame-type combinations must keep encoding.
file_size() {
  if stat -c%s "$1" >/dev/null 2>&1; then
    stat -c%s "$1"
  else
    stat -f%z "$1"
  fi
}
for spec in "switch:--frames 2 --video-switch" "sef:--frames 4 --video-sef"; do
  name=${spec%%:*}
  read -r -a args <<< "${spec#*:}"
  if "$BIN" "$INPUT" -o "${OUT}.${name}.ivf" --video "${args[@]}" \
      --rc crf --crf 32 >"${OUT}.${name}.log" 2>&1; then
    if [ "$(file_size "${OUT}.${name}.ivf")" -gt 32 ]; then
      echo "PASS accepted: --rc crf + ${name}"
    else
      echo "FAIL accepted: --rc crf + ${name}: header-only output" >&2
      usage_failures=$((usage_failures + 1))
    fi
  else
    echo "FAIL accepted: --rc crf + ${name}: encode failed" >&2
    usage_failures=$((usage_failures + 1))
  fi
done
if [ "$usage_failures" -ne 0 ]; then
  exit 1
fi

# Accepted reference-model combinations must keep encoding.  The GPU parity
# gate (src/cuda/pipeline/video/gate_video_gpu.sh) pins the bytes for the same
# configurations; this keeps a CPU-only build covering the CLI surface.
for spec in "refs7:--video-refs 7" \
            "nrf4:--video-refs 4 --video-num-ref-frames 4"; do
  name=${spec%%:*}
  read -r -a args <<< "${spec#*:}"
  if "$BIN" "$INPUT" -o "${OUT}.${name}.ivf" --video --frames 4 \
      "${args[@]}" >"${OUT}.${name}.log" 2>&1 &&
      [ "$(file_size "${OUT}.${name}.ivf")" -gt 32 ]; then
    echo "PASS accepted: reference model ${name}"
  else
    echo "FAIL accepted: reference model ${name}" >&2
    usage_failures=$((usage_failures + 1))
  fi
done
if [ "$usage_failures" -ne 0 ]; then
  exit 1
fi


# Long-term mode codes the frame index as OrderHint with OrderHintBits = 4.
reject "long-term frame cap" "at most 16 coded frames" \
  --video --frames 17 --video-long-term
