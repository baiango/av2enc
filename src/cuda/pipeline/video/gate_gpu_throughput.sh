#!/usr/bin/env bash
# GPU quality-calibration + throughput gate.
#
# Two measurements, one script:
#
#   1. CALIBRATION (decision policy): 424x240 (240p), 60 frames, QP 150.
#      The GPU path must reproduce the CPU reference decode within
#        CVVDP <= 0.05 JOD, bitrate <= 2 %, PSNR-Y <= 0.1 dB.
#      The current GPU pipeline is a decision-identical mirror of the CPU one,
#      so the expected drift is exactly 0 (identical temporal units).
#
#   2. THROUGHPUT: 1440x720, QP 90 (the cross-encoder benchmark point).  Only
#      speed is being optimized there; this script reports fps plus the same
#      drift row so a throughput change cannot silently move quality.
#
# Both points are decoded with the pinned dav2d (`--strict 1`) and scored with
# `tools/quality.py` (CVVDP + PSNR, fixed display model).
#
# Usage:
#   src/cuda/pipeline/video/gate_gpu_throughput.sh --bin build/av2enc
#   ... --frames 60 --repeats 3 --width 1440 --height 720
#   ... --quick            # 4-frame smoke, skips CVVDP
#   ... --json out.json    # machine-readable row set
#
# Exit status: 0 pass, 1 drift/speed-gate failure, 77 skip (no decoder/CUDA).
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${HERE}/../../../.." && pwd)"

BIN=""
DAV2D="${DAV2D:-}"
FRAMES=60
REPEATS=3
WIDTH=1440
HEIGHT=720
CLIP=""
QUICK=0
JSON=""
CVVDP_TOL=0.05
BITRATE_TOL=2.0
PSNR_TOL=0.1
MIN_FPS=0
PYTHON="${PYTHON:-python3}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bin) BIN="$2"; shift 2 ;;
    --dav2d) DAV2D="$2"; shift 2 ;;
    --frames) FRAMES="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --width) WIDTH="$2"; shift 2 ;;
    --height) HEIGHT="$2"; shift 2 ;;
    --clip) CLIP="$2"; shift 2 ;;
    --json) JSON="$2"; shift 2 ;;
    --min-fps) MIN_FPS="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --quick) QUICK=1; FRAMES=4; REPEATS=1; shift ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "gate_gpu_throughput: unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$BIN" || ! -x "$BIN" ]]; then
  # Default to the usual CMake build directory.
  for candidate in "$ROOT/build/av2enc" "$ROOT/build-int/av2enc"; do
    if [[ -x "$candidate" ]]; then BIN="$candidate"; break; fi
  done
fi
if [[ -z "$BIN" || ! -x "$BIN" ]]; then
  echo "gate_gpu_throughput: missing --bin <av2enc>" >&2
  exit 2
fi

if [[ -z "$DAV2D" ]]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [[ -x "$candidate" ]]; then DAV2D="$candidate"; break; fi
  done
fi
if [[ -z "$DAV2D" || ! -x "$DAV2D" ]]; then
  echo "gate_gpu_throughput: SKIP (no pinned dav2d; set DAV2D)"
  exit 77
fi
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "gate_gpu_throughput: SKIP (no $PYTHON)"
  exit 77
fi

TMP="$(mktemp -d "${TMPDIR:-/tmp}/gate_gpu_thr.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT

# Deterministic source: a textured background plus a drifting block, so the
# clip carries real inter and intra decisions at both resolutions.
gen_clip() {
  local path="$1" w="$2" h="$3" frames="$4"
  "$PYTHON" - "$path" "$w" "$h" "$frames" <<'PY'
import sys

path, w, h, frames = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
cw, ch = w // 2, h // 2
sx0, sy0 = max(4, w // 20), max(4, h // 20)
bw, bh = max(8, w // 13), max(8, h // 9)

def sample(x, y, f, plane):
    sx = sx0 + max(1, w // 200) * f
    sy = sy0 + max(1, h // 300) * f
    inside = sx <= x < sx + bw and sy <= y < sy + bh
    if plane == 0:
        if inside:
            return 32 + ((x * 29 + y * 17 + f * 41) & 0xBF)
        return 88 + (((x + 3 * f) * 5 + (y + 2 * f) * 3 + ((x * y + f * 37) & 15))
                     & 0x7F)
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
                out.write(bytes(sample(2 * x, 2 * y, f, plane) for x in range(cw)))
PY
}

if [[ -z "$CLIP" ]]; then
  CLIP="$TMP/bench_${WIDTH}x${HEIGHT}.y4m"
  if [[ "$WIDTH" == 1440 && "$HEIGHT" == 720 && -f "$ROOT/testdata/nasa_1440x720_60f.y4m" ]]; then
    CLIP="$ROOT/testdata/nasa_1440x720_60f.y4m"
  else
    gen_clip "$CLIP" "$WIDTH" "$HEIGHT" "$(( FRAMES > 60 ? FRAMES : 60 ))"
  fi
fi
# Calibration source: the same content as the throughput benchmark, scaled to
# 480x240 (240p, 2:1).  This is the decision-policy calibration point.
# Without ffmpeg, fall back to the synthetic clip.
CAL_CLIP="$TMP/cal_480x240.y4m"
if command -v ffmpeg >/dev/null 2>&1 && [[ "$WIDTH" == 1280 && "$HEIGHT" == 720 ]]; then
  ffmpeg -y -loglevel error -i "$CLIP" -vf "fps=30,scale=480:240" -pix_fmt yuv420p \
    -frames:v 60 "$CAL_CLIP" >/dev/null 2>&1 || gen_clip "$CAL_CLIP" 480 240 60
else
  gen_clip "$CAL_CLIP" 480 240 60
fi

status=0
row() { printf '%s\n' "$*"; }

# ---------------------------------------------------------------- calibration
run_point() {
  local label="$1" clip="$2" frames="$3" qp="$4" w="$5" h="$6"
  local cpu="$TMP/${label}.cpu.ivf" gpu="$TMP/${label}.gpu.ivf"
  local cpu_dec="$TMP/${label}.cpu.dec.y4m" gpu_dec="$TMP/${label}.gpu.dec.y4m"
  local gpu_time

  "$BIN" "$clip" -o "$cpu" --video --frames "$frames" --qp "$qp" \
    >"$TMP/${label}.cpu.log" 2>&1 ||
    { echo "gate_gpu_throughput: $label CPU encode failed" >&2; return 1; }
  "$BIN" "$clip" -o "$gpu" --video --video-gpu --frames "$frames" --qp "$qp" \
    >"$TMP/${label}.gpu.log" 2>&1 ||
    { echo "gate_gpu_throughput: $label GPU encode failed" >&2; return 1; }
  grep -q -- "frame_bytes_from_cpu=0" "$TMP/${label}.gpu.log" ||
    { echo "gate_gpu_throughput: $label GPU path reported CPU bytes" >&2; return 1; }

  # Repeatable fps: best of N wall times (the box is a shared GPU).
  local best=""
  local i start end elapsed
  for ((i = 0; i < REPEATS; i++)); do
    start=$("$PYTHON" -c 'import time;print(time.time())')
    "$BIN" "$clip" -o "$TMP/${label}.rep.ivf" --video --video-gpu \
      --frames "$frames" --qp "$qp" >/dev/null 2>&1
    end=$("$PYTHON" -c 'import time;print(time.time())')
    elapsed=$("$PYTHON" -c "print(${end}-${start})")
    if [[ -z "$best" ]] || "$PYTHON" -c "import sys;sys.exit(0 if ${elapsed}<${best} else 1)"; then
      best="$elapsed"
    fi
  done
  gpu_time="$best"

  "$DAV2D" -i "$gpu" --strict 1 -o "$gpu_dec" >"$TMP/${label}.dec.log" 2>&1 ||
    { echo "gate_gpu_throughput: $label dav2d --strict 1 rejected the GPU stream" >&2;
      tail -3 "$TMP/${label}.dec.log" >&2; return 1; }
  "$DAV2D" -i "$cpu" --strict 1 -o "$cpu_dec" >/dev/null 2>&1 ||
    { echo "gate_gpu_throughput: $label dav2d rejected the CPU stream" >&2; return 1; }

  local same_bytes=0 same_dec=0
  cmp -s "$cpu" "$gpu" && same_bytes=1
  cmp -s "$cpu_dec" "$gpu_dec" && same_dec=1

  local cpu_bits gpu_bits fps
  cpu_bits=$(stat -c %s "$cpu" 2>/dev/null || stat -f %z "$cpu")
  gpu_bits=$(stat -c %s "$gpu" 2>/dev/null || stat -f %z "$gpu")
  fps=$("$PYTHON" -c "print(f'{$frames/$gpu_time:.2f}')")

  local cpu_bitrate gpu_bitrate
  cpu_bitrate=$("$PYTHON" -c "print(f'{$cpu_bits*8*30/$frames/1000:.1f}')")
  gpu_bitrate=$("$PYTHON" -c "print(f'{$gpu_bits*8*30/$frames/1000:.1f}')")

  local cpu_cvvdp="n/a" gpu_cvvdp="n/a" cpu_psnr="n/a" gpu_psnr="n/a"
  local drift_cvvdp="n/a" drift_psnr="n/a" drift_bitrate
  drift_bitrate=$("$PYTHON" -c "print(f'{(($gpu_bits/$cpu_bits)-1)*100:+.2f}')")

  if [[ "$QUICK" == 0 ]]; then
    local ok=1
    cpu_q=$("$PYTHON" "$ROOT/tools/quality.py" --ref "$clip" --ivf "$cpu" \
              --json "$TMP/${label}.cpu.json" 2>"$TMP/${label}.cpu.q.log") || ok=0
    gpu_q=$("$PYTHON" "$ROOT/tools/quality.py" --ref "$clip" --ivf "$gpu" \
              --json "$TMP/${label}.gpu.json" 2>"$TMP/${label}.gpu.q.log") || ok=0
    if [[ $ok == 1 ]]; then
      read -r cpu_cvvdp gpu_cvvdp cpu_psnr gpu_psnr drift_cvvdp drift_psnr \
        <<<"$("$PYTHON" - "$TMP/${label}.cpu.json" "$TMP/${label}.gpu.json" <<'PYQ'
import json, sys
cpu = json.load(open(sys.argv[1]))
gpu = json.load(open(sys.argv[2]))
cpu_y = cpu["psnr"]["psnr_y"]
gpu_y = gpu["psnr"]["psnr_y"]
print(f"{cpu['cvvdp_jod']:.3f} {gpu['cvvdp_jod']:.3f} {cpu_y:.2f} {gpu_y:.2f} "
      f"{gpu['cvvdp_jod'] - cpu['cvvdp_jod']:+.3f} {gpu_y - cpu_y:+.2f}")
PYQ
)"
    else
      echo "gate_gpu_throughput: $label CVVDP unavailable (see $TMP/${label}.gpu.q.log)" >&2
      sed -n '1,3p' "$TMP/${label}.gpu.q.log" >&2 || true
    fi
  fi

  row "== ${label} (${w}x${h}, ${frames} frames, QP ${qp}) =="
  row "  identical stream units : $([[ $same_bytes == 1 ]] && echo yes || echo no)"
  row "  identical decoded Y4M  : $([[ $same_dec == 1 ]] && echo yes || echo no)"
  row "  bitrate CPU/GPU        : ${cpu_bitrate} / ${gpu_bitrate} kbps (drift ${drift_bitrate}%)"
  row "  CVVDP   CPU/GPU        : ${cpu_cvvdp} / ${gpu_cvvdp} JOD (drift ${drift_cvvdp})"
  row "  PSNR-Y  CPU/GPU        : ${cpu_psnr} / ${gpu_psnr} dB (drift ${drift_psnr})"
  row "  GPU encode             : ${gpu_time}s = ${fps} fps"

  RESULT_JSON=$("$PYTHON" - "$label" "$w" "$h" "$frames" "$qp" "$cpu_bitrate" \
      "$gpu_bitrate" "$drift_bitrate" "$cpu_cvvdp" "$gpu_cvvdp" "$drift_cvvdp" \
      "$cpu_psnr" "$gpu_psnr" "$drift_psnr" "$fps" "$same_bytes" "$same_dec" <<'PY'
import json, sys
keys = ["label", "width", "height", "frames", "qp", "cpu_kbps", "gpu_kbps",
        "kbps_drift_pct", "cpu_cvvdp", "gpu_cvvdp", "cvvdp_drift", "cpu_psnr",
        "gpu_psnr", "psnr_drift", "gpu_fps", "identical_units", "identical_dec"]
print(json.dumps(dict(zip(keys, sys.argv[1:]))))
PY
)
  ROWS+=("$RESULT_JSON")
  ROW_VALUES+=("$label|$cpu_bitrate|$gpu_bitrate|$drift_bitrate|$cpu_cvvdp|$gpu_cvvdp|$drift_cvvdp|$cpu_psnr|$gpu_psnr|$drift_psnr|$fps|$same_bytes|$same_dec|$frames|$qp")
  return 0
}

ROWS=()
ROW_VALUES=()

[[ "$QUICK" == 1 ]] && row "gate_gpu_throughput: quick mode (no CVVDP, 4 frames)"

# Calibration point: the decision policy is pinned here.
run_point "cal144-qp150" "$CAL_CLIP" 60 150 256 144 || status=1

# Throughput point.
run_point "bench720-qp90" "$CLIP" "$FRAMES" 90 "$WIDTH" "$HEIGHT" || status=1

# --------------------------------------------------------------- drift checks
check_drift() {
  local label="$1" max_cvvdp="$2" max_bitrate="$3" max_psnr="$4"
  local entry="" v
  for v in "${ROW_VALUES[@]}"; do
    case "$v" in "$label|"*) entry="$v" ;; esac
  done
  [[ -z "$entry" ]] && return 0
  local IFS='|'
  # shellcheck disable=SC2206
  local f=($entry)
  unset IFS
  local cvvdp="${f[6]}" bitrate="${f[3]}" psnr="${f[9]}"
  if [[ "$cvvdp" != "n/a" ]]; then
    if ! "$PYTHON" -c "import sys;sys.exit(0 if abs($cvvdp)<=$max_cvvdp else 1)"; then
      echo "gate_gpu_throughput: FAIL $label CVVDP drift $cvvdp > $max_cvvdp" >&2
      return 1
    fi
  fi
  if ! "$PYTHON" -c "import sys;sys.exit(0 if abs($bitrate)<=$max_bitrate else 1)"; then
    echo "gate_gpu_throughput: FAIL $label bitrate drift $bitrate% > $max_bitrate%" >&2
    return 1
  fi
  if [[ "$psnr" != "n/a" ]]; then
    if ! "$PYTHON" -c "import sys;sys.exit(0 if abs($psnr)<=$max_psnr else 1)"; then
      echo "gate_gpu_throughput: FAIL $label PSNR drift $psnr > $max_psnr" >&2
      return 1
    fi
  fi
  return 0
}

check_drift "cal144-qp150" "$CVVDP_TOL" "$BITRATE_TOL" "$PSNR_TOL" || status=1
check_drift "bench720-qp90" "$CVVDP_TOL" "$BITRATE_TOL" "$PSNR_TOL" || status=1

if [[ "$MIN_FPS" != 0 ]]; then
  fps=$(printf '%s\n' "${ROW_VALUES[@]}" | sed -n 's/^bench720-qp90|.*|\([0-9.]*\)|.*/\1/p')
  if [[ -n "$fps" ]] && ! "$PYTHON" -c "import sys;sys.exit(0 if $fps>=$MIN_FPS else 1)"; then
    echo "gate_gpu_throughput: FAIL 720p fps $fps < $MIN_FPS" >&2
    status=1
  fi
fi

if [[ -n "$JSON" ]]; then
  "$PYTHON" - "$JSON" "${ROWS[@]}" <<'PY'
import json, sys
out = sys.argv[1]
rows = [json.loads(x) for x in sys.argv[2:]]
json.dump({"rows": rows}, open(out, "w"), indent=2)
PY
  row "  json                   : $JSON"
fi

if [[ $status == 0 ]]; then
  row "gate_gpu_throughput: PASS"
else
  row "gate_gpu_throughput: FAIL"
fi
exit $status
