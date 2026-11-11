#!/usr/bin/env bash
# CPU-vs-GPU gate for the M1 still-picture path.
#
#   CPU:  av2enc <in.y4m> -o cpu.ivf --m1
#   GPU:  av2enc <in.y4m> -o gpu.ivf --m1-gpu
#
# Gate (all must hold for PASS):
#   1. the GPU execution is *attested* (stats file, output marker, or Nsight
#      Systems shows >=1 CUDA kernel) so a silent CPU fallback cannot pass;
#   2. IVF **frame payloads** are byte-identical (not PSNR);
#   3. both streams decode with the pinned dav2d;
#   4. the decoded first frames are plane-identical;
#   5. in --recon mode, GPU recon planes equal the CPU M1Result.recon_* dump
#      when the orchestrator exposes one (otherwise the decoded-frame compare
#      above is the fallback).
#
# Verdicts / exit codes:
#   0  PASS
#   1  FAIL            (real byte/plane/decode difference, or GPU crash)
#   2  usage/setup     (missing encoder/dav2d/ffmpeg/input)
#   3  SKIP            (no --m1-gpu in this build; GPU run is a stub/blocked;
#                       or GPU execution could not be attested -> possible
#                       CPU fallback).  A SKIP is never a pass.
#
# ALLOW_SKIP=1 maps SKIP to exit 0 for CI that wants "not a failure".
#
# Usage:
#   tools/m1_gpu/verify_gpu.sh [input.y4m] [options]
#
#   tools/m1_gpu/verify_gpu.sh                         # generate 640x360
#   tools/m1_gpu/verify_gpu.sh in.y4m --recon
#   tools/m1_gpu/verify_gpu.sh --size 1920x1080 --keep
#   tools/m1_gpu/verify_gpu.sh --allow-skip
#
# Options:
#   --encoder PATH        av2enc binary (default $AV2ENC or build-cuda/av2enc)
#   --decoder PATH        dav2d (default $DAV2D, tools/bin/dav2d[-macos])
#   --size WxH            generate this input when no input is given
#                         (default 640x360 testsrc2)
#   --workdir DIR         artifact directory (default mktemp -d)
#   --qp N                base_q_idx for both encodes (default 90)
#   --cpu-args "...".     extra args for the CPU run only
#   --gpu-args "..."      extra args for the GPU run only
#   --recon               compare encoder-side recon planes too
#   --harness PATH        m1_dump binary for the CPU recon dump
#   --harness-args "..."  extra m1_dump args (must match the CLI options)
#   --gpu-dump PATH       GPU M1DUMP1 file to compare (implies --recon)
#   --gpu-dump-arg FLAG   append "FLAG <workdir>/gpu-recon.dump" to the GPU run
#   --probe PATH          optional m1_gpu_probe binary (auto: tools/m1_gpu/
#                         m1_gpu_probe, build-cuda/m1_gpu_probe); used for
#                         recon parity when the CLI exposes no dump
#   --probe-args "..."    extra probe args (must match the CLI options)
#   --require-recon       SKIP if no GPU recon dump is available
#   --no-attestation      PASS on identical bytes even without GPU evidence
#                         (explicitly unsafe: a CPU fallback would pass)
#   --no-nsys             do not use Nsight Systems as the last-resort
#                         attestation source (faster; more SKIPs)
#   --keep                keep the work directory
#   --timeout SECONDS     per-encode timeout (default 1800)
#   -h, --help
#
# Environment:
#   AV2ENC_GPU_STATS / AV2ENC_M1_GPU_STATS  stats JSON path (see README)
#   AV2ENC_M1_GPU_DUMP                     recon dump path (see README)
#   NSYS                                   nsys binary (default: PATH lookup)
#   PYTHON                                 python3 interpreter
#
set -uo pipefail

SELF=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$SELF/../.." && pwd)
PY=${PYTHON:-python3}
NSYS=${NSYS:-nsys}

ENCODER=${AV2ENC:-$ROOT/build-cuda/av2enc}
DECODER=${DAV2D:-}
WORKDIR=""
INPUT=""
SIZE="640x360"
QP=90
KEEP=0
ATTEST="auto"
RECON=0
REQUIRE_RECON=0
HARNESS=${M1_DUMP:-}
HARNESS_ARGS=""
CPU_ARGS=""
GPU_ARGS=""
GPU_DUMP=""
GPU_DUMP_ARG=""
PROBE=""
PROBE_ARGS=""
NO_ATTESTATION=0
TIMEOUT=${TIMEOUT:-1800}

usage() {
    sed -n '2,/^set -uo pipefail/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'
}

while [ $# -gt 0 ]; do
    case "$1" in
        --encoder) ENCODER=$2; shift 2 ;;
        --decoder) DECODER=$2; shift 2 ;;
        --size) SIZE=$2; shift 2 ;;
        --workdir) WORKDIR=$2; shift 2 ;;
        --qp) QP=$2; shift 2 ;;
        --cpu-args) CPU_ARGS=$2; shift 2 ;;
        --gpu-args) GPU_ARGS=$2; shift 2 ;;
        --recon) RECON=1; shift ;;
        --require-recon) RECON=1; REQUIRE_RECON=1; shift ;;
        --harness) HARNESS=$2; shift 2 ;;
        --harness-args) HARNESS_ARGS=$2; shift 2 ;;
        --gpu-dump) GPU_DUMP=$2; RECON=1; shift 2 ;;
        --gpu-dump-arg) GPU_DUMP_ARG=$2; RECON=1; shift 2 ;;
        --probe) PROBE=$2; RECON=1; shift 2 ;;
        --probe-args) PROBE_ARGS=$2; RECON=1; shift 2 ;;
        --no-attestation) NO_ATTESTATION=1; shift ;;
        --no-nsys) ATTEST=off; shift ;;
        --keep) KEEP=1; shift ;;
        --timeout) TIMEOUT=$2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        --*) echo "verify_gpu: unknown option: $1" >&2; exit 2 ;;
        *) INPUT=$1; shift ;;
    esac
done

say() { printf '%s\n' "$*"; }

skip() {
    local reason=$1
    say ""
    say "RESULT: SKIP ($reason)"
    if [ "${ALLOW_SKIP:-0}" = "1" ]; then
        exit 0
    fi
    exit 3
}

setup_error() {
    say "verify_gpu: error: $*" >&2
    exit 2
}

command -v "$PY" >/dev/null 2>&1 || setup_error "python3 not found"
[ -x "$ENCODER" ] || setup_error "encoder not executable: $ENCODER (build it first)"
command -v ffmpeg >/dev/null 2>&1 || [ -n "$INPUT" ] || \
    setup_error "ffmpeg not found; pass an input Y4M or install ffmpeg"

# ---------------------------------------------------------------------------
# 1. Does this build even have the GPU path?  No flag -> SKIP, never PASS.
# ---------------------------------------------------------------------------
HELP=$("$ENCODER" --help 2>&1 || true)
if ! grep -q -- '--m1-gpu' <<< "$HELP"; then
    say "== M1 GPU gate =="
    say "encoder:  $ENCODER"
    say "gate:     this build has no --m1-gpu option"
    skip "GPU path not present in this encoder build"
fi

# ---------------------------------------------------------------------------
# 2. Decoder, input, work directory.
# ---------------------------------------------------------------------------
if [ -z "$DECODER" ]; then
    for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
        if [ -x "$candidate" ]; then DECODER=$candidate; break; fi
    done
fi
[ -n "$DECODER" ] && [ -x "$DECODER" ] || \
    setup_error "dav2d not found; set DAV2D=/path/to/dav2d"

if [ -z "$WORKDIR" ]; then
    WORKDIR=$(mktemp -d "${TMPDIR:-/tmp}/m1_gpu_verify.XXXXXX")
    CLEAN=1
else
    mkdir -p "$WORKDIR" || setup_error "cannot create $WORKDIR"
    CLEAN=0
fi
if [ "$KEEP" = "1" ]; then CLEAN=0; fi
cleanup() { if [ "$CLEAN" = "1" ]; then rm -rf "$WORKDIR"; fi; }
trap cleanup EXIT

if [ -z "$INPUT" ]; then
    INPUT="$WORKDIR/input-$SIZE.y4m"
    MATCH=$(printf '%s' "$SIZE" | sed -n 's/^\([0-9][0-9]*\)x\([0-9][0-9]*\)$/\1 \2/p')
    [ -n "$MATCH" ] || setup_error "bad --size '$SIZE' (want WxH)"
    W=${MATCH% *}
    H=${MATCH#* }
    say "input:    generating $SIZE testsrc2"
    if [ "$W" -gt 640 ]; then
        ffmpeg -hide_banner -v error -y -f lavfi \
            -i "testsrc2=size=640x360:rate=1" -frames:v 1 -pix_fmt yuv420p \
            -f yuv4mpegpipe "$WORKDIR/.src.y4m" || setup_error "ffmpeg input generation failed"
        ffmpeg -hide_banner -v error -y -i "$WORKDIR/.src.y4m" \
            -vf "scale=$W:$H:flags=lanczos" -frames:v 1 -pix_fmt yuv420p \
            -f yuv4mpegpipe "$INPUT" || setup_error "ffmpeg input generation failed"
    else
        ffmpeg -hide_banner -v error -y -f lavfi \
            -i "testsrc2=size=${W}x${H}:rate=1" -frames:v 1 -pix_fmt yuv420p \
            -f yuv4mpegpipe "$INPUT" || setup_error "ffmpeg input generation failed"
    fi
fi
[ -f "$INPUT" ] || setup_error "no such input: $INPUT"

say "== M1 GPU gate =="
say "encoder:  $ENCODER"
say "decoder:  $DECODER"
say "input:    $INPUT"
say "workdir:  $WORKDIR"

# ---------------------------------------------------------------------------
# 3. Encode CPU reference and GPU candidate.
# ---------------------------------------------------------------------------
CPU_IVF="$WORKDIR/cpu.ivf"
GPU_IVF="$WORKDIR/gpu.ivf"
GPU_LOG="$WORKDIR/gpu.log"
GPU_STATS="$WORKDIR/gpu-stats.json"
AUTO_DUMP="$WORKDIR/gpu-recon.dump"

# shellcheck disable=SC2086
"$ENCODER" "$INPUT" -o "$CPU_IVF" --m1 --qp "$QP" $CPU_ARGS \
    >"$WORKDIR/cpu.log" 2>&1
CPU_RC=$?
if [ $CPU_RC -ne 0 ] || [ ! -s "$CPU_IVF" ]; then
    say ""
    say "cpu log:"
    sed 's/^/  /' "$WORKDIR/cpu.log" | tail -20
    say "RESULT: FAIL (CPU reference encode failed, exit $CPU_RC)"
    exit 1
fi

GPU_ENV=(env "AV2ENC_GPU_STATS=$GPU_STATS" \
          "AV2ENC_M1_GPU_STATS=$GPU_STATS" \
          "AV2ENC_M1_GPU_DUMP=$AUTO_DUMP")
GPU_CMD=("$ENCODER" "$INPUT" -o "$GPU_IVF" --m1-gpu --qp "$QP")
if [ -n "$GPU_ARGS" ]; then
    # shellcheck disable=SC2206
    GPU_CMD+=($GPU_ARGS)
fi
if [ -n "$GPU_DUMP_ARG" ]; then
    if [ -z "$GPU_DUMP" ]; then GPU_DUMP="$AUTO_DUMP"; fi
    GPU_CMD+=("$GPU_DUMP_ARG" "$GPU_DUMP")
fi
"${GPU_ENV[@]}" "${GPU_CMD[@]}" >"$GPU_LOG" 2>&1
GPU_RC=$?

# ---------------------------------------------------------------------------
# 4. Classify GPU run: pass through real work, SKIP stubs/blockers.
# ---------------------------------------------------------------------------
if [ $GPU_RC -ne 0 ]; then
    say ""
    say "gpu log:"
    sed 's/^/  /' "$GPU_LOG" | tail -20
    if grep -Eqi '(not[ _-]+implemented|unsupported|stub|not[ _-]+built|not[ _-]+available|no[ _-]+cuda[ _-]+device|CUDA_ERROR_NO_DEVICE|disabled|blocked|TODO|requires[ _-]+a[ _-]+cuda[ _-]+build|cpu[ _-]+only)' "$GPU_LOG"; then
        skip "GPU run reported an unimplemented/blocked path (exit $GPU_RC)"
    fi
    say "RESULT: FAIL (GPU encode failed, exit $GPU_RC)"
    exit 1
fi

if [ ! -s "$GPU_IVF" ]; then
    say ""
    say "gpu log:"
    sed 's/^/  /' "$GPU_LOG" | tail -20
    skip "GPU run exited 0 but wrote no IVF (stub/blocked path)"
fi

# The orchestrator stages CPU fallback while stage hooks are missing and can
# still exit 0 with byte-identical output. That is a stub, not a GPU result.
if grep -Eqi 'frame bytes came from the CPU|cpu[ _-]+(m1[ _-]+)?reference|stage hooks are not linked|frame_bytes_from_cpu' "$GPU_LOG"; then
    say ""
    say "gpu log:"
    sed 's/^/  /' "$GPU_LOG" | tail -20
    skip "GPU run used CPU-reference staging; frame bytes did not come from GPU stages"
fi

# ---------------------------------------------------------------------------
# 5. Byte gate: IVF frame payloads (not PSNR) + MD5.
# ---------------------------------------------------------------------------
say ""
say "-- frame bytes --"
"$PY" "$SELF/ivf_compare.py" "$CPU_IVF" "$GPU_IVF" | tee "$WORKDIR/ivf_compare.txt"
BYTES_RC=${PIPESTATUS[0]}
BYTES_STATUS=$([ $BYTES_RC -eq 0 ] && echo IDENTICAL || echo DIFFER)

# ---------------------------------------------------------------------------
# 6. Attestation that the GPU path actually ran.
# ---------------------------------------------------------------------------
ATTEST_RESULT="none"

stats_attests() {
    [ -s "$1" ] || return 1
    "$PY" - "$1" <<'PY'
import json, sys
try:
    data = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(1)
backend = str(data.get("backend", data.get("path", ""))).lower()
if data.get("frame_bytes_from_cpu") is True:
    sys.exit(1)
if backend in ("gpu", "m1-gpu", "cuda"):
    sys.exit(0)
for key in ("kernels", "kernel_count", "instances", "launches", "host_launches"):
    value = data.get(key)
    if isinstance(value, (int, float)) and value > 0:
        sys.exit(0)
sys.exit(1)
PY
}

nsys_kernels() {
    local rep="$1"
    "$PY" - "$rep" <<'PY'
import csv, subprocess, sys
path = sys.argv[1]
proc = subprocess.run(["nsys", "stats", "--force-export=true",
                       "--report", "cuda_gpu_kern_sum",
                       "--format", "csv", path],
                      stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                      text=True)
lines = proc.stdout.splitlines()
header = next((i for i, line in enumerate(lines)
               if line.startswith("Time (%)")), None)
if header is None:
    print(0)
    raise SystemExit(0)
total = 0
for row in csv.DictReader(lines[header:]):
    if not row.get("Name"):
        continue
    try:
        total += int(row.get("Instances", "0"))
    except ValueError:
        pass
print(total)
PY
}

if stats_attests "$GPU_STATS"; then
    ATTEST_RESULT="stats"
elif grep -Eqi 'backend[[:space:]]*[=:][[:space:]]*gpu|using[[:space:]]+the[[:space:]]+GPU|GPU[[:space:]]+path[[:space:]]+(ran|active|enabled)|stages?[[:space:]]*:[^\\n]*gpu' "$GPU_LOG"; then
    ATTEST_RESULT="output marker"
elif [ "$ATTEST" != "off" ] && { command -v "$NSYS" >/dev/null 2>&1 || [ -x "$NSYS" ]; }; then
    say ""
    say "-- attestation (nsys) --"
    NSYS_PREFIX="$WORKDIR/attest"
    rm -f "$NSYS_PREFIX.nsys-rep" "$NSYS_PREFIX.sqlite"

    # The attestation run must not overwrite the artifact that was byte-gated
    # and will be decoded: redirect its IVF, stats and recon dump to attest.*.
    ATTEST_IVF="$WORKDIR/attest.ivf"
    ATTEST_DUMP="$WORKDIR/attest-recon.dump"
    ATTEST_CMD=()
    next_output=0
    next_dump=0
    for part in "${GPU_CMD[@]}"; do
        if [ $next_output -eq 1 ]; then ATTEST_CMD+=("$ATTEST_IVF"); next_output=0; continue; fi
        if [ $next_dump -eq 1 ]; then ATTEST_CMD+=("$ATTEST_DUMP"); next_dump=0; continue; fi
        if [ "$part" = "-o" ] || [ "$part" = "--output" ]; then
            ATTEST_CMD+=("$part"); next_output=1; continue
        fi
        if [ -n "$GPU_DUMP_ARG" ] && [ "$part" = "$GPU_DUMP_ARG" ]; then
            ATTEST_CMD+=("$part"); next_dump=1; continue
        fi
        ATTEST_CMD+=("$part")
    done
    ATTEST_ENV=(env
        "AV2ENC_GPU_STATS=$WORKDIR/attest-stats.json"
        "AV2ENC_M1_GPU_STATS=$WORKDIR/attest-stats.json"
        "AV2ENC_M1_GPU_DUMP=$ATTEST_DUMP")
    if "$NSYS" profile --trace=cuda --sample=none --cpuctxsw=none \
            --force-overwrite true -o "$NSYS_PREFIX" \
            "${ATTEST_ENV[@]}" "${ATTEST_CMD[@]}" >"$WORKDIR/attest.log" 2>&1; then
        if [ -f "$NSYS_PREFIX.nsys-rep" ]; then
            KERNELS=$(nsys_kernels "$NSYS_PREFIX.nsys-rep")
            if [ "${KERNELS:-0}" -gt 0 ] 2>/dev/null; then
                ATTEST_RESULT="nsys ($KERNELS kernels)"
                say "kernels:  $KERNELS (GPU work observed)"
            else
                say "kernels:  0 (no CUDA kernel in the GPU run)"
            fi
        else
            say "note:     nsys produced no report; see $WORKDIR/attest.log"
        fi
    else
        say "note:     nsys profile failed; see $WORKDIR/attest.log"
    fi
fi

# ---------------------------------------------------------------------------
# 7. Decode gate with the pinned dav2d, then plane compare.
# ---------------------------------------------------------------------------
DECODE_STATUS="not run"
PLANE_STATUS="not run"
CPU_Y4M="$WORKDIR/cpu-decoded.y4m"
GPU_Y4M="$WORKDIR/gpu-decoded.y4m"
say ""
say "-- dav2d decode --"
CPU_DEC_RC=0
GPU_DEC_RC=0
"$DECODER" -q -i "$CPU_IVF" -o "$CPU_Y4M" --threads 0 \
    >"$WORKDIR/cpu-decode.log" 2>&1 || CPU_DEC_RC=$?
"$DECODER" -q -i "$GPU_IVF" -o "$GPU_Y4M" --threads 0 \
    >"$WORKDIR/gpu-decode.log" 2>&1 || GPU_DEC_RC=$?
if [ $CPU_DEC_RC -ne 0 ] || [ ! -s "$CPU_Y4M" ]; then
    say "cpu decode: FAILED (exit $CPU_DEC_RC)"
    sed 's/^/  /' "$WORKDIR/cpu-decode.log" | tail -10
    DECODE_STATUS="FAIL (cpu)"
elif [ $GPU_DEC_RC -ne 0 ] || [ ! -s "$GPU_Y4M" ]; then
    say "gpu decode: FAILED (exit $GPU_DEC_RC)"
    sed 's/^/  /' "$WORKDIR/gpu-decode.log" | tail -10
    DECODE_STATUS="FAIL (gpu)"
else
    say "cpu decode: ok"
    say "gpu decode: ok"
    DECODE_STATUS="ok"
    say ""
    say "-- decoded planes --"
    "$PY" "$SELF/plane_compare.py" "$CPU_Y4M" "$GPU_Y4M" \
        | tee "$WORKDIR/plane_compare.txt"
    PLANE_RC=${PIPESTATUS[0]}
    PLANE_STATUS=$([ $PLANE_RC -eq 0 ] && echo IDENTICAL || echo DIFFER)
fi

# ---------------------------------------------------------------------------
# 8. Recon parity (GPU M1GpuResult vs CPU M1Result.recon_*), if available.
# ---------------------------------------------------------------------------
RECON_STATUS="not requested"
if [ "$RECON" = "1" ]; then
    RECON_STATUS="unavailable"
    say ""
    say "-- recon parity --"
    if [ -z "$HARNESS" ]; then
        for candidate in "$ROOT/build-cuda/m1_dump" "$ROOT/tools/m1_verify/m1_dump"; do
            if [ -x "$candidate" ]; then HARNESS=$candidate; break; fi
        done
    fi
    CPU_DUMP="$WORKDIR/cpu-recon.dump"
    GPU_SOURCE=""
    if [ -z "$GPU_DUMP" ] && [ -f "$AUTO_DUMP" ]; then
        GPU_DUMP="$AUTO_DUMP"
        GPU_SOURCE="CLI dump"
    fi

    if [ -z "$HARNESS" ] || [ ! -x "$HARNESS" ]; then
        say "note:     no m1_dump harness; using the decoded-frame compare above"
        RECON_STATUS="fallback (no CPU recon harness)"
    elif ! "$HARNESS" -i "$INPUT" -o "$CPU_DUMP" $HARNESS_ARGS \
            >"$WORKDIR/harness.log" 2>&1; then
        say "note:     m1_dump failed; using the decoded-frame compare above"
        sed 's/^/  /' "$WORKDIR/harness.log" | tail -10
        RECON_STATUS="fallback (CPU recon dump failed)"
    else
        # Optional probe: calls encode_m1_gpu directly when the CLI exposes no
        # recon dump. Only its recon is used here; attestation stays tied to
        # the CLI run that produced the byte-gated IVF.
        if [ -z "$GPU_DUMP" ]; then
            if [ -z "$PROBE" ]; then
                for candidate in "$ROOT/tools/m1_gpu/m1_gpu_probe" \
                                 "$ROOT/build-cuda/m1_gpu_probe"; do
                    if [ -x "$candidate" ]; then PROBE=$candidate; break; fi
                done
            fi
            if [ -n "$PROBE" ] && [ -x "$PROBE" ]; then
                PROBE_STATS="$WORKDIR/gpu-probe-stats.json"
                PROBE_DUMP="$WORKDIR/gpu-probe.dump"
                say "probe:    $PROBE"
                # shellcheck disable=SC2086
                if "$PROBE" -i "$INPUT" -o "$PROBE_DUMP" --stats "$PROBE_STATS" \
                        --qp "$QP" $PROBE_ARGS >"$WORKDIR/probe.log" 2>&1; then
                    if grep -Eq '"backend":[[:space:]]*"gpu"' "$PROBE_STATS" 2>/dev/null; then
                        GPU_DUMP="$PROBE_DUMP"
                        GPU_SOURCE="probe"
                        say "probe:    backend=gpu"
                    else
                        say "note:     probe ran the CPU reference (no kGpu stage)"
                    fi
                else
                    say "note:     probe failed; see the log below"
                    sed 's/^/  /' "$WORKDIR/probe.log" | tail -10
                fi
            fi
        fi

        if [ -z "$GPU_DUMP" ] || [ ! -f "$GPU_DUMP" ]; then
            say "note:     orchestrator exposes no GPU recon dump; using the"
            say "          decoded-frame compare above (set --gpu-dump/--gpu-dump-arg,"
            say "          AV2ENC_M1_GPU_DUMP, or build m1_gpu_probe)"
            RECON_STATUS="fallback (no GPU recon dump)"
        else
            say "dump:     $GPU_SOURCE"
            "$PY" "$SELF/plane_compare.py" "$CPU_DUMP" "$GPU_DUMP" \
                | tee "$WORKDIR/recon_compare.txt"
            RECON_RC=${PIPESTATUS[0]}
            RECON_STATUS=$([ $RECON_RC -eq 0 ] && echo IDENTICAL || echo DIFFER)
        fi
    fi
fi

# ---------------------------------------------------------------------------
# 9. Verdict.
# ---------------------------------------------------------------------------
say ""
say "-- summary --"
say "frame bytes:  $BYTES_STATUS"
say "attestation:  ${ATTEST_RESULT:-none}"
say "decode:       $DECODE_STATUS"
say "decoded YUV:  $PLANE_STATUS"
say "recon parity: $RECON_STATUS"

if [ "$BYTES_STATUS" = "DIFFER" ] || [ "$DECODE_STATUS" = "FAIL (cpu)" ] || \
   [ "$DECODE_STATUS" = "FAIL (gpu)" ] || [ "$PLANE_STATUS" = "DIFFER" ] || \
   [ "$RECON_STATUS" = "DIFFER" ]; then
    say "RESULT: FAIL"
    exit 1
fi

if [ "$ATTEST_RESULT" = "none" ] && [ "$NO_ATTESTATION" = "0" ]; then
    say ""
    say "note: bytes matched, but nothing proves the GPU path executed. A CPU"
    say "      fallback would produce identical bytes. Add AV2ENC_GPU_STATS"
    say "      (JSON, see README) or an 'M1 GPU' log marker; nsys was not"
    say "      available or saw no kernels. Use --no-attestation to override."
    if [ "$REQUIRE_RECON" = "1" ] && [ "$RECON_STATUS" != "IDENTICAL" ]; then
        skip "GPU execution unattested and no recon dump"
    fi
    skip "bytes matched but GPU execution is unattested (possible CPU fallback)"
fi

if [ "$REQUIRE_RECON" = "1" ] && [ "$RECON_STATUS" != "IDENTICAL" ]; then
    skip "recon dump required but not available ($RECON_STATUS)"
fi

say "RESULT: PASS"
exit 0
