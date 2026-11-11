#!/usr/bin/env bash
# CPU-vs-GPU gate for the M3 feature flags on the M1 still-picture path.
#
# M3 tools covered: CDEF, CCSO (band offset), RDOQ, intra TX type, rate-aware
# intra (RDO), full UV intra.  Each case encodes the same Y4M twice:
#
#   CPU:  av2enc <in.y4m> -o cpu.ivf --m1      [feature flags]
#   GPU:  av2enc <in.y4m> -o gpu.ivf --m1-gpu  [feature flags]
#
# PASS requires ALL of:
#   1. the GPU run was attested by Nsight Systems (>=1 CUDA kernel instance)
#      by default -- a stats file or log marker alone is not enough;
#   2. the GPU run did not reject a feature flag and did not stage CPU bytes;
#   3. IVF frame payloads are byte-identical (not PSNR);
#   4. both streams decode with `dav2d --strict 1`;
#   5. the decoded Y/U/V planes have zero differing samples;
#   6. the feature actually changed the CPU bitstream (no vacuous PASS);
#   7. the feature also changed the GPU bitstream (catches a flag the GPU
#      silently ignores, e.g. an unplumbed option);
#   8. with --determinism, two GPU encodes are byte-identical;
#   9. with --sanitize, compute-sanitizer memcheck reports zero errors.
#
# A GPU path that rejects a flag is SKIP, never PASS.  A CPU-staged GPU run is
# SKIP, never PASS.  Missing Nsight evidence is SKIP, never PASS.
#
# Verdicts / exit codes:
#   0  PASS            (or CPU-ONLY-PASS with --cpu-only)
#   1  FAIL            (byte/plane/decode difference, silent-ignore, GPU crash,
#                       nondeterminism, sanitizer error)
#   2  usage/setup
#   3  SKIP            (rejected flag, CPU staging, no attestation, no GPU build)
#
# ALLOW_SKIP=1 maps SKIP to exit 0.
#
# Usage:
#   tools/m3_gpu/verify_m3_gpu.sh [input.y4m] [options]
#
#   tools/m3_gpu/verify_m3_gpu.sh --size 64x64 --features cdef
#   tools/m3_gpu/verify_m3_gpu.sh --size 640x360 \
#       --features cdef,ccso,rdoq,tx-type
#   tools/m3_gpu/verify_m3_gpu.sh --size 1920x1080 --features all \
#       --determinism --sanitize
#   tools/m3_gpu/verify_m3_gpu.sh --cpu-only --size 640x360 --features all
#
#   # Feature matrix: 18 modes x 4 sizes by default.
#   tools/m3_gpu/verify_m3_gpu.sh --matrix
#   tools/m3_gpu/verify_m3_gpu.sh --matrix --sizes 64x64,640x360 \
#       --modes cdef,ccso,uv-full,all --determinism
#
# Options:
#   --encoder PATH        av2enc binary (default $AV2ENC or build-cuda/av2enc)
#   --decoder PATH        dav2d (default $DAV2D, tools/bin/dav2d[-macos])
#   --size WxH            generate this input when no input is given (640x360)
#   --features LIST       comma list: cdef,cdef1,ccso,ccso4,rdoq,tx-type,
#                         rdo-intra,uv-full,all (default: none = baseline)
#   --m3-args "..."       extra encoder flags applied to BOTH sides
#   --cpu-args "..."      extra args for the CPU run only
#   --gpu-args "..."      extra args for the GPU run only
#   --qp N                base_q_idx (default 90; --m1-gpu pins 90, so other
#                         values are a setup error unless --cpu-only)
#   --cpu-only            validate the CPU reference only; never claims GPU
#   --determinism         encode twice with the GPU CLI; bytes must match
#   --sanitize            run the GPU encode under compute-sanitizer memcheck
#   --racecheck           additionally run racecheck at 64x64
#   --attest MODE         nsys (default, required), stats, or auto
#   --no-ignore-probe     skip the GPU no-feature comparison
#   --no-noop-check       skip the CPU feature-effect check
#   --no-nsys             alias for --attest stats
#   --workdir DIR         artifact directory (default mktemp -d)
#   --keep                keep the work directory
#   --timeout SECONDS     per-command timeout (default 1800)
#   -h, --help
#
# Matrix options:
#   --matrix              run the feature matrix
#   --sizes LIST          default 64x64,640x360,1280x720,1920x1080
#   --modes LIST          subset of the mode names printed by --list-modes
#   --matrix-budget SEC   stop starting new cases after SEC (default 3600)
#   --max-cases N         refuse to start above N cases (default 0)
#   --matrix-keep         keep per-case work directories
#   --list-modes          print mode names + flags and exit
#
# Environment:
#   AV2ENC_GPU_STATS / AV2ENC_M1_GPU_STATS  stats JSON path
#   NSYS                                    nsys binary
#   COMPUTE_SANITIZER                       compute-sanitizer binary
#   DAV2D_THREADS                           decoder threads (default 1)
#   PYTHON                                  python3 interpreter
#
set -uo pipefail

SELF=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$SELF/../.." && pwd)
PY=${PYTHON:-python3}
NSYS=${NSYS:-nsys}
CS_TOOL=${COMPUTE_SANITIZER:-compute-sanitizer}
DECD_THREADS=${DAV2D_THREADS:-1}

ENCODER=${AV2ENC:-$ROOT/build-cuda/av2enc}
DECODER=${DAV2D:-}
WORKDIR=""
INPUT=""
SIZE="640x360"
QP=90
FEATURES=""
M3_ARGS=""
CPU_ARGS=""
GPU_ARGS=""
KEEP=0
CPU_ONLY=0
DETERMINISM=0
SANITIZE=0
RACECHECK=0
ATTEST_MODE="nsys"
IGNORE_PROBE=1
NOOP_CHECK=1
TIMEOUT=${TIMEOUT:-1800}

MATRIX=0
MATRIX_SIZES="64x64,640x360,1280x720,1920x1080"
MATRIX_MODES="all"
MATRIX_BUDGET=${MATRIX_BUDGET:-3600}
MATRIX_MAX_CASES=${MATRIX_MAX_CASES:-0}
MATRIX_KEEP=0
LIST_MODES=0

usage() {
    sed -n '2,/^set -uo pipefail/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'
}

while [ $# -gt 0 ]; do
    case "$1" in
        --encoder) ENCODER=$2; shift 2 ;;
        --decoder) DECODER=$2; shift 2 ;;
        --size) SIZE=$2; shift 2 ;;
        --features) FEATURES=$2; shift 2 ;;
        --m3-args) M3_ARGS=$2; shift 2 ;;
        --cpu-args) CPU_ARGS=$2; shift 2 ;;
        --gpu-args) GPU_ARGS=$2; shift 2 ;;
        --qp) QP=$2; shift 2 ;;
        --cpu-only) CPU_ONLY=1; shift ;;
        --determinism) DETERMINISM=1; shift ;;
        --sanitize) SANITIZE=1; shift ;;
        --racecheck) RACECHECK=1; shift ;;
        --attest) ATTEST_MODE=$2; shift 2 ;;
        --no-ignore-probe) IGNORE_PROBE=0; shift ;;
        --no-noop-check) NOOP_CHECK=0; shift ;;
        --no-nsys) ATTEST_MODE=stats; shift ;;
        --workdir) WORKDIR=$2; shift 2 ;;
        --keep) KEEP=1; shift ;;
        --timeout) TIMEOUT=$2; shift 2 ;;
        --matrix) MATRIX=1; shift ;;
        --sizes) MATRIX_SIZES=$2; shift 2 ;;
        --modes) MATRIX_MODES=$2; shift 2 ;;
        --matrix-budget) MATRIX_BUDGET=$2; shift 2 ;;
        --max-cases) MATRIX_MAX_CASES=$2; shift 2 ;;
        --matrix-keep) MATRIX_KEEP=1; shift ;;
        --list-modes) LIST_MODES=1; shift ;;
        -h|--help) usage; exit 0 ;;
        --*) echo "verify_m3_gpu: unknown option: $1" >&2; exit 2 ;;
        *) INPUT=$1; shift ;;
    esac
done

case "$ATTEST_MODE" in
    nsys|stats|auto) ;;
    *) echo "verify_m3_gpu: --attest must be nsys|stats|auto" >&2; exit 2 ;;
esac

say() { printf '%s\n' "$*"; }

class_line() { printf 'CLASS: %s\n' "$*"; }

skip() {
    local class=$1 reason=$2
    say ""
    class_line "SKIP-$class"
    say "RESULT: SKIP ($reason)"
    if [ "${ALLOW_SKIP:-0}" = "1" ]; then exit 0; fi
    exit 3
}

setup_error() {
    say "verify_m3_gpu: error: $*" >&2
    exit 2
}

# ---------------------------------------------------------------------------
# Feature mapping.  Keep every M3 flag in one place.
# ---------------------------------------------------------------------------
ALL_FEATURES=(
    cdef cdef1 ccso ccso4 rdoq tx-type rdo-intra uv-full
)

build_feature_args() {
    local csv=$1 name
    FEATURE_ARGS=()
    [ -z "$csv" ] && return 0
    local -a names=()
    IFS=',' read -r -a names <<< "$csv"
    for name in "${names[@]}"; do
        name=$(printf '%s' "$name" | tr -d '[:space:]')
        [ -z "$name" ] && continue
        case "$name" in
            baseline) ;;
            cdef) FEATURE_ARGS+=(--m1-cdef) ;;
            cdef1) FEATURE_ARGS+=(--m1-cdef --m1-cdef-strengths 1) ;;
            ccso) FEATURE_ARGS+=(--m1-ccso) ;;
            ccso4) FEATURE_ARGS+=(--m1-ccso --m1-ccso-bands 4) ;;
            rdoq) FEATURE_ARGS+=(--m1-rdoq) ;;
            tx-type) FEATURE_ARGS+=(--m1-tx-type) ;;
            rdo-intra) FEATURE_ARGS+=(--m1-rdo-intra) ;;
            uv-full) FEATURE_ARGS+=(--m1-uv-full) ;;
            all)
                FEATURE_ARGS+=(--m1-cdef --m1-ccso --m1-rdoq --m1-tx-type
                               --m1-rdo-intra --m1-uv-full) ;;
            *) setup_error "unknown feature '$name' (try --list-modes)" ;;
        esac
    done
}

build_feature_args "$FEATURES"

MODES=(
    "baseline|"
    "cdef|cdef"
    "cdef1|cdef1"
    "ccso|ccso"
    "ccso4|ccso4"
    "rdoq|rdoq"
    "tx-type|tx-type"
    "rdo-intra|rdo-intra"
    "uv-full|uv-full"
    "cdef+ccso|cdef,ccso"
    "cdef+rdoq|cdef,rdoq"
    "cdef+tx-type|cdef,tx-type"
    "rdoq+tx-type|rdoq,tx-type"
    "cdef+ccso+rdoq|cdef,ccso,rdoq"
    "cdef+rdoq+tx-type|cdef,rdoq,tx-type"
    "rdo-intra+uv-full|rdo-intra,uv-full"
    "cdef+rdoq+tx-type+uv-full|cdef,rdoq,tx-type,uv-full"
    "all|all"
)
ALONE_MODES=(cdef cdef1 ccso ccso4 rdoq tx-type rdo-intra uv-full)

if [ "$LIST_MODES" = "1" ]; then
    say "modes:"
    local_name=""
    for entry in "${MODES[@]}"; do
        local_name=${entry%%|*}
        say "  $(printf '%-30s' "$local_name") ${entry#*|}"
    done
    say "features: ${ALL_FEATURES[*]}"
    exit 0
fi

# Run a command under the per-command timeout when `timeout` is available.
run_timed() {
    if [ "${TIMEOUT:-0}" -gt 0 ] 2>/dev/null && command -v timeout >/dev/null 2>&1; then
        timeout "$TIMEOUT" "$@"
    else
        "$@"
    fi
}

md5_file() {
    if command -v md5 >/dev/null 2>&1; then
        md5 -q "$1"
    else
        md5sum "$1" | awk '{print $1}'
    fi
}

# Generate a native testsrc2 Y4M at WxH (even dimensions for 4:2:0).
generate_y4m() {
    local dims=$1 path=$2 w h
    w=${dims%x*}
    h=${dims#*x}
    case "$w" in ''|*[!0-9]*) setup_error "bad size '$dims' (want WxH)" ;; esac
    case "$h" in ''|*[!0-9]*) setup_error "bad size '$dims' (want WxH)" ;; esac
    if [ "$w" -lt 16 ] || [ "$h" -lt 16 ]; then
        setup_error "size $dims is too small for M1 (min 16x16)"
    fi
    if [ $((w % 2)) -ne 0 ] || [ $((h % 2)) -ne 0 ]; then
        setup_error "4:2:0 needs even dimensions, got $dims"
    fi
    say "input:    generating $dims testsrc2 (native)"
    ffmpeg -hide_banner -v error -y -f lavfi \
        -i "testsrc2=size=${w}x${h}:rate=1" -frames:v 1 -pix_fmt yuv420p \
        -f yuv4mpegpipe "$path" || setup_error "ffmpeg input generation failed"
}

input_dims() {
    "$PY" - "$1" <<'PY'
import sys
with open(sys.argv[1], "rb") as handle:
    header = handle.readline()
width = height = 0
for token in header.split()[1:]:
    if token[:1] == b"W":
        width = int(token[1:])
    elif token[:1] == b"H":
        height = int(token[1:])
print(f"{width}x{height}")
PY
}

# Rewrite the -o target in GPU_CMD so a secondary run cannot overwrite the
# artifact that is byte-gated and decoded.
rewrite_output() {
    local new_ivf=$1 part next_output=0
    RET_CMD=()
    for part in "${GPU_CMD[@]}"; do
        if [ $next_output -eq 1 ]; then RET_CMD+=("$new_ivf"); next_output=0; continue; fi
        if [ "$part" = "-o" ] || [ "$part" = "--output" ]; then
            RET_CMD+=("$part"); next_output=1; continue
        fi
        RET_CMD+=("$part")
    done
}

stats_cpu_staged() {
    [ -s "$1" ] || return 1
    "$PY" - "$1" <<'PY'
import json, sys
try:
    data = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(1)
sys.exit(0 if data.get("frame_bytes_from_cpu") is True else 1)
PY
}

stats_attests() {
    [ -s "$1" ] || return 1
    "$PY" - "$1" <<'PY'
import json, sys
try:
    data = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(1)
if data.get("frame_bytes_from_cpu") is True:
    sys.exit(1)
backend = str(data.get("backend", data.get("path", ""))).lower()
if backend in ("gpu", "m1-gpu", "cuda"):
    sys.exit(0)
for key in ("kernels", "kernel_count", "instances", "launches", "host_launches"):
    value = data.get(key)
    if isinstance(value, (int, float)) and value > 0:
        sys.exit(0)
sys.exit(1)
PY
}

stats_kernels() {
    [ -s "$1" ] || { printf '0'; return 0; }
    "$PY" - "$1" <<'PY'
import json, sys
try:
    data = json.load(open(sys.argv[1]))
except Exception:
    print(0); raise SystemExit(0)
for key in ("kernels", "kernel_count", "instances", "launches", "host_launches"):
    value = data.get(key)
    if isinstance(value, (int, float)) and value > 0:
        print(int(value)); raise SystemExit(0)
print(0)
PY
}

nsys_kernels() {
    local rep=$1
    "$PY" - "$rep" "$NSYS" <<'PY'
import csv, subprocess, sys
path, nsys = sys.argv[1], sys.argv[2]
proc = subprocess.run([nsys, "stats", "--force-export=true",
                       "--report", "cuda_gpu_kern_sum",
                       "--format", "csv", path],
                      stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                      text=True)
lines = proc.stdout.splitlines()
header = next((i for i, line in enumerate(lines)
               if line.startswith("Time (%)")), None)
if header is None:
    print(0); raise SystemExit(0)
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

# Does a GPU log look like a rejected/unavailable flag or a CPU staging path?
gpu_rejection_detail() {
    grep -Eoi -- '--m1-[a-z-]+' "$1" 2>/dev/null | sort -u | tr '\n' ' '
}

gpu_unavailable() {
    grep -Eqi '(requires[ _-]+a[ _-]+cuda[ _-]+build|cpu[ _-]+only[ _-]+reference[ _-]+build|not[ _-]+available[ _-]+in[ _-]+this[ _-]+build|no[ _-]+cuda[ _-]+device|CUDA_ERROR_NO_DEVICE|cudaErrorNoDevice)' "$1"
}

gpu_rejected() {
    grep -Eqi '(not[ _-]+ported|not[ _-]+implemented|not[ _-]+supported)' "$1" &&
        grep -q -- '--m1-' "$1"
}

gpu_cpu_staged() {
    grep -Eqi '(frame_bytes_from_cpu[[:space:]]*[=:][[:space:]]*1|"frame_bytes_from_cpu"[[:space:]]*:[[:space:]]*true|frame[ _-]+bytes[ _-]+came[ _-]+from[ _-]+the[ _-]+cpu|stage[ _-]+hooks[ _-]+are[ _-]+not[ _-]+linked)' "$1"
}

# ---------------------------------------------------------------------------
# Matrix mode.
# ---------------------------------------------------------------------------
run_matrix() {
    command -v ffmpeg >/dev/null 2>&1 || setup_error "ffmpeg not found (needed for matrix inputs)"
    command -v "$PY" >/dev/null 2>&1 || setup_error "python3 not found"
    if [ "$CPU_ONLY" = "0" ] && [ "$QP" != "90" ]; then
        setup_error "--m1-gpu pins --qp 90; use --cpu-only to test another QP"
    fi

    say "== M3 GPU feature matrix =="
    say "encoder:  $ENCODER"
    say "decoder:  $DECODER"
    say "cpu-only: $CPU_ONLY"
    say "attest:   $ATTEST_MODE"

    local help
    help=$("$ENCODER" --help 2>&1 || true)
    if [ "$CPU_ONLY" = "0" ] && ! grep -q -- '--m1-gpu' <<< "$help"; then
        skip "UNAVAILABLE" "GPU path not present in this encoder build"
    fi

    local mroot
    if [ -n "$WORKDIR" ]; then
        mroot="$WORKDIR/matrix"
    else
        mroot=$(mktemp -d "${TMPDIR:-/tmp}/m3_gpu_matrix.XXXXXX")
    fi
    mkdir -p "$mroot/logs" "$mroot/inputs" || setup_error "cannot create $mroot"

    for numeric in "$MATRIX_BUDGET" "$MATRIX_MAX_CASES"; do
        case "$numeric" in
            ''|*[!0-9]*) setup_error "matrix numeric option is not an integer: '$numeric'" ;;
        esac
    done
    [ -n "$MATRIX_SIZES" ] || setup_error "no matrix sizes"
    local -a sizes=() parts=() token
    IFS=',' read -r -a parts <<< "$MATRIX_SIZES"
    for token in "${parts[@]}"; do
        token=$(printf '%s' "$token" | tr -d '[:space:]')
        [ -n "$token" ] && sizes+=("$token")
    done
    [ ${#sizes[@]} -gt 0 ] || setup_error "no matrix sizes"

    local -a mode_names=() mode_features=()
    local entry name feats found want idx
    if [ "$MATRIX_MODES" = "all" ]; then
        for entry in "${MODES[@]}"; do
            mode_names+=("${entry%%|*}")
            mode_features+=("${entry#*|}")
        done
    else
        local -a wanted=()
        IFS=',' read -r -a wanted <<< "$MATRIX_MODES"
        for want in "${wanted[@]}"; do
            want=$(printf '%s' "$want" | tr -d '[:space:]')
            found=0
            for idx in "${!MODES[@]}"; do
                name=${MODES[$idx]%%|*}
                if [ "$name" = "$want" ]; then
                    mode_names+=("$name")
                    mode_features+=("${MODES[$idx]#*|}")
                    found=1
                    break
                fi
            done
            [ "$found" = 1 ] || setup_error "unknown --modes entry '$want' (see --list-modes)"
        done
    fi
    [ ${#mode_names[@]} -gt 0 ] || setup_error "no matrix modes"

    local total=$(( ${#mode_names[@]} * ${#sizes[@]} ))
    say "sizes:    ${sizes[*]}"
    say "modes:    ${mode_names[*]}"
    say "cases:    $total (${#mode_names[@]} modes x ${#sizes[@]} sizes)"
    say "logs:     $mroot/logs"
    if [ "$MATRIX_MAX_CASES" -gt 0 ] && [ "$total" -gt "$MATRIX_MAX_CASES" ]; then
        setup_error "matrix has $total cases, above --max-cases $MATRIX_MAX_CASES"
    fi

    local size input
    for size in "${sizes[@]}"; do
        input="$mroot/inputs/in-$size.y4m"
        if [ ! -s "$input" ]; then
            generate_y4m "$size" "$input" >/dev/null
        fi
    done

    local start now elapsed pass=0 fail=0 skipped=0 idx=0
    local -a failures=()
    local skip_file="$mroot/skip-reasons.txt"
    local class_file="$mroot/classes.tsv"
    : > "$skip_file"
    : > "$class_file"
    start=$(date +%s)

    local mode_i case_log case_cmd rc class reason
    for size in "${sizes[@]}"; do
        input="$mroot/inputs/in-$size.y4m"
        for mode_i in "${!mode_names[@]}"; do
            idx=$((idx + 1))
            name=${mode_names[$mode_i]}
            feats=${mode_features[$mode_i]}
            case_log="$mroot/logs/${size}-${name}.log"

            now=$(date +%s)
            elapsed=$((now - start))
            if [ "$MATRIX_BUDGET" -gt 0 ] && [ "$elapsed" -ge "$MATRIX_BUDGET" ]; then
                local remaining=$((total - idx + 1))
                skipped=$((skipped + remaining))
                printf 'budget:   stopping after %ss; %d case(s) not run\n' \
                    "$elapsed" "$remaining"
                local budget_i=0
                while [ $budget_i -lt "$remaining" ]; do
                    printf 'matrix budget exceeded after %ss\n' "$elapsed" >> "$skip_file"
                    printf '%s\t%s\tSKIP-BUDGET\n' "$size" "$name" >> "$class_file"
                    budget_i=$((budget_i + 1))
                done
                break 2
            fi

            case_cmd=("$SELF/verify_m3_gpu.sh" --encoder "$ENCODER" \
                --decoder "$DECODER" --qp "$QP" --timeout "$TIMEOUT"
                --attest "$ATTEST_MODE")
            [ -n "$feats" ] && case_cmd+=(--features "$feats")
            [ -n "$M3_ARGS" ] && case_cmd+=(--m3-args "$M3_ARGS")
            [ -n "$CPU_ARGS" ] && case_cmd+=(--cpu-args "$CPU_ARGS")
            [ -n "$GPU_ARGS" ] && case_cmd+=(--gpu-args "$GPU_ARGS")
            [ "$CPU_ONLY" = 1 ] && case_cmd+=(--cpu-only)
            [ "$DETERMINISM" = 1 ] && case_cmd+=(--determinism)
            [ "$SANITIZE" = 1 ] && case_cmd+=(--sanitize)
            if [ "$RACECHECK" = 1 ] && [ "$size" = "64x64" ]; then
                case_cmd+=(--racecheck)
            fi
            [ "$IGNORE_PROBE" = 0 ] && case_cmd+=(--no-ignore-probe)
            if [ "$MATRIX_KEEP" = 1 ]; then
                case_cmd+=(--keep --workdir "$mroot/cases/$size-$name")
            fi
            case_cmd+=("$input")

            printf '[%3d/%3d] %-10s %-28s ' "$idx" "$total" "$size" "$name"
            "${case_cmd[@]}" >"$case_log" 2>&1
            rc=$?
            class=$(grep -E '^CLASS: ' "$case_log" 2>/dev/null | tail -1 | sed 's/^CLASS: //')
            [ -z "$class" ] && class="UNCLASSIFIED(exit $rc)"
            printf '%s\t%s\t%s\n' "$size" "$name" "$class" >> "$class_file"

            case $rc in
                0)
                    case "$class" in
                        SKIP-*)
                            skipped=$((skipped + 1))
                            reason=$(grep -E '^RESULT: ' "$case_log" 2>/dev/null | tail -1)
                            printf 'SKIP  %s\n' "${reason#RESULT: }"
                            printf '%s: %s\n' "$name" "${reason#RESULT: }" >> "$skip_file"
                            ;;
                        *)
                            pass=$((pass + 1))
                            printf 'PASS\n'
                            ;;
                    esac
                    ;;
                3)
                    skipped=$((skipped + 1))
                    reason=$(grep -E '^RESULT: ' "$case_log" 2>/dev/null | tail -1)
                    printf 'SKIP  %s\n' "${reason#RESULT: }"
                    printf '%s: %s\n' "$name" "${reason#RESULT: }" >> "$skip_file"
                    ;;
                *)
                    fail=$((fail + 1))
                    printf 'FAIL  %s\n' "$class"
                    failures+=("$size|$name|$case_log")
                    ;;
            esac
        done
    done

    now=$(date +%s)
    elapsed=$((now - start))
    say ""
    say "-- matrix summary --"
    say "cases:    $total (${#mode_names[@]} modes x ${#sizes[@]} sizes)"
    if [ "$CPU_ONLY" = "1" ]; then
        say "result:   $pass CPU-ONLY-PASS, $fail FAIL, $skipped SKIP (GPU NOT tested)"
    else
        say "result:   $pass PASS, $fail FAIL, $skipped SKIP"
    fi
    say "elapsed:  ${elapsed}s (budget ${MATRIX_BUDGET}s)"
    say "logs:     $mroot/logs"
    say "classes:  $class_file"

    if [ "$fail" -gt 0 ]; then
        say ""
        say "FAILURES (exact diff)"
        local entry2 f_size f_name f_log detail
        for entry2 in "${failures[@]}"; do
            IFS='|' read -r f_size f_name f_log <<< "$entry2"
            say ""
            say "[$f_size $f_name] $(grep -E '^CLASS: |^RESULT: ' "$f_log" | tail -2 | tr '\n' ' ')"
            detail=$(grep -E 'first byte offset|first sample|plane [YUV]:.*differ|MISMATCH|RESULT: .*(DIFFER|FAIL)|ERROR SUMMARY|RACECHECK SUMMARY|decode: +FAILED|Error decoding frame|No data decoded|silently ignored|did not change' "$f_log" 2>/dev/null | head -8)
            if [ -n "$detail" ]; then
                printf '%s\n' "$detail" | sed 's/^/  /'
            else
                tail -8 "$f_log" | sed 's/^/  /'
            fi
            say "  log: $f_log"
        done
    fi

    # ---- per-feature readiness -------------------------------------------
    say ""
    if [ "$CPU_ONLY" = "1" ]; then
        say "-- CPU feature effect (GPU NOT tested) --"
    else
        say "-- feature readiness (alone modes across all sizes) --"
    fi
    local fmode fm_idx n_pass n_fail n_skip classes classlist verdict
    local -a ready=() not_ready=()
    for fmode in "${ALONE_MODES[@]}"; do
        fm_idx=-1
        for idx in "${!mode_names[@]}"; do
            [ "${mode_names[$idx]}" = "$fmode" ] && fm_idx=$idx
        done
        if [ "$fm_idx" -lt 0 ]; then
            continue
        fi
        n_pass=$(awk -F'\t' -v m="$fmode" -v c="$CPU_ONLY" \
            '$2==m && ($3=="PASS" || (c=="1" && $3=="CPU-ONLY-PASS")) {n++} END {print n+0}' "$class_file")
        n_fail=$(awk -F'\t' -v m="$fmode" -v c="$CPU_ONLY" \
            '$2==m && $3!="PASS" && !(c=="1" && $3=="CPU-ONLY-PASS") && $3!~/^SKIP/ {n++} END {print n+0}' "$class_file")
        n_skip=$(awk -F'\t' -v m="$fmode" '$2==m && $3~/^SKIP/ {n++} END {print n+0}' "$class_file")
        classes=$(awk -F'\t' -v m="$fmode" '$2==m {print $3}' "$class_file" | sort -u | tr '\n' ',' | sed 's/,$//')
        if [ "$n_fail" -eq 0 ] && [ "$n_skip" -eq 0 ] && [ "$n_pass" -gt 0 ]; then
            verdict=$([ "$CPU_ONLY" = "1" ] && echo "CHANGES-BITSTREAM" || echo "READY")
            ready+=("$fmode")
        elif [ "$n_fail" -eq 0 ] && [ "$n_pass" -gt 0 ] && [ "$n_skip" -gt 0 ]; then
            verdict="PARTIAL"
            not_ready+=("$fmode ($classes)")
        else
            verdict="NOT-READY"
            not_ready+=("$fmode ($classes)")
        fi
        say "  $(printf '%-12s' "$fmode") $verdict  pass=$n_pass fail=$n_fail skip=$n_skip  [$classes]"
    done

    say ""
    if [ "$CPU_ONLY" = "1" ]; then
        say "CPU FEATURES CHANGING THE BITSTREAM: ${ready[*]:-(none)}"
        if [ ${#not_ready[@]} -gt 0 ]; then
            say "CPU FEATURES PARTIAL/NOT-READY:     ${not_ready[*]}"
        fi
    else
        if [ ${#ready[@]} -gt 0 ]; then
            say "GPU-READY FEATURES:     ${ready[*]}"
        else
            say "GPU-READY FEATURES:     (none)"
        fi
        if [ ${#not_ready[@]} -gt 0 ]; then
            say "NOT GPU-READY:          ${not_ready[*]}"
        fi
    fi

    if [ "$skipped" -gt 0 ]; then
        say ""
        say "SKIPPED (grouped by reason)"
        sort "$skip_file" 2>/dev/null | sed 's/^[^:]*: //' | sort | uniq -c | sort -rn | sed 's/^/  /'
    fi

    say ""
    if [ "$fail" -gt 0 ]; then
        say "RESULT: FAIL ($fail case(s))"
        return 1
    fi
    if [ "$skipped" -gt 0 ]; then
        say "RESULT: SKIP ($skipped case(s) not verified)"
        if [ "${ALLOW_SKIP:-0}" = "1" ]; then return 0; fi
        return 3
    fi
    if [ "$CPU_ONLY" = "1" ]; then
        say "RESULT: CPU-ONLY-PASS ($total case(s); GPU was NOT verified)"
        return 0
    fi
    say "RESULT: PASS ($total case(s))"
    return 0
}

# ---------------------------------------------------------------------------
# Common setup.
# ---------------------------------------------------------------------------
command -v "$PY" >/dev/null 2>&1 || setup_error "python3 not found"
[ -x "$ENCODER" ] || setup_error "encoder not executable: $ENCODER (build it first)"

if [ -z "$DECODER" ]; then
    for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
        if [ -x "$candidate" ]; then DECODER=$candidate; break; fi
    done
fi
[ -n "$DECODER" ] && [ -x "$DECODER" ] || \
    setup_error "dav2d not found; set DAV2D=/path/to/dav2d"

if [ "$MATRIX" = "1" ]; then
    run_matrix
    exit $?
fi

command -v ffmpeg >/dev/null 2>&1 || [ -n "$INPUT" ] || \
    setup_error "ffmpeg not found; pass an input Y4M or install ffmpeg"

if [ "$CPU_ONLY" = "0" ] && [ "$QP" != "90" ]; then
    setup_error "--m1-gpu pins --qp 90; use --cpu-only to test another QP"
fi

HELP=$("$ENCODER" --help 2>&1 || true)
if [ "$CPU_ONLY" = "0" ] && ! grep -q -- '--m1-gpu' <<< "$HELP"; then
    say "== M3 GPU feature gate =="
    say "encoder:  $ENCODER"
    say "gate:     this build has no --m1-gpu option"
    skip "UNAVAILABLE" "GPU path not present in this encoder build"
fi
for needed in --m1-cdef --m1-ccso --m1-rdoq --m1-tx-type --m1-rdo-intra --m1-uv-full; do
    if ! grep -q -- "$needed" <<< "$HELP"; then
        say "gate:     this build has no $needed option"
        skip "UNAVAILABLE" "encoder build lacks M3 flag $needed"
    fi
done

if [ -z "$WORKDIR" ]; then
    WORKDIR=$(mktemp -d "${TMPDIR:-/tmp}/m3_gpu_verify.XXXXXX")
    CLEAN=1
else
    mkdir -p "$WORKDIR" || setup_error "cannot create $WORKDIR"
    CLEAN=0
fi
[ "$KEEP" = "1" ] && CLEAN=0
cleanup() { if [ "$CLEAN" = "1" ]; then rm -rf "$WORKDIR"; fi; }
trap cleanup EXIT

if [ -z "$INPUT" ]; then
    INPUT="$WORKDIR/input-$SIZE.y4m"
    generate_y4m "$SIZE" "$INPUT"
fi
[ -f "$INPUT" ] || setup_error "no such input: $INPUT"
DIMS=$(input_dims "$INPUT")

if [ "$SANITIZE" = "1" ] || [ "$RACECHECK" = "1" ]; then
    command -v "$CS_TOOL" >/dev/null 2>&1 || [ -x "$CS_TOOL" ] || \
        skip "UNAVAILABLE" "compute-sanitizer not found (set COMPUTE_SANITIZER)"
fi

say "== M3 GPU feature gate =="
say "encoder:  $ENCODER"
say "decoder:  $DECODER"
say "input:    $INPUT ($DIMS)"
say "features: ${FEATURES:--}"
say "m3 args:  ${M3_ARGS:--}"
say "cpu-only: $CPU_ONLY"
say "attest:   $ATTEST_MODE"
say "workdir:  $WORKDIR"

# ---------------------------------------------------------------------------
# CPU reference encode.  Always validate it with dav2d --strict 1 before the
# GPU path is allowed to make any claim.
# ---------------------------------------------------------------------------
CPU_IVF="$WORKDIR/cpu.ivf"
CPU_Y4M="$WORKDIR/cpu-decoded.y4m"
GPU_IVF="$WORKDIR/gpu.ivf"
GPU_LOG="$WORKDIR/gpu.log"
GPU_STATS="$WORKDIR/gpu-stats.json"

CPU_CMD=("$ENCODER" "$INPUT" -o "$CPU_IVF" --m1 --qp "$QP")
CPU_CMD+=(${FEATURE_ARGS[@]+"${FEATURE_ARGS[@]}"})
if [ -n "$M3_ARGS" ]; then
    # shellcheck disable=SC2206
    CPU_CMD+=($M3_ARGS)
fi
if [ -n "$CPU_ARGS" ]; then
    # shellcheck disable=SC2206
    CPU_CMD+=($CPU_ARGS)
fi

say ""
say "-- CPU reference encode --"
run_timed "${CPU_CMD[@]}" >"$WORKDIR/cpu.log" 2>&1
CPU_RC=$?
if [ $CPU_RC -ne 0 ] || [ ! -s "$CPU_IVF" ]; then
    sed 's/^/  /' "$WORKDIR/cpu.log" | tail -20
    class_line "CPU-REF-FAIL"
    say "RESULT: FAIL (CPU reference encode failed, exit $CPU_RC)"
    exit 1
fi
CPU_MD5=$(md5_file "$CPU_IVF")
CPU_BYTES=$(wc -c < "$CPU_IVF" | tr -d ' ')
say "cpu ivf:  $CPU_BYTES B  md5 $CPU_MD5"

# Strict decode is the reference-validity preflight.
CPU_DEC_RC=0
run_timed "$DECODER" -q -i "$CPU_IVF" -o "$CPU_Y4M" \
    --threads "$DECD_THREADS" --strict 1 \
    >"$WORKDIR/cpu-decode.log" 2>&1 || CPU_DEC_RC=$?
if [ $CPU_DEC_RC -ne 0 ] || [ ! -s "$CPU_Y4M" ]; then
    sed 's/^/  /' "$WORKDIR/cpu-decode.log" | tail -10
    class_line "CPU-REF-DECODE-FAIL"
    say "RESULT: FAIL (CPU reference stream fails dav2d --strict 1, exit $CPU_DEC_RC)"
    exit 1
fi
say "cpu decode: ok (dav2d --strict 1)"

# Does the feature set actually change the CPU bitstream?  If not, a byte match
# proves nothing about the feature.
if [ "$NOOP_CHECK" = "1" ] && [ ${#FEATURE_ARGS[@]} -gt 0 ]; then
    BASE_CMD=("$ENCODER" "$INPUT" -o "$WORKDIR/cpu-base.ivf" --m1 --qp "$QP")
    if [ -n "$M3_ARGS" ]; then
        # shellcheck disable=SC2206
        BASE_CMD+=($M3_ARGS)
    fi
    if [ -n "$CPU_ARGS" ]; then
        # shellcheck disable=SC2206
        BASE_CMD+=($CPU_ARGS)
    fi
    run_timed "${BASE_CMD[@]}" >"$WORKDIR/cpu-base.log" 2>&1
    BASE_RC=$?
    if [ $BASE_RC -ne 0 ] || [ ! -s "$WORKDIR/cpu-base.ivf" ]; then
        sed 's/^/  /' "$WORKDIR/cpu-base.log" | tail -10
        class_line "CPU-REF-FAIL"
        say "RESULT: FAIL (CPU no-feature baseline encode failed, exit $BASE_RC)"
        exit 1
    fi
    BASE_MD5=$(md5_file "$WORKDIR/cpu-base.ivf")
    if [ "$BASE_MD5" = "$CPU_MD5" ]; then
        say "feature:  CPU bitstream UNCHANGED vs no-feature baseline"
        class_line "NO-EFFECT"
        skip "NO-EFFECT" "feature set did not change the CPU reference bitstream; case would be vacuous"
    fi
    say "feature:  CPU bitstream changed (baseline md5 $BASE_MD5)"

    # Tuning-parameter modes must also differ from their default variant, or a
    # GPU that drops the tuning flag could look correct where the two variants
    # coincide (e.g. cdef strengths 1 vs 4 at 64x64).
    variant=""
    case "$(printf '%s' "$FEATURES" | tr -d '[:space:]')" in
        cdef1) variant=cdef ;;
        ccso4) variant=ccso ;;
    esac
    if [ -n "$variant" ]; then
        build_feature_args "$variant"
        VAR_CMD=("$ENCODER" "$INPUT" -o "$WORKDIR/cpu-variant.ivf" --m1 --qp "$QP")
        VAR_CMD+=(${FEATURE_ARGS[@]+"${FEATURE_ARGS[@]}"})
        if [ -n "$M3_ARGS" ]; then
            # shellcheck disable=SC2206
            VAR_CMD+=($M3_ARGS)
        fi
        if [ -n "$CPU_ARGS" ]; then
            # shellcheck disable=SC2206
            VAR_CMD+=($CPU_ARGS)
        fi
        run_timed "${VAR_CMD[@]}" >"$WORKDIR/cpu-variant.log" 2>&1
        VAR_RC=$?
        build_feature_args "$FEATURES"
        if [ $VAR_RC -ne 0 ] || [ ! -s "$WORKDIR/cpu-variant.ivf" ]; then
            sed 's/^/  /' "$WORKDIR/cpu-variant.log" | tail -10
            class_line "CPU-REF-FAIL"
            say "RESULT: FAIL (CPU variant baseline '$variant' encode failed, exit $VAR_RC)"
            exit 1
        fi
        VAR_MD5=$(md5_file "$WORKDIR/cpu-variant.ivf")
        if [ "$VAR_MD5" = "$CPU_MD5" ]; then
            say "feature:  variant '$variant' produces the same bitstream at this size"
            class_line "NO-EFFECT"
            skip "NO-EFFECT" "tuning variant '$FEATURES' does not differ from '$variant' here; cannot verify the parameter"
        fi
        say "feature:  tuning variant differs from '$variant' (md5 $VAR_MD5)"
    fi
fi

if [ "$CPU_ONLY" = "1" ]; then
    say ""
    class_line "CPU-ONLY-PASS"
    say "RESULT: CPU-ONLY-PASS (CPU reference + dav2d --strict 1; GPU NOT tested)"
    exit 0
fi

# ---------------------------------------------------------------------------
# GPU encode.  Same feature flags as the CPU side.
# ---------------------------------------------------------------------------
GPU_ENV=(env "AV2ENC_GPU_STATS=$GPU_STATS" \
          "AV2ENC_M1_GPU_STATS=$GPU_STATS")
GPU_CMD=("$ENCODER" "$INPUT" -o "$GPU_IVF" --m1-gpu --qp "$QP")
GPU_CMD+=(${FEATURE_ARGS[@]+"${FEATURE_ARGS[@]}"})
if [ -n "$M3_ARGS" ]; then
    # shellcheck disable=SC2206
    GPU_CMD+=($M3_ARGS)
fi
if [ -n "$GPU_ARGS" ]; then
    # shellcheck disable=SC2206
    GPU_CMD+=($GPU_ARGS)
fi

say ""
say "-- GPU encode --"
SAN_STATUS="not requested"
RACE_STATUS="not requested"
if [ "$SANITIZE" = "1" ]; then
    GPU_LOG="$WORKDIR/sanitize-memcheck.log"
    say "memcheck: compute-sanitizer --tool memcheck"
    run_timed "$CS_TOOL" --tool memcheck --error-exitcode 99 \
        --target-processes all "${GPU_ENV[@]}" "${GPU_CMD[@]}" >"$GPU_LOG" 2>&1
    GPU_RC=$?
    SAN_ERRORS=$(grep -oE 'ERROR SUMMARY: [0-9]+ error' "$GPU_LOG" | tail -1 \
        | grep -oE '[0-9]+' | head -1)
    if [ -n "$SAN_ERRORS" ] && [ "$SAN_ERRORS" -gt 0 ]; then
        grep -E 'Invalid|out-of-bounds|Uninitialized|ERROR SUMMARY' "$GPU_LOG" \
            | head -12 | sed 's/^/  /'
        class_line "SANITIZER"
        say "RESULT: FAIL (compute-sanitizer memcheck: $SAN_ERRORS errors)"
        exit 1
    fi
    if grep -q 'terminated before first instrumented API call' "$GPU_LOG"; then
        class_line "CPU-STAGED"
        skip "CPU-STAGED" "compute-sanitizer saw no CUDA work (CPU/staged path)"
    fi
    if [ -z "$SAN_ERRORS" ]; then
        if gpu_unavailable "$GPU_LOG"; then
            class_line "UNAVAILABLE"
            skip "UNAVAILABLE" "GPU path unavailable in this build/host under sanitizer"
        fi
        if gpu_rejected "$GPU_LOG"; then
            say "detail:   $(gpu_rejection_detail "$GPU_LOG")"
            class_line "REJECTED"
            skip "REJECTED" "GPU rejected feature flag(s) under sanitizer"
        fi
        sed 's/^/  /' "$GPU_LOG" | tail -12
        class_line "SANITIZER"
        say "RESULT: FAIL (compute-sanitizer produced no ERROR SUMMARY)"
        exit 1
    fi
    SAN_STATUS="memcheck 0 errors"
    say "memcheck: 0 errors"
else
    run_timed "${GPU_ENV[@]}" "${GPU_CMD[@]}" >"$GPU_LOG" 2>&1
    GPU_RC=$?
fi

if [ $GPU_RC -ne 0 ]; then
    say "gpu log:"
    sed 's/^/  /' "$GPU_LOG" | tail -20
    if gpu_unavailable "$GPU_LOG"; then
        class_line "UNAVAILABLE"
        skip "UNAVAILABLE" "GPU path unavailable in this build/host (see log)"
    fi
    if gpu_rejected "$GPU_LOG"; then
        say "detail:   rejected $(gpu_rejection_detail "$GPU_LOG")"
        class_line "REJECTED"
        skip "REJECTED" "GPU rejected feature flag(s): $(gpu_rejection_detail "$GPU_LOG")"
    fi
    class_line "GPU-ERROR"
    say "RESULT: FAIL (GPU encode failed, exit $GPU_RC)"
    exit 1
fi

if [ ! -s "$GPU_IVF" ]; then
    say "gpu log:"
    sed 's/^/  /' "$GPU_LOG" | tail -20
    class_line "CPU-STAGED"
    skip "CPU-STAGED" "GPU run exited 0 but wrote no IVF (stub/blocked path)"
fi

# Frame bytes from the CPU reference are never a GPU result, even when they are
# byte-identical.  Check both the log and the stats channel.
if gpu_cpu_staged "$GPU_LOG"; then
    say "gpu log:"
    sed 's/^/  /' "$GPU_LOG" | tail -20
    class_line "CPU-STAGED"
    skip "CPU-STAGED" "GPU run used CPU-reference staging; frame bytes did not come from GPU stages"
fi
if stats_cpu_staged "$GPU_STATS"; then
    class_line "CPU-STAGED"
    skip "CPU-STAGED" "stats file reports frame_bytes_from_cpu=true (CPU staging)"
fi

# ---------------------------------------------------------------------------
# Silent-ignore probe: the GPU must react to the feature flags.  CCSO is the
# known case where main.cpp does not plumb a field into GpuOptions, so the flag
# can be accepted and dropped.  A dropped flag must never look like a PASS.
# ---------------------------------------------------------------------------
if [ "$IGNORE_PROBE" = "1" ] && [ ${#FEATURE_ARGS[@]} -gt 0 ]; then
    say ""
    say "-- GPU no-feature probe --"
    BASE_GPU=("$ENCODER" "$INPUT" -o "$WORKDIR/gpu-base.ivf" --m1-gpu --qp "$QP")
    if [ -n "$M3_ARGS" ]; then
        # shellcheck disable=SC2206
        BASE_GPU+=($M3_ARGS)
    fi
    if [ -n "$GPU_ARGS" ]; then
        # shellcheck disable=SC2206
        BASE_GPU+=($GPU_ARGS)
    fi
    run_timed "${BASE_GPU[@]}" >"$WORKDIR/gpu-base.log" 2>&1
    BASE_GPU_RC=$?
    if [ $BASE_GPU_RC -eq 0 ] && [ -s "$WORKDIR/gpu-base.ivf" ]; then
        "$PY" "$SELF/ivf_compare.py" --label-a "gpu+features" \
            --label-b "gpu-base" "$GPU_IVF" "$WORKDIR/gpu-base.ivf" \
            >"$WORKDIR/gpu-base-compare.txt" 2>&1
        PROBE_RC=$?
        if [ $PROBE_RC -eq 0 ]; then
            cat "$WORKDIR/gpu-base-compare.txt" | sed 's/^/  /'
            say "probe:    the GPU bitstream is identical with and without the feature flags"
            class_line "IGNORED"
            say "RESULT: FAIL (GPU silently ignored the feature flag; not a GPU implementation)"
            exit 1
        fi
        say "probe:    GPU responds to the feature flags (bitstream differs)"
    else
        say "probe:    baseline GPU encode failed; skipped (see gpu-base.log)"
    fi
fi

# ---------------------------------------------------------------------------
# Racecheck (64x64 only; separate from memcheck).
# ---------------------------------------------------------------------------
if [ "$RACECHECK" = "1" ]; then
    if [ "$DIMS" != "64x64" ]; then
        RACE_STATUS="skipped (only 64x64)"
        say ""
        say "racecheck: skipped (input is $DIMS)"
    else
        say ""
        say "-- compute-sanitizer (racecheck) --"
        rewrite_output "$WORKDIR/race.ivf"
        RACE_ENV=(env "AV2ENC_GPU_STATS=$WORKDIR/race-stats.json" \
                  "AV2ENC_M1_GPU_STATS=$WORKDIR/race-stats.json")
        RACE_LOG="$WORKDIR/sanitize-racecheck.log"
        run_timed "$CS_TOOL" --tool racecheck --error-exitcode 99 \
            --target-processes all "${RACE_ENV[@]}" "${RET_CMD[@]}" >"$RACE_LOG" 2>&1
        RACE_RC=$?
        RACE_LINE=$(grep -E 'RACECHECK SUMMARY' "$RACE_LOG" | tail -1)
        RACE_ERRORS=$(printf '%s' "$RACE_LINE" | grep -oE '[0-9]+ error' | grep -oE '[0-9]+' | head -1)
        RACE_WARNINGS=$(printf '%s' "$RACE_LINE" | grep -oE '[0-9]+ warning' | grep -oE '[0-9]+' | head -1)
        if grep -q 'terminated before first instrumented API call' "$RACE_LOG"; then
            RACE_STATUS="skipped (no CUDA work)"
            say "racecheck: no instrumented CUDA API call"
        elif [ -z "$RACE_LINE" ]; then
            RACE_STATUS="FAIL (no summary)"
            sed 's/^/  /' "$RACE_LOG" | tail -10
            say "racecheck: no RACECHECK SUMMARY (exit $RACE_RC)"
        elif [ "${RACE_ERRORS:-0}" -gt 0 ] || [ "${RACE_WARNINGS:-0}" -gt 0 ]; then
            RACE_STATUS="FAIL (${RACE_ERRORS:-0} errors, ${RACE_WARNINGS:-0} warnings)"
            say "racecheck: ${RACE_ERRORS:-0} errors, ${RACE_WARNINGS:-0} warnings"
        else
            RACE_STATUS="0 hazards"
            say "racecheck: 0 hazards"
        fi
    fi
fi

# ---------------------------------------------------------------------------
# Byte gate: IVF frame payloads.
# ---------------------------------------------------------------------------
say ""
say "-- frame bytes --"
"$PY" "$SELF/ivf_compare.py" "$CPU_IVF" "$GPU_IVF" | tee "$WORKDIR/ivf_compare.txt"
BYTES_RC=${PIPESTATUS[0]}
BYTES_STATUS=$([ $BYTES_RC -eq 0 ] && echo IDENTICAL || echo DIFFER)

# ---------------------------------------------------------------------------
# Nsight Systems attestation.  Default: a PASS needs >=1 CUDA kernel instance
# observed by nsys.  A stats file cannot replace that unless --attest stats.
# ---------------------------------------------------------------------------
ATTEST_RESULT="none"
NSYS_KERNELS=0
stats_seen=0
if [ -s "$GPU_STATS" ]; then
    stats_seen=1
    if stats_attests "$GPU_STATS"; then
        ATTEST_RESULT="stats ($(stats_kernels "$GPU_STATS") kernels)"
    fi
fi

if [ "$ATTEST_MODE" = "nsys" ] || [ "$ATTEST_MODE" = "auto" ]; then
    if [ "$ATTEST_MODE" = "nsys" ] || [ "$ATTEST_RESULT" = "none" ]; then
        if ! command -v "$NSYS" >/dev/null 2>&1 && [ ! -x "$NSYS" ]; then
            if [ "$ATTEST_MODE" = "nsys" ]; then
                class_line "UNATTESTED"
                skip "UNATTESTED" "nsys not found; Nsight attestation is required (or pass --attest stats)"
            fi
        else
            say ""
            say "-- attestation (nsys) --"
            NSYS_PREFIX="$WORKDIR/attest"
            rm -f "$NSYS_PREFIX.nsys-rep" "$NSYS_PREFIX.sqlite"
            rewrite_output "$WORKDIR/attest.ivf"
            ATTEST_ENV=(env "AV2ENC_GPU_STATS=$WORKDIR/attest-stats.json" \
                        "AV2ENC_M1_GPU_STATS=$WORKDIR/attest-stats.json")
            if run_timed "$NSYS" profile --trace=cuda --sample=none --cpuctxsw=none \
                    --force-overwrite true -o "$NSYS_PREFIX" \
                    "${ATTEST_ENV[@]}" "${RET_CMD[@]}" >"$WORKDIR/attest.log" 2>&1; then
                if [ -f "$NSYS_PREFIX.nsys-rep" ]; then
                    NSYS_KERNELS=$(nsys_kernels "$NSYS_PREFIX.nsys-rep")
                    if [ "${NSYS_KERNELS:-0}" -gt 0 ] 2>/dev/null; then
                        ATTEST_RESULT="nsys ($NSYS_KERNELS kernels)"
                        say "kernels:  $NSYS_KERNELS (GPU work observed)"
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
    fi
fi

# ---------------------------------------------------------------------------
# Determinism: two independent GPU encodes must match byte for byte.
# ---------------------------------------------------------------------------
DET_STATUS="not requested"
if [ "$DETERMINISM" = "1" ]; then
    say ""
    say "-- determinism (two GPU encodes) --"
    RUN2_IVF="$WORKDIR/gpu-run2.ivf"
    rewrite_output "$RUN2_IVF"
    RUN2_ENV=(env "AV2ENC_GPU_STATS=$WORKDIR/gpu2-stats.json" \
              "AV2ENC_M1_GPU_STATS=$WORKDIR/gpu2-stats.json")
    run_timed "${RUN2_ENV[@]}" "${RET_CMD[@]}" >"$WORKDIR/gpu-run2.log" 2>&1
    RUN2_RC=$?
    if [ $RUN2_RC -ne 0 ] || [ ! -s "$RUN2_IVF" ]; then
        sed 's/^/  /' "$WORKDIR/gpu-run2.log" | tail -10
        DET_STATUS="FAIL (second encode exit $RUN2_RC)"
    elif gpu_cpu_staged "$WORKDIR/gpu-run2.log"; then
        DET_STATUS="FAIL (second run used CPU staging)"
    elif [ -s "$WORKDIR/gpu2-stats.json" ] && ! stats_attests "$WORKDIR/gpu2-stats.json"; then
        DET_STATUS="FAIL (second run not GPU-attested)"
    else
        "$PY" "$SELF/ivf_compare.py" --label-a run1 --label-b run2 \
            "$GPU_IVF" "$RUN2_IVF" | tee "$WORKDIR/determinism.txt"
        DET_RC=${PIPESTATUS[0]}
        DET_STATUS=$([ $DET_RC -eq 0 ] && echo IDENTICAL || echo DIFFER)
    fi
    say "determinism: $DET_STATUS"
fi

# ---------------------------------------------------------------------------
# Decode gate with dav2d --strict 1, then bit-exact plane compare.
# ---------------------------------------------------------------------------
DECODE_STATUS="not run"
PLANE_STATUS="not run"
GPU_Y4M="$WORKDIR/gpu-decoded.y4m"
say ""
say "-- dav2d decode (--strict 1) --"
CPU_DEC2_RC=0
GPU_DEC_RC=0
run_timed "$DECODER" -q -i "$CPU_IVF" -o "$CPU_Y4M" \
    --threads "$DECD_THREADS" --strict 1 \
    >"$WORKDIR/cpu-decode2.log" 2>&1 || CPU_DEC2_RC=$?
run_timed "$DECODER" -q -i "$GPU_IVF" -o "$GPU_Y4M" \
    --threads "$DECD_THREADS" --strict 1 \
    >"$WORKDIR/gpu-decode.log" 2>&1 || GPU_DEC_RC=$?
if [ $CPU_DEC2_RC -ne 0 ] || [ ! -s "$CPU_Y4M" ]; then
    say "cpu decode: FAILED (exit $CPU_DEC2_RC)"
    sed 's/^/  /' "$WORKDIR/cpu-decode2.log" | tail -10
    DECODE_STATUS="FAIL (cpu)"
elif [ $GPU_DEC_RC -ne 0 ] || [ ! -s "$GPU_Y4M" ]; then
    say "cpu decode: ok"
    say "gpu decode: FAILED (exit $GPU_DEC_RC)"
    sed 's/^/  /' "$WORKDIR/gpu-decode.log" | tail -10
    DECODE_STATUS="FAIL (gpu)"
else
    say "cpu decode: ok (strict 1, threads=$DECD_THREADS)"
    say "gpu decode: ok (strict 1, threads=$DECD_THREADS)"
    DECODE_STATUS="ok"
    say ""
    say "-- decoded planes (Y/U/V) --"
    "$PY" "$SELF/plane_compare.py" "$CPU_Y4M" "$GPU_Y4M" \
        | tee "$WORKDIR/plane_compare.txt"
    PLANE_RC=${PIPESTATUS[0]}
    PLANE_STATUS=$([ $PLANE_RC -eq 0 ] && echo IDENTICAL || echo DIFFER)

    if [ "$PLANE_STATUS" = "DIFFER" ] && [ "$BYTES_STATUS" = "IDENTICAL" ]; then
        say ""
        say "note:     identical payloads decoded differently; retrying both"
        CPU_RETRY_RC=0
        GPU_RETRY_RC=0
        run_timed "$DECODER" -q -i "$CPU_IVF" -o "$WORKDIR/cpu-retry.y4m" \
            --threads "$DECD_THREADS" --strict 1 \
            >"$WORKDIR/cpu-retry.log" 2>&1 || CPU_RETRY_RC=$?
        run_timed "$DECODER" -q -i "$GPU_IVF" -o "$WORKDIR/gpu-retry.y4m" \
            --threads "$DECD_THREADS" --strict 1 \
            >"$WORKDIR/gpu-retry.log" 2>&1 || GPU_RETRY_RC=$?
        if [ $CPU_RETRY_RC -eq 0 ] && [ $GPU_RETRY_RC -eq 0 ]; then
            "$PY" "$SELF/plane_compare.py" "$WORKDIR/cpu-retry.y4m" \
                "$WORKDIR/gpu-retry.y4m" | tee "$WORKDIR/plane_compare_retry.txt"
            RETRY_RC=${PIPESTATUS[0]}
            if [ $RETRY_RC -eq 0 ]; then
                PLANE_STATUS="IDENTICAL (decoder retry)"
                say "note:     retry agreed; first decode was inconsistent"
            fi
        fi
    fi
fi

# ---------------------------------------------------------------------------
# Verdict.
# ---------------------------------------------------------------------------
say ""
say "-- summary --"
say "features:     ${FEATURES:--}"
say "frame bytes:  $BYTES_STATUS"
say "attestation:  ${ATTEST_RESULT:-none}"
say "nsys kernels: $NSYS_KERNELS"
say "decode:       $DECODE_STATUS"
say "decoded YUV:  $PLANE_STATUS"
say "determinism:  $DET_STATUS"
say "sanitizer:    $SAN_STATUS"
say "racecheck:    $RACE_STATUS"

if [ "$BYTES_STATUS" = "DIFFER" ]; then
    class_line "BYTE-DIFF"
    say "RESULT: FAIL (IVF frame payloads differ)"
    exit 1
fi
if [ "$DECODE_STATUS" = "FAIL (cpu)" ]; then
    class_line "CPU-REF-DECODE-FAIL"
    say "RESULT: FAIL (CPU reference stream does not decode)"
    exit 1
fi
if [ "$DECODE_STATUS" = "FAIL (gpu)" ]; then
    class_line "GPU-DECODE-FAIL"
    say "RESULT: FAIL (GPU stream does not decode with dav2d --strict 1)"
    exit 1
fi
if [ "$PLANE_STATUS" = "DIFFER" ]; then
    class_line "PLANE-DIFF"
    say "RESULT: FAIL (decoded Y/U/V planes differ)"
    exit 1
fi
case "$DET_STATUS" in
    DIFFER|FAIL*)
        class_line "NONDETERMINISTIC"
        say "RESULT: FAIL ($DET_STATUS)"
        exit 1 ;;
esac
case "$SAN_STATUS" in
    FAIL*)
        class_line "SANITIZER"
        say "RESULT: FAIL ($SAN_STATUS)"
        exit 1 ;;
esac
case "$RACE_STATUS" in
    FAIL*)
        class_line "RACECHECK"
        say "RESULT: FAIL ($RACE_STATUS)"
        exit 1 ;;
esac

# Attestation is the last gate: byte/plane checks passing without proof of GPU
# execution is exactly the silent-CPU-fallback failure mode.
if [ "$ATTEST_MODE" = "stats" ]; then
    if ! stats_attests "$GPU_STATS"; then
        if [ "$stats_seen" = "1" ]; then
            class_line "UNATTESTED"
            skip "UNATTESTED" "stats file present but does not attest GPU execution"
        fi
        class_line "UNATTESTED"
        skip "UNATTESTED" "no stats JSON attesting GPU execution (--attest stats)"
    fi
elif [ "$ATTEST_MODE" = "auto" ] && [ "$ATTEST_RESULT" != "none" ]; then
    :  # stats attested and nsys was not required
else
    if [ "${NSYS_KERNELS:-0}" -le 0 ] 2>/dev/null; then
        class_line "UNATTESTED"
        say "note: bytes matched, but nsys saw no CUDA kernel. A CPU fallback would"
        say "      produce identical bytes. This is not a pass."
        skip "UNATTESTED" "bytes matched but Nsight Systems observed no CUDA kernel"
    fi
fi

class_line "PASS"
say "RESULT: PASS"
exit 0
