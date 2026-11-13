#!/usr/bin/env bash
# CPU-vs-GPU gate for the M1 still-picture path.
#
#   CPU:  av2enc <in.y4m> -o cpu.ivf --still      [M1 options]
#   GPU:  av2enc <in.y4m> -o gpu.ivf --still-gpu  [M1 options]
#
# Gate (all must hold for PASS):
#   1. the GPU execution is *attested* (stats file, output marker, or Nsight
#      Systems shows >=1 CUDA kernel) so a silent CPU fallback cannot pass;
#   2. IVF **frame payloads** are byte-identical (not PSNR);
#   3. both streams decode with the pinned dav2d;
#   4. the decoded first frames are plane-identical;
#   5. in --recon mode, GPU recon planes equal the CPU StillResult.recon_* dump
#      when the orchestrator exposes one (otherwise the decoded-frame compare
#      above is the fallback);
#   6. with --determinism, two GPU encodes are byte-identical;
#   7. with --sanitize/--racecheck, Compute Sanitizer reports zero errors.
#
# A CPU-reference staging path (frame_bytes_from_cpu) is SKIP, never PASS.
#
# Verdicts / exit codes:
#   0  PASS
#   1  FAIL            (real byte/plane/decode difference, GPU crash, sanitizer
#                       error, nondeterminism)
#   2  usage/setup     (missing encoder/dav2d/ffmpeg/input, bad matrix)
#   3  SKIP            (no --still-gpu; blocked/stub; CPU-reference staging; or
#                       GPU execution could not be attested). A SKIP is never
#                       a pass.
#
# ALLOW_SKIP=1 maps SKIP to exit 0 for CI that wants "not a failure".
#
# Usage:
#   tools/still_gpu/verify_gpu.sh [input.y4m] [options]
#
#   tools/still_gpu/verify_gpu.sh                         # generate 640x360
#   tools/still_gpu/verify_gpu.sh in.y4m --recon
#   tools/still_gpu/verify_gpu.sh --size 1920x1080 --keep
#   tools/still_gpu/verify_gpu.sh --still-args "--still-depth 2 --no-deblock"
#   tools/still_gpu/verify_gpu.sh --determinism --sanitize
#
#   # Full option matrix (12 modes x 8 sizes by default). Prints a progress
#   # line per case, a PASS/FAIL/SKIP table, every failure with the exact
#   # byte offset / first sample, and grouped SKIP reasons.
#   tools/still_gpu/verify_gpu.sh --matrix
#   tools/still_gpu/verify_gpu.sh --matrix --sizes 64x64,136x88 \
#       --modes default,full-intra,no-cfl --determinism
#
# Options:
#   --encoder PATH        av2enc binary (default $AV2ENC or build-cuda/av2enc)
#   --decoder PATH        dav2d (default $DAV2D, tools/bin/dav2d[-macos])
#   --size WxH            generate this input when no input is given; native
#                         testsrc2 (default 640x360)
#   --workdir DIR         artifact directory (default mktemp -d)
#   --qp N                base_q_idx for both encodes (default 90)
#   --still-args "..."       M1 flags for BOTH encodes (depth, full-intra, no-cfl,
#                         no-deblock, single, dc, adaptive-cdf, vram-cap);
#                         mapped automatically to still_dump/probe arguments
#   --cpu-args "...".     extra args for the CPU run only
#   --gpu-args "..."      extra args for the GPU run only
#   --recon               compare encoder-side recon planes too
#   --harness PATH        still_dump binary for the CPU recon dump
#   --harness-args "..."  extra still_dump args (overrides --still-args mapping)
#   --gpu-dump PATH       GPU StillDUMP1 file to compare (implies --recon)
#   --gpu-dump-arg FLAG   append "FLAG <workdir>/gpu-recon.dump" to the GPU run
#   --probe PATH          optional still_gpu_probe binary (auto: tools/still_gpu/
#                         still_gpu_probe, build-cuda/still_gpu_probe)
#   --probe-args "..."    extra probe args
#   --require-recon       SKIP if no GPU recon dump is available
#   --determinism         encode twice with the CLI; bytes must match
#   --determinism-probe   also encode twice in one process via the probe
#   --sanitize            run the GPU encode under compute-sanitizer memcheck
#   --racecheck           additionally run compute-sanitizer racecheck at 64x64
#   --no-attestation      PASS on identical bytes even without GPU evidence
#                         (explicitly unsafe: a CPU fallback would pass)
#   --no-nsys             do not use Nsight Systems as the last-resort
#                         attestation source (faster; more SKIPs)
#   --keep                keep the work directory
#   --timeout SECONDS     per-command timeout (default 1800; uses `timeout`)
#   -h, --help
#
# Matrix options:
#   --matrix              run the option matrix (this is the batch mode)
#   --sizes LIST          comma list; token `random` expands to random even
#                         sizes. Default: 64x64,136x88,640x360,1280x720,
#                         1920x1080,random
#   --modes LIST          subset of: default,full-intra,no-cfl,no-deblock,
#                         depth-0..4,8,single,dc,adaptive-cdf,vram-cap-64,
#                         wiener,chroma-lr,lr,gdf,gdf-lr,delta-q,qm,cfl-ds,
#                         sb128,sb256,sb128-filters,sb256-filters,
#                         segmentation,neutral-delta-q,per-segment-q,
#                         ext,ext-sb128,sdp,sdp-full,sub8,sub8-full,
#                         dip,mrl,mrl-dip,ibp,ibp-full,
#                         cfl-explicit,mhccp,mhccp-explicit,uv-full,
#                         palette,ibc,palette-ibc,
#                         tx-type,idtx,ist,cctx,tcq,rdoq,tx-type-ist,ist-cctx
#                         (default: all)
#   --seed N              random-size seed (default 20261111)
#   --random-count N      number of random even sizes (default 3)
#   --matrix-budget SEC   stop starting new cases after SEC (default 1800,
#                         0 = unlimited); unfinished cases count as SKIP
#   --max-cases N         refuse to start if the matrix exceeds N (default 0)
#   --matrix-keep         keep per-case work directories
#
# Environment:
#   AV2ENC_GPU_STATS / AV2ENC_STILL_GPU_STATS  stats JSON path (see README)
#   AV2ENC_STILL_GPU_DUMP                     recon dump path (see README)
#   NSYS                                   nsys binary (default: PATH lookup)
#   COMPUTE_SANITIZER                      compute-sanitizer binary
#   PYTHON                                 python3 interpreter
#
set -uo pipefail

SELF=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$SELF/../.." && pwd)
PY=${PYTHON:-python3}
NSYS=${NSYS:-nsys}
CS_TOOL=${COMPUTE_SANITIZER:-compute-sanitizer}
CXX=${CXX:-g++}
DECD_THREADS=${DAV2D_THREADS:-1}
CUDART_LIB_DIR=${CUDART_LIB_DIR:-/usr/local/cuda/lib64}
AV2ENC_LIB=${AV2ENC_LIB:-}

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
HARNESS=${STILL_DUMP:-}
HARNESS_ARGS=""
CPU_ARGS=""
GPU_ARGS=""
STILL_ARGS=""
GPU_DUMP=""
GPU_DUMP_ARG=""
PROBE=""
PROBE_ARGS=""
PROBE_STILL_ARGS=""
NO_ATTESTATION=0
DETERMINISM=0
DETERMINISM_PROBE=0
SANITIZE=0
RACECHECK=0
TIMEOUT=${TIMEOUT:-1800}

MATRIX=0
MATRIX_SIZES="64x64,136x88,640x360,1280x720,1920x1080,random"
MATRIX_MODES="all"
MATRIX_BUDGET=${MATRIX_BUDGET:-1800}
MATRIX_MAX_CASES=${MATRIX_MAX_CASES:-0}
MATRIX_KEEP=0
RANDOM_SEED=20261111
RANDOM_COUNT=3

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
        --still-args) STILL_ARGS=$2; shift 2 ;;
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
        --determinism) DETERMINISM=1; shift ;;
        --determinism-probe) DETERMINISM_PROBE=1; shift ;;
        --sanitize) SANITIZE=1; shift ;;
        --racecheck) RACECHECK=1; shift ;;
        --no-attestation) NO_ATTESTATION=1; shift ;;
        --no-nsys) ATTEST=off; shift ;;
        --keep) KEEP=1; shift ;;
        --timeout) TIMEOUT=$2; shift 2 ;;
        --matrix) MATRIX=1; shift ;;
        --sizes) MATRIX_SIZES=$2; shift 2 ;;
        --modes) MATRIX_MODES=$2; shift 2 ;;
        --seed) RANDOM_SEED=$2; shift 2 ;;
        --random-count) RANDOM_COUNT=$2; shift 2 ;;
        --matrix-budget) MATRIX_BUDGET=$2; shift 2 ;;
        --max-cases) MATRIX_MAX_CASES=$2; shift 2 ;;
        --matrix-keep) MATRIX_KEEP=1; shift ;;
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

# Run a command under the per-command timeout when `timeout` is available.
run_timed() {
    if [ "${TIMEOUT:-0}" -gt 0 ] 2>/dev/null && command -v timeout >/dev/null 2>&1; then
        timeout "$TIMEOUT" "$@"
    else
        "$@"
    fi
}

# Map av2enc M1 flags to still_dump / still_gpu_probe flags. Keeping this in one
# place avoids the depth/option mismatch that used to fake transform bugs.
map_still_args() {
    local mode=$1 tokens out="" tok val i
    local -a t
    read -r -a t <<< "$2"
    i=0
    while [ $i -lt ${#t[@]} ]; do
        tok=${t[$i]}
        case "$tok" in
            --still-full-intra) out+=" --full-intra" ;;
            --still-depth)
                i=$((i + 1)); out+=" --depth ${t[$i]:-}" ;;
            --still-depth=*) out+=" --depth ${tok#*=}" ;;
            --still-single|--still-dc|--no-cfl|--no-deblock|--adaptive-cdf)
                out+=" $tok" ;;
            --vram-cap)
                i=$((i + 1))
                if [ "$mode" = "probe" ]; then out+=" --vram-cap ${t[$i]:-}"; fi ;;
            --vram-cap=*)
                if [ "$mode" = "probe" ]; then out+=" $tok"; fi ;;
            --qp)
                i=$((i + 1)) ;;  # verify passes --qp separately
            --qp=*) ;;
            *)
                # Unknown flags are forwarded so a new encoder flag surfaces in
                # the harness rather than being silently dropped. The matrix
                # only uses the mapping above.
                out+=" $tok" ;;
        esac
        i=$((i + 1))
    done
    printf '%s' "$out" | sed 's/^ *//'
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

# Rewrite the output IVF (and the recon dump, when one is passed) in GPU_CMD.
# Used by the nsys attestation run, the determinism second run and racecheck so
# none of them overwrite the artifact that is byte-gated and decoded.
rewrite_output() {
    local new_ivf=$1 new_dump=$2 part next_output=0 next_dump=0
    RET_CMD=()
    for part in "${GPU_CMD[@]}"; do
        if [ $next_output -eq 1 ]; then RET_CMD+=("$new_ivf"); next_output=0; continue; fi
        if [ $next_dump -eq 1 ]; then RET_CMD+=("$new_dump"); next_dump=0; continue; fi
        if [ "$part" = "-o" ] || [ "$part" = "--output" ]; then
            RET_CMD+=("$part"); next_output=1; continue
        fi
        if [ -n "$GPU_DUMP_ARG" ] && [ "$part" = "$GPU_DUMP_ARG" ]; then
            RET_CMD+=("$part"); next_dump=1; continue
        fi
        RET_CMD+=("$part")
    done
}

reason_from_log() {
    local log=$1 rc=$2 line
    line=$(grep -E '^RESULT: ' "$log" 2>/dev/null | tail -1)
    if [ -n "$line" ]; then
        printf '%s' "${line#RESULT: }"
    else
        printf 'no RESULT line (exit %s)' "$rc"
    fi
}

# Build still_dump against the same library as the encoder. A stale harness
# reconstructs a different tree and looks like a transform bug; rebuilding makes
# harness/library skew impossible.
build_harness() {
    local src="$ROOT/tools/still_verify/still_dump.cpp"
    local lib="${AV2ENC_LIB:-}"
    local out="$WORKDIR/still_dump"
    if [ -z "$lib" ]; then
        lib="$(dirname "$ENCODER")/libav2enc_core.a"
        [ -f "$lib" ] || lib="$ROOT/build-cuda/libav2enc_core.a"
        [ -f "$lib" ] || lib="$ROOT/build/libav2enc_core.a"
    fi
    [ -f "$src" ] && [ -f "$lib" ] || return 1
    command -v "$CXX" >/dev/null 2>&1 || return 1
    local -a link=()
    if [ -d "$CUDART_LIB_DIR" ]; then
        link=(-L"$CUDART_LIB_DIR" -lcudart)
    fi
    if [ ${#link[@]} -gt 0 ]; then
        "$CXX" -O2 -std=c++20 -I"$ROOT/src" "$src" -o "$out" "$lib" \
            "${link[@]}" >"$WORKDIR/harness-build.log" 2>&1
    else
        "$CXX" -O2 -std=c++20 -I"$ROOT/src" "$src" -o "$out" "$lib" \
            >"$WORKDIR/harness-build.log" 2>&1
    fi
}

# ---------------------------------------------------------------------------
# Matrix mode. Runs each (size, mode) as an independent verify_gpu.sh case.
# ---------------------------------------------------------------------------
run_matrix() {
    command -v ffmpeg >/dev/null 2>&1 || setup_error "ffmpeg not found (needed to generate matrix inputs)"
    command -v "$PY" >/dev/null 2>&1 || setup_error "python3 not found"

    # Encoder capability is checked once; without the flag every case is SKIP.
    local help
    help=$("$ENCODER" --help 2>&1 || true)
    say "== M1 GPU matrix =="
    say "encoder:  $ENCODER"
    if ! grep -q -- '--still-gpu' <<< "$help"; then
        say "gate:     this build has no --still-gpu option"
        skip "GPU path not present in this encoder build"
    fi

    local mroot
    if [ -n "$WORKDIR" ]; then
        mroot="$WORKDIR/matrix"
    else
        mroot=$(mktemp -d "${TMPDIR:-/tmp}/still_gpu_matrix.XXXXXX")
    fi
    mkdir -p "$mroot/logs" "$mroot/inputs" || setup_error "cannot create $mroot"

    # ---- sizes -------------------------------------------------------------
    for numeric in "$RANDOM_COUNT" "$MATRIX_BUDGET" "$MATRIX_MAX_CASES"; do
        case "$numeric" in
            ''|*[!0-9]*) setup_error "matrix numeric option is not an integer: '$numeric'" ;;
        esac
    done
    if [ "$MATRIX_SIZES" = "" ]; then setup_error "no matrix sizes"; fi
    local -a sizes=() parts=()
    IFS=',' read -r -a parts <<< "$MATRIX_SIZES"
    local token
    for token in "${parts[@]}"; do
        token=$(printf '%s' "$token" | tr -d '[:space:]')
        [ -z "$token" ] && continue
        if [ "$token" = "random" ]; then
            local random_list
            random_list=$("$PY" - "$RANDOM_SEED" "$RANDOM_COUNT" <<'PY'
import random, sys
seed, count = int(sys.argv[1]), int(sys.argv[2])
fixed = {(64, 64), (136, 88), (640, 360), (1280, 720), (1920, 1080)}
rng = random.Random(seed)
out = []
while len(out) < count:
    w = rng.randrange(64, 960, 2)
    h = rng.randrange(48, 720, 2)
    if (w, h) in fixed or (w <= 64 and h <= 64):
        continue
    if w % 64 == 0 and h % 64 == 0:
        continue  # prefer an edge/odd-superblock case
    out.append(f"{w}x{h}")
print(",".join(out))
PY
)
            local -a random_parts=()
            IFS=',' read -r -a random_parts <<< "$random_list"
            local rp
            for rp in "${random_parts[@]}"; do
                [ -n "$rp" ] && sizes+=("$rp")
            done
        else
            sizes+=("$token")
        fi
    done
    [ ${#sizes[@]} -gt 0 ] || setup_error "no matrix sizes"

    # ---- modes -------------------------------------------------------------
    local -a all_names=(default full-intra no-cfl no-deblock \
        depth-0 depth-1 depth-2 depth-3 depth-4 depth-8 single dc \
        adaptive-cdf vram-cap-64 wiener chroma-lr lr gdf gdf-lr delta-q qm cfl-ds \
        sb128 sb256 sb128-filters sb256-filters segmentation neutral-delta-q \
        per-segment-q ext ext-sb128 sdp sdp-full sub8 sub8-full dip mrl mrl-dip \
        ibp ibp-full cfl-explicit mhccp mhccp-explicit uv-full \
        palette ibc palette-ibc tx-type idtx ist cctx tcq rdoq tx-type-ist ist-cctx)
    local -a all_args=("" "--still-full-intra" "--no-cfl" "--no-deblock" \
        "--still-depth 0" "--still-depth 1" "--still-depth 2" "--still-depth 3" \
        "--still-depth 4" "--still-depth 8" "--still-single" "--still-dc" \
        "--adaptive-cdf" "--vram-cap 64" "--still-wiener" \
        "--still-chroma-lr" "--still-wiener --still-chroma-lr" \
        "--still-gdf" \
        "--still-cdef --still-ccso --still-gdf --still-wiener --still-chroma-lr" \
        "--still-dc-ac-delta-q" "--still-qm" "--still-cfl-ds 2" \
        "--still-sb128" "--still-sb256" \
        "--still-sb128 --still-cdef --still-ccso --still-wiener --still-chroma-lr --still-gdf" \
        "--still-sb256 --still-cdef --still-ccso --still-pc-wiener --still-gdf" \
        "--still-segmentation" "--still-delta-q" "--still-per-segment-q" \
        "--still-ext-partitions" \
        "--still-ext-partitions --still-sb128" "--still-sdp" \
        "--still-sdp --still-full-intra" "--still-sub8" \
        "--still-sub8 --still-full-intra" "--still-dip" "--still-mrl" \
        "--still-mrl --still-dip" "--still-ibp" \
        "--still-ibp --still-full-intra" "--still-cfl-explicit" \
        "--still-mhccp" "--still-mhccp --still-cfl-explicit" \
        "--still-uv-full" "--still-palette" "--still-ibc" \
        "--still-palette --still-ibc" \
        "--still-tx-type" "--still-idtx" "--still-ist" "--still-cctx" \
        "--still-tcq" "--still-rdoq" \
        "--still-tx-type --still-ist" "--still-ist --still-cctx")
    local -a mode_names=() mode_args=()
    if [ "$MATRIX_MODES" = "all" ]; then
        mode_names=("${all_names[@]}")
        mode_args=("${all_args[@]}")
    else
        local -a wanted=()
        IFS=',' read -r -a wanted <<< "$MATRIX_MODES"
        local want found
        for want in "${wanted[@]}"; do
            want=$(printf '%s' "$want" | tr -d '[:space:]')
            found=0
            local idx
            for idx in "${!all_names[@]}"; do
                if [ "${all_names[$idx]}" = "$want" ]; then
                    mode_names+=("${all_names[$idx]}")
                    mode_args+=("${all_args[$idx]}")
                    found=1
                    break
                fi
            done
            if [ "$found" = 0 ]; then
                setup_error "unknown --modes entry '$want'"
            fi
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
    if [ "$total" -gt 40 ]; then
        say "WARNING:  large matrix. Cases run sequentially; each case does a CPU"
        say "          encode, a GPU encode, a dav2d decode and byte/plane"
        say "          compares. Runtime is bounded to ${MATRIX_BUDGET}s of new"
        say "          cases (--matrix-budget). Narrow with --sizes/--modes."
    fi
    if [ "$SANITIZE" = 1 ] && [ "$total" -gt 8 ]; then
        say "WARNING:  compute-sanitizer under $total cases can be very slow."
        say "          Consider --sizes 64x64 --modes default --sanitize."
    fi

    # ---- inputs ------------------------------------------------------------
    local size input
    for size in "${sizes[@]}"; do
        input="$mroot/inputs/in-$size.y4m"
        if [ ! -s "$input" ]; then
            generate_y4m "$size" "$input" >/dev/null
        fi
    done

    # ---- run ---------------------------------------------------------------
    local start now elapsed pass=0 fail=0 skipped=0 idx=0
    local -a failures=()
    local skip_file="$mroot/skip-reasons.txt"
    : > "$skip_file"
    start=$(date +%s)

    local mode_i mode_name mode_arg case_log case_cmd combined rc reason
    for size in "${sizes[@]}"; do
        input="$mroot/inputs/in-$size.y4m"
        for mode_i in "${!mode_names[@]}"; do
            idx=$((idx + 1))
            mode_name=${mode_names[$mode_i]}
            mode_arg=${mode_args[$mode_i]}
            case_log="$mroot/logs/${size}-${mode_name}.log"

            now=$(date +%s)
            elapsed=$((now - start))
            if [ "$MATRIX_BUDGET" -gt 0 ] && [ "$elapsed" -ge "$MATRIX_BUDGET" ]; then
                local remaining=$((total - idx + 1))
                skipped=$((skipped + remaining))
                printf 'budget:   stopping after %ss; %d case(s) not run\n' \
                    "$elapsed" "$remaining"
                local budget_i=0
                while [ $budget_i -lt "$remaining" ]; do
                    printf 'matrix budget exceeded after %ss\n' "$elapsed" \
                        >> "$skip_file"
                    budget_i=$((budget_i + 1))
                done
                idx=$((total + 1))
                break 2
            fi

            combined="$STILL_ARGS $mode_arg"
            case_cmd=("$SELF/verify_gpu.sh" --encoder "$ENCODER" \
                --decoder "$DECODER" --qp "$QP" --timeout "$TIMEOUT")
            [ -n "$combined" ] && case_cmd+=(--still-args "$combined")
            [ -n "$CPU_ARGS" ] && case_cmd+=(--cpu-args "$CPU_ARGS")
            [ -n "$GPU_ARGS" ] && case_cmd+=(--gpu-args "$GPU_ARGS")
            [ "$RECON" = 1 ] && case_cmd+=(--recon)
            [ "$REQUIRE_RECON" = 1 ] && case_cmd+=(--require-recon)
            [ "$DETERMINISM" = 1 ] && case_cmd+=(--determinism)
            [ "$DETERMINISM_PROBE" = 1 ] && case_cmd+=(--determinism-probe)
            [ "$SANITIZE" = 1 ] && case_cmd+=(--sanitize)
            if [ "$RACECHECK" = 1 ] && [ "$size" = "64x64" ]; then
                case_cmd+=(--racecheck)
            fi
            [ -n "$PROBE" ] && case_cmd+=(--probe "$PROBE")
            [ -n "$PROBE_ARGS" ] && case_cmd+=(--probe-args "$PROBE_ARGS")
            if [ "$MATRIX_KEEP" = 1 ]; then
                case_cmd+=(--keep --workdir "$mroot/cases/$size-$mode_name")
            fi
            case_cmd+=("$input")

            printf '[%3d/%3d] %-10s %-13s ' "$idx" "$total" "$size" "$mode_name"
            "${case_cmd[@]}" >"$case_log" 2>&1
            rc=$?
            reason=$(reason_from_log "$case_log" "$rc")
            case $rc in
                0)
                    pass=$((pass + 1))
                    printf 'PASS\n'
                    ;;
                3)
                    skipped=$((skipped + 1))
                    printf 'SKIP  %s\n' "$reason"
                    printf '%s\n' "$reason" >> "$skip_file"
                    ;;
                *)
                    fail=$((fail + 1))
                    printf 'FAIL  %s\n' "$reason"
                    failures+=("$size|$mode_name|$case_log")
                    ;;
            esac
        done
    done

    now=$(date +%s)
    elapsed=$((now - start))
    say ""
    say "-- matrix summary --"
    say "cases:    $total (${#mode_names[@]} modes x ${#sizes[@]} sizes)"
    say "result:   $pass PASS, $fail FAIL, $skipped SKIP"
    say "elapsed:  ${elapsed}s (budget ${MATRIX_BUDGET}s)"
    say "logs:     $mroot/logs"

    if [ "$fail" -gt 0 ]; then
        say ""
        say "FAILURES (exact diff)"
        local entry f_size f_mode f_log detail
        for entry in "${failures[@]}"; do
            IFS='|' read -r f_size f_mode f_log <<< "$entry"
            say ""
            say "[$f_size $f_mode] $(reason_from_log "$f_log" 1)"
            detail=$(grep -E 'first byte offset|first sample|plane [YUV]:.*differ|MISMATCH|RESULT: .*(DIFFER|FAIL)|ERROR SUMMARY|RACECHECK SUMMARY|decode: +FAILED|Error decoding frame|No data decoded|repeat' "$f_log" 2>/dev/null | head -8)
            if [ -n "$detail" ]; then
                printf '%s\n' "$detail" | sed 's/^/  /'
            else
                tail -8 "$f_log" | sed 's/^/  /'
            fi
            say "  log: $f_log"
        done
    fi

    # Attribute failures so an invalid CPU reference is not mistaken for a GPU
    # parity bug. These counts overlap when a case fails several checks.
    if [ "$fail" -gt 0 ]; then
        local cpu_enc=0 cpu_dec=0 gpu_dec=0 byte_diff=0 san_fail=0
        local fentry
        for fentry in "${failures[@]}"; do
            IFS='|' read -r f_size f_mode f_log <<< "$fentry"
            grep -q 'CPU reference encode failed' "$f_log" 2>/dev/null && cpu_enc=$((cpu_enc + 1))
            grep -q 'cpu decode: FAILED' "$f_log" 2>/dev/null && cpu_dec=$((cpu_dec + 1))
            grep -q 'gpu decode: FAILED' "$f_log" 2>/dev/null && gpu_dec=$((gpu_dec + 1))
            grep -q 'RESULT: IVF FRAME BYTES DIFFER' "$f_log" 2>/dev/null && byte_diff=$((byte_diff + 1))
            grep -q 'compute-sanitizer.*errors\|RACECHECK.*error' "$f_log" 2>/dev/null && san_fail=$((san_fail + 1))
        done
        say ""
        say "ATTRIBUTION (counts per failing case; may overlap)"
        say "  cpu reference encode failures: $cpu_enc (encoder/geometry bug, not GPU parity)"
        say "  cpu reference decode failures: $cpu_dec (encoder/geometry bug, not GPU parity)"
        say "  gpu decode failures:           $gpu_dec"
        say "  frame-byte differences:        $byte_diff"
        say "  sanitizer/racecheck failures:  $san_fail"
    fi

    if [ "$skipped" -gt 0 ]; then
        say ""
        say "SKIPPED (grouped by reason)"
        sort "$skip_file" 2>/dev/null | uniq -c | sort -rn | sed 's/^/  /'
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
    say "RESULT: PASS ($total case(s))"
    return 0
}

# ---------------------------------------------------------------------------
# Common setup.
# ---------------------------------------------------------------------------
command -v "$PY" >/dev/null 2>&1 || setup_error "python3 not found"
[ -x "$ENCODER" ] || setup_error "encoder not executable: $ENCODER (build it first)"
command -v ffmpeg >/dev/null 2>&1 || [ -n "$INPUT" ] || \
    setup_error "ffmpeg not found; pass an input Y4M or install ffmpeg"

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

# Does this build even have the GPU path?  No flag -> SKIP, never PASS.
HELP=$("$ENCODER" --help 2>&1 || true)
if ! grep -q -- '--still-gpu' <<< "$HELP"; then
    say "== M1 GPU gate =="
    say "encoder:  $ENCODER"
    say "gate:     this build has no --still-gpu option"
    skip "GPU path not present in this encoder build"
fi

# Map common M1 flags onto both encodes and the recon harness/probe.
if [ -n "$STILL_ARGS" ]; then
    CPU_ARGS="$CPU_ARGS $STILL_ARGS"
    GPU_ARGS="$GPU_ARGS $STILL_ARGS"
    PROBE_STILL_ARGS=$(map_still_args probe "$STILL_ARGS")
    if [ -z "$HARNESS_ARGS" ]; then
        HARNESS_ARGS=$(map_still_args harness "$STILL_ARGS")
    fi
fi

if [ -z "$WORKDIR" ]; then
    WORKDIR=$(mktemp -d "${TMPDIR:-/tmp}/still_gpu_verify.XXXXXX")
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
    generate_y4m "$SIZE" "$INPUT"
fi
[ -f "$INPUT" ] || setup_error "no such input: $INPUT"
DIMS=$(input_dims "$INPUT")

if [ "$SANITIZE" = "1" ] || [ "$RACECHECK" = "1" ]; then
    command -v "$CS_TOOL" >/dev/null 2>&1 || [ -x "$CS_TOOL" ] || \
        skip "compute-sanitizer not found (set COMPUTE_SANITIZER)"
fi

say "== M1 GPU gate =="
say "encoder:  $ENCODER"
say "decoder:  $DECODER"
say "input:    $INPUT ($DIMS)"
say "m1 args:  ${STILL_ARGS:--}"
say "workdir:  $WORKDIR"

# ---------------------------------------------------------------------------
# Encode CPU reference and GPU candidate.
# ---------------------------------------------------------------------------
CPU_IVF="$WORKDIR/cpu.ivf"
CPU_Y4M="$WORKDIR/cpu-decoded.y4m"
GPU_IVF="$WORKDIR/gpu.ivf"
GPU_LOG="$WORKDIR/gpu.log"
GPU_STATS="$WORKDIR/gpu-stats.json"
AUTO_DUMP="$WORKDIR/gpu-recon.dump"

# shellcheck disable=SC2086
run_timed "$ENCODER" "$INPUT" -o "$CPU_IVF" --still --qp "$QP" $CPU_ARGS \
    >"$WORKDIR/cpu.log" 2>&1
CPU_RC=$?
if [ $CPU_RC -ne 0 ] || [ ! -s "$CPU_IVF" ]; then
    say ""
    say "cpu log:"
    sed 's/^/  /' "$WORKDIR/cpu.log" | tail -20
    say "RESULT: FAIL (CPU reference encode failed, exit $CPU_RC)"
    exit 1
fi

# Validate the reference stream before touching the GPU path. If the CPU
# reference does not decode there is nothing to compare against, and a staged
# GPU run must not hide that behind SKIP. This is how the matrix surfaces
# geometry bugs even when the GPU path is not wired yet.
CPU_DEC_RC=0
run_timed "$DECODER" -q -i "$CPU_IVF" -o "$CPU_Y4M" --threads "$DECD_THREADS" \
    >"$WORKDIR/cpu-decode.log" 2>&1 || CPU_DEC_RC=$?
if [ $CPU_DEC_RC -ne 0 ] || [ ! -s "$CPU_Y4M" ]; then
    say ""
    say "-- dav2d decode (CPU reference) --"
    say "cpu decode: FAILED (exit $CPU_DEC_RC; reference stream is invalid)"
    sed 's/^/  /' "$WORKDIR/cpu-decode.log" | tail -10
    say "RESULT: FAIL (CPU reference stream does not decode)"
    exit 1
fi

GPU_ENV=(env "AV2ENC_GPU_STATS=$GPU_STATS" \
          "AV2ENC_STILL_GPU_STATS=$GPU_STATS" \
          "AV2ENC_STILL_GPU_DUMP=$AUTO_DUMP")
GPU_CMD=("$ENCODER" "$INPUT" -o "$GPU_IVF" --still-gpu --qp "$QP")
if [ -n "$GPU_ARGS" ]; then
    # shellcheck disable=SC2206
    GPU_CMD+=($GPU_ARGS)
fi
if [ -n "$GPU_DUMP_ARG" ]; then
    if [ -z "$GPU_DUMP" ]; then GPU_DUMP="$AUTO_DUMP"; fi
    GPU_CMD+=("$GPU_DUMP_ARG" "$GPU_DUMP")
fi

SAN_STATUS="not requested"
RACE_STATUS="not requested"
if [ "$SANITIZE" = "1" ]; then
    GPU_LOG="$WORKDIR/sanitize-memcheck.log"
    say ""
    say "-- compute-sanitizer (memcheck) --"
    run_timed "$CS_TOOL" --tool memcheck --error-exitcode 99 \
        --target-processes all "${GPU_ENV[@]}" "${GPU_CMD[@]}" >"$GPU_LOG" 2>&1
    GPU_RC=$?
    SAN_ERRORS=$(grep -oE 'ERROR SUMMARY: [0-9]+ error' "$GPU_LOG" | tail -1 \
        | grep -oE '[0-9]+' | head -1)
    if [ -n "$SAN_ERRORS" ] && [ "$SAN_ERRORS" -gt 0 ]; then
        say "memcheck: $SAN_ERRORS error(s)"
        grep -E 'Invalid|out-of-bounds|Uninitialized|ERROR SUMMARY' "$GPU_LOG" \
            | head -12 | sed 's/^/  /'
        say "RESULT: FAIL (compute-sanitizer memcheck: $SAN_ERRORS errors)"
        exit 1
    fi
    if grep -q 'terminated before first instrumented API call' "$GPU_LOG"; then
        say "memcheck: no instrumented CUDA API call"
        skip "compute-sanitizer saw no CUDA work (CPU/staged path)"
    fi
    if [ -z "$SAN_ERRORS" ]; then
        if grep -Eqi '(not[ _-]+implemented|unsupported|stub|not[ _-]+built|not[ _-]+available|no[ _-]+cuda[ _-]+device|CUDA_ERROR_NO_DEVICE|disabled|blocked|TODO|requires[ _-]+a[ _-]+cuda[ _-]+build|cpu[ _-]+only)' "$GPU_LOG"; then
            skip "GPU run reported an unimplemented/blocked path under sanitizer"
        fi
        say ""
        sed 's/^/  /' "$GPU_LOG" | tail -12
        say "RESULT: FAIL (compute-sanitizer produced no ERROR SUMMARY)"
        exit 1
    fi
    SAN_STATUS="memcheck 0 errors"
    say "memcheck: 0 errors"
else
    run_timed "${GPU_ENV[@]}" "${GPU_CMD[@]}" >"$GPU_LOG" 2>&1
    GPU_RC=$?
fi

# ---------------------------------------------------------------------------
# Classify GPU run: pass through real work, SKIP stubs/blockers.
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
# Racecheck (64x64 only; deliberately separate from memcheck).
# ---------------------------------------------------------------------------
if [ "$RACECHECK" = "1" ]; then
    if [ "$DIMS" != "64x64" ]; then
        RACE_STATUS="skipped (only 64x64)"
        say ""
        say "-- compute-sanitizer (racecheck) --"
        say "racecheck: skipped (input is $DIMS, racecheck runs at 64x64)"
    else
        say ""
        say "-- compute-sanitizer (racecheck) --"
        rewrite_output "$WORKDIR/race.ivf" "$WORKDIR/race-recon.dump"
        RACE_ENV=(env "AV2ENC_GPU_STATS=$WORKDIR/race-stats.json" \
                  "AV2ENC_STILL_GPU_STATS=$WORKDIR/race-stats.json" \
                  "AV2ENC_STILL_GPU_DUMP=$WORKDIR/race-recon.dump")
        RACE_LOG="$WORKDIR/sanitize-racecheck.log"
        run_timed "$CS_TOOL" --tool racecheck --error-exitcode 99 \
            --target-processes all "${RACE_ENV[@]}" "${RET_CMD[@]}" >"$RACE_LOG" 2>&1
        RACE_RC=$?
        RACE_LINE=$(grep -E 'RACECHECK SUMMARY' "$RACE_LOG" | tail -1)
        RACE_ERRORS=$(printf '%s' "$RACE_LINE" | grep -oE '[0-9]+ error' | grep -oE '[0-9]+' | head -1)
        RACE_WARNINGS=$(printf '%s' "$RACE_LINE" | grep -oE '[0-9]+ warning' | grep -oE '[0-9]+' | head -1)
        if grep -q 'terminated before first instrumented API call' "$RACE_LOG"; then
            say "racecheck: no instrumented CUDA API call"
            RACE_STATUS="skipped (no CUDA work)"
        elif [ -z "$RACE_LINE" ]; then
            say "racecheck: no RACECHECK SUMMARY (exit $RACE_RC)"
            sed 's/^/  /' "$RACE_LOG" | tail -10
            RACE_STATUS="FAIL (no summary)"
        elif [ "${RACE_ERRORS:-0}" -gt 0 ] || [ "${RACE_WARNINGS:-0}" -gt 0 ]; then
            say "racecheck: ${RACE_ERRORS:-0} errors, ${RACE_WARNINGS:-0} warnings"
            RACE_STATUS="FAIL (${RACE_ERRORS:-0} errors, ${RACE_WARNINGS:-0} warnings)"
        else
            say "racecheck: 0 hazards"
            RACE_STATUS="0 hazards"
        fi
    fi
fi

# ---------------------------------------------------------------------------
# Byte gate: IVF frame payloads (not PSNR) + MD5.
# ---------------------------------------------------------------------------
say ""
say "-- frame bytes --"
"$PY" "$SELF/ivf_compare.py" "$CPU_IVF" "$GPU_IVF" | tee "$WORKDIR/ivf_compare.txt"
BYTES_RC=${PIPESTATUS[0]}
BYTES_STATUS=$([ $BYTES_RC -eq 0 ] && echo IDENTICAL || echo DIFFER)

# ---------------------------------------------------------------------------
# Attestation that the GPU path actually ran.
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
elif grep -Eqi 'backend[[:space:]]*[=:][[:space:]]*gpu|using[[:space:]]+the[[:space:]]+GPU|GPU[[:space:]]+path[[:space:]]+(ran|active|enabled)|stages?[[:space:]]*:[^\n]*gpu' "$GPU_LOG"; then
    ATTEST_RESULT="output marker"
elif [ "$ATTEST" != "off" ] && { command -v "$NSYS" >/dev/null 2>&1 || [ -x "$NSYS" ]; }; then
    say ""
    say "-- attestation (nsys) --"
    NSYS_PREFIX="$WORKDIR/attest"
    rm -f "$NSYS_PREFIX.nsys-rep" "$NSYS_PREFIX.sqlite"

    # The attestation run must not overwrite the artifact that was byte-gated
    # and will be decoded: redirect its IVF, stats and recon dump to attest.*.
    rewrite_output "$WORKDIR/attest.ivf" "$WORKDIR/attest-recon.dump"
    ATTEST_CMD=("${RET_CMD[@]}")
    ATTEST_ENV=(env
        "AV2ENC_GPU_STATS=$WORKDIR/attest-stats.json"
        "AV2ENC_STILL_GPU_STATS=$WORKDIR/attest-stats.json"
        "AV2ENC_STILL_GPU_DUMP=$WORKDIR/attest-recon.dump")
    if run_timed "$NSYS" profile --trace=cuda --sample=none --cpuctxsw=none \
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
# Determinism: two independent GPU encodes must match, byte for byte.
# ---------------------------------------------------------------------------
DET_STATUS="not requested"
DET_PROBE_STATUS="not requested"
if [ "$DETERMINISM" = "1" ]; then
    say ""
    say "-- determinism (two CLI encodes) --"
    RUN2_IVF="$WORKDIR/gpu-run2.ivf"
    rewrite_output "$RUN2_IVF" "$WORKDIR/gpu-run2-recon.dump"
    RUN2_ENV=(env "AV2ENC_GPU_STATS=$WORKDIR/gpu2-stats.json" \
              "AV2ENC_STILL_GPU_STATS=$WORKDIR/gpu2-stats.json" \
              "AV2ENC_STILL_GPU_DUMP=$WORKDIR/gpu-run2-recon.dump")
    run_timed "${RUN2_ENV[@]}" "${RET_CMD[@]}" >"$WORKDIR/gpu-run2.log" 2>&1
    RUN2_RC=$?
    if [ $RUN2_RC -ne 0 ] || [ ! -s "$RUN2_IVF" ]; then
        sed 's/^/  /' "$WORKDIR/gpu-run2.log" | tail -10
        DET_STATUS="FAIL (second encode exit $RUN2_RC)"
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

if [ "$DETERMINISM_PROBE" = "1" ]; then
    say ""
    say "-- determinism (two encodes in one process, probe) --"
    if [ -z "$PROBE" ]; then
        for candidate in "$ROOT/tools/still_gpu/still_gpu_probe" \
                         "$ROOT/build-cuda/still_gpu_probe"; do
            if [ -x "$candidate" ]; then PROBE=$candidate; break; fi
        done
    fi
    if [ -z "$PROBE" ] || [ ! -x "$PROBE" ]; then
        say "note:     no still_gpu_probe; in-process determinism not measured"
        DET_PROBE_STATUS="unavailable (no probe)"
    else
        DET_PROBE_STATS="$WORKDIR/determinism-probe.json"
        # shellcheck disable=SC2086
        if run_timed "$PROBE" -i "$INPUT" -o "$WORKDIR/determinism-probe.dump" \
                --stats "$DET_PROBE_STATS" --repeat 2 --qp "$QP" \
                $PROBE_ARGS $PROBE_STILL_ARGS >"$WORKDIR/determinism-probe.log" 2>&1; then
            DET_PROBE_LINE=$("$PY" - "$DET_PROBE_STATS" <<'PY'
import json, sys
try:
    data = json.load(open(sys.argv[1]))
except Exception:
    print("unreadable")
    raise SystemExit(0)
if data.get("frame_bytes_from_cpu") is True:
    print("cpu-reference")
elif data.get("repeat_identical") is True:
    print("IDENTICAL")
elif data.get("repeat_identical") is False:
    print(f"DIFFER offset {data.get('repeat_first_diff_offset', -1)}")
else:
    print("no-repeat-data")
PY
)
            case "$DET_PROBE_LINE" in
                IDENTICAL)
                    DET_PROBE_STATUS="IDENTICAL" ;;
                DIFFER*)
                    DET_PROBE_STATUS="$DET_PROBE_LINE" ;;
                cpu-reference)
                    DET_PROBE_STATUS="ignored (probe used CPU reference)" ;;
                *)
                    DET_PROBE_STATUS="unavailable ($DET_PROBE_LINE)" ;;
            esac
            say "probe:    $DET_PROBE_STATUS"
        else
            sed 's/^/  /' "$WORKDIR/determinism-probe.log" | tail -10
            DET_PROBE_STATUS="unavailable (probe failed)"
            say "probe:    $DET_PROBE_STATUS"
        fi
    fi
fi

# ---------------------------------------------------------------------------
# Decode gate with the pinned dav2d, then plane compare.
# ---------------------------------------------------------------------------
DECODE_STATUS="not run"
PLANE_STATUS="not run"
GPU_Y4M="$WORKDIR/gpu-decoded.y4m"
say ""
say "-- dav2d decode --"
CPU_DEC2_RC=0
GPU_DEC_RC=0
# Decode both streams back-to-back with one thread. The early decode above is a
# reference-validity preflight; this pair is what the plane gate compares, so
# both sides see the same decoder state.
run_timed "$DECODER" -q -i "$CPU_IVF" -o "$CPU_Y4M" --threads "$DECD_THREADS" \
    >"$WORKDIR/cpu-decode2.log" 2>&1 || CPU_DEC2_RC=$?
run_timed "$DECODER" -q -i "$GPU_IVF" -o "$GPU_Y4M" --threads "$DECD_THREADS" \
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
    say "cpu decode: ok (threads=$DECD_THREADS)"
    say "gpu decode: ok (threads=$DECD_THREADS)"
    DECODE_STATUS="ok"
    say ""
    say "-- decoded planes --"
    "$PY" "$SELF/plane_compare.py" "$CPU_Y4M" "$GPU_Y4M" \
        | tee "$WORKDIR/plane_compare.txt"
    PLANE_RC=${PIPESTATUS[0]}
    PLANE_STATUS=$([ $PLANE_RC -eq 0 ] && echo IDENTICAL || echo DIFFER)

    # Identical payloads must decode identically. If they did not, retry the
    # pair once before declaring a parity failure: a rare decoder/threading
    # inconsistency is not a GPU bug, and the byte gate is authoritative.
    if [ "$PLANE_STATUS" = "DIFFER" ] && [ "$BYTES_STATUS" = "IDENTICAL" ]; then
        say ""
        say "note:     identical payloads decoded differently; retrying both"
        CPU_RETRY_RC=0
        GPU_RETRY_RC=0
        run_timed "$DECODER" -q -i "$CPU_IVF" -o "$WORKDIR/cpu-retry.y4m" \
            --threads "$DECD_THREADS" >"$WORKDIR/cpu-retry.log" 2>&1 || CPU_RETRY_RC=$?
        run_timed "$DECODER" -q -i "$GPU_IVF" -o "$WORKDIR/gpu-retry.y4m" \
            --threads "$DECD_THREADS" >"$WORKDIR/gpu-retry.log" 2>&1 || GPU_RETRY_RC=$?
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
# Recon parity (GPU StillGpuResult vs CPU StillResult.recon_*), if available.
# ---------------------------------------------------------------------------
RECON_STATUS="not requested"
if [ "$RECON" = "1" ]; then
    RECON_STATUS="unavailable"
    say ""
    say "-- recon parity --"
    if [ -z "$HARNESS" ]; then
        if build_harness; then
            HARNESS="$WORKDIR/still_dump"
            say "harness:  rebuilt still_dump against the current library"
        else
            for candidate in "$ROOT/build-cuda/still_dump" "$ROOT/tools/still_verify/still_dump"; do
                if [ -x "$candidate" ]; then HARNESS=$candidate; break; fi
            done
            if [ -n "$HARNESS" ]; then
                say "harness:  using prebuilt $HARNESS (rebuild failed; see harness-build.log)"
            fi
        fi
    fi
    CPU_DUMP="$WORKDIR/cpu-recon.dump"
    GPU_SOURCE=""
    if [ -z "$GPU_DUMP" ] && [ -f "$AUTO_DUMP" ]; then
        GPU_DUMP="$AUTO_DUMP"
        GPU_SOURCE="CLI dump"
    fi

    if [ -z "$HARNESS" ] || [ ! -x "$HARNESS" ]; then
        say "note:     no still_dump harness; using the decoded-frame compare above"
        RECON_STATUS="fallback (no CPU recon harness)"
    elif ! run_timed "$HARNESS" -i "$INPUT" -o "$CPU_DUMP" $HARNESS_ARGS \
            >"$WORKDIR/harness.log" 2>&1; then
        say "note:     still_dump failed; using the decoded-frame compare above"
        sed 's/^/  /' "$WORKDIR/harness.log" | tail -10
        RECON_STATUS="fallback (CPU recon dump failed)"
    else
        # Optional probe: calls encode_still_gpu directly when the CLI exposes no
        # recon dump. Only its recon is used here; attestation stays tied to
        # the CLI run that produced the byte-gated IVF.
        if [ -z "$GPU_DUMP" ]; then
            if [ -z "$PROBE" ]; then
                for candidate in "$ROOT/tools/still_gpu/still_gpu_probe" \
                                 "$ROOT/build-cuda/still_gpu_probe"; do
                    if [ -x "$candidate" ]; then PROBE=$candidate; break; fi
                done
                if [ -n "$PROBE" ] && [ "$PROBE" -ot "$ENCODER" ] && \
                        [ -x "$SELF/build_probe.sh" ]; then
                    say "probe:    rebuilding stale $PROBE"
                    if ! OUT="$PROBE" "$SELF/build_probe.sh" \
                            >"$WORKDIR/probe-build.log" 2>&1; then
                        say "probe:    rebuild failed; using the existing binary"
                    fi
                fi
            fi
            if [ -n "$PROBE" ] && [ -x "$PROBE" ]; then
                PROBE_STATS="$WORKDIR/gpu-probe-stats.json"
                PROBE_DUMP="$WORKDIR/gpu-probe.dump"
                say "probe:    $PROBE"
                # shellcheck disable=SC2086
                if run_timed "$PROBE" -i "$INPUT" -o "$PROBE_DUMP" \
                        --stats "$PROBE_STATS" --qp "$QP" \
                        $PROBE_ARGS $PROBE_STILL_ARGS >"$WORKDIR/probe.log" 2>&1; then
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
            say "          AV2ENC_STILL_GPU_DUMP, or build still_gpu_probe)"
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
# Verdict.
# ---------------------------------------------------------------------------
say ""
say "-- summary --"
say "frame bytes:  $BYTES_STATUS"
say "attestation:  ${ATTEST_RESULT:-none}"
say "decode:       $DECODE_STATUS"
say "decoded YUV:  $PLANE_STATUS"
if [ "$PLANE_STATUS" = "IDENTICAL (decoder retry)" ]; then
    say "decoder:      first pair differed on identical payloads; retry agreed"
fi
say "recon parity: $RECON_STATUS"
say "determinism:  $DET_STATUS"
say "det. probe:   $DET_PROBE_STATUS"
say "sanitizer:    $SAN_STATUS"
say "racecheck:    $RACE_STATUS"

FAIL_FLAG=0
[ "$BYTES_STATUS" = "DIFFER" ] && FAIL_FLAG=1
[ "$DECODE_STATUS" = "FAIL (cpu)" ] && FAIL_FLAG=1
[ "$DECODE_STATUS" = "FAIL (gpu)" ] && FAIL_FLAG=1
[ "$PLANE_STATUS" = "DIFFER" ] && FAIL_FLAG=1
[ "$RECON_STATUS" = "DIFFER" ] && FAIL_FLAG=1
case "$DET_STATUS" in DIFFER|FAIL*) FAIL_FLAG=1 ;; esac
case "$DET_PROBE_STATUS" in DIFFER*) FAIL_FLAG=1 ;; esac
case "$SAN_STATUS" in FAIL*) FAIL_FLAG=1 ;; esac
case "$RACE_STATUS" in FAIL*) FAIL_FLAG=1 ;; esac
if [ "$FAIL_FLAG" = "1" ]; then
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
