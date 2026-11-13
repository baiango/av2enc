#!/usr/bin/env bash
# One runner for the av2enc product gates.
#
# Configures and builds nothing: point it at an existing av2enc binary (CPU or
# CUDA build) and it runs the pinned CTest unit tests plus the product gates in
# one pass, with a summary. Every gate uses the same decoder discovery as
# tools/validate_dav2d.sh ($DAV2D, tools/bin/dav2d, tools/bin/dav2d-macos) and
# exits 77 when its prerequisites are missing; those count as skips here.
#
# Usage:
#   tools/run_gates.sh [--bin build-int/av2enc] [--build-dir DIR]
#       [--dav2d PATH] [--quick] [--no-ctest] [--jobs N] [--verbose]
#
#   --bin PATH       av2enc binary (default: first of build-int/build/build-cuda/
#                    build-gpu/refactor/build/cpu)
#   --build-dir DIR  CTest build directory (default: the directory of --bin)
#   --dav2d PATH     pinned decoder (default: $DAV2D, then tools/bin/dav2d[-macos])
#   --quick          skip the 1080p/720p video gates (gate_rc, gate_video_cli)
#                    and the 1080p cases inside gate_tiles/gate_loop_restoration
#   --no-ctest       run only the product gates
#   --jobs N         CTest parallelism (default: CPU count)
#   --verbose        stream each gate's output instead of capturing it
#
# Env: DAV2D, AV2ENC_TESTDATA (override the gitignored testdata/ directory).
# Exit: 0 all selected tests pass or skip, 1 a real failure, 2 usage/setup.
set -uo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
BIN=""
BUILD=""
DAV2D="${DAV2D:-}"
DAV2D_EXPLICIT=0
QUICK=0
RUN_CTEST=1
JOBS=""
VERBOSE=0

usage() { sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; }

die() { echo "run_gates: error: $*" >&2; exit 2; }

while [ $# -gt 0 ]; do
    case "$1" in
        --bin) BIN=${2:-}; shift 2 ;;
        --build-dir) BUILD=${2:-}; shift 2 ;;
        --dav2d) DAV2D=${2:-}; DAV2D_EXPLICIT=1; shift 2 ;;
        --quick) QUICK=1; shift ;;
        --no-ctest) RUN_CTEST=0; shift ;;
        --jobs) JOBS=${2:-}; shift 2 ;;
        --verbose) VERBOSE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument: $1 (try --help)" ;;
    esac
done

if [ -z "$BIN" ]; then
    for candidate in "$ROOT/build-int/av2enc" "$ROOT/build/av2enc" \
                     "$ROOT/build-cuda/av2enc" "$ROOT/build-gpu/av2enc" \
                     "$ROOT/refactor/build/cpu/av2enc"; do
        if [ -x "$candidate" ]; then BIN=$candidate; break; fi
    done
fi
[ -n "$BIN" ] && [ -x "$BIN" ] || die "no av2enc binary; pass --bin PATH"
BIN=$(cd -- "$(dirname -- "$BIN")" && pwd)/$(basename -- "$BIN")

if [ -z "$BUILD" ]; then BUILD=$(dirname -- "$BIN"); fi
[ -d "$BUILD" ] || die "build directory does not exist: $BUILD"

if [ -n "$DAV2D" ]; then
    if [ ! -x "$DAV2D" ]; then
        if [ "$DAV2D_EXPLICIT" -eq 1 ]; then
            die "--dav2d is not executable: $DAV2D"
        fi
        echo "run_gates: warning: DAV2D=$DAV2D is not executable; using discovery" >&2
        DAV2D=""
    fi
fi
if [ -z "$DAV2D" ]; then
    for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
        if [ -x "$candidate" ]; then DAV2D=$candidate; break; fi
    done
fi
[ -n "$DAV2D" ] && export DAV2D

if [ -z "$JOBS" ]; then
    if command -v nproc >/dev/null 2>&1; then JOBS=$(nproc)
    elif command -v sysctl >/dev/null 2>&1; then JOBS=$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
    else JOBS=4; fi
fi

TMP=$(mktemp -d "${TMPDIR:-/tmp}/av2enc-gates.XXXXXX")
cleanup() {
    if [ "$VERBOSE" = "1" ] || [ -n "${KEEP:-}" ]; then
        echo "run_gates: logs kept in $TMP"
    else
        rm -rf "$TMP"
    fi
}
trap cleanup EXIT

STILL_RECON_DUMP="$BUILD/still_recon_dump"
GATE_STILL_DUMP="$BUILD/still_dump"

echo "== av2enc gate runner =="
echo "  binary:  $BIN"
echo "  build:   $BUILD"
echo "  decoder: ${DAV2D:-none (decoder gates will skip)}"
[ "$QUICK" -eq 1 ] && echo "  mode:    quick (1080p/720p video skipped)"

CTEST_RC=0
CTEST_NOTE="skipped"
if [ "$RUN_CTEST" -eq 0 ]; then
    CTEST_NOTE="disabled (--no-ctest)"
elif [ ! -f "$BUILD/CTestTestfile.cmake" ]; then
    CTEST_NOTE="skipped (build was not configured with CTest)"
    echo "run_gates: warning: no $BUILD/CTestTestfile.cmake" >&2
else
    echo
    echo "== ctest (unit tests; gate_* run below) =="
    CTEST_LOG="$TMP/ctest.log"
    if [ "$VERBOSE" -eq 1 ]; then
        ctest --test-dir "$BUILD" --output-on-failure -E '^gate_' -j "$JOBS" \
            2>&1 | tee "$CTEST_LOG"
        CTEST_RC=${PIPESTATUS[0]}
    else
        ctest --test-dir "$BUILD" --output-on-failure -E '^gate_' -j "$JOBS" \
            >"$CTEST_LOG" 2>&1
        CTEST_RC=$?
        tail -n 4 "$CTEST_LOG"
    fi
    if [ "$CTEST_RC" -eq 0 ]; then
        CTEST_NOTE="passed"
    else
        CTEST_NOTE="FAILED (rc=$CTEST_RC)"
        [ "$VERBOSE" -eq 1 ] || cat "$CTEST_LOG" >&2
    fi
fi

echo
echo "== product gates =="
PASSED=0
SKIPPED=0
FAILED=0
INDEX=0
TOTAL=17
[ "$QUICK" -eq 1 ] && TOTAL=6
declare -a RESULTS=()

now() { date +%s; }

record_result() {
    RESULTS+=("$1|$2|$3|$4")
}

skip_gate() {
    local name=$1 reason=$2
    INDEX=$((INDEX + 1))
    printf '[%d/%d] %-22s SKIP  %s\n' "$INDEX" "$TOTAL" "$name" "$reason"
    record_result "$name" SKIP "0.00" "$reason"
    SKIPPED=$((SKIPPED + 1))
}

# run_gate <name> <script> [gate args...]
run_gate() {
    local name=$1 script=$2
    shift 2
    INDEX=$((INDEX + 1))
    local label
    label=$(printf '[%d/%d] %-22s' "$INDEX" "$TOTAL" "$name")
    local args=(--bin "$BIN")
    [ -n "$DAV2D" ] && args+=(--dav2d "$DAV2D")
    args+=("$@")

    local log="$TMP/$name.log"
    local t0 t1 secs rc
    t0=$(now)
    if [ "$VERBOSE" -eq 1 ]; then
        bash "$ROOT/$script" "${args[@]}" 2>&1 | tee "$log"
        rc=${PIPESTATUS[0]}
    else
        bash "$ROOT/$script" "${args[@]}" >"$log" 2>&1
        rc=$?
    fi
    t1=$(now)
    secs=$((t1 - t0))

    if [ "$rc" -eq 0 ]; then
        printf '%s PASS  %ss\n' "$label" "$secs"
        record_result "$name" PASS "$secs.00" ""
        PASSED=$((PASSED + 1))
    elif [ "$rc" -eq 77 ]; then
        printf '%s SKIP  %ss (prerequisites missing)\n' "$label" "$secs"
        record_result "$name" SKIP "$secs.00" "prerequisites missing"
        SKIPPED=$((SKIPPED + 1))
    else
        printf '%s FAIL  %ss (rc=%s)\n' "$label" "$secs" "$rc"
        local data
        data=$(grep -E '^(FAIL|SKIP|error)' "$log" | tail -n 5)
        [ -n "$data" ] && printf '%s\n' "$data" | sed 's/^/    /'
        record_result "$name" FAIL "$secs.00" "rc=$rc"
        FAILED=$((FAILED + 1))
    fi
}

run_gate gate_format src/formats/format_gate.sh
if [ -x "$STILL_RECON_DUMP" ]; then
    run_gate gate_intra tools/still_verify/intra_gate.sh --dump "$STILL_RECON_DUMP"
else
    skip_gate gate_intra "still_recon_dump is not built"
fi

LOOP_ARGS=()
[ -x "$GATE_STILL_DUMP" ] && LOOP_ARGS+=(--dump "$GATE_STILL_DUMP")
[ "$QUICK" -eq 1 ] && LOOP_ARGS+=(--quick)
run_gate gate_loop_restoration tools/still_verify/loop_restoration_gate.sh \
    ${LOOP_ARGS[@]+"${LOOP_ARGS[@]}"}

TILES_ARGS=()
[ "$QUICK" -eq 1 ] && TILES_ARGS+=(--quick)
run_gate gate_tiles src/pipeline/tiles/stream/gate_cli_tiles.sh \
    ${TILES_ARGS[@]+"${TILES_ARGS[@]}"}
run_gate gate_host_plan src/pipeline/dispatch/gate_host_plan.sh
run_gate gate_10bit_filters src/formats/high_bitdepth/gate_10bit_filters.sh
if [ "$QUICK" -eq 0 ]; then
    run_gate gate_rc src/pipeline/video/post/gate_cli_rc.sh
    run_gate gate_video_cli src/pipeline/video/video_cli_gate.sh --frames 4
    run_gate gate_wrapper_cdf src/pipeline/video/wrapper_cdf_gate.sh
    run_gate gate_video_inter_residual src/pipeline/video/video_cli_gate.sh \
        --frames 4 --illumination-shift --expect-inter-residual \
        -- --video-inter-residual
    run_gate gate_video_tcq src/pipeline/video/video_tcq_gate.sh
    run_gate gate_video_cctx src/pipeline/video/video_cctx_gate.sh
    run_gate gate_implicit_ref_map src/pipeline/video/gate_implicit_ref_map.sh
    run_gate gate_video_near_near src/pipeline/video/gate_video_near_near.sh
    run_gate gate_video_tip src/pipeline/video/video_tip_gate.sh
    run_gate gate_video_drl_reorder src/pipeline/video/gate_video_drl_reorder.sh
    run_gate gate_video_opfl_refine src/pipeline/video/gate_video_opfl_refine.sh
fi

echo
echo "== summary =="
printf '  %-22s %-6s %6s  %s\n' gate result secs note
for row in "${RESULTS[@]}"; do
    IFS='|' read -r name result secs note <<<"$row"
    printf '  %-22s %-6s %6s  %s\n' "$name" "$result" "$secs" "$note"
done
printf '  %-22s %-6s\n' ctest "$CTEST_NOTE"
echo "  gates: $PASSED passed, $SKIPPED skipped, $FAILED failed"

if [ "$FAILED" -ne 0 ] || [ "$CTEST_RC" -ne 0 ]; then
    echo "run_gates: FAILED" >&2
    exit 1
fi
echo "run_gates: OK"
exit 0
