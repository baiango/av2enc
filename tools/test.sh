#!/usr/bin/env bash
# Fast test entry point.
#
#   tools/test.sh                 # unit tests only: ctest -LE gate (~4 s)
#   tools/test.sh --gates         # fast product gates (~20 s; quick subset)
#   tools/test.sh --gates-full    # all product gates (~2 min)
#   tools/test.sh --all           # everything, parallel (~2.5 min)
#   tools/test.sh --unit -V       # extra ctest args are forwarded
#
# The default CTest inventory contains 10 unit tests and 7 product gates.
# The gates encode real streams (including 1080p loop restoration and VBR),
# so they dominate wall time; they are not unit tests and should not be run
# on every edit.
set -euo pipefail

ROOT=$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel)
BUILD="$ROOT/build"
MODE="unit"
JOBS=""
PASSTHRU=()

usage() {
    sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'
}

while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --unit) MODE="unit"; shift ;;
        --gates) MODE="gates"; shift ;;
        --gates-full) MODE="gates-full"; shift ;;
        --all) MODE="all"; shift ;;
        --build) BUILD=$2; shift 2 ;;
        -j) JOBS=$2; shift 2 ;;
        -j*) JOBS=${1#-j}; shift ;;
        *) PASSTHRU+=("$1"); shift ;;
    esac
done

if [ -z "$JOBS" ]; then
    if command -v nproc >/dev/null 2>&1; then
        JOBS=$(nproc)
    else
        JOBS=$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
    fi
    [ "$JOBS" -gt 8 ] && JOBS=8
fi

case "$MODE" in
unit)
    echo "unit tests only (use --gates or --all for product gates)"
    exec ctest --test-dir "$BUILD" -LE '^gate$' ${PASSTHRU[@]+"${PASSTHRU[@]}"}
    ;;
gates)
    echo "fast product gates (quick subset; use --gates-full for everything)"
    exec "$ROOT/tools/run_gates.sh" --bin "$BUILD/av2enc" --no-ctest --quick
    ;;
gates-full)
    if ctest --test-dir "$BUILD" -N -L '^gate$' 2>/dev/null | grep -q 'Test #'; then
        echo "all product gates (parallel -j$JOBS)"
        exec ctest --test-dir "$BUILD" -L '^gate$' -j "$JOBS" ${PASSTHRU[@]+"${PASSTHRU[@]}"}
    fi
    echo "product gates are not registered in this build; running them directly"
    exec "$ROOT/tools/run_gates.sh" --bin "$BUILD/av2enc" --no-ctest
    ;;
all)
    echo "all tests (parallel -j$JOBS)"
    exec ctest --test-dir "$BUILD" -j "$JOBS" ${PASSTHRU[@]+"${PASSTHRU[@]}"}
    ;;
esac
