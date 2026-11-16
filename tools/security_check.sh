#!/usr/bin/env bash
# On-demand security checks (no CI required).
#
#   1. ASan + UBSan build: unit tests, then a real 240p encode.
#   2. TSan build: a trace-enabled 8-thread encode, which is the schedule that
#      previously produced a data race in --video-trace output.
#   3. Short fuzzer run (delegated to tools/fuzz_smoke.sh; skipped when no
#      libFuzzer-capable clang is available).
#
# Usage: tools/security_check.sh [--fuzz-seconds N] [--keep]
# Exit: 0 clean, 1 a sanitizer/fuzzer finding, 2 setup problem.
set -uo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
JOBS=$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)
FUZZ_SECONDS=60
KEEP=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --fuzz-seconds) FUZZ_SECONDS=${2:-60}; shift 2 ;;
        --keep) KEEP=1; shift ;;
        -h|--help) sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "security_check: unknown argument: $1" >&2; exit 2 ;;
    esac
done

ASAN_BUILD=${ASAN_BUILD:-$ROOT/build-asan}
TSAN_BUILD=${TSAN_BUILD:-$ROOT/build-tsan}
NASA="$ROOT/testdata/nasa_480x240_30fps_60f.y4m"

export UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1
export ASAN_OPTIONS=detect_leaks=0:abort_on_error=1
export TSAN_OPTIONS=halt_on_error=1

fail=0
note() { printf '\n== %s ==\n' "$*"; }

configure_build() { # <dir> <sanitizer>
    local dir=$1 san=$2
    cmake -S "$ROOT" -B "$dir" -G Ninja \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DAV2ENC_ENABLE_CUDA=OFF -DAV2ENC_ALLOW_CPU_ONLY=ON \
        -DCMAKE_CXX_FLAGS="-O1 -g -fno-omit-frame-pointer -fsanitize=$san" \
        -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=$san" \
        >"$dir/cfg.log" 2>&1 && cmake --build "$dir" -j"$JOBS" >"$dir/build.log" 2>&1
}

note "ASan+UBSan: build"
if configure_build "$ASAN_BUILD" "address,undefined"; then
    note "ASan+UBSan: unit tests"
    if ! ctest --test-dir "$ASAN_BUILD" -E '^gate_' --output-on-failure -j"$JOBS"; then
        echo "FAIL: unit tests under ASan+UBSan" >&2; fail=1
    fi
    note "ASan+UBSan: 240p encode"
    if [ -f "$NASA" ]; then
        if "$ASAN_BUILD/av2enc" "$NASA" -o "$ASAN_BUILD/smoke.ivf" \
            --video --frames 8 --cpu-threads 8 --qp 106 --nightly \
            >"$ASAN_BUILD/smoke.log" 2>&1; then
            echo "240p encode: ok ($(wc -c <"$ASAN_BUILD/smoke.ivf" | tr -d ' ') bytes)"
        else
            echo "FAIL: 240p encode under ASan+UBSan" >&2
            tail -10 "$ASAN_BUILD/smoke.log" >&2; fail=1
        fi
    else
        echo "skip: $NASA missing"
    fi
else
    echo "FAIL: ASan+UBSan build" >&2
    tail -10 "$ASAN_BUILD/build.log" >&2; fail=1
fi

note "TSan: build"
if configure_build "$TSAN_BUILD" "thread"; then
    note "TSan: trace-enabled 8-thread encode"
    if [ -f "$NASA" ]; then
        if "$TSAN_BUILD/av2enc" "$NASA" -o "$TSAN_BUILD/smoke.ivf" \
            --video --frames 8 --cpu-threads 8 --qp 106 --nightly \
            --video-trace >"$TSAN_BUILD/smoke.log" 2>"$TSAN_BUILD/smoke.trace"; then
            echo "trace encode: ok ($(wc -l <"$TSAN_BUILD/smoke.trace" | tr -d ' ') trace lines)"
        else
            echo "FAIL: trace encode under TSan (see $TSAN_BUILD/smoke.trace)" >&2
            fail=1
        fi
        if grep -q 'WARNING: ThreadSanitizer' "$TSAN_BUILD/smoke.trace"; then
            echo "FAIL: ThreadSanitizer reported a race" >&2
            grep -A6 'WARNING: ThreadSanitizer' "$TSAN_BUILD/smoke.trace" | head -20 >&2
            fail=1
        fi
    else
        echo "skip: $NASA missing"
    fi
else
    echo "FAIL: TSan build" >&2
    tail -10 "$TSAN_BUILD/build.log" >&2; fail=1
fi

note "fuzzer: ${FUZZ_SECONDS}s"
if ! "$ROOT/tools/fuzz_smoke.sh" "$FUZZ_SECONDS" >"$ROOT/.fuzz_smoke.log" 2>&1; then
    rc=$?
    if [ "$rc" -eq 2 ]; then
        echo "skip: no libFuzzer-capable clang (see .fuzz_smoke.log)"
    else
        echo "FAIL: fuzzer found a crash (see .fuzz_smoke.log)" >&2
        tail -20 "$ROOT/.fuzz_smoke.log" >&2; fail=1
    fi
else
    tail -3 "$ROOT/.fuzz_smoke.log"
fi

if [ "$KEEP" -eq 0 ]; then
    rm -f "$ROOT/.fuzz_smoke.log"
fi

note "result"
if [ "$fail" -ne 0 ]; then
    echo "security_check: FAILED" >&2
    exit 1
fi
echo "security_check: OK"
