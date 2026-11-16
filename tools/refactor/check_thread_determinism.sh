#!/usr/bin/env bash
# Thread-determinism guard for the persistent worker pool.
#
#   tools/refactor/check_thread_determinism.sh [--bin PATH] [--fixtures DIR]
#
# Encodes each measured operating point twice -- `--cpu-threads 1` and
# `--cpu-threads 8` -- and requires the two IVFs to be `cmp`-identical.  The
# pool changes which thread evaluates a given RDO candidate; it must not change
# a single coded byte, so a per-index slot merge that silently depends on timing
# shows up here and nowhere else.
#
# The concurrent sites are the per-block loops in
# `src/pipeline/video/video_tx_search.cpp` (TX-type and TX-partition
# candidates) and `src/pipeline/video/video_intra_search.h` (61 intra modes).
# Both are selected with the historical comparator in candidate order, so a tie
# keeps the first candidate exactly as the serial loop did.
#
# The 720p point is also compared against its published pin (36,572 B,
# c4ba2813...): a determinism script that accepts any two matching streams
# would still pass if both were wrong.
#
# Not registered as a gate yet; the integrator reviews and registers it.
# Exit: 0 identical, 1 mismatch or encode failure, 2 usage/missing input.
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BIN=""
FIXTURES="$ROOT/testdata"
THREADS=8

usage() { sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; }
fail() { echo "check_thread_determinism: FAIL: $*" >&2; exit 1; }

while [ "$#" -gt 0 ]; do
    case "$1" in
        --bin) BIN=${2:-}; shift 2 ;;
        --fixtures) FIXTURES=${2:-}; shift 2 ;;
        # Accepted for tools/run_gates.sh compatibility; this check never decodes.
        --dav2d) shift 2 ;;
        --threads) THREADS=${2:-}; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "check_thread_determinism: unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [ -z "$BIN" ]; then
    for candidate in "$ROOT/build/av2enc" "$ROOT/build-t/av2enc"; do
        if [ -x "$candidate" ]; then BIN=$candidate; break; fi
    done
fi
[ -n "$BIN" ] && [ -x "$BIN" ] || {
    echo "check_thread_determinism: no av2enc binary; pass --bin PATH" >&2
    exit 2
}
[ "$THREADS" -ge 2 ] 2>/dev/null ||
    { echo "check_thread_determinism: --threads must be >= 2" >&2; exit 2; }

TMP=$(mktemp -d "${TMPDIR:-/tmp}/av2enc-threads.XXXXXX")
trap 'rm -rf "$TMP"' EXIT

md5_of() {
    if command -v md5 >/dev/null 2>&1; then md5 -q "$1";
    else md5sum "$1" | awk '{print $1}'; fi
}

# name | fixture | encode args | expected bytes (or -) | expected md5 (or -)
CASES=(
    "240p_qp106|nasa_480x240_30fps_60f.y4m|--video --frames 60 --qp 106 --nightly|-|-"
    "720p_qp217|space8k_1440x720_60f.y4m|--video --frames 60 --qp 217 --nightly --video-key-qp-offset -57 --video-leaf-rd-lambda 250 --video-ext-partitions|36572|c4ba28138d8f6569d5e08721a66d755c"
)

FAILURES=0
for spec in "${CASES[@]}"; do
    IFS='|' read -r name fixture args want_bytes want_md5 <<< "$spec"
    input="$FIXTURES/$fixture"
    [ -f "$input" ] || {
        # Gates exit 77 when prerequisites are absent so the suite can run on a
        # tree without the gitignored testdata.
        echo "check_thread_determinism: missing fixture $input (skipping)" >&2
        exit 77
    }
    read -r -a ENCODE_ARGS <<< "$args"

    outputs=()
    for threads in 1 "$THREADS"; do
        out="$TMP/$name-$threads.ivf"
        log="$TMP/$name-$threads.log"
        start=$(python3 -c 'import time; print(f"{time.time():.3f}")')
        if ! "$BIN" "$input" -o "$out" "${ENCODE_ARGS[@]}" \
                --cpu-threads "$threads" >"$log" 2>&1; then
            tail -n 15 "$log" >&2
            fail "$name: encode at --cpu-threads $threads failed"
        fi
        end=$(python3 -c 'import time; print(f"{time.time():.3f}")')
        seconds=$(python3 -c "print(f'{float($end) - float($start):.2f}')")
        bytes=$(wc -c <"$out" | tr -d ' ')
        echo "  $name t=$threads ${seconds}s ${bytes} B $(md5_of "$out")"
        outputs+=("$out")
        if [ "$want_bytes" != "-" ] && [ "$bytes" != "$want_bytes" ]; then
            echo "check_thread_determinism: FAIL: $name at $threads thread(s) is" \
                 "$bytes B, pinned $want_bytes" >&2
            FAILURES=$((FAILURES + 1))
        fi
        if [ "$want_md5" != "-" ]; then
            got=$(md5_of "$out")
            if [ "$got" != "$want_md5" ]; then
                echo "check_thread_determinism: FAIL: $name at $threads thread(s)" \
                     "md5 is $got, pinned $want_md5" >&2
                FAILURES=$((FAILURES + 1))
            fi
        fi
    done

    if ! cmp -s "${outputs[0]}" "${outputs[1]}"; then
        echo "check_thread_determinism: FAIL: $name differs between" \
             "--cpu-threads 1 and $THREADS" >&2
        cmp "${outputs[0]}" "${outputs[1]}" >&2 || true
        FAILURES=$((FAILURES + 1))
    else
        echo "  $name: --cpu-threads 1 == $THREADS (bit-identical)"
    fi
done

if [ "$FAILURES" -ne 0 ]; then
    echo "check_thread_determinism: FAILED ($FAILURES check(s))" >&2
    exit 1
fi
echo "check_thread_determinism: OK (thread count does not change the bitstream)"
