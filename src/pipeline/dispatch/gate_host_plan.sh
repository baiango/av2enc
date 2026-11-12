#!/usr/bin/env bash
# Gate for the host-only multi-GPU scheduling contract (--gpu-plan).
#
# The planner has no CUDA dependency; this gate checks that the CLI prints the
# documented deterministic assignments (docs/multi-gpu.md, dispatch README) and
# that the dry run never writes a stream or allocates a device.
#
# Usage: src/pipeline/dispatch/gate_host_plan.sh --bin build-int/av2enc
# Exit codes: 0 pass, 1 failure, 2 usage, 77 missing fixture.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${HERE}/../../.." && pwd)"
BIN=""
DAV2D="${DAV2D:-}"
TESTDATA="${AV2ENC_TESTDATA:-$ROOT/testdata}"

while [ $# -gt 0 ]; do
    case "$1" in
        --bin) BIN=$2; shift 2 ;;
        # Accepted for uniform gate invocation; the planner never decodes.
        --dav2d) DAV2D=$2; shift 2 ;;
        -h|--help)
            sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "gate_host_plan: unknown argument $1" >&2; exit 2 ;;
    esac
done
if [ -z "$BIN" ] || [ ! -x "$BIN" ]; then
    echo "gate_host_plan: --bin must point at an av2enc executable" >&2
    exit 2
fi

F1080="$TESTDATA/shine_1920x1080_10f.y4m"
F360="$ROOT/tests/data/basic_640x360_1f.y4m"
for f in "$F1080" "$F360"; do
    if [ ! -f "$f" ]; then
        echo "SKIP: missing $f" >&2
        exit 77
    fi
done

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
failures=0
pass=0
fail() { echo "FAIL: $*" >&2; failures=$((failures + 1)); }
ok() { echo "PASS $*"; pass=$((pass + 1)); }

# expect_line <input> <label> <substring> <args...>
expect_line() {
    local input=$1 label=$2 needle=$3
    shift 3
    if ! "$BIN" "$input" "$@" >"$TMP/out" 2>"$TMP/err"; then
        cat "$TMP/err" >&2
        fail "$label: command failed"
        return 0
    fi
    if ! grep -qF -e "$needle" "$TMP/out"; then
        cat "$TMP/out" >&2
        fail "$label: output does not contain '$needle'"
        return 0
    fi
    ok "$label"
}

expect_line "$F1080" "1 GPU / 4x2 layout" "layout:  4x2 (8 tiles)" --gpu-plan 1
expect_line "$F1080" "1 GPU owns all 8 tiles" \
    "gpu 0:  tiles 0 1 2 3 4 5 6 7 (8," --gpu-plan 1
expect_line "$F1080" "2 GPU / 8x2 layout" "layout:  8x2 (16 tiles)" --gpu-plan 2
expect_line "$F1080" "2 GPU even set" \
    "gpu 0:  tiles 0 2 4 6 8 10 12 14 (8," --gpu-plan 2
expect_line "$F1080" "2 GPU odd set" \
    "gpu 1:  tiles 1 3 5 7 9 11 13 15 (8," --gpu-plan 2
expect_line "$F1080" "4 GPU / 16x2 layout" "layout:  16x2 (32 tiles)" --gpu-plan 4
expect_line "$F1080" "4 GPU set 0" \
    "gpu 0:  tiles 4 8 12 14 16 20 24 28 (8," --gpu-plan 4
expect_line "$F1080" "4 GPU set 1" \
    "gpu 1:  tiles 5 9 13 15 17 21 25 29 (8," --gpu-plan 4
expect_line "$F1080" "4 GPU set 2" \
    "gpu 2:  tiles 0 2 6 10 18 22 26 30 (8," --gpu-plan 4
expect_line "$F1080" "4 GPU set 3" \
    "gpu 3:  tiles 1 3 7 11 19 23 27 31 (8," --gpu-plan 4
expect_line "$F360" "640x360 2 GPU / 8x2" "layout:  8x2 (16 tiles)" --gpu-plan 2
expect_line "$F360" "explicit --tiles override" "layout:  2x2 (4 tiles)" \
    --gpu-plan 2 --tiles 2x2
expect_line "$F1080" "no-measurement disclaimer" "measured=0" --gpu-plan 2
expect_line "$F1080" "evidence flag" "evidence=estimated" --gpu-plan 2
expect_line "$F1080" "per-tile dispatch (1 GPU)" \
    "dispatch: 0:0 1:0 2:0 3:0 4:0 5:0 6:0 7:0 (raster merge order" \
    --gpu-plan 1
expect_line "$F1080" "per-tile dispatch (2 GPU)" \
    "dispatch: 0:0 1:1 2:0 3:1 4:0 5:1 6:0 7:1 8:0 9:1 10:0 11:1 12:0 13:1 14:0 15:1 (raster merge order" \
    --gpu-plan 2
expect_line "$F1080" "no device allocation note" "no device allocation, no stream" \
    --gpu-plan 2

# A dry run must not create the output file even when -o is supplied.
"$BIN" "$F1080" -o "$TMP/dry.ivf" --gpu-plan 2 >/dev/null
if [ -e "$TMP/dry.ivf" ]; then
    fail "dry run wrote an output stream"
else
    ok "dry run writes no output stream"
fi

# Rejections.
expect_reject() {
    local needle=$1 label=$2
    shift 2
    if "$BIN" "$F360" "$@" >/dev/null 2>"$TMP/err"; then
        fail "$label: expected a rejection"
        return 0
    fi
    if ! grep -qF -e "$needle" "$TMP/err"; then
        cat "$TMP/err" >&2
        fail "$label: error does not mention '$needle'"
        return 0
    fi
    ok "$label (rejected by name)"
}
expect_reject "not representable" "640x360 3 GPU / 16x2" --gpu-plan 3
expect_reject "host-planning dry run" "gpu plan is not a stream mode" \
    --gpu-plan 2 --video
expect_reject "--gpu-plan must be" "gpu count range" --gpu-plan 9

echo
if [ "$failures" -ne 0 ]; then
    echo "gate_host_plan: $failures failure(s), $pass passed" >&2
    exit 1
fi
echo "gate_host_plan: $pass/$pass cases passed"
