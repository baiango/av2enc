#!/usr/bin/env bash
# M1 feature gate: pinned IVF MD5s + dav2d --strict 1 decode + optional
# encoder-recon exactness.
#
# Every case is opt-in on the CLI (no feature is default-on), so the pinned
# `--m1` baseline must never move.  The reference frames are the two repository
# fixtures; the MD5s below were pinned from a CPU build on 2026-11-11.
#
# Usage:
#   tools/m1_verify/feature_gate.sh --bin build-m1/av2enc
#   tools/m1_verify/feature_gate.sh --bin build-cuda/av2enc --dump build-cuda/m1_dump
#
# Env: DAV2D overrides the decoder path.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
BIN=""
DUMP=""
DAV2D="${DAV2D:-}"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

while [ $# -gt 0 ]; do
    case "$1" in
        --bin) BIN=$2; shift 2 ;;
        --dump) DUMP=$2; shift 2 ;;
        --dav2d) DAV2D=$2; shift 2 ;;
        -h|--help)
            sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "feature_gate: unknown argument $1" >&2; exit 2 ;;
    esac
done

if [ -z "$BIN" ] || [ ! -x "$BIN" ]; then
    echo "feature_gate: --bin must point at an av2enc executable" >&2
    exit 2
fi
if [ -z "$DAV2D" ]; then
    for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
        if [ -x "$candidate" ]; then DAV2D=$candidate; break; fi
    done
fi
if [ -z "$DAV2D" ] || [ ! -x "$DAV2D" ]; then
    echo "feature_gate: no dav2d binary (set DAV2D=...)" >&2
    exit 77
fi

failures=0
pass=0

# run_case <fixture> <name> <expected_md5> <encode args...>
run_case() {
    local fixture=$1 name=$2 expected=$3
    shift 3
    local input="$ROOT/$fixture"
    local out="$TMP/${name}.ivf"
    local dec="$TMP/${name}.y4m"
    if [ ! -f "$input" ]; then
        echo "SKIP $name: missing $fixture" >&2
        return 0
    fi
    if ! "$BIN" "$input" -o "$out" --frames 1 "$@" >"$TMP/${name}.log" 2>&1; then
        echo "FAIL $name: encode failed (see below)" >&2
        cat "$TMP/${name}.log" >&2
        failures=$((failures + 1))
        return 0
    fi
    local got
    got=$(md5 -q "$out" 2>/dev/null || md5sum "$out" | awk '{print $1}')
    if [ "$got" != "$expected" ]; then
        echo "FAIL $name: IVF md5 $got, expected $expected" >&2
        failures=$((failures + 1))
        return 0
    fi
    if ! "$DAV2D" -q -i "$out" -o "$dec" --threads 1 --strict 1 \
            >"$TMP/${name}.dec.log" 2>&1; then
        echo "FAIL $name: dav2d --strict 1 rejected the stream" >&2
        cat "$TMP/${name}.dec.log" >&2
        failures=$((failures + 1))
        return 0
    fi
    if [ -n "$DUMP" ]; then
        # Re-encode through the harness with the same options.  The caller maps
        # CLI flags to harness flags in DUMP_ARGS.
        if ! "$DUMP" -i "$input" -o "$TMP/${name}.dump" \
                ${DUMP_ARGS[@]+"${DUMP_ARGS[@]}"} \
                >"$TMP/${name}.dump.log" 2>&1; then
            echo "FAIL $name: m1_dump failed" >&2
            cat "$TMP/${name}.dump.log" >&2
            failures=$((failures + 1))
            return 0
        fi
        if ! python3 "$ROOT/tools/m1_verify/verify.py" \
                --ivf "$out" --dump "$TMP/${name}.dump" --decoder "$DAV2D" \
                --tsv >"$TMP/${name}.verify.log" 2>&1; then
            echo "FAIL $name: recon/decoder mismatch" >&2
            cat "$TMP/${name}.verify.log" >&2
            failures=$((failures + 1))
            return 0
        fi
    fi
    echo "PASS $name ($(wc -c <"$out" | tr -d ' ') B, md5 $expected)"
    pass=$((pass + 1))
}

DUMP_ARGS=()

# --- 64x64 -----------------------------------------------------------------
DUMP_ARGS=(); run_case tests/data/m0_64x64_1f.y4m m1_64_base \
    0ef9825062a25c4d950ee924113c6f89 --m1
DUMP_ARGS=(--cdef --cdef-strengths 1); run_case tests/data/m0_64x64_1f.y4m m1_64_cdef1 \
    b795a6f998c7037c95b926d6afdfc89e --m1 --m1-cdef --m1-cdef-strengths 1
DUMP_ARGS=(--cdef --cdef-strengths 4); run_case tests/data/m0_64x64_1f.y4m m1_64_cdef4 \
    b795a6f998c7037c95b926d6afdfc89e --m1 --m1-cdef --m1-cdef-strengths 4
DUMP_ARGS=(--rdoq); run_case tests/data/m0_64x64_1f.y4m m1_64_rdoq \
    a7ee3a0ac75781ce2a2c7390c3503019 --m1 --m1-rdoq
DUMP_ARGS=(--tx-type); run_case tests/data/m0_64x64_1f.y4m m1_64_txtype \
    15681508a144eeea88bb8223185d75e9 --m1 --m1-tx-type
DUMP_ARGS=(--full-intra --rdo-intra); run_case tests/data/m0_64x64_1f.y4m m1_64_rdointra \
    45b4a209a8c3e889c91dfa201f26d2d4 --m1 --m1-full-intra --m1-rdo-intra
DUMP_ARGS=(--uv-full); run_case tests/data/m0_64x64_1f.y4m m1_64_uvfull \
    2ed7f1ca8d146be8345ed95fee65f9bf --m1 --m1-uv-full
DUMP_ARGS=(--cdef --rdoq --tx-type); run_case tests/data/m0_64x64_1f.y4m m1_64_combo \
    aa885652b8c57ecc25b14b87c97862d8 --m1 --m1-cdef --m1-rdoq --m1-tx-type
DUMP_ARGS=(--ccso); run_case tests/data/m0_64x64_1f.y4m m1_64_ccso \
    980e5c2194091faf2347e3b05dfdea69 --m1 --m1-ccso
DUMP_ARGS=(--cdef --ccso); run_case tests/data/m0_64x64_1f.y4m m1_64_ccso_cdef \
    a90c525e480e8ff52db3f47f77fff887 --m1 --m1-cdef --m1-ccso

# --- 640x360 ---------------------------------------------------------------
DUMP_ARGS=(); run_case tests/data/m0_640x360_1f.y4m m1_360_base \
    e61de628b0812b716b0efa7dcafcdd06 --m1
DUMP_ARGS=(--cdef --cdef-strengths 1); run_case tests/data/m0_640x360_1f.y4m m1_360_cdef1 \
    dec10527a779e7664ea78e53def9dbf1 --m1 --m1-cdef --m1-cdef-strengths 1
DUMP_ARGS=(--cdef --cdef-strengths 4); run_case tests/data/m0_640x360_1f.y4m m1_360_cdef4 \
    b37bed7eb785d0c6b2e07219e888ff89 --m1 --m1-cdef --m1-cdef-strengths 4
DUMP_ARGS=(--rdoq); run_case tests/data/m0_640x360_1f.y4m m1_360_rdoq \
    ac668d497012333b6aeecb82c2e70337 --m1 --m1-rdoq
DUMP_ARGS=(--tx-type); run_case tests/data/m0_640x360_1f.y4m m1_360_txtype \
    21c39aec98507dba7c24e5d320e1bb3c --m1 --m1-tx-type
DUMP_ARGS=(--full-intra --rdo-intra); run_case tests/data/m0_640x360_1f.y4m m1_360_rdointra \
    211da270c5baf34f7bd40da019ffd84b --m1 --m1-full-intra --m1-rdo-intra
DUMP_ARGS=(--uv-full); run_case tests/data/m0_640x360_1f.y4m m1_360_uvfull \
    400dd3ad02db9e0e3e366091a7e6962f --m1 --m1-uv-full
DUMP_ARGS=(--cdef --rdoq --tx-type); run_case tests/data/m0_640x360_1f.y4m m1_360_combo \
    84ab3934c08e6c90b876d067bcbb3606 --m1 --m1-cdef --m1-rdoq --m1-tx-type
DUMP_ARGS=(--ccso); run_case tests/data/m0_640x360_1f.y4m m1_360_ccso \
    54e853e24b21bf1237b8b51a846b1c35 --m1 --m1-ccso
DUMP_ARGS=(--cdef --ccso); run_case tests/data/m0_640x360_1f.y4m m1_360_ccso_cdef \
    a272550a72c387a8a4f8cc8d3186f8da --m1 --m1-cdef --m1-ccso
DUMP_ARGS=(--ccso --ccso-bands 6); run_case tests/data/m0_640x360_1f.y4m m1_360_ccso6 \
    54e853e24b21bf1237b8b51a846b1c35 --m1 --m1-ccso --m1-ccso-bands 6
DUMP_ARGS=(--ccso --rdoq); run_case tests/data/m0_640x360_1f.y4m m1_360_ccso_rdoq \
    7fd7a2a0e078665614f8b6d2c5fe1b57 --m1 --m1-ccso --m1-rdoq

echo "feature_gate: $pass passed, $failures failed"
[ "$failures" -eq 0 ]
