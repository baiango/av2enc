#!/usr/bin/env bash
# M1 feature gate: pinned IVF MD5s + dav2d --strict 1 decode + optional
# encoder-recon exactness.
#
# Every case is opt-in on the CLI (no feature is default-on), so the pinned
# `--still` baseline must never move.  The reference frames are the two repository
# fixtures; the MD5s below were pinned from a CPU build on 2026-11-11.
#
# Usage:
#   tools/still_verify/feature_gate.sh --bin build-m1/av2enc
#   tools/still_verify/feature_gate.sh --bin build-cuda/av2enc --dump build-cuda/still_dump
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
# RUN_QP selects a base_q_idx for both the encoder and the recon harness; unset
# means the pinned QP 90.
run_case() {
    local fixture=$1 name=$2 expected=$3
    shift 3
    local input="$ROOT/$fixture"
    local out="$TMP/${name}.ivf"
    local dec="$TMP/${name}.y4m"
    local qp_args=()
    if [ -n "${RUN_QP:-}" ]; then
        qp_args=(--qp "$RUN_QP")
    fi
    if [ ! -f "$input" ]; then
        echo "SKIP $name: missing $fixture" >&2
        return 0
    fi
    if ! "$BIN" "$input" -o "$out" --frames 1 \
            ${qp_args[@]+"${qp_args[@]}"} "$@" >"$TMP/${name}.log" 2>&1; then
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
                ${qp_args[@]+"${qp_args[@]}"} \
                ${DUMP_ARGS[@]+"${DUMP_ARGS[@]}"} \
                >"$TMP/${name}.dump.log" 2>&1; then
            echo "FAIL $name: still_dump failed" >&2
            cat "$TMP/${name}.dump.log" >&2
            failures=$((failures + 1))
            return 0
        fi
        if ! python3 "$ROOT/tools/still_verify/verify.py" \
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
DUMP_ARGS=(); run_case tests/data/basic_64x64_1f.y4m still_64_base \
    0ef9825062a25c4d950ee924113c6f89 --still
DUMP_ARGS=(--cdef --cdef-strengths 1); run_case tests/data/basic_64x64_1f.y4m still_64_cdef1 \
    b795a6f998c7037c95b926d6afdfc89e --still --still-cdef --still-cdef-strengths 1
DUMP_ARGS=(--cdef --cdef-strengths 4); run_case tests/data/basic_64x64_1f.y4m still_64_cdef4 \
    b795a6f998c7037c95b926d6afdfc89e --still --still-cdef --still-cdef-strengths 4
DUMP_ARGS=(--rdoq); run_case tests/data/basic_64x64_1f.y4m still_64_rdoq \
    a7ee3a0ac75781ce2a2c7390c3503019 --still --still-rdoq
DUMP_ARGS=(--tx-type); run_case tests/data/basic_64x64_1f.y4m still_64_txtype \
    15681508a144eeea88bb8223185d75e9 --still --still-tx-type
DUMP_ARGS=(--full-intra --rdo-intra); run_case tests/data/basic_64x64_1f.y4m still_64_rdointra \
    45b4a209a8c3e889c91dfa201f26d2d4 --still --still-full-intra --still-rdo-intra
DUMP_ARGS=(--uv-full); run_case tests/data/basic_64x64_1f.y4m still_64_uvfull \
    2ed7f1ca8d146be8345ed95fee65f9bf --still --still-uv-full
DUMP_ARGS=(--cdef --rdoq --tx-type); run_case tests/data/basic_64x64_1f.y4m still_64_combo \
    aa885652b8c57ecc25b14b87c97862d8 --still --still-cdef --still-rdoq --still-tx-type
DUMP_ARGS=(--ccso); run_case tests/data/basic_64x64_1f.y4m still_64_ccso \
    980e5c2194091faf2347e3b05dfdea69 --still --still-ccso
DUMP_ARGS=(--cdef --ccso); run_case tests/data/basic_64x64_1f.y4m still_64_ccso_cdef \
    a90c525e480e8ff52db3f47f77fff887 --still --still-cdef --still-ccso

# --- 640x360 ---------------------------------------------------------------
DUMP_ARGS=(); run_case tests/data/basic_640x360_1f.y4m still_360_base \
    e61de628b0812b716b0efa7dcafcdd06 --still
DUMP_ARGS=(--cdef --cdef-strengths 1); run_case tests/data/basic_640x360_1f.y4m still_360_cdef1 \
    dec10527a779e7664ea78e53def9dbf1 --still --still-cdef --still-cdef-strengths 1
DUMP_ARGS=(--cdef --cdef-strengths 4); run_case tests/data/basic_640x360_1f.y4m still_360_cdef4 \
    b37bed7eb785d0c6b2e07219e888ff89 --still --still-cdef --still-cdef-strengths 4
DUMP_ARGS=(--rdoq); run_case tests/data/basic_640x360_1f.y4m still_360_rdoq \
    ac668d497012333b6aeecb82c2e70337 --still --still-rdoq
DUMP_ARGS=(--tx-type); run_case tests/data/basic_640x360_1f.y4m still_360_txtype \
    21c39aec98507dba7c24e5d320e1bb3c --still --still-tx-type
DUMP_ARGS=(--full-intra --rdo-intra); run_case tests/data/basic_640x360_1f.y4m still_360_rdointra \
    211da270c5baf34f7bd40da019ffd84b --still --still-full-intra --still-rdo-intra
DUMP_ARGS=(--uv-full); run_case tests/data/basic_640x360_1f.y4m still_360_uvfull \
    400dd3ad02db9e0e3e366091a7e6962f --still --still-uv-full
DUMP_ARGS=(--cdef --rdoq --tx-type); run_case tests/data/basic_640x360_1f.y4m still_360_combo \
    84ab3934c08e6c90b876d067bcbb3606 --still --still-cdef --still-rdoq --still-tx-type
DUMP_ARGS=(--ccso); run_case tests/data/basic_640x360_1f.y4m still_360_ccso \
    54e853e24b21bf1237b8b51a846b1c35 --still --still-ccso
DUMP_ARGS=(--cdef --ccso); run_case tests/data/basic_640x360_1f.y4m still_360_ccso_cdef \
    a272550a72c387a8a4f8cc8d3186f8da --still --still-cdef --still-ccso
DUMP_ARGS=(--ccso --ccso-bands 6); run_case tests/data/basic_640x360_1f.y4m still_360_ccso6 \
    54e853e24b21bf1237b8b51a846b1c35 --still --still-ccso --still-ccso-bands 6
DUMP_ARGS=(--ccso --rdoq); run_case tests/data/basic_640x360_1f.y4m still_360_ccso_rdoq \
    7fd7a2a0e078665614f8b6d2c5fe1b57 --still --still-ccso --still-rdoq

# --- coefficient-CDF qctx slices 1..3 (base_q_idx > 90) -------------------
# §8.3.1 selects one of four COEFF_CDF_Q_CTXS slices from the raw frame
# base_q_idx: <=90 slice 0, <=140 slice 1, <=190 slice 2, else slice 3.  One
# case at each slice's lower bound proves the selection and the qlookup-based
# quantiser against the pinned decoder; the default cases above cover slice 0.
DUMP_ARGS=()
RUN_QP=91;  run_case tests/data/basic_64x64_1f.y4m  still_64_qp91 \
    f50e18c4e08eab3ca6437ed6283a9216 --still
RUN_QP=141; run_case tests/data/basic_64x64_1f.y4m  still_64_qp141 \
    c47ea7a5e8f826addb8ef90c527c917a --still
RUN_QP=191; run_case tests/data/basic_64x64_1f.y4m  still_64_qp191 \
    3768deaaad9a67f7bece568d743993e5 --still
RUN_QP=255; run_case tests/data/basic_64x64_1f.y4m  still_64_qp255 \
    d1e1ebebf2885e575c55508f8b7157d1 --still
RUN_QP=91;  run_case tests/data/basic_640x360_1f.y4m still_360_qp91 \
    f5762980cda6c08ec82e131bac487a77 --still
RUN_QP=141; run_case tests/data/basic_640x360_1f.y4m still_360_qp141 \
    6db7ffba200e2b90075fcb826aaef3c1 --still
RUN_QP=191; run_case tests/data/basic_640x360_1f.y4m still_360_qp191 \
    df6894744772f7107cad88381ae82e2e --still
RUN_QP=255; run_case tests/data/basic_640x360_1f.y4m still_360_qp255 \
    d0c61808ae8dd6e011046bbab34256af --still
RUN_QP=

echo "feature_gate: $pass passed, $failures failed"
[ "$failures" -eq 0 ]
