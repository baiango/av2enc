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
    2b31e378787363641c836fdf563bbd0e --still
DUMP_ARGS=(--cdef --cdef-strengths 1); run_case tests/data/basic_64x64_1f.y4m still_64_cdef1 \
    5cd9182e9d6faf86889d13f14d59dd42 --still --still-cdef --still-cdef-strengths 1
DUMP_ARGS=(--cdef --cdef-strengths 4); run_case tests/data/basic_64x64_1f.y4m still_64_cdef4 \
    5cd9182e9d6faf86889d13f14d59dd42 --still --still-cdef --still-cdef-strengths 4
DUMP_ARGS=(--rdoq); run_case tests/data/basic_64x64_1f.y4m still_64_rdoq \
    cc330b8b5d528da885ee2ff6c4228e1c --still --still-rdoq
DUMP_ARGS=(--tx-type); run_case tests/data/basic_64x64_1f.y4m still_64_txtype \
    4965b4a03e5b456c7a364a012df417dc --still --still-tx-type
DUMP_ARGS=(--full-intra --rdo-intra); run_case tests/data/basic_64x64_1f.y4m still_64_rdointra \
    070c452082d9b71e00abc63db4e2a959 --still --still-full-intra --still-rdo-intra
DUMP_ARGS=(--uv-full); run_case tests/data/basic_64x64_1f.y4m still_64_uvfull \
    ffcb8859679f810220f36315726d9b2a --still --still-uv-full
DUMP_ARGS=(--cdef --rdoq --tx-type); run_case tests/data/basic_64x64_1f.y4m still_64_combo \
    cbc82d8f2b19d2131d7581c666d1f86d --still --still-cdef --still-rdoq --still-tx-type
DUMP_ARGS=(--ccso); run_case tests/data/basic_64x64_1f.y4m still_64_ccso \
    71ca9ce6ad19fe2675419cd317361b4a --still --still-ccso
DUMP_ARGS=(--cdef --ccso); run_case tests/data/basic_64x64_1f.y4m still_64_ccso_cdef \
    b5e0a05a48833b59a22729de72ccc3b1 --still --still-cdef --still-ccso

# --- 640x360 ---------------------------------------------------------------
DUMP_ARGS=(); run_case tests/data/basic_640x360_1f.y4m still_360_base \
    8ffbe059cfeac3e6ee255f3bb7cf72b8 --still
DUMP_ARGS=(--cdef --cdef-strengths 1); run_case tests/data/basic_640x360_1f.y4m still_360_cdef1 \
    086e90b1645f7d7778235fea94fba656 --still --still-cdef --still-cdef-strengths 1
DUMP_ARGS=(--cdef --cdef-strengths 4); run_case tests/data/basic_640x360_1f.y4m still_360_cdef4 \
    5bf1100af41190b447c0741bcca780dd --still --still-cdef --still-cdef-strengths 4
DUMP_ARGS=(--rdoq); run_case tests/data/basic_640x360_1f.y4m still_360_rdoq \
    fa5f528bc75d4d6f5039286641a24bb1 --still --still-rdoq
DUMP_ARGS=(--tx-type); run_case tests/data/basic_640x360_1f.y4m still_360_txtype \
    93f2e4cb5098f55e0991480d820a32cb --still --still-tx-type
DUMP_ARGS=(--full-intra --rdo-intra); run_case tests/data/basic_640x360_1f.y4m still_360_rdointra \
    f7106fa0147446621528e244a9d85949 --still --still-full-intra --still-rdo-intra
DUMP_ARGS=(--uv-full); run_case tests/data/basic_640x360_1f.y4m still_360_uvfull \
    67d79e56c10ab4cf4792a5e4ee88895e --still --still-uv-full
DUMP_ARGS=(--cdef --rdoq --tx-type); run_case tests/data/basic_640x360_1f.y4m still_360_combo \
    d7f83b6f8df0ef9b9c97a9298840065a --still --still-cdef --still-rdoq --still-tx-type
DUMP_ARGS=(--ccso); run_case tests/data/basic_640x360_1f.y4m still_360_ccso \
    cd270c0e55fdb4ef75ce54aaae4abdc6 --still --still-ccso
DUMP_ARGS=(--cdef --ccso); run_case tests/data/basic_640x360_1f.y4m still_360_ccso_cdef \
    1a1f0d163de13f5b9f477adb9ca4fb13 --still --still-cdef --still-ccso
DUMP_ARGS=(--ccso --ccso-bands 6); run_case tests/data/basic_640x360_1f.y4m still_360_ccso6 \
    cd270c0e55fdb4ef75ce54aaae4abdc6 --still --still-ccso --still-ccso-bands 6
DUMP_ARGS=(--ccso --rdoq); run_case tests/data/basic_640x360_1f.y4m still_360_ccso_rdoq \
    4d0d10514d2caf17f5797ed65444551b --still --still-ccso --still-rdoq

# --- coefficient-CDF qctx slices 1..3 (base_q_idx > 90) -------------------
# §8.3.1 selects one of four COEFF_CDF_Q_CTXS slices from the raw frame
# base_q_idx: <=90 slice 0, <=140 slice 1, <=190 slice 2, else slice 3.  One
# case at each slice's lower bound proves the selection and the qlookup-based
# quantiser against the pinned decoder; the default cases above cover slice 0.
DUMP_ARGS=()
RUN_QP=91;  run_case tests/data/basic_64x64_1f.y4m  still_64_qp91 \
    d2779b50951674778c10a6fdbd644aa7 --still
RUN_QP=141; run_case tests/data/basic_64x64_1f.y4m  still_64_qp141 \
    ac39c106370f67cb5720280f650338a9 --still
RUN_QP=191; run_case tests/data/basic_64x64_1f.y4m  still_64_qp191 \
    c3feceb50ec5cfc2dc3340704ffada38 --still
RUN_QP=255; run_case tests/data/basic_64x64_1f.y4m  still_64_qp255 \
    f41f9d1556e56b69ecd07026815f7a7c --still
RUN_QP=91;  run_case tests/data/basic_640x360_1f.y4m still_360_qp91 \
    b6fdefabaf2781f62d6dae44d7a69ebd --still
RUN_QP=141; run_case tests/data/basic_640x360_1f.y4m still_360_qp141 \
    cef364ae9fe9408063208348aff4402b --still
RUN_QP=191; run_case tests/data/basic_640x360_1f.y4m still_360_qp191 \
    c2d2fed17bf56510382f2f8faadbcd75 --still
RUN_QP=255; run_case tests/data/basic_640x360_1f.y4m still_360_qp255 \
    a5866b16adf351580d5de21a1752f04a --still
RUN_QP=

echo "feature_gate: $pass passed, $failures failed"
[ "$failures" -eq 0 ]
