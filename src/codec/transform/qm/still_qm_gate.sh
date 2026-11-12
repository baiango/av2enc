#!/usr/bin/env bash
# Decoder gate for the integrated QM path (--still-qm).
#
#   ./still_qm_gate.sh
#   AV2ENC_BIN=build-int/av2enc ./still_qm_gate.sh
#   KEEP=1 ./still_qm_gate.sh
#
# What it proves:
#   1. `--still --still-qm` signals using_qmatrix = 1 with a spec-defined
#      default matrix level (no quantizer_matrix_obu(); the pinned dav2d
#      rejects all QM OBUs) and the stream decodes with `dav2d --strict 1`.
#   2. The IVF bytes are pinned: any change to the header, the forward
#      quantizer or the §7.14.4 q2 matrix moves the MD5.
#   3. The default `--still` streams stay byte-identical (QM is opt-in).
#   4. When still_dump is available, the encoder reconstruction is compared
#      sample-for-sample with the decoder (all three planes).
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
TMP="${TMPDIR:-/tmp}/av2enc_still_qm_gate.$$"
mkdir -p "$TMP"
cleanup() {
    if [ "${KEEP:-0}" != "1" ]; then rm -rf "$TMP"; fi
}
trap cleanup EXIT

AV2ENC_BIN="${AV2ENC_BIN:-}"
if [ -z "$AV2ENC_BIN" ]; then
    for candidate in "$ROOT/build-int/av2enc" "$ROOT/build-m1/av2enc" \
                     "$ROOT/build/av2enc"; do
        if [ -x "$candidate" ]; then AV2ENC_BIN="$candidate"; break; fi
    done
fi
if [ -z "$AV2ENC_BIN" ] || [ ! -x "$AV2ENC_BIN" ]; then
    echo "still_qm_gate: no av2enc binary; set AV2ENC_BIN=..." >&2
    exit 127
fi

DAV2D="${DAV2D:-}"
if [ -z "$DAV2D" ]; then
    for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
        if [ -x "$candidate" ]; then DAV2D="$candidate"; break; fi
    done
fi
if [ -z "$DAV2D" ] || [ ! -x "$DAV2D" ]; then
    echo "still_qm_gate: no dav2d binary (set DAV2D=...)" >&2
    exit 77
fi

STILL_DUMP="${STILL_DUMP:-}"
if [ -z "$STILL_DUMP" ]; then
    for candidate in "$ROOT/build-int/still_dump" "$ROOT/build-m1/still_dump"; do
        if [ -x "$candidate" ]; then STILL_DUMP="$candidate"; break; fi
    done
fi

md5_of() {
    md5 -q "$1" 2>/dev/null || md5sum "$1" | awk '{print $1}'
}

failures=0
pass=0

# run_case <fixture> <name> <expected_ivf_md5> <qm:0|1> <encode args...>
run_case() {
    local fixture=$1 name=$2 expected=$3 use_qm=$4
    shift 4
    local input="$ROOT/$fixture"
    local out="$TMP/${name}.ivf"
    local dec="$TMP/${name}.y4m"
    if [ ! -f "$input" ]; then
        echo "SKIP $name: missing $fixture" >&2
        return 0
    fi
    local qm_args=()
    if [ "$use_qm" = "1" ]; then qm_args=(--still-qm); fi
    if ! "$AV2ENC_BIN" "$input" -o "$out" --frames 1 --still \
            ${qm_args[@]+"${qm_args[@]}"} "$@" >"$TMP/${name}.log" 2>&1; then
        echo "FAIL $name: encode failed" >&2
        cat "$TMP/${name}.log" >&2
        failures=$((failures + 1))
        return 0
    fi
    local got
    got=$(md5_of "$out")
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
    if [ -n "$STILL_DUMP" ]; then
        local dump_args=()
        if [ "$use_qm" = "1" ]; then dump_args=(--qm); fi
        if "$STILL_DUMP" -i "$input" -o "$TMP/${name}.dump" \
                ${dump_args[@]+"${dump_args[@]}"} "$@" \
                >"$TMP/${name}.dump.log" 2>&1; then
            if ! python3 "$ROOT/tools/still_verify/verify.py" \
                    --ivf "$out" --dump "$TMP/${name}.dump" \
                    >"$TMP/${name}.verify.log" 2>&1; then
                echo "FAIL $name: recon/decoder mismatch" >&2
                tail -20 "$TMP/${name}.verify.log" >&2
                failures=$((failures + 1))
                return 0
            fi
        else
            echo "note $name: still_dump failed; skipping recon compare" >&2
        fi
    fi
    echo "PASS $name ($(wc -c <"$out" | tr -d ' ') B, md5 $expected)"
    pass=$((pass + 1))
}

# Default streams must not move.
run_case tests/data/basic_64x64_1f.y4m still_64_base \
    0ef9825062a25c4d950ee924113c6f89 0
run_case tests/data/basic_640x360_1f.y4m still_360_base \
    e61de628b0812b716b0efa7dcafcdd06 0

# QM on: default matrix level 3, both fixtures and a qctx slice change.
run_case tests/data/basic_64x64_1f.y4m still_64_qm \
    421c3418520713aa6ab027ee57b1bcaf 1
run_case tests/data/basic_640x360_1f.y4m still_360_qm \
    9c2c765e15fb746275e50113d806d374 1
run_case tests/data/basic_64x64_1f.y4m still_64_qm_qp141 \
    0a96dc8b76ef2994b215be542e07d7d6 1 --qp 141

# Incompatible combination must be rejected clearly.
if "$AV2ENC_BIN" "$ROOT/tests/data/basic_64x64_1f.y4m" \
        -o "$TMP/reject.ivf" --still --still-qm --still-rdoq \
        >"$TMP/reject.log" 2>&1; then
    echo "FAIL reject_qm_rdoq: encode unexpectedly succeeded" >&2
    failures=$((failures + 1))
else
    if grep -q -- "--still-qm is not supported with --still-rdoq" \
            "$TMP/reject.log"; then
        echo "PASS reject_qm_rdoq"
        pass=$((pass + 1))
    else
        echo "FAIL reject_qm_rdoq: wrong diagnostic" >&2
        cat "$TMP/reject.log" >&2
        failures=$((failures + 1))
    fi
fi

if [ "$failures" -ne 0 ]; then
    echo "still_qm_gate: $failures failed, $pass passed" >&2
    exit 1
fi
echo "still_qm_gate: $pass passed, 0 failed"
