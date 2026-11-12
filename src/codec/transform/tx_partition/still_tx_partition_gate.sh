#!/usr/bin/env bash
# Decoder gate for the integrated TX_MODE_SELECT partition path
# (--still-tx-partition).
#
#   ./still_tx_partition_gate.sh
#   AV2ENC_BIN=build-int/av2enc STILL_RECON=build-int/still_recon_dump \
#       ./still_tx_partition_gate.sh
#   KEEP=1 ./still_tx_partition_gate.sh
#
# What it proves:
#   1. `--still-tx-partition` sets tx_mode_select = 1 and writes the
#      §5.20.6.3 tx_do_partition / tx_partition_type symbols before the
#      coefficients, with one transform per part.
#   2. Prediction and reconstruction run per transform part, in decode order,
#      with each part's reconstruction feeding the next part's prediction; the
#      result is sample-exact against the pinned dav2d on all three planes.
#   3. The default `--still` streams stay byte-identical and the natural and
#      forced NONE/SPLIT/HORZ/VERT streams are IVF-pinned.
#   4. TX partitioning composes with RDOQ, TX_TYPE, IST, TCQ, CCTX and
#      RDO-intra, and QM / DC-AC delta-Q / DIP / MRL / IBP are rejected by name.
#
# AV2ENC_STILL_FORCE_TX_PARTITION=none|split|horz|vert is the debug hook used to
# pin all four signalling shapes without relying on the SSE decision.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
TMP="${TMPDIR:-/tmp}/av2enc_still_tx_partition_gate.$$"
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
    echo "still_tx_partition_gate: no av2enc binary; set AV2ENC_BIN=..." >&2
    exit 127
fi

DAV2D="${DAV2D:-}"
if [ -z "$DAV2D" ]; then
    for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
        if [ -x "$candidate" ]; then DAV2D="$candidate"; break; fi
    done
fi
if [ -z "$DAV2D" ] || [ ! -x "$DAV2D" ]; then
    echo "still_tx_partition_gate: no dav2d binary (set DAV2D=...)" >&2
    exit 77
fi

STILL_RECON="${STILL_RECON:-}"
if [ -z "$STILL_RECON" ]; then
    for candidate in "$ROOT/build-int/still_recon_dump" \
                     "$ROOT/build-m1/still_recon_dump"; do
        if [ -x "$candidate" ]; then STILL_RECON="$candidate"; break; fi
    done
fi

md5_of() {
    md5 -q "$1" 2>/dev/null || md5sum "$1" | awk '{print $1}'
}

compare_recon() {
    if [ -z "$STILL_RECON" ]; then
        echo "note $3: still_recon_dump unavailable; skipping sample-exact compare" >&2
        return 0
    fi
    python3 - "$1" "$2" "$3" <<'PY'
import sys
def payload(path):
    with open(path, "rb") as f:
        data = f.read()
    marker = b"FRAME\n"
    index = data.find(marker)
    if index < 0:
        raise SystemExit(f"{path}: no FRAME marker")
    return data[index + len(marker):]
a = payload(sys.argv[1])
b = payload(sys.argv[2])
if len(a) != len(b):
    raise SystemExit(f"{sys.argv[3]}: plane size {len(a)} != {len(b)}")
diffs = [abs(x - y) for x, y in zip(a, b)]
bad = sum(1 for d in diffs if d)
if bad:
    raise SystemExit(
        f"{sys.argv[3]}: {bad}/{len(a)} samples differ, max {max(diffs)}")
print(f"exact {sys.argv[3]}: {len(a)} samples")
PY
}

failures=0
pass=0

# run_case <fixture> <name> <expected_ivf_md5|-> <forced:-> [encode args...]
run_case() {
    local fixture=$1 name=$2 expected=$3 forced=$4
    shift 4
    local input="$ROOT/$fixture"
    local out="$TMP/${name}.ivf"
    local dec="$TMP/${name}.dec.y4m"
    local recon="$TMP/${name}.recon.y4m"
    if [ ! -f "$input" ]; then
        echo "SKIP $name: missing $fixture" >&2
        return 0
    fi
    local force_env=()
    if [ "$forced" != "-" ]; then
        force_env=(env "AV2ENC_STILL_FORCE_TX_PARTITION=$forced")
    fi
    if ! ${force_env[@]+"${force_env[@]}"} "$AV2ENC_BIN" "$input" -o "$out" --frames 1 --still \
            "$@" >"$TMP/${name}.log" 2>&1; then
        echo "FAIL $name: encode failed" >&2
        cat "$TMP/${name}.log" >&2
        failures=$((failures + 1))
        return 0
    fi
    if [ "$expected" != "-" ]; then
        local got
        got=$(md5_of "$out")
        if [ "$got" != "$expected" ]; then
            echo "FAIL $name: IVF md5 $got, expected $expected" >&2
            failures=$((failures + 1))
            return 0
        fi
    fi
    if ! "$DAV2D" -q -i "$out" -o "$dec" --threads 1 --strict 1 \
            >"$TMP/${name}.dec.log" 2>&1; then
        echo "FAIL $name: dav2d --strict 1 rejected the stream" >&2
        cat "$TMP/${name}.dec.log" >&2
        failures=$((failures + 1))
        return 0
    fi
    if [ -n "$STILL_RECON" ]; then
        if ! ${force_env[@]+"${force_env[@]}"} "$STILL_RECON" -i "$input" -o "$recon" --still \
                "$@" >"$TMP/${name}.recon.log" 2>&1; then
            echo "FAIL $name: still_recon_dump failed" >&2
            cat "$TMP/${name}.recon.log" >&2
            failures=$((failures + 1))
            return 0
        fi
        if ! compare_recon "$recon" "$dec" "$name" >>"$TMP/${name}.cmp.log" 2>&1; then
            echo "FAIL $name: encoder reconstruction is not decoder-exact" >&2
            cat "$TMP/${name}.cmp.log" >&2
            failures=$((failures + 1))
            return 0
        fi
    fi
    echo "PASS $name ($(wc -c <"$out" | tr -d ' ') B, md5 $expected)"
    pass=$((pass + 1))
}

reject_case() {
    local name=$1 needle=$2
    shift 2
    if "$AV2ENC_BIN" "$ROOT/tests/data/basic_64x64_1f.y4m" \
            -o "$TMP/${name}.ivf" --frames 1 --still "$@" \
            >"$TMP/${name}.log" 2>&1; then
        echo "FAIL $name: encode unexpectedly succeeded" >&2
        failures=$((failures + 1))
        return 0
    fi
    if grep -q -- "$needle" "$TMP/${name}.log"; then
        echo "PASS $name"
        pass=$((pass + 1))
    else
        echo "FAIL $name: wrong diagnostic" >&2
        cat "$TMP/${name}.log" >&2
        failures=$((failures + 1))
    fi
}

# Default streams must not move.
run_case tests/data/basic_64x64_1f.y4m still_64_base \
    0ef9825062a25c4d950ee924113c6f89 - 
run_case tests/data/basic_640x360_1f.y4m still_360_base \
    e61de628b0812b716b0efa7dcafcdd06 -

# Natural SSE decision (small streams, only SPLIT is chosen at this QP).
run_case tests/data/basic_64x64_1f.y4m still_64_txp \
    78efef38377b249ba3a4d2fcd4fb0399 - --still-tx-partition
run_case tests/data/basic_640x360_1f.y4m still_360_txp \
    29c5b0b100f512090719846eb9b3c516 - --still-tx-partition

# All four signalling shapes, forced for the gate.
run_case tests/data/basic_64x64_1f.y4m still_64_txp_split \
    d8726d73d52d2e011996f6eab555815f split --still-tx-partition
run_case tests/data/basic_64x64_1f.y4m still_64_txp_horz \
    4f311b979bab5aa56cef7b0f7dc0f452 horz --still-tx-partition
run_case tests/data/basic_64x64_1f.y4m still_64_txp_vert \
    82cc7b5e92aaafbb1d9f18ffe08165a4 vert --still-tx-partition

# Compositions are sample-exact (their own gates pin the decisions).
run_case tests/data/basic_64x64_1f.y4m still_64_txp_rdoq - split \
    --still-tx-partition --still-rdoq
run_case tests/data/basic_64x64_1f.y4m still_64_txp_txtype - split \
    --still-tx-partition --still-tx-type
run_case tests/data/basic_64x64_1f.y4m still_64_txp_ist - split \
    --still-tx-partition --still-ist
run_case tests/data/basic_64x64_1f.y4m still_64_txp_tcq - split \
    --still-tx-partition --still-tcq
run_case tests/data/basic_64x64_1f.y4m still_64_txp_cctx - split \
    --still-tx-partition --still-cctx
run_case tests/data/basic_64x64_1f.y4m still_64_txp_rdointra - split \
    --still-tx-partition --still-full-intra --still-rdo-intra

reject_case reject_txp_qm \
    "--still-tx-partition is not supported with --still-qm" \
    --still-tx-partition --still-qm
reject_case reject_txp_deltaq \
    "--still-tx-partition is not supported with --still-dc-ac-delta-q" \
    --still-tx-partition --still-dc-ac-delta-q
reject_case reject_txp_dip \
    "--still-tx-partition is not supported with --still-dip" \
    --still-tx-partition --still-dip
reject_case reject_txp_mrl \
    "--still-tx-partition is not supported with --still-mrl" \
    --still-tx-partition --still-mrl
reject_case reject_txp_ibp \
    "--still-tx-partition is not supported with --still-ibp" \
    --still-tx-partition --still-ibp


# The named AV2 gate wrapper must also accept the representative stream.
if [ -x "$ROOT/tools/validate_dav2d.sh" ] && [ -f "$TMP/still_64_txp.ivf" ]; then
    if "$ROOT/tools/validate_dav2d.sh" "$TMP/still_64_txp.ivf" \
            "$ROOT/tests/data/basic_64x64_1f.y4m" >"$TMP/validate_dav2d.log" 2>&1; then
        echo "PASS validate_dav2d (TX partition)"
        pass=$((pass + 1))
    else
        echo "FAIL validate_dav2d (TX partition)" >&2
        tail -20 "$TMP/validate_dav2d.log" >&2
        failures=$((failures + 1))
    fi
fi

if [ "$failures" -ne 0 ]; then
    echo "still_tx_partition_gate: $failures failed, $pass passed" >&2
    exit 1
fi
echo "still_tx_partition_gate: $pass passed, 0 failed"
