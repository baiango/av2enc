#!/usr/bin/env bash
# Decoder gate for the integrated TCQ path (--still-tcq).
#
#   ./still_tcq_gate.sh
#   AV2ENC_BIN=build-int/av2enc STILL_RECON=build-int/still_recon_dump \
#       ./still_tcq_gate.sh
#   KEEP=1 ./still_tcq_gate.sh
#
# What it proves:
#   1. `--still-tcq` writes enable_tcq = 1 and omits the enable_parity_hiding
#      bit (§5.4.8 single-picture inference: choose_tcq_per_frame = 0).
#   2. The §5.20.7.27 syntax (syntax level for the full level, (tcqState>>1)&1
#      coeff_base selectors, read_quant escape with x<<1) and the §7.14.4
#      extra dqDenom shift reconstruct sample-for-sample with the pinned dav2d.
#   3. The default `--still` streams stay byte-identical, and the IVF bytes of
#      every TCQ case are pinned.
#   4. TCQ composes with TX_TYPE, RDO-intra and CDEF, and the documented
#      incompatible combinations are rejected by name.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
TMP="${TMPDIR:-/tmp}/av2enc_still_tcq_gate.$$"
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
    echo "still_tcq_gate: no av2enc binary; set AV2ENC_BIN=..." >&2
    exit 127
fi

DAV2D="${DAV2D:-}"
if [ -z "$DAV2D" ]; then
    for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
        if [ -x "$candidate" ]; then DAV2D="$candidate"; break; fi
    done
fi
if [ -z "$DAV2D" ] || [ ! -x "$DAV2D" ]; then
    echo "still_tcq_gate: no dav2d binary (set DAV2D=...)" >&2
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

# compare_recon <encoder-recon.y4m> <decoded.y4m> <name>
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

# run_case <fixture> <name> <expected_ivf_md5|-> <encode args...>
run_case() {
    local fixture=$1 name=$2 expected=$3
    shift 3
    local input="$ROOT/$fixture"
    local out="$TMP/${name}.ivf"
    local dec="$TMP/${name}.dec.y4m"
    local recon="$TMP/${name}.recon.y4m"
    if [ ! -f "$input" ]; then
        echo "SKIP $name: missing $fixture" >&2
        return 0
    fi
    if ! "$AV2ENC_BIN" "$input" -o "$out" --frames 1 --still "$@" \
            >"$TMP/${name}.log" 2>&1; then
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
        if ! "$STILL_RECON" -i "$input" -o "$recon" --still "$@" \
                >"$TMP/${name}.recon.log" 2>&1; then
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

# reject_case <name> <substring> <args...>
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
    0ef9825062a25c4d950ee924113c6f89
run_case tests/data/basic_640x360_1f.y4m still_360_base \
    e61de628b0812b716b0efa7dcafcdd06

# TCQ on: 64x64 QP 90 and the 640x360 QP 60/90/120 sweep.
run_case tests/data/basic_64x64_1f.y4m still_64_tcq \
    2da56a430a0b445ea3bb9687729eaacd --still-tcq
run_case tests/data/basic_640x360_1f.y4m still_360_tcq_qp60 \
    9395e72bab6ff7c18f70ef2aa12a96a4 --still-tcq --qp 60
run_case tests/data/basic_640x360_1f.y4m still_360_tcq_qp90 \
    2743603540f41c46947219c5441ed455 --still-tcq
run_case tests/data/basic_640x360_1f.y4m still_360_tcq_qp120 \
    d3c0c73d69a922355d2639a26aab8aa3 --still-tcq --qp 120

# Documented compositions: sample-exact, not pinned (the other tool's own gate
# pins its decisions).
run_case tests/data/basic_64x64_1f.y4m still_64_tcq_txtype - \
    --still-tcq --still-tx-type
run_case tests/data/basic_64x64_1f.y4m still_64_tcq_rdo - \
    --still-tcq --still-full-intra --still-rdo-intra
run_case tests/data/basic_64x64_1f.y4m still_64_tcq_cdef - \
    --still-tcq --still-cdef

# Incompatible quantiser/tool combinations are rejected by name.
reject_case reject_tcq_qm "--still-tcq is not supported with --still-qm" \
    --still-tcq --still-qm
reject_case reject_tcq_rdoq "--still-tcq is not supported with --still-rdoq" \
    --still-tcq --still-rdoq
reject_case reject_tcq_deltaq \
    "--still-tcq is not supported with --still-dc-ac-delta-q" \
    --still-tcq --still-dc-ac-delta-q
reject_case reject_tcq_ist "--still-tcq is not supported with --still-ist" \
    --still-tcq --still-ist


# The named AV2 gate wrapper must also accept the representative stream.
if [ -x "$ROOT/tools/validate_dav2d.sh" ] && [ -f "$TMP/still_64_tcq.ivf" ]; then
    if "$ROOT/tools/validate_dav2d.sh" "$TMP/still_64_tcq.ivf" \
            "$ROOT/tests/data/basic_64x64_1f.y4m" >"$TMP/validate_dav2d.log" 2>&1; then
        echo "PASS validate_dav2d (TCQ)"
        pass=$((pass + 1))
    else
        echo "FAIL validate_dav2d (TCQ)" >&2
        tail -20 "$TMP/validate_dav2d.log" >&2
        failures=$((failures + 1))
    fi
fi

if [ "$failures" -ne 0 ]; then
    echo "still_tcq_gate: $failures failed, $pass passed" >&2
    exit 1
fi
echo "still_tcq_gate: $pass passed, 0 failed"
