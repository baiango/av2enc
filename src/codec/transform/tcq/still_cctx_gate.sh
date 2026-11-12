#!/usr/bin/env bash
# Decoder gate for the integrated CCTX path (--still-cctx).
#
#   ./still_cctx_gate.sh
#   AV2ENC_BIN=build-int/av2enc STILL_RECON=build-int/still_recon_dump \
#       ./still_cctx_gate.sh
#   KEEP=1 ./still_cctx_gate.sh
#
# What it proves:
#   1. `--still-cctx` writes enable_cctx = 1 and codes the §5.20.7.27
#      cctx_type symbol after the U EOB (when that EOB is neither 0 nor 1).
#   2. The §7.14.6 joint dequant rotation reconstructs sample-for-sample with
#      the pinned dav2d on all three planes.
#   3. The default `--still` streams stay byte-identical and every CCTX case
#      is IVF-pinned.
#   4. CCTX composes with TCQ, IST, TX_TYPE, RDOQ and CDEF, and QM / DC-AC
#      delta-Q are rejected by name.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
TMP="${TMPDIR:-/tmp}/av2enc_still_cctx_gate.$$"
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
    echo "still_cctx_gate: no av2enc binary; set AV2ENC_BIN=..." >&2
    exit 127
fi

DAV2D="${DAV2D:-}"
if [ -z "$DAV2D" ]; then
    for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
        if [ -x "$candidate" ]; then DAV2D="$candidate"; break; fi
    done
fi
if [ -z "$DAV2D" ] || [ ! -x "$DAV2D" ]; then
    echo "still_cctx_gate: no dav2d binary (set DAV2D=...)" >&2
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

# CCTX on, QP 60/90/120.
run_case tests/data/basic_64x64_1f.y4m still_64_cctx \
    5b8b0dcbd2ffb2a3a5afbc41c81f5329 --still-cctx
run_case tests/data/basic_640x360_1f.y4m still_360_cctx_qp60 \
    221172cae583919dd9becbdb9384656d --still-cctx --qp 60
run_case tests/data/basic_640x360_1f.y4m still_360_cctx_qp90 \
    909ee3792f3f1f1be436670a18ae2372 --still-cctx
run_case tests/data/basic_640x360_1f.y4m still_360_cctx_qp120 \
    7fb75da63815fa03607b734977a5caf4 --still-cctx --qp 120

# Compositions are sample-exact but not pinned here (their own gates pin the
# decisions).
run_case tests/data/basic_64x64_1f.y4m still_64_cctx_tcq - \
    --still-cctx --still-tcq
run_case tests/data/basic_64x64_1f.y4m still_64_cctx_ist - \
    --still-cctx --still-ist
run_case tests/data/basic_64x64_1f.y4m still_64_cctx_txtype - \
    --still-cctx --still-tx-type
run_case tests/data/basic_64x64_1f.y4m still_64_cctx_rdoq - \
    --still-cctx --still-rdoq
run_case tests/data/basic_64x64_1f.y4m still_64_cctx_cdef - \
    --still-cctx --still-cdef

reject_case reject_cctx_qm "--still-cctx is not supported with --still-qm" \
    --still-cctx --still-qm
reject_case reject_cctx_deltaq \
    "--still-cctx is not supported with --still-dc-ac-delta-q" \
    --still-cctx --still-dc-ac-delta-q


# The named AV2 gate wrapper must also accept the representative stream.
if [ -x "$ROOT/tools/validate_dav2d.sh" ] && [ -f "$TMP/still_64_cctx.ivf" ]; then
    if "$ROOT/tools/validate_dav2d.sh" "$TMP/still_64_cctx.ivf" \
            "$ROOT/tests/data/basic_64x64_1f.y4m" >"$TMP/validate_dav2d.log" 2>&1; then
        echo "PASS validate_dav2d (CCTX)"
        pass=$((pass + 1))
    else
        echo "FAIL validate_dav2d (CCTX)" >&2
        tail -20 "$TMP/validate_dav2d.log" >&2
        failures=$((failures + 1))
    fi
fi

if [ "$failures" -ne 0 ]; then
    echo "still_cctx_gate: $failures failed, $pass passed" >&2
    exit 1
fi
echo "still_cctx_gate: $pass passed, 0 failed"
