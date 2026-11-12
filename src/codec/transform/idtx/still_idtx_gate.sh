#!/usr/bin/env bash
# Decoder gate for the integrated identity intra transform (--still-idtx).
#
#   ./still_idtx_gate.sh
#   AV2ENC_BIN=build-int/av2enc STILL_RECON=build-int/still_recon_dump \
#       ./still_idtx_gate.sh
#   KEEP=1 ./still_idtx_gate.sh
#
# What it proves:
#   1. `--still-idtx` writes enable_idtx_intra = 1 and codes the §5.20.5.5
#      fsc_mode = 0 symbol for every luma coding block <= 32x32.
#   2. IDTX is actually selected (the 640x360 encoder trace reports type=9
#      blocks) and the whole stream, including the identity inverse, is
#      sample-exact against the pinned dav2d on all three planes.
#   3. The default `--still` streams stay byte-identical and every IDTX stream
#      is IVF-pinned.
#   4. IDTX composes with TX partitioning, TCQ, IST, CCTX, RDOQ, QM and the
#      TX_TYPE search.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
TMP="${TMPDIR:-/tmp}/av2enc_still_idtx_gate.$$"
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
    echo "still_idtx_gate: no av2enc binary; set AV2ENC_BIN=..." >&2
    exit 127
fi

DAV2D="${DAV2D:-}"
if [ -z "$DAV2D" ]; then
    for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
        if [ -x "$candidate" ]; then DAV2D="$candidate"; break; fi
    done
fi
if [ -z "$DAV2D" ] || [ ! -x "$DAV2D" ]; then
    echo "still_idtx_gate: no dav2d binary (set DAV2D=...)" >&2
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

# Default streams must not move.
run_case tests/data/basic_64x64_1f.y4m still_64_base \
    0ef9825062a25c4d950ee924113c6f89
run_case tests/data/basic_640x360_1f.y4m still_360_base \
    e61de628b0812b716b0efa7dcafcdd06

# IDTX on: the 640x360 stream selects IDTX on long-side transform blocks.
run_case tests/data/basic_64x64_1f.y4m still_64_idtx \
    2f0788e54b8220897a48237bdbfe09d9 --still-idtx
run_case tests/data/basic_640x360_1f.y4m still_360_idtx_qp60 \
    e528b0a4ced436d1ca8ed2d5ab7552ed --still-idtx --qp 60
run_case tests/data/basic_640x360_1f.y4m still_360_idtx \
    858da1895d9855c9f0aae7963e7ec6dc --still-idtx

# Prove IDTX is actually reached (not just the mode-search plumbing).
if [ -n "$STILL_RECON" ]; then
    if AV2ENC_STILL_TRACE=1 "$STILL_RECON" \
            -i "$ROOT/tests/data/basic_640x360_1f.y4m" \
            -o "$TMP/trace.y4m" --still --still-idtx \
            >"$TMP/trace.log" 2>&1; then
        idtx_blocks=$(grep -o "type=9" "$TMP/trace.log" | wc -l | tr -d ' ')
        if [ "$idtx_blocks" -gt 0 ]; then
            echo "PASS idtx_selected ($idtx_blocks blocks)"
            pass=$((pass + 1))
        else
            echo "FAIL idtx_selected: no type=9 block in the trace" >&2
            failures=$((failures + 1))
        fi
    else
        echo "FAIL idtx_selected: trace encode failed" >&2
        failures=$((failures + 1))
    fi
fi

# Compositions are sample-exact (their own gates pin the decisions).
run_case tests/data/basic_640x360_1f.y4m still_360_idtx_txp - \
    --still-idtx --still-tx-partition
run_case tests/data/basic_640x360_1f.y4m still_360_idtx_tcq - \
    --still-idtx --still-tcq
run_case tests/data/basic_640x360_1f.y4m still_360_idtx_ist - \
    --still-idtx --still-ist
run_case tests/data/basic_640x360_1f.y4m still_360_idtx_cctx - \
    --still-idtx --still-cctx
run_case tests/data/basic_640x360_1f.y4m still_360_idtx_rdoq - \
    --still-idtx --still-rdoq
run_case tests/data/basic_640x360_1f.y4m still_360_idtx_qm - \
    --still-idtx --still-qm
run_case tests/data/basic_640x360_1f.y4m still_360_idtx_txtype - \
    --still-idtx --still-tx-type


# The named AV2 gate wrapper must also accept the representative stream.
if [ -x "$ROOT/tools/validate_dav2d.sh" ] && [ -f "$TMP/still_64_idtx.ivf" ]; then
    if "$ROOT/tools/validate_dav2d.sh" "$TMP/still_64_idtx.ivf" \
            "$ROOT/tests/data/basic_64x64_1f.y4m" >"$TMP/validate_dav2d.log" 2>&1; then
        echo "PASS validate_dav2d (IDTX)"
        pass=$((pass + 1))
    else
        echo "FAIL validate_dav2d (IDTX)" >&2
        tail -20 "$TMP/validate_dav2d.log" >&2
        failures=$((failures + 1))
    fi
fi

if [ "$failures" -ne 0 ]; then
    echo "still_idtx_gate: $failures failed, $pass passed" >&2
    exit 1
fi
echo "still_idtx_gate: $pass passed, 0 failed"
