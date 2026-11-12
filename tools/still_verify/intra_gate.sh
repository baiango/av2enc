#!/usr/bin/env bash
# Intra-tool gate: encode -> dav2d --strict 1 decode -> encoder-reconstruction
# exactness, for every opt-in tool integrated on the `integration/intra` branch.
#
# Usage:
#   tools/still_verify/intra_gate.sh --bin build-int/av2enc \
#       [--dump build-int/still_recon_dump] [--pin] [--only <substring>]
#
#   --pin     print the IVF MD5 of every case (for docs/m1-features.md)
#   --only S  run only cases whose name contains S
#
# Every case is default-off on the CLI, so the pinned `--still` baseline cases
# also run here; a moved baseline MD5 fails the case.
#
# Env: DAV2D overrides the decoder path.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
BIN=""
DUMP=""
PIN=0
ONLY=""
DAV2D="${DAV2D:-}"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

while [ $# -gt 0 ]; do
    case "$1" in
        --bin) BIN=$2; shift 2 ;;
        --dump) DUMP=$2; shift 2 ;;
        --pin) PIN=1; shift ;;
        --only) ONLY=$2; shift 2 ;;
        --dav2d) DAV2D=$2; shift 2 ;;
        -h|--help)
            sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "intra_gate: unknown argument $1" >&2; exit 2 ;;
    esac
done

if [ -z "$BIN" ] || [ ! -x "$BIN" ]; then
    echo "intra_gate: --bin must point at an av2enc executable" >&2
    exit 2
fi
if [ -z "$DAV2D" ]; then
    for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
        if [ -x "$candidate" ]; then DAV2D=$candidate; break; fi
    done
fi
if [ -z "$DAV2D" ] || [ ! -x "$DAV2D" ]; then
    echo "intra_gate: no dav2d binary (set DAV2D=...)" >&2
    exit 77
fi
if [ -z "$DUMP" ] || [ ! -x "$DUMP" ]; then
    echo "intra_gate: --dump must point at still_recon_dump" >&2
    exit 2
fi

failures=0
pass=0
skipped=0

# run_case <fixture> <name> <encode flags...>
run_case() {
    local fixture=$1 name=$2
    shift 2
    if [ -n "$ONLY" ] && [[ "$name" != *"$ONLY"* ]]; then
        return 0
    fi
    local input="$ROOT/$fixture"
    if [[ "$fixture" == /* ]]; then
        input="$fixture"
    fi
    local out="$TMP/${name}.ivf"
    local dec="$TMP/${name}.dec.y4m"
    local recon="$TMP/${name}.recon.y4m"
    if [ ! -f "$input" ]; then
        echo "SKIP $name: missing $fixture" >&2
        skipped=$((skipped + 1))
        return 0
    fi
    if ! "$BIN" "$input" -o "$out" --frames 1 "$@" >"$TMP/${name}.enc.log" 2>&1; then
        echo "FAIL $name: encode failed" >&2
        cat "$TMP/${name}.enc.log" >&2
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
    if ! "$DUMP" -i "$input" -o "$recon" "$@" >"$TMP/${name}.dump.log" 2>&1; then
        echo "FAIL $name: still_recon_dump failed" >&2
        cat "$TMP/${name}.dump.log" >&2
        failures=$((failures + 1))
        return 0
    fi
    local cmp
    cmp=$(python3 - "$dec" "$recon" <<'PY'
import sys

def read_y4m(path):
    with open(path, "rb") as f:
        data = f.read()
    pos = data.index(b"\n") + 1
    frames = []
    while pos < len(data):
        if data[pos:pos + 5] != b"FRAME":
            raise SystemExit("bad frame marker in %s" % path)
        pos = data.index(b"\n", pos) + 1
        frames.append(data[pos:])
        break
    return frames[0] if frames else b""

a = read_y4m(sys.argv[1])
b = read_y4m(sys.argv[2])
if len(a) != len(b):
    print("size %d != %d" % (len(a), len(b)))
    raise SystemExit(0)
diffs = [abs(x - y) for x, y in zip(a, b) if x != y]
if not diffs:
    print("exact")
else:
    print("%d diffs, max %d" % (len(diffs), max(diffs)))
PY
)
    local md5
    md5=$(md5 -q "$out" 2>/dev/null || md5sum "$out" | awk '{print $1}')
    if [ "$cmp" != "exact" ]; then
        echo "FAIL $name: recon mismatch ($cmp)" >&2
        failures=$((failures + 1))
        return 0
    fi
    if [ "$PIN" = "1" ]; then
        echo "PASS $name ($(wc -c <"$out" | tr -d ' ') B, md5 $md5, recon exact)"
    else
        echo "PASS $name ($(wc -c <"$out" | tr -d ' ') B, recon exact)"
    fi
    pass=$((pass + 1))
}

# run_reject <name> <expected stderr substring> <encode flags...>
run_reject() {
    local name=$1 expected=$2
    shift 2
    if [ -n "$ONLY" ] && [[ "$name" != *"$ONLY"* ]]; then
        return 0
    fi
    local out="$TMP/${name}.ivf"
    if "$BIN" "$ROOT/tests/data/basic_64x64_1f.y4m" -o "$out" --frames 1 \
            "$@" >"$TMP/${name}.log" 2>&1; then
        echo "FAIL $name: expected a rejection, encode succeeded" >&2
        failures=$((failures + 1))
        return 0
    fi
    if ! grep -q "$expected" "$TMP/${name}.log"; then
        echo "FAIL $name: rejection message does not mention '$expected'" >&2
        cat "$TMP/${name}.log" >&2
        failures=$((failures + 1))
        return 0
    fi
    echo "PASS $name (rejected at the CLI)"
    pass=$((pass + 1))
}

BASELINE=(--still)

# Synthetic screen-content fixtures (flat fills, stripes, icon squares).  The
# natural fixtures almost never select palette, so the tool needs its own
# content to prove the syntax path.
python3 - "$TMP" <<'PY'
import sys, pathlib
out = pathlib.Path(sys.argv[1])

def write_y4m(path, w, h):
    y = bytearray(w * h)
    for yy in range(h):
        for xx in range(w):
            if yy < h // 2:
                v = 32 if xx < w // 2 else 200
                if (xx // 8) % 2 == 0 and yy >= h // 8:
                    v = 96
            else:
                v = 64 if (xx // 16) % 2 == 0 else 160
                if (yy // 8) % 4 == 0:
                    v = 240
            y[yy * w + xx] = v
    cw, ch = (w + 1) // 2, (h + 1) // 2
    u = bytes([96]) * (cw * ch)
    v = bytes([160]) * (cw * ch)
    with open(path, "wb") as f:
        f.write(b"YUV4MPEG2 W%d H%d F25:1 Ip A0:0 C420\n" % (w, h))
        f.write(b"FRAME\n")
        f.write(bytes(y))
        f.write(u)
        f.write(v)

write_y4m(out / "scc_64x64.y4m", 64, 64)
write_y4m(out / "scc_128x96.y4m", 128, 96)
PY
SCC64="$TMP/scc_64x64.y4m"
SCC128="$TMP/scc_128x96.y4m"

# --- default-off regressions (must stay byte-identical everywhere) ----------
run_case tests/data/basic_64x64_1f.y4m baseline_64 "${BASELINE[@]}"
run_case tests/data/basic_640x360_1f.y4m baseline_360 "${BASELINE[@]}"

# --- integrated intra tools -------------------------------------------------
run_case tests/data/basic_64x64_1f.y4m ibp_64 "${BASELINE[@]}" --still-ibp
run_case tests/data/basic_640x360_1f.y4m ibp_360 "${BASELINE[@]}" --still-ibp
run_case tests/data/basic_64x64_1f.y4m ibp_fullintra_64 "${BASELINE[@]}" \
    --still-ibp --still-full-intra
run_case tests/data/basic_640x360_1f.y4m ibp_fullintra_360 "${BASELINE[@]}" \
    --still-ibp --still-full-intra
run_case tests/data/basic_64x64_1f.y4m ibp_rdoq_64 "${BASELINE[@]}" \
    --still-ibp --still-rdoq
run_case tests/data/basic_640x360_1f.y4m ibp_cdef_360 "${BASELINE[@]}" \
    --still-ibp --still-cdef
run_case tests/data/basic_64x64_1f.y4m mrl_64 "${BASELINE[@]}" --still-mrl
run_case tests/data/basic_640x360_1f.y4m mrl_360 "${BASELINE[@]}" --still-mrl
run_case tests/data/basic_64x64_1f.y4m mrl_txtype_64 "${BASELINE[@]}" \
    --still-mrl --still-tx-type
run_case tests/data/basic_640x360_1f.y4m mrl_deblock_360 "${BASELINE[@]}" \
    --still-mrl --still-cdef
run_case tests/data/basic_64x64_1f.y4m cfl_ds1_64 "${BASELINE[@]}" \
    --still-cfl-ds 1
run_case tests/data/basic_640x360_1f.y4m cfl_ds1_360 "${BASELINE[@]}" \
    --still-cfl-ds 1
run_case tests/data/basic_64x64_1f.y4m cfl_ds2_64 "${BASELINE[@]}" \
    --still-cfl-ds 2
run_case tests/data/basic_640x360_1f.y4m cfl_ds2_360 "${BASELINE[@]}" \
    --still-cfl-ds 2
run_case tests/data/basic_64x64_1f.y4m cfl_explicit_64 "${BASELINE[@]}" \
    --still-cfl-explicit
run_case tests/data/basic_640x360_1f.y4m cfl_explicit_360 "${BASELINE[@]}" \
    --still-cfl-explicit
run_case tests/data/basic_64x64_1f.y4m cfl_explicit_ds1_64 "${BASELINE[@]}" \
    --still-cfl-explicit --still-cfl-ds 1
run_case tests/data/basic_64x64_1f.y4m dip_64 "${BASELINE[@]}" --still-dip
run_case tests/data/basic_640x360_1f.y4m dip_360 "${BASELINE[@]}" --still-dip
run_case tests/data/basic_64x64_1f.y4m dip_cdef_64 "${BASELINE[@]}" \
    --still-dip --still-cdef
run_case tests/data/basic_640x360_1f.y4m dip_ibp_360 "${BASELINE[@]}" \
    --still-dip --still-ibp
run_reject palette_disabled "not integrated" --still --still-palette
run_case tests/data/basic_64x64_1f.y4m mhccp_64 "${BASELINE[@]}" --still-mhccp
run_case tests/data/basic_640x360_1f.y4m mhccp_360 "${BASELINE[@]}" --still-mhccp
run_case tests/data/basic_64x64_1f.y4m mhccp_explicit_64 "${BASELINE[@]}" \
    --still-mhccp --still-cfl-explicit
run_case "$SCC128" mhccp_scc128 "${BASELINE[@]}" --still-mhccp

echo
echo "intra_gate: $pass passed, $failures failed, $skipped skipped"
[ "$failures" -eq 0 ]
