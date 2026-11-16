#!/usr/bin/env bash
# User-defined quantizer matrix gate (--still-user-qm).
#
#   ./still_user_qm_gate.sh
#   AV2ENC_BIN=build-int/av2enc STILL_RECON=build-int/still_recon_dump \
#       ./still_user_qm_gate.sh
#   KEEP=1 ./still_user_qm_gate.sh
#
# What it proves:
#   1. `--still-user-qm` emits a real Annex B.2 OBU_QUANTIZATION_MATRIX
#      (type 22) carrying an identity (all-32) user matrix for level 3, and
#      sets QmDataPresent/user_qm in the encoder state.
#   2. The pinned dav2d **rejects** every QM OBU ("QM OBUs with user-defined
#      QMs are currently unsupported"), so the product path keeps the
#      spec-defined default matrices (`--still-qm`), which do decode
#      sample-exactly.  This is the documented rejection, not a silent
#      fallback.
#   3. The user-QM carrier is additive: with `--still-single` every transform
#      is >8x8, so §7.14.4 never consults UserQm; the sequence header and CLK
#      OBU are then byte-identical to `--still-qm` and the IVF is exactly the
#      default-QM stream plus the type-22 OBU.
#   4. Default `--still` streams stay byte-identical; QM and user-QM streams
#      are IVF-pinned.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
TMP="${TMPDIR:-/tmp}/av2enc_still_user_qm_gate.$$"
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
    echo "still_user_qm_gate: no av2enc binary; set AV2ENC_BIN=..." >&2
    exit 127
fi

DAV2D="${DAV2D:-}"
if [ -z "$DAV2D" ]; then
    for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
        if [ -x "$candidate" ]; then DAV2D="$candidate"; break; fi
    done
fi
if [ -z "$DAV2D" ] || [ ! -x "$DAV2D" ]; then
    echo "still_user_qm_gate: no dav2d binary (set DAV2D=...)" >&2
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

# list_obus <ivf> -> "type:length" per line
list_obus() {
    python3 - "$1" <<'PY'
import struct, sys
data = open(sys.argv[1], "rb").read()
if data[:4] != b"DKIF":
    raise SystemExit("not an IVF file")
size = struct.unpack_from("<I", data, 32)[0]
payload = data[44:44 + size]
i = 0
while i < len(payload):
    value = 0
    shift = 0
    while True:
        byte = payload[i]
        i += 1
        value |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            break
        shift += 7
    obu = payload[i:i + value]
    i += value
    obu_type = (obu[0] >> 2) & 0x1F
    print(f"{obu_type}:{len(obu)}")
PY
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

# run_case <fixture> <name> <expected_ivf_md5|-> <mode:qm|user> [args...]
run_case() {
    local fixture=$1 name=$2 expected=$3 mode=$4
    shift 4
    local input="$ROOT/$fixture"
    local out="$TMP/${name}.ivf"
    local dec="$TMP/${name}.dec.y4m"
    local recon="$TMP/${name}.recon.y4m"
    if [ ! -f "$input" ]; then
        echo "SKIP $name: missing $fixture" >&2
        return 0
    fi
    local tool_args=(--still)
    local recon_args=(--still)
    if [ "$mode" = "qm" ]; then
        tool_args+=(--still-qm)
        recon_args+=(--still-qm)
    elif [ "$mode" = "user" ]; then
        tool_args+=(--still-user-qm)
        recon_args+=(--still-user-qm)
    fi
    if ! "$AV2ENC_BIN" "$input" -o "$out" --frames 1 "${tool_args[@]}" "$@" \
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
    if [ "$mode" != "user" ]; then
        if ! "$DAV2D" -q -i "$out" -o "$dec" --threads 1 --strict 1 \
                >"$TMP/${name}.dec.log" 2>&1; then
            echo "FAIL $name: dav2d --strict 1 rejected the stream" >&2
            cat "$TMP/${name}.dec.log" >&2
            failures=$((failures + 1))
            return 0
        fi
        if [ -n "$STILL_RECON" ]; then
            if ! "$STILL_RECON" -i "$input" -o "$recon" "${recon_args[@]}" \
                    "$@" >"$TMP/${name}.recon.log" 2>&1; then
                echo "FAIL $name: still_recon_dump failed" >&2
                cat "$TMP/${name}.recon.log" >&2
                failures=$((failures + 1))
                return 0
            fi
            if ! compare_recon "$recon" "$dec" "$name" \
                    >>"$TMP/${name}.cmp.log" 2>&1; then
                echo "FAIL $name: encoder reconstruction is not decoder-exact" >&2
                cat "$TMP/${name}.cmp.log" >&2
                failures=$((failures + 1))
                return 0
            fi
        fi
        echo "PASS $name ($(wc -c <"$out" | tr -d ' ') B, md5 $expected)"
        pass=$((pass + 1))
        return 0
    fi
    # User QM: the pinned decoder must reject the OBU with the documented
    # message.  Any successful decode would mean the pin changed.
    if "$DAV2D" -q -i "$out" -o "$dec" --threads 1 --strict 1 \
            >"$TMP/${name}.dec.log" 2>&1; then
        echo "FAIL $name: dav2d unexpectedly accepted a user-QM OBU" >&2
        failures=$((failures + 1))
        return 0
    fi
    if ! grep -q "QM OBUs with user-defined QMs are currently unsupported" \
            "$TMP/${name}.dec.log"; then
        echo "FAIL $name: rejection message changed" >&2
        cat "$TMP/${name}.dec.log" >&2
        failures=$((failures + 1))
        return 0
    fi
    if ! list_obus "$out" | grep -q "^22:"; then
        echo "FAIL $name: no OBU_QUANTIZATION_MATRIX (type 22) in the stream" >&2
        failures=$((failures + 1))
        return 0
    fi
    echo "PASS $name (rejected by dav2d, $(wc -c <"$out" | tr -d ' ') B, md5 $expected)"
    pass=$((pass + 1))
}

# Default streams must not move.
run_case tests/data/basic_64x64_1f.y4m still_64_base \
    2b31e378787363641c836fdf563bbd0e - 
run_case tests/data/basic_640x360_1f.y4m still_360_base \
    8ffbe059cfeac3e6ee255f3bb7cf72b8 -

# Spec-defined default matrices: decode exact.
run_case tests/data/basic_64x64_1f.y4m still_64_qm \
    e26cf28c5a8c73d5fc92b0d89d508ab9 qm
run_case tests/data/basic_640x360_1f.y4m still_360_qm \
    8f8521a1af468de3dd4482d43955f363 qm

# Real user QM OBU: produced, pinned, and rejected by the pinned decoder.
run_case tests/data/basic_64x64_1f.y4m still_64_user_qm \
    1d6158f41835cb62a914a2e7fc810c75 user
run_case tests/data/basic_640x360_1f.y4m still_360_user_qm \
    3c568831a501e4ee22af3fc61910393d user

# Additive carrier check: with only >8x8 transforms the sequence header and
# CLK OBU are identical, and the user stream is the default stream + type-22.
"$AV2ENC_BIN" "$ROOT/tests/data/basic_64x64_1f.y4m" -o "$TMP/qm_single.ivf" \
    --frames 1 --still --still-single --still-qm >/dev/null 2>&1
"$AV2ENC_BIN" "$ROOT/tests/data/basic_64x64_1f.y4m" -o "$TMP/user_single.ivf" \
    --frames 1 --still --still-single --still-user-qm >/dev/null 2>&1
if python3 - "$TMP/qm_single.ivf" "$TMP/user_single.ivf" <<'PY'
import struct, sys
def obus(path):
    data = open(path, "rb").read()
    size = struct.unpack_from("<I", data, 32)[0]
    payload = data[44:44 + size]
    i = 0
    out = {}
    while i < len(payload):
        value = 0
        shift = 0
        while True:
            byte = payload[i]
            i += 1
            value |= (byte & 0x7F) << shift
            if not (byte & 0x80):
                break
            shift += 7
        obu = payload[i:i + value]
        i += value
        obu_type = (obu[0] >> 2) & 0x1F
        out.setdefault(obu_type, []).append(obu)
    return out
a = obus(sys.argv[1])
b = obus(sys.argv[2])
assert 1 in a and 1 in b, "missing sequence header"
assert 4 in a and 4 in b, "missing CLK"
assert 22 not in a, "default-QM stream unexpectedly carries a QM OBU"
assert 22 in b, "user-QM stream is missing the QM OBU"
assert a[1] == b[1], "sequence header changed"
assert a[4] == b[4], "CLK OBU changed"
assert list(a) == [2, 1, 4], f"unexpected default OBU list {list(a)}"
assert list(b) == [2, 1, 22, 4], f"unexpected user OBU list {list(b)}"
print("user QM carrier is additive")
PY
then
    echo "PASS user_qm_carrier"
    pass=$((pass + 1))
else
    echo "FAIL user_qm_carrier" >&2
    failures=$((failures + 1))
fi

if [ "$failures" -ne 0 ]; then
    echo "still_user_qm_gate: $failures failed, $pass passed" >&2
    exit 1
fi
echo "still_user_qm_gate: $pass passed, 0 failed"
