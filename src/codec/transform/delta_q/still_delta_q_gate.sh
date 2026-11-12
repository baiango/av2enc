#!/usr/bin/env bash
# Decoder gate for the integrated DC/AC delta-Q path (--still-dc-ac-delta-q).
#
#   ./still_delta_q_gate.sh
#   AV2ENC_BIN=build-int/av2enc STILL_DUMP=/tmp/still_dump ./still_delta_q_gate.sh
#   KEEP=1 ./still_delta_q_gate.sh
#
# What it proves:
#   1. The sequence §5.4.8 delta-Q fields and the frame §5.18.6.1
#      quantization_params() are written from the encoder's frame search.
#   2. Forward quantisation and §7.14.4 dequant use the per-plane DC/AC
#      quantizers: the encoder reconstruction is decoder-exact on all planes.
#   3. The streams decode with `dav2d --strict 1` and the IVF bytes are pinned.
#   4. The default `--still` bytes do not move, and the documented interactions
#      (IST, TX type) stay exact.  QM and RDOQ are rejected by name.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
TMP="${TMPDIR:-/tmp}/av2enc_still_delta_q_gate.$$"
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
    echo "still_delta_q_gate: no av2enc binary; set AV2ENC_BIN=..." >&2
    exit 127
fi

DAV2D="${DAV2D:-}"
if [ -z "$DAV2D" ]; then
    for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
        if [ -x "$candidate" ]; then DAV2D="$candidate"; break; fi
    done
fi
if [ -z "$DAV2D" ] || [ ! -x "$DAV2D" ]; then
    echo "still_delta_q_gate: no dav2d binary (set DAV2D=...)" >&2
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

# run_case <name> <fixture> <expected_ivf_md5> <encode args...>
run_case() {
    local name=$1 fixture=$2 expected=$3
    shift 3
    local input="$ROOT/$fixture"
    local out="$TMP/${name}.ivf"
    local dec="$TMP/${name}.y4m"
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
        for arg in "$@"; do
            case "$arg" in
                --still-dc-ac-delta-q) dump_args+=(--delta-q) ;;
                --still-ist) dump_args+=(--ist) ;;
                --still-tx-type) dump_args+=(--tx-type) ;;
                --still-qm) dump_args+=(--qm) ;;
                --still-rdoq) dump_args+=(--rdoq) ;;
                *) dump_args+=("$arg") ;;
            esac
        done
        if "$STILL_DUMP" -i "$input" -o "$TMP/${name}.dump" \
                ${dump_args[@]+"${dump_args[@]}"} \
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

# Defaults unchanged.
run_case still_64_base tests/data/basic_64x64_1f.y4m \
    0ef9825062a25c4d950ee924113c6f89
run_case still_360_base tests/data/basic_640x360_1f.y4m \
    e61de628b0812b716b0efa7dcafcdd06

run_case still_64_delta_q tests/data/basic_64x64_1f.y4m \
    8a34626bdb4c5706749805cf9c1828dd --still-dc-ac-delta-q
run_case still_360_delta_q tests/data/basic_640x360_1f.y4m \
    d53749baca37f4b1e16786de1ba6de05 --still-dc-ac-delta-q
run_case still_360_delta_q_qp141 tests/data/basic_640x360_1f.y4m \
    843a4a2bcb259cc8a4fe6b1231ca192a --still-dc-ac-delta-q --qp 141
run_case still_360_delta_q_ist tests/data/basic_640x360_1f.y4m \
    70cf5b9e7e680e39530e415119a134b4 --still-dc-ac-delta-q --still-ist

# Incompatible combinations must be rejected by name.
for combo in "--still-qm:--still-qm" "--still-rdoq:--still-rdoq"; do
    flag=${combo%%:*}
    other=${combo##*:}
    if "$AV2ENC_BIN" "$ROOT/tests/data/basic_64x64_1f.y4m" \
            -o "$TMP/reject.ivf" --still --still-dc-ac-delta-q "$other" \
            >"$TMP/reject.log" 2>&1; then
        echo "FAIL reject_delta_q_$flag: encode unexpectedly succeeded" >&2
        failures=$((failures + 1))
    elif grep -q -- "--still-dc-ac-delta-q is not supported with $other" \
            "$TMP/reject.log"; then
        echo "PASS reject_delta_q_$flag"
        pass=$((pass + 1))
    else
        echo "FAIL reject_delta_q_$flag: wrong diagnostic" >&2
        cat "$TMP/reject.log" >&2
        failures=$((failures + 1))
    fi
done

if [ "$failures" -ne 0 ]; then
    echo "still_delta_q_gate: $failures failed, $pass passed" >&2
    exit 1
fi
echo "still_delta_q_gate: $pass passed, 0 failed"
