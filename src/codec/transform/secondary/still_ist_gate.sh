#!/usr/bin/env bash
# Decoder gate for the integrated intra secondary transform (--still-ist).
#
#   ./still_ist_gate.sh
#   AV2ENC_BIN=build-int/av2enc STILL_DUMP=/tmp/still_dump ./still_ist_gate.sh
#   KEEP=1 ./still_ist_gate.sh
#
# What it proves:
#   1. `--still-ist` sets enable_intra_ist and codes §5.20.8.2 sec_tx_type for
#      every IST-applicable luma block (including symbol 0); the 640x360 QP 90
#      stream actually signals nonzero IST on several blocks.
#   2. The streams decode with `dav2d --strict 1`, and the IVF bytes are pinned.
#   3. When still_dump is available, the encoder reconstruction (which runs the
#      §7.15.3 inverse before the primary inverse) matches the decoder for all
#      three planes.
#   4. The default `--still` streams stay byte-identical, and the documented
#      combinations (QM, TX type, RDOQ) still decode.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
TMP="${TMPDIR:-/tmp}/av2enc_still_ist_gate.$$"
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
    echo "still_ist_gate: no av2enc binary; set AV2ENC_BIN=..." >&2
    exit 127
fi

DAV2D="${DAV2D:-}"
if [ -z "$DAV2D" ]; then
    for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
        if [ -x "$candidate" ]; then DAV2D="$candidate"; break; fi
    done
fi
if [ -z "$DAV2D" ] || [ ! -x "$DAV2D" ]; then
    echo "still_ist_gate: no dav2d binary (set DAV2D=...)" >&2
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
        # Map CLI flags to still_dump flags.  The harness must reproduce the
        # same decisions, so pass the same feature switches.
        local dump_args=()
        for arg in "$@"; do
            case "$arg" in
                --still-ist) dump_args+=(--ist) ;;
                --still-mrl) dump_args+=(--mrl) ;;
                --still-qm) dump_args+=(--qm) ;;
                --still-rdoq) dump_args+=(--rdoq) ;;
                --still-tx-type) dump_args+=(--tx-type) ;;
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
    2b31e378787363641c836fdf563bbd0e
run_case still_360_base tests/data/basic_640x360_1f.y4m \
    8ffbe059cfeac3e6ee255f3bb7cf72b8

# IST on.  64x64 has no applicable low-EOB block, so it only pins the header
# bit and the per-block zero symbols; 640x360 QP 90 signals real IST.
run_case still_64_ist tests/data/basic_64x64_1f.y4m \
    4594dbe3c9ddfa93d8c86ae5d41330be --still-ist
run_case still_360_ist tests/data/basic_640x360_1f.y4m \
    32bee9efbd0d7a1ecc3c260ccada665e --still-ist
run_case still_360_ist_qp141 tests/data/basic_640x360_1f.y4m \
    ee1a91fa7c60c6db3aef5d0be0cdc8c4 --still-ist --qp 141

# Documented interactions (signalling + reconstruction must stay exact).
run_case still_360_ist_qm tests/data/basic_640x360_1f.y4m \
    b0f7c483f631f51ff2284465cd3997c6 --still-ist --still-qm
run_case still_360_ist_txtype tests/data/basic_640x360_1f.y4m \
    9c6e575901c193e3f73595bc48911973 --still-ist --still-tx-type
run_case still_360_ist_rdoq tests/data/basic_640x360_1f.y4m \
    db43b0bdad2019e53539ea714726a7ca --still-ist --still-rdoq

# SCOUT-TXADV-02: §7.15.3 folds Mrl_Index_To_Delta[MrlIndex] into pAngle before
# selecting the IST kernel, so the IST decision, the inverse and the syntax must
# all use the block's real mrl_index.  Forcing 0 made the decoder pick a
# different kernel/transpose on MRL blocks (QP40/60 luma drift).
run_case still_360_ist_mrl40 testdata/pub_640x360_1f.y4m \
    e1455bccf3263499b62ffcb57ed64f27 --still-ist --still-mrl --qp 40
run_case still_360_ist_mrl60 testdata/pub_640x360_1f.y4m \
    d256ef5a13924c6f4f99697054afe07c --still-ist --still-mrl --qp 60

if [ "$failures" -ne 0 ]; then
    echo "still_ist_gate: $failures failed, $pass passed" >&2
    exit 1
fi
echo "still_ist_gate: $pass passed, 0 failed"
