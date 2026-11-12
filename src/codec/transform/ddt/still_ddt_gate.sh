#!/usr/bin/env bash
# DDT / DDTX / FDDT gate for the completion wave.
#
#   ./still_ddt_gate.sh
#   AV2ENC_BIN=build-int/av2enc ./still_ddt_gate.sh
#
# DDT is inter-only in AV2 v1.0.0:
#   * §5.4.8 omits enable_inter_ddt when single_picture_header_flag = 1, and
#     there is no frame/block DDT syntax;
#   * §7.15.4.1 substitutes DDTX/FDDT only when is_inter is true.
# Therefore a still picture cannot signal it.  This gate proves that the
# product CLI rejects --still-ddt by name, that the pinned `--still` streams
# stay byte-identical, and then runs the existing M2 inter-DDT component gate
# (`gate/build_gate.sh`), which decodes a real inter stream with the pinned
# dav2d and requires sample-exact DDTX/FDDT reconstruction.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "$0")" && pwd)"
ROOT="$HERE"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done

AV2ENC_BIN="${AV2ENC_BIN:-}"
if [ -z "$AV2ENC_BIN" ]; then
    for candidate in "$ROOT/build-int/av2enc" "$ROOT/build-m1/av2enc" \
                     "$ROOT/build/av2enc"; do
        if [ -x "$candidate" ]; then AV2ENC_BIN="$candidate"; break; fi
    done
fi
if [ -z "$AV2ENC_BIN" ] || [ ! -x "$AV2ENC_BIN" ]; then
    echo "still_ddt_gate: no av2enc binary; set AV2ENC_BIN=..." >&2
    exit 127
fi

DAV2D="${DAV2D:-}"
if [ -z "$DAV2D" ]; then
    for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
        if [ -x "$candidate" ]; then DAV2D="$candidate"; break; fi
    done
fi
if [ -z "$DAV2D" ] || [ ! -x "$DAV2D" ]; then
    echo "still_ddt_gate: no dav2d binary (set DAV2D=...)" >&2
    exit 77
fi
export DAV2D

TMP="${TMPDIR:-/tmp}/av2enc_still_ddt_gate.$$"
mkdir -p "$TMP"
cleanup() {
    if [ "${KEEP:-0}" != "1" ]; then rm -rf "$TMP"; fi
}
trap cleanup EXIT

md5_of() {
    md5 -q "$1" 2>/dev/null || md5sum "$1" | awk '{print $1}'
}

# The DDT component gate compiles video_frame.cpp from source; the product
# archive supplies the post/tile/rate-control symbols that TU references after
# the completion-wave merges.  Point it at the archive next to --bin.
AV2ENC_CORE_LIB="${AV2ENC_CORE_LIB:-$(dirname "$AV2ENC_BIN")/libav2enc_core.a}"
if [ -f "$AV2ENC_CORE_LIB" ]; then export AV2ENC_CORE_LIB; fi

failures=0
pass=0

# Pinned default streams must not move.
check_pin() {
    local fixture=$1 expected=$2
    local out="$TMP/base.ivf"
    if ! "$AV2ENC_BIN" "$ROOT/$fixture" -o "$out" --frames 1 --still \
            >"$TMP/base.log" 2>&1; then
        echo "FAIL base($fixture): encode failed" >&2
        failures=$((failures + 1))
        return 0
    fi
    local got
    got=$(md5_of "$out")
    if [ "$got" = "$expected" ]; then
        echo "PASS base($fixture) ($got)"
        pass=$((pass + 1))
    else
        echo "FAIL base($fixture): md5 $got, expected $expected" >&2
        failures=$((failures + 1))
    fi
}
check_pin tests/data/basic_64x64_1f.y4m 0ef9825062a25c4d950ee924113c6f89
check_pin tests/data/basic_640x360_1f.y4m e61de628b0812b716b0efa7dcafcdd06

# --still-ddt is not signallable and must be rejected with the spec reason.
if "$AV2ENC_BIN" "$ROOT/tests/data/basic_64x64_1f.y4m" \
        -o "$TMP/ddt.ivf" --frames 1 --still --still-ddt \
        >"$TMP/ddt.log" 2>&1; then
    echo "FAIL reject_still_ddt: encode unexpectedly succeeded" >&2
    failures=$((failures + 1))
else
    if grep -q -- "--still-ddt is not signallable in a still picture" \
            "$TMP/ddt.log"; then
        echo "PASS reject_still_ddt"
        pass=$((pass + 1))
    else
        echo "FAIL reject_still_ddt: wrong diagnostic" >&2
        cat "$TMP/ddt.log" >&2
        failures=$((failures + 1))
    fi
fi

# The component gate is the end-to-end DDT proof: an M2 inter stream with
# enable_inter_ddt = 1, decode + sample-exact comparison for DDTX and FDDT.
if ! KEEP=1 OUT="$TMP/ddt_component" \
        bash "$HERE/gate/build_gate.sh" --quick >"$TMP/component.log" 2>&1; then
    echo "FAIL ddt_component_gate: see $TMP/component.log" >&2
    tail -20 "$TMP/component.log" >&2
    failures=$((failures + 1))
else
    echo "PASS ddt_component_gate (DDTX + FDDT decode exact)"
    pass=$((pass + 1))
fi

if [ "$failures" -ne 0 ]; then
    echo "still_ddt_gate: $failures failed, $pass passed" >&2
    exit 1
fi
echo "still_ddt_gate: $pass passed, 0 failed"
