#!/usr/bin/env bash
# Gate for the opt-in CPU video RD preset (--video-rd-preset quality).
#
# What it checks:
#   * the preset expands to exactly the documented flag set (same IVF bytes as
#     spelling every flag out), on a 4-frame clip;
#   * the flag-less stream is untouched: the pinned 480x240 60-frame default
#     stream keeps its recorded size and MD5 when the fixture is available;
#   * the preset stream decodes with the pinned dav2d under --strict 1;
#   * the whole-frame decoder output equals the encoder reconstruction
#     (video_cli_gate.sh compares every sample, so a loop-filter or TX-search
#     mismatch fails here);
#   * usage errors: an unknown preset name and a preset on a non---video run
#     exit 2.
#
# Usage:
#   tools/gate_video_rd_preset.sh [--bin build-rd/av2enc] [--dav2d PATH]
#
# Exit codes: 0 pass, 1 failure, 2 usage, 77 missing fixture/decoder.
set -uo pipefail

ROOT=$(cd -- "$(dirname -- "$0")/.." && pwd)
BIN=""
DAV2D="${DAV2D:-}"
TESTDATA="${AV2ENC_TESTDATA:-$ROOT/testdata}"

while [ $# -gt 0 ]; do
    case "$1" in
        --bin) BIN=${2:-}; shift 2 ;;
        --dav2d) DAV2D=${2:-}; shift 2 ;;
        -h|--help) sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "gate_video_rd_preset: unknown argument $1" >&2; exit 2 ;;
    esac
done

BIN=${BIN:-"$ROOT/build-rd/av2enc"}
[ -x "$BIN" ] || { echo "gate_video_rd_preset: --bin must point at av2enc" >&2; exit 2; }
if [ -z "$DAV2D" ]; then
    for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
        if [ -x "$candidate" ]; then DAV2D=$candidate; break; fi
    done
fi
[ -n "$DAV2D" ] && [ -x "$DAV2D" ] || {
    echo "gate_video_rd_preset: SKIP (exit 77): no dav2d binary" >&2; exit 77; }
export DAV2D

NASA="$TESTDATA/nasa_480x240_30fps_60f.y4m"
SHINE="$TESTDATA/shine_640x360_4f.y4m"
[ -f "$SHINE" ] || SHINE="$TESTDATA/shine_640x360_1f.y4m"
if [ ! -f "$NASA" ] && [ ! -f "$SHINE" ]; then
    echo "gate_video_rd_preset: SKIP (exit 77): no 4-frame clip in $TESTDATA" >&2
    exit 77
fi

TMP=$(mktemp -d "${TMPDIR:-/tmp}/av2enc-rd-preset.XXXXXX")
trap 'rm -rf "$TMP"' EXIT
FAILED=0
fail() { echo "FAIL: $*" >&2; FAILED=1; }

# The preset is documented as this exact expansion.  Keep the two in sync.
PRESET_FLAGS=(--video-inter-residual --video-deblock --video-ccso
              --video-tx-search --video-intra-full)

echo "== video RD preset gate =="

# --- usage errors ---------------------------------------------------------
if "$BIN" "$SHINE" -o "$TMP/reject.ivf" --video --frames 1 \
        --video-rd-preset bogus >/dev/null 2>"$TMP/reject1.err"; then
    fail "an unknown --video-rd-preset name was accepted"
fi
grep -q "video-rd-preset expects 'quality'" "$TMP/reject1.err" ||
    fail "unknown preset name did not report the accepted value"

# The rate-aware partition decision is CPU --video only and owns the base
# tree; reject the carriers whose trees it does not model.
if "$BIN" "$SHINE" -o "$TMP/reject.ivf" --still --video-partition-rd \
        >/dev/null 2>"$TMP/rejectp1.err"; then
    fail "--video-partition-rd was accepted on the --still writer"
fi
grep -q "pass --video" "$TMP/rejectp1.err" ||
    fail "--still did not report the --video-only partition rejection"

if "$BIN" "$SHINE" -o "$TMP/reject.ivf" --video --frames 1 \
        --video-partition-rd --video-sdp >/dev/null 2>"$TMP/rejectp2.err"; then
    fail "--video-partition-rd composed with --video-sdp"
fi
grep -q "does not compose with --video-sdp" "$TMP/rejectp2.err" ||
    fail "--video-sdp did not report the partition-cost rejection"

if "$BIN" "$SHINE" -o "$TMP/reject.ivf" --still --video-rd-preset quality \
        >/dev/null 2>"$TMP/reject2.err"; then
    fail "--video-rd-preset was accepted on the --still writer"
fi
grep -q "applies to the CPU --video encoder" "$TMP/reject2.err" ||
    fail "--still did not report the --video-only preset rejection"
[ -e "$TMP/reject.ivf" ] && fail "a rejected run created its output file"

# --- preset == explicit flag list ----------------------------------------
FRAMES=4
if ! "$BIN" "$SHINE" -o "$TMP/preset.ivf" --video --frames "$FRAMES" \
        --qp 189 --video-rd-preset quality >"$TMP/preset.log" 2>&1; then
    fail "the preset encode failed"
    cat "$TMP/preset.log" >&2
fi
if ! "$BIN" "$SHINE" -o "$TMP/explicit.ivf" --video --frames "$FRAMES" \
        --qp 189 "${PRESET_FLAGS[@]}" >"$TMP/explicit.log" 2>&1; then
    fail "the explicit-flags encode failed"
fi
if [ -f "$TMP/preset.ivf" ] && [ -f "$TMP/explicit.ivf" ]; then
    cmp -s "$TMP/preset.ivf" "$TMP/explicit.ivf" ||
        fail "the preset stream differs from the documented flag expansion"
fi

# --- flag-less 480x240 stream is untouched -------------------------------
if [ -f "$NASA" ]; then
    if ! "$BIN" "$NASA" -o "$TMP/default.ivf" --video --frames 60 \
            --qp 189 --cpu-threads 8 >"$TMP/default.log" 2>&1; then
        fail "the flag-less baseline encode failed"
    fi
    if [ -f "$TMP/default.ivf" ]; then
        BYTES=$(wc -c <"$TMP/default.ivf" | tr -d ' ')
        MD5=$(python3 - "$TMP/default.ivf" <<'PY'
import hashlib, sys
print(hashlib.md5(open(sys.argv[1], 'rb').read()).hexdigest())
PY
)
        [ "$BYTES" = "25009" ] ||
            fail "default 480x240 stream is $BYTES B, expected 25009 B"
        [ "$MD5" = "a1d0dba1259b20a4b628d4cd1e14c33c" ] ||
            fail "default 480x240 stream md5 is $MD5, expected a1d0dba1..."
    fi
fi

# --- decode + encoder-reconstruction comparison --------------------------
if ! bash "$ROOT/tools/validate_dav2d.sh" "$TMP/preset.ivf" \
        >"$TMP/validate.log" 2>&1; then
    fail "the preset stream did not decode"
    cat "$TMP/validate.log" >&2
fi
# The shared validator decodes without --strict; the spec-violation abort is
# checked here so a non-conforming header cannot pass on lenient parsing.
if ! "$DAV2D" -q --strict 1 -i "$TMP/preset.ivf" -o "$TMP/decoded.y4m" \
        --threads 1 >"$TMP/strict.log" 2>&1; then
    fail "the preset stream failed dav2d --strict 1"
    cat "$TMP/strict.log" >&2
elif [ ! -s "$TMP/decoded.y4m" ]; then
    fail "dav2d --strict 1 produced no output"
fi

# Whole-frame recon check: the encoder dump must match the decoder sample for
# sample (loop filters and the TX-search reconstruction included).
if ! "$BIN" "$SHINE" -o "$TMP/recon.ivf" --video --frames "$FRAMES" \
        --qp 189 --video-recon "$TMP/recon.y4m" --video-rd-preset quality \
        >"$TMP/recon-encode.log" 2>&1; then
    fail "the --video-recon encode failed"
elif ! python3 - "$TMP/recon.y4m" "$TMP/decoded.y4m" <<'PY'
import sys

def payload(path):
    data = open(path, 'rb').read()
    # Y4M: one header line, then FRAME markers interleaved with raw planes.
    nl = data.index(b'\n') + 1
    body = data[nl:].replace(b'FRAME\n', b'')
    return body

a, b = payload(sys.argv[1]), payload(sys.argv[2])
if len(a) != len(b):
    print(f"recon payload {len(a)} B != decoded {len(b)} B", file=sys.stderr)
    sys.exit(1)
if a != b:
    diff = sum(1 for x, y in zip(a, b) if x != y)
    print(f"recon differs from decoded output in {diff} byte(s)", file=sys.stderr)
    sys.exit(1)
print(f"recon matches decoded output ({len(a)} B)")
PY
then
    fail "decoder output differs from the encoder reconstruction"
fi

if [ "$FAILED" -ne 0 ]; then
    echo "gate_video_rd_preset: FAILED" >&2
    exit 1
fi
echo "gate_video_rd_preset: OK (preset expansion, default identity, decode, recon)"
exit 0
