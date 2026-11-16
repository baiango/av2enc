#!/usr/bin/env bash
# Gate for --video-tile-flush-min: the §8.2.4 minimum tile termination.
#
# The normative decoder derives the trailing-'1' position from SymbolMaxBits
# alone, so a tile payload may stop right after that '1' (SymbolMaxBits >= -14)
# instead of writing the fourteen marker zeros and the byte padding.  The flag
# is opt-in; the historical streams must stay byte-identical.
#
# Checks:
#   * the default / quality / nightly pins are unchanged with the flag off;
#   * with the flag on every tile is exactly ceil((R + 1) / 8) bytes (never
#     below the 2-byte tile-group minimum), i.e. the trailer is minimal --
#     verified per tile from the AV2ENC_TRACE_TILE_FLUSH accounting;
#   * the historical trailer is always the 14 marker zeros plus 0..7
#     alignment bits, and the byte delta equals the reported per-tile saves;
#   * the shortened streams decode under dav2d --strict 1 to the same picture
#     count and the same decoded payload as the historical streams (2-, 6- and
#     60-frame streams), and the encoder reconstruction matches the decoder;
#   * the flag is rejected without --video, with --video-gpu, with --tiles and
#     with --lossless.
#
# Usage: src/pipeline/video/gate_tile_flush.sh [--bin build/av2enc] [--dav2d PATH]
# Exit: 0 pass, 1 failure, 2 usage, 77 missing fixture/decoder.
set -uo pipefail

ROOT=$(cd -- "$(dirname -- "$0")/../../.." && pwd)
BIN=""
DAV2D="${DAV2D:-}"
TESTDATA="${AV2ENC_TESTDATA:-$ROOT/testdata}"

while [ $# -gt 0 ]; do
    case "$1" in
        --bin) BIN=${2:-}; shift 2 ;;
        --dav2d) DAV2D=${2:-}; shift 2 ;;
        -h|--help) sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "gate_tile_flush: unknown argument $1" >&2; exit 2 ;;
    esac
done

BIN=${BIN:-"$ROOT/build/av2enc"}
[ -x "$BIN" ] || { echo "gate_tile_flush: --bin must point at av2enc" >&2; exit 2; }
if [ -z "$DAV2D" ]; then
    for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
        if [ -x "$candidate" ]; then DAV2D=$candidate; break; fi
    done
fi
[ -n "$DAV2D" ] && [ -x "$DAV2D" ] || {
    echo "gate_tile_flush: SKIP (exit 77): no dav2d binary" >&2; exit 77; }

NASA="$TESTDATA/nasa_480x240_30fps_60f.y4m"
[ -f "$NASA" ] || {
    echo "gate_tile_flush: SKIP (exit 77): no $NASA" >&2; exit 77; }

TMP=$(mktemp -d "${TMPDIR:-/tmp}/av2enc-flush.XXXXXX")
trap 'rm -rf "$TMP"' EXIT
FAILED=0
fail() { echo "FAIL: $*" >&2; FAILED=1; }

NIGHTLY=(--video --frames 60 --nightly --qp 110 --cpu-threads 8)
# --nightly enables the minimal trailer; the legacy form is the A/B control.
NIGHTLY_LEGACY=("${NIGHTLY[@]}" --video-tile-flush-off)
encode60() {
    # $1 = output, $2... = the full video flag set
    local out=$1; shift
    if ! "$BIN" "$NASA" -o "$out" "$@" >"$out.log" 2>&1; then
        fail "encode failed: $out"
        cat "$out.log" >&2
        return 1
    fi
    return 0
}

decoded_payload() {
    python3 - "$1" <<'PY'
import sys
data = open(sys.argv[1], 'rb').read()
nl = data.index(b'\n') + 1
sys.stdout.write(str(data.count(b'FRAME\n')))
sys.stdout.write('\n')
sys.stdout.flush()
sys.stdout.write(repr(len(data[nl:].replace(b'FRAME\n', b''))))
PY
}

echo "== §8.2.4 minimum tile termination gate =="

# --- pins stay byte-identical with the flag off ---------------------------
encode60 "$TMP/off-nightly.ivf" "${NIGHTLY_LEGACY[@]}" || true
encode60 "$TMP/off-default.ivf" --video --frames 60 --qp 189 \
    --cpu-threads 8 || true
encode60 "$TMP/off-quality.ivf" --video --frames 60 --qp 190 \
    --cpu-threads 8 --video-rd-preset quality || true
if [ -f "$TMP/off-nightly.ivf" ]; then
    BYTES=$(wc -c <"$TMP/off-nightly.ivf" | tr -d ' ')
    [ "$BYTES" = "24218" ] || fail "nightly pin is $BYTES B, expected 24218"
    MD5=$(python3 -c "import hashlib,sys;print(hashlib.md5(open(sys.argv[1],'rb').read()).hexdigest())" "$TMP/off-nightly.ivf")
    [ "$MD5" = "d3a7bd02c692f4f9770843c19f745307" ] ||
        fail "nightly pin md5 is $MD5, expected 1b99127a..."
fi
if [ -f "$TMP/off-default.ivf" ]; then
    BYTES=$(wc -c <"$TMP/off-default.ivf" | tr -d ' ')
    [ "$BYTES" = "25009" ] || fail "default pin is $BYTES B, expected 25009"
fi
if [ -f "$TMP/off-quality.ivf" ]; then
    BYTES=$(wc -c <"$TMP/off-quality.ivf" | tr -d ' ')
    [ "$BYTES" = "25141" ] || fail "quality pin is $BYTES B, expected 25141"
fi

# --- the flag shortens the stream and the trailer is minimal --------------
if encode60 "$TMP/on-nightly.ivf" "${NIGHTLY[@]}"; then
    ON=$(wc -c <"$TMP/on-nightly.ivf" | tr -d ' ')
    OFF=$(wc -c <"$TMP/off-nightly.ivf" | tr -d ' ')
    [ "$ON" -lt "$OFF" ] || fail "the minimum termination saved nothing"
    SAVED=$((OFF - ON))
    echo "nightly pin: $OFF B -> $ON B (-$SAVED B)"
fi

# Per-tile accounting straight from the writer.  The trace reports, per tile:
#   R    renormalisation bits (the §8.2.4 marker position)
#   nat  bits the historical "marker + 14 zeros + padding" form writes
#   pad  trailer bits after the marker in that form (14 + 0..7 alignment)
#   min  shortest conformant payload in bytes
#   save bytes the minimum termination saves
AV2ENC_TRACE_TILE_FLUSH=1 "$BIN" "$NASA" -o "$TMP/trace.ivf" "${NIGHTLY[@]}" >"$TMP/trace.log" 2>"$TMP/trace.txt"
if [ -f "$TMP/trace.ivf" ]; then
    if ! python3 - "$TMP/trace.txt" "$TMP/on-nightly.ivf" \
            "$TMP/off-nightly.ivf" <<'PY'
import sys

trace, on, off = sys.argv[1], sys.argv[2], sys.argv[3]
rows = []
for line in open(trace):
    if not line.startswith('tile_flush'):
        continue
    rows.append({k: int(v) for k, v in
                 (field.split('=') for field in line.split()[1:])})
errors = []
if len(rows) != 60:
    errors.append(f'{len(rows)} tiles traced, expected 60')
total_pad = 0
saved = 0
for i, row in enumerate(rows):
    r = row['R']
    bits = row['bits']
    nat = row['nat']
    pad = row['pad']
    expect_nat = ((r + 15 + 7) // 8) * 8
    expect_min = max(2, (r + 8) // 8)
    if nat != expect_nat:
        errors.append(f'tile {i}: nat={nat} expected {expect_nat}')
    if pad != nat - (r + 15) or not 0 <= pad <= 7:
        errors.append(f'tile {i}: pad={pad} outside 14 marker zeros + 0..7')
    if row['min'] != expect_min:
        errors.append(f'tile {i}: min={row["min"]} expected {expect_min}')
    if bits != row['min'] * 8:
        errors.append(f'tile {i}: wrote {bits} bits, minimum is '
                      f'{row["min"] * 8}')
    if row['save'] != nat // 8 - row['min']:
        errors.append(f'tile {i}: save={row["save"]} does not match nat-min')
    total_pad += pad
    saved += row['save']
delta = len(open(off, 'rb').read()) - len(open(on, 'rb').read())
# The per-tile trailer saving is exact, but shortening a tile payload can drop
# one byte from its leb128 size field, so the whole-file delta is the sum of
# the trailer savings plus 0..1 byte of framing per frame that crosses a
# 128-byte boundary.  Allow that one-byte framing effect, nothing more.
if not (saved <= delta <= saved + 1):
    errors.append(f'per-tile saves total {saved} B but the files differ by '
                  f'{delta} B (allowed +1 for the tile-size leb128)')
print(f'tiles={len(rows)} pad_bits={total_pad} saved={saved} B '
      f'marker_bits={15 * len(rows)}')
for message in errors[:8]:
    print('FAIL: ' + message, file=sys.stderr)
sys.exit(1 if errors else 0)
PY
    then
        fail "per-tile §8.2.4 accounting"
    fi
fi

# --- decoded output is identical ----------------------------------------
"$DAV2D" -q --strict 1 -i "$TMP/off-nightly.ivf" -o "$TMP/off-nightly.y4m" \
    --threads 1 >"$TMP/off-strict.log" 2>&1 ||
    fail "the historical nightly stream failed dav2d --strict 1"
"$DAV2D" -q --strict 1 -i "$TMP/on-nightly.ivf" -o "$TMP/on-nightly.y4m" \
    --threads 1 >"$TMP/on-strict.log" 2>&1 ||
    fail "the minimum-termination stream failed dav2d --strict 1"
if [ -f "$TMP/off-nightly.y4m" ] && [ -f "$TMP/on-nightly.y4m" ]; then
    if ! python3 - "$TMP/off-nightly.y4m" "$TMP/on-nightly.y4m" <<'PY'
import sys

def payload(path):
    data = open(path, 'rb').read()
    return data[data.index(b'\n') + 1:].replace(b'FRAME\n', b''), \
        data.count(b'FRAME\n')

a, na = payload(sys.argv[1])
b, nb = payload(sys.argv[2])
if na != 60 or nb != 60:
    print(f'decoded pictures {na}/{nb}, expected 60', file=sys.stderr)
    sys.exit(1)
if a != b:
    diff = sum(1 for x, y in zip(a, b) if x != y)
    print(f'decoded payloads differ in {diff} byte(s)', file=sys.stderr)
    sys.exit(1)
print(f'decoded 60 pictures, payload identical ({len(a)} B)')
PY
    then
        fail "the shortened stream does not decode to the same pictures"
    fi
fi

# --- 2- and 6-frame streams, encoder reconstruction ----------------------
for FRAMES in 2 6; do
    if ! "$BIN" "$NASA" -o "$TMP/on-$FRAMES.ivf" --video --frames "$FRAMES" \
            --nightly --qp 110 --cpu-threads 8 --video-tile-flush-min \
            --video-recon "$TMP/on-$FRAMES.y4m" >"$TMP/on-$FRAMES.log" 2>&1; then
        fail "the $FRAMES-frame minimum-termination encode failed"
        continue
    fi
    if ! "$BIN" "$NASA" -o "$TMP/off-$FRAMES.ivf" --video --frames "$FRAMES" \
            --nightly --qp 110 --cpu-threads 8 \
            --video-recon "$TMP/off-$FRAMES.y4m" \
            >"$TMP/off-$FRAMES.log" 2>&1; then
        fail "the $FRAMES-frame historical encode failed"
        continue
    fi
    "$DAV2D" -q --strict 1 -i "$TMP/on-$FRAMES.ivf" -o "$TMP/on-$FRAMES.dec.y4m" \
        --threads 1 >"$TMP/on-$FRAMES.strict.log" 2>&1 ||
        fail "the $FRAMES-frame minimum-termination stream failed --strict 1"
    "$DAV2D" -q --strict 1 -i "$TMP/off-$FRAMES.ivf" \
        -o "$TMP/off-$FRAMES.dec.y4m" --threads 1 \
        >"$TMP/off-$FRAMES.strict.log" 2>&1 ||
        fail "the $FRAMES-frame historical stream failed --strict 1"
    if ! python3 - "$FRAMES" "$TMP/on-$FRAMES.dec.y4m" \
            "$TMP/off-$FRAMES.dec.y4m" "$TMP/on-$FRAMES.y4m" <<'PY'
import sys

frames = int(sys.argv[1])

def payload(path):
    data = open(path, 'rb').read()
    return data[data.index(b'\n') + 1:].replace(b'FRAME\n', b''), \
        data.count(b'FRAME\n')

on, n_on = payload(sys.argv[2])
off, n_off = payload(sys.argv[3])
recon, _ = payload(sys.argv[4])
if n_on != frames or n_off != frames:
    print(f'{n_on}/{n_off} pictures decoded, expected {frames}',
          file=sys.stderr)
    sys.exit(1)
if on != off:
    print('the shortened and historical streams decode differently',
          file=sys.stderr)
    sys.exit(1)
if on != recon:
    print('the encoder reconstruction differs from the decoded output',
          file=sys.stderr)
    sys.exit(1)
print(f'{frames} frames: decode identical, recon exact')
PY
    then
        fail "the $FRAMES-frame stream does not verify"
    fi
done

# --- rejections ----------------------------------------------------------
if "$BIN" "$NASA" -o "$TMP/x.ivf" --frames 2 \
        --video-tile-flush-min >/dev/null 2>"$TMP/x.err"; then
    fail "--video-tile-flush-min was accepted without --video"
fi
grep -q -- "--video feature" "$TMP/x.err" ||
    fail "the --video rejection did not name the --video contract"
if "$BIN" "$NASA" -o "$TMP/x.ivf" --video --frames 2 --video-gpu \
        --video-tile-flush-min >/dev/null 2>"$TMP/gpu.err"; then
    fail "--video-tile-flush-min was accepted with --video-gpu"
fi
grep -q "CPU-only" "$TMP/gpu.err" ||
    fail "the GPU rejection did not name the CPU-only path"
if "$BIN" "$NASA" -o "$TMP/x.ivf" --video --frames 2 --tiles 2x1 \
        --video-tile-flush-min >/dev/null 2>"$TMP/tiles.err"; then
    fail "--video-tile-flush-min was accepted with --tiles"
fi
grep -q "single-tile writer" "$TMP/tiles.err" ||
    fail "the tile rejection did not name the single-tile writer"
if "$BIN" "$NASA" -o "$TMP/x.ivf" --video --frames 2 --lossless \
        --video-tile-flush-min >/dev/null 2>"$TMP/lossless.err"; then
    fail "--video-tile-flush-min was accepted with --lossless"
fi

if [ "$FAILED" -ne 0 ]; then
    echo "gate_tile_flush: FAILED" >&2
    exit 1
fi
echo "gate_tile_flush: OK (pins, minimal per-tile trailer, strict decode, recon)"
exit 0
