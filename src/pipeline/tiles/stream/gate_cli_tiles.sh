#!/usr/bin/env bash
# CLI gate for multi-tile stills (--tiles).
#
# Every case encodes through the real `av2enc` CLI, decodes with the pinned
# dav2d (`--strict 1 --threads 1`), and requires the decoded Y/U/V planes to be
# byte-identical to the encoder reconstruction dumped by `--recon`. The same
# IVF also runs `tools/validate_dav2d.sh` (the repository decode policy).
#
# Also checked:
#   * `--still` and `--still --tiles 1x1` produce the same pinned bytes;
#   * the tiled writer rejects unsupported combinations by name.
#
# Usage:
#   src/pipeline/tiles/stream/gate_cli_tiles.sh --bin build-int/av2enc [--quick]
#   DAV2D=/path/to/dav2d src/pipeline/tiles/stream/gate_cli_tiles.sh --bin ...
#
#   --quick   skip the 1080p layouts (testdata is gitignored); the 640x360
#             layouts, default-stream check and rejections still run.
#
# Exit codes: 0 pass, 1 failure, 2 usage, 77 missing fixture/decoder.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${HERE}/../../../.." && pwd)"
BIN=""
DAV2D="${DAV2D:-}"
TESTDATA="${AV2ENC_TESTDATA:-$ROOT/testdata}"
QUICK=0

while [ $# -gt 0 ]; do
    case "$1" in
        --bin) BIN=$2; shift 2 ;;
        --dav2d) DAV2D=$2; shift 2 ;;
        --quick) QUICK=1; shift ;;
        -h|--help)
            sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "gate_cli_tiles: unknown argument $1" >&2; exit 2 ;;
    esac
done

if [ -z "$BIN" ] || [ ! -x "$BIN" ]; then
    echo "gate_cli_tiles: --bin must point at an av2enc executable" >&2
    exit 2
fi
if [ -z "$DAV2D" ]; then
    for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
        if [ -x "$candidate" ]; then DAV2D=$candidate; break; fi
    done
fi
if [ -z "$DAV2D" ] || [ ! -x "$DAV2D" ]; then
    echo "gate_cli_tiles: no dav2d binary (set DAV2D=...)" >&2
    exit 77
fi
export DAV2D
if ! command -v python3 >/dev/null 2>&1; then
    echo "gate_cli_tiles: python3 is required for the recon comparison" >&2
    exit 77
fi

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

F360="$ROOT/tests/data/basic_640x360_1f.y4m"
F1080="$TESTDATA/shine_1920x1080_10f.y4m"
if [ ! -f "$F360" ]; then
    echo "SKIP: missing $F360" >&2
    exit 77
fi

md5_of() {
    if command -v md5 >/dev/null 2>&1; then
        md5 -q "$1"
    else
        md5sum "$1" | awk '{print $1}'
    fi
}

failures=0
pass=0

fail() {
    echo "FAIL: $*" >&2
    failures=$((failures + 1))
}

# compare_recon <recon.y4m> <decoded.y4m> <label>
compare_recon() {
    python3 - "$1" "$2" "$3" <<'PY'
import sys

def parse(path):
    data = open(path, "rb").read()
    newline = data.index(b"\n")
    header = data[:newline].decode("ascii")
    fields = header.split()
    width = int(next(f[1:] for f in fields if f.startswith("W")))
    height = int(next(f[1:] for f in fields if f.startswith("H")))
    pos = newline + 1
    newline = data.index(b"\n", pos)
    if not data[pos:newline].startswith(b"FRAME"):
        raise SystemExit("not a frame header")
    return width, height, data[newline + 1:]

wa, ha, a = parse(sys.argv[1])
wb, hb, b = parse(sys.argv[2])
label = sys.argv[3]
if (wa, ha) != (wb, hb):
    print(f"FAIL {label}: decoded {wb}x{hb} vs recon {wa}x{ha}", file=sys.stderr)
    sys.exit(1)
expected = wa * ha * 3 // 2
if len(a) != expected or len(b) != expected:
    print(f"FAIL {label}: frame bytes {len(a)} vs {len(b)}, expected {expected}",
          file=sys.stderr)
    sys.exit(1)
if a != b:
    diffs = sum(1 for x, y in zip(a, b) if x != y)
    print(f"FAIL {label}: {diffs} sample(s) differ", file=sys.stderr)
    sys.exit(1)
PY
}

# run_case <fixture> <layout> <label>
run_case() {
    local fixture=$1 layout=$2 label=$3
    local input="$ROOT/$fixture"
    if [ ! -f "$input" ]; then
        echo "SKIP: missing $fixture" >&2
        return 77
    fi
    local ivf="$TMP/${label}.ivf"
    local recon="$TMP/${label}-recon.y4m"
    local dec="$TMP/${label}-dec.y4m"

    if ! "$BIN" "$input" -o "$ivf" --still --no-deblock --tiles "$layout" \
            --recon "$recon" >/dev/null; then
        fail "$label: encode failed"
        return 0
    fi
    if ! "$ROOT/tools/validate_dav2d.sh" "$ivf" >/dev/null; then
        fail "$label: validate_dav2d.sh failed"
        return 0
    fi
    if ! "$DAV2D" --strict 1 -q -i "$ivf" -o "$dec" --threads 1 >/dev/null 2>&1; then
        fail "$label: dav2d --strict 1 failed"
        return 0
    fi
    if ! compare_recon "$recon" "$dec" "$label"; then
        failures=$((failures + 1))
        return 0
    fi
    echo "PASS $label ($layout, $(wc -c <"$ivf" | tr -d ' ') B)"
    pass=$((pass + 1))
}

# expect_reject <needle> <label> <encode args...>
expect_reject() {
    local needle=$1 label=$2
    shift 2
    if "$BIN" "$@" >/dev/null 2>"$TMP/reject.err"; then
        fail "$label: expected a rejection"
        return 0
    fi
    if ! grep -qF -e "$needle" "$TMP/reject.err"; then
        echo "--- $label stderr ---" >&2
        cat "$TMP/reject.err" >&2
        fail "$label: error does not mention '$needle'"
        return 0
    fi
    echo "PASS $label (rejected by name)"
    pass=$((pass + 1))
}

echo "encoder: $BIN"
echo "decoder: $($DAV2D --version 2>&1 | tail -1)"

# --- default stream must not move -----------------------------------------
"$BIN" "$F360" -o "$TMP/still.ivf" --still >/dev/null
still_md5=$(md5_of "$TMP/still.ivf")
if [ "$still_md5" != "e61de628b0812b716b0efa7dcafcdd06" ]; then
    fail "default --still 640x360 MD5 moved: $still_md5"
else
    echo "PASS default --still 640x360 MD5 $still_md5"
    pass=$((pass + 1))
fi
"$BIN" "$F360" -o "$TMP/t11.ivf" --still --tiles 1x1 >/dev/null
if ! cmp -s "$TMP/still.ivf" "$TMP/t11.ivf"; then
    fail "--tiles 1x1 is not byte-identical to --still"
else
    echo "PASS --tiles 1x1 is byte-identical to --still"
    pass=$((pass + 1))
fi

# --- exact-recon matrix (deblock off is enforced by the CLI) --------------
set +e
for layout in 2x1 2x2 4x2 8x2; do
    run_case tests/data/basic_640x360_1f.y4m "$layout" "360_${layout}"
done
if [ "$QUICK" -eq 0 ]; then
    for layout in 2x1 2x2 4x2 8x2 16x2; do
        run_case "$F1080" "$layout" "1080_${layout}"
    done
else
    echo "gate_cli_tiles: --quick, skipping 1080p layouts"
fi
set -e

# --- rejected combinations stay rejected by name --------------------------
set +e
expect_reject "--tiles requires --no-deblock" "tiles_needs_no_deblock" \
    "$F360" -o "$TMP/rej1.ivf" --still --tiles 2x2
expect_reject "--tiles > 1 requires --still" "tiles_needs_still" \
    "$F360" -o "$TMP/rej2.ivf" --tiles 2x2 --no-deblock
expect_reject "does not implement --still-cdef" "tiles_rejects_cdef" \
    "$F360" -o "$TMP/rej3.ivf" --still --no-deblock --tiles 2x2 --still-cdef
expect_reject "not representable" "tiles_rejects_bad_grid" \
    "$F360" -o "$TMP/rej4.ivf" --still --no-deblock --tiles 16x2
expect_reject "--tiles must be" "tiles_rejects_unknown_layout" \
    "$F360" -o "$TMP/rej5.ivf" --still --no-deblock --tiles 3x3
expect_reject "8-bit 4:2:0 only" "tiles_rejects_10bit" \
    "$F360" -o "$TMP/rej6.ivf" --still --no-deblock --tiles 2x2 \
    --bit-depth 10
set -e

echo
if [ "$failures" -ne 0 ]; then
    echo "gate_cli_tiles: $failures failure(s), $pass passed" >&2
    exit 1
fi
echo "gate_cli_tiles: $pass/$pass cases passed"
