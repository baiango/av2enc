#!/usr/bin/env bash
# Gate for the C++ lpl36 port against the Python reference implementation.
#
# The refresh planner consumes the C++ header-only metric, so the port must
# agree with the reference package on the same Y planes.  This gate:
#   * encodes a short lossy stream and decodes it with dav2d to Y4M;
#   * runs tools/lpl36_dump on the source/decoded luma planes;
#   * runs the Python `lpl36` package on the identical Y4M frames;
#   * requires the per-frame relative error |cpp-python| / |python| <= 1e-3;
#   * checks that identical planes score exactly 0.0 in both implementations.
#
# Set LPL36_SRC to the Python lpl36 package's src directory.  Unset, or a
# missing Python/cv2/lpl36/dav2d/fixture, => exit 77 (skip).
#
# Usage: src/pipeline/video/video_lpl36_parity_gate.sh [--bin build/av2enc]
#                                                      [--dav2d PATH]
#                                                      [--metric-bin PATH]
# Exit: 0 pass, 1 failure, 2 usage, 77 missing prerequisites.
set -uo pipefail

ROOT=$(cd -- "$(dirname -- "$0")/../../.." && pwd)
BIN=""
DAV2D="${DAV2D:-}"
METRIC=""
TESTDATA="${AV2ENC_TESTDATA:-$ROOT/testdata}"
LPL36_SRC="${LPL36_SRC:-}"

while [ $# -gt 0 ]; do
    case "$1" in
        --bin) BIN=${2:-}; shift 2 ;;
        --dav2d) DAV2D=${2:-}; shift 2 ;;
        --metric-bin) METRIC=${2:-}; shift 2 ;;
        -h|--help) sed -n '2,21p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "video_lpl36_parity_gate: unknown argument $1" >&2; exit 2 ;;
    esac
done

BIN=${BIN:-"$ROOT/build/av2enc"}
[ -x "$BIN" ] || {
    echo "video_lpl36_parity_gate: --bin must point at av2enc" >&2; exit 2; }
if [ -z "$METRIC" ]; then
    METRIC="$(cd -- "$(dirname -- "$BIN")" && pwd)/lpl36_dump"
fi
[ -x "$METRIC" ] || {
    echo "video_lpl36_parity_gate: SKIP (exit 77): no lpl36_dump next to av2enc" >&2
    exit 77; }
if [ -z "$DAV2D" ]; then
    for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
        if [ -x "$candidate" ]; then DAV2D=$candidate; break; fi
    done
fi
[ -n "$DAV2D" ] && [ -x "$DAV2D" ] || {
    echo "video_lpl36_parity_gate: SKIP (exit 77): no dav2d binary" >&2
    exit 77; }
if [ -z "$LPL36_SRC" ] || [ ! -d "$LPL36_SRC" ]; then
    echo "video_lpl36_parity_gate: SKIP (exit 77): set LPL36_SRC to the lpl36 src directory" >&2
    exit 77
fi
if ! PYTHONPATH="$LPL36_SRC${PYTHONPATH:+:$PYTHONPATH}" python3 -c \
        'import cv2, numpy, lpl36' >/dev/null 2>&1; then
    echo "video_lpl36_parity_gate: SKIP (exit 77): Python lpl36/cv2/numpy missing" >&2
    exit 77
fi

NASA="$TESTDATA/nasa_480x240_30fps_60f.y4m"
[ -f "$NASA" ] || {
    echo "video_lpl36_parity_gate: SKIP (exit 77): no $NASA" >&2
    exit 77; }

TMP=$(mktemp -d "${TMPDIR:-/tmp}/av2enc-lpl36-parity.XXXXXX")
trap 'rm -rf "$TMP"' EXIT
FAILED=0
fail() { echo "FAIL: $*" >&2; FAILED=1; }

FRAMES=4
echo "== lpl36 C++/Python parity gate =="

if ! "$BIN" "$NASA" -o "$TMP/parity.ivf" --video --frames "$FRAMES" \
        --qp 111 --cpu-threads 4 --nightly >"$TMP/enc.log" 2>&1; then
    fail "the parity encode failed"
    cat "$TMP/enc.log" >&2
fi
if ! "$DAV2D" -q --strict 1 -i "$TMP/parity.ivf" -o "$TMP/parity.y4m" \
        --threads 1 >"$TMP/dec.log" 2>&1; then
    fail "the parity stream failed dav2d --strict 1"
    cat "$TMP/dec.log" >&2
fi

if ! "$METRIC" --ref "$NASA" --test "$TMP/parity.y4m" --frames "$FRAMES" \
        >"$TMP/cpp.txt" 2>"$TMP/cpp.err"; then
    fail "lpl36_dump failed on the decoded stream"
    cat "$TMP/cpp.err" >&2
fi

# Identical planes: both implementations must report exactly 0.0.
if ! "$METRIC" --ref "$NASA" --test "$NASA" --frames "$FRAMES" \
        >"$TMP/ident.txt" 2>"$TMP/ident.err"; then
    fail "lpl36_dump failed on the identical-plane control"
    cat "$TMP/ident.err" >&2
fi
if ! python3 - "$TMP/ident.txt" <<'PY'
import re
import sys

bad = 0
for line in open(sys.argv[1]):
    m = re.search(r"lpl_error=([-0-9.eE+]+)", line)
    if m:
        value = abs(float(m.group(1)))
        if value > 1e-12:
            print(f"identical-plane lpl_error={value}", file=sys.stderr)
            bad += 1
if bad:
    sys.exit(1)
PY
then
    fail "the C++ port does not score identical planes as 0.0"
fi

if ! PYTHONPATH="$LPL36_SRC${PYTHONPATH:+:$PYTHONPATH}" \
        python3 - "$NASA" "$TMP/parity.y4m" "$TMP/cpp.txt" "$FRAMES" <<'PY'
import re
import sys

import numpy as np
from lpl36 import lpl36

ref_path, test_path, cpp_path, frame_limit = sys.argv[1:]
limit = int(frame_limit)


def read_luma(path, maximum):
    frames = []
    with open(path, "rb") as f:
        header = f.readline().split()
        width = int([t for t in header if t.startswith(b"W")][0][1:])
        height = int([t for t in header if t.startswith(b"H")][0][1:])
        chroma = [t for t in header if t.startswith(b"C")][0]
        if not chroma.startswith(b"C420"):
            raise SystemExit(f"{path}: expected 4:2:0, got {chroma!r}")
        while len(frames) < maximum:
            line = f.readline()
            if not line:
                break
            if not line.startswith(b"FRAME"):
                raise SystemExit(f"{path}: expected FRAME, got {line[:16]!r}")
            y = f.read(width * height)
            f.read(width * height // 2)
            if len(y) != width * height:
                raise SystemExit(f"{path}: truncated frame")
            frames.append(np.frombuffer(y, dtype=np.uint8).reshape(height, width))
        return frames


cpp = {}
for line in open(cpp_path):
    m = re.match(r"frame=(\d+) lpl_error=([-0-9.eE+]+)", line)
    if m:
        cpp[int(m.group(1))] = float(m.group(2))

ref = read_luma(ref_path, limit)
test = read_luma(test_path, limit)
if len(ref) != len(test) or not ref:
    raise SystemExit(f"frame count mismatch: {len(ref)} vs {len(test)}")
if len(cpp) != len(ref):
    raise SystemExit(f"C++ reported {len(cpp)} frames, expected {len(ref)}")

worst = 0.0
total = 0.0
for i, (a, b) in enumerate(zip(ref, test)):
    error = -lpl36(a, b)
    if i not in cpp:
        raise SystemExit(f"missing C++ frame {i}")
    rel = abs(cpp[i] - error) / max(abs(error), 1e-12)
    total += rel
    worst = max(worst, rel)
    print(f"frame={i} python={error:.9f} cpp={cpp[i]:.9f} rel={rel:.3e}")
mean = total / len(ref)
print(f"parity frames={len(ref)} mean_rel={mean:.3e} max_rel={worst:.3e}")
if worst > 1e-3:
    raise SystemExit(f"parity relative error {worst:.3e} > 1e-3")
PY
then
    fail "the C++ port does not match the Python lpl36 reference"
fi

if [ "$FAILED" -ne 0 ]; then
    echo "video_lpl36_parity_gate: FAILED" >&2
    exit 1
fi
echo "video_lpl36_parity_gate: OK (C++/Python parity, identical-plane zero)"
