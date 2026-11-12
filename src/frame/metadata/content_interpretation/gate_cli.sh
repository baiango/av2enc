#!/usr/bin/env bash
# End-to-end gate for the CLI-integrated content interpretation OBU.
#
# `--still-content-interpretation` inserts one default-parameter
# OBU_CONTENT_INTERPRETATION (§5.15) between the sequence header and the frame.
# The default parameters are the spec-default all-flags-clear form: a 4-byte
# OBU.  The stream must decode under the pinned dav2d, the decoded planes must
# be identical to the plain `--still` stream, and the default stream must not
# move.
#
# Usage: src/frame/metadata/content_interpretation/gate_cli.sh --bin build-int/av2enc
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "$0")/../../../.." && pwd)
BIN=""
DAV2D="${DAV2D:-}"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

while [ $# -gt 0 ]; do
  case "$1" in
    --bin) BIN=$2; shift 2 ;;
    --dav2d) DAV2D=$2; shift 2 ;;
    -h|--help) sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "gate_cli: unknown argument $1" >&2; exit 2 ;;
  esac
done
[ -n "$BIN" ] && [ -x "$BIN" ] || {
  echo "gate_cli: --bin must point at an av2enc executable" >&2; exit 2;
}
if [ -z "$DAV2D" ]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [ -x "$candidate" ]; then DAV2D=$candidate; break; fi
  done
fi
[ -n "$DAV2D" ] && [ -x "$DAV2D" ] || {
  echo "gate_cli: no dav2d binary (set DAV2D=...)" >&2; exit 77;
}

fail=0
pass=0
for fixture in tests/data/basic_64x64_1f.y4m tests/data/basic_640x360_1f.y4m; do
  [ -f "$ROOT/$fixture" ] || continue
  for qp in 90 141; do
    base="$TMP/base_${qp}.ivf"
    ci="$TMP/ci_${qp}.ivf"
    "$BIN" "$ROOT/$fixture" -o "$base" --frames 1 --still --qp "$qp" \
      >/dev/null 2>&1
    "$BIN" "$ROOT/$fixture" -o "$ci" --frames 1 --still --qp "$qp" \
      --still-content-interpretation >/dev/null 2>&1
    bsize=$(stat -c%s "$base" 2>/dev/null || stat -f%z "$base")
    csize=$(stat -c%s "$ci" 2>/dev/null || stat -f%z "$ci")
    if [ $((csize - bsize)) -ne 4 ]; then
      echo "FAIL $fixture qp$qp: CI delta $((csize - bsize)) bytes, expected 4" >&2
      fail=$((fail + 1)); continue
    fi
    "$DAV2D" --strict 1 --threads 1 -q -i "$base" -o "$TMP/base.y4m" \
      >/dev/null 2>&1
    if ! "$DAV2D" --strict 1 --threads 1 -q -i "$ci" -o "$TMP/ci.y4m" \
        >/dev/null 2>&1; then
      echo "FAIL $fixture qp$qp: dav2d rejected the CI stream" >&2
      fail=$((fail + 1)); continue
    fi
    if ! cmp -s "$TMP/base.y4m" "$TMP/ci.y4m"; then
      echo "FAIL $fixture qp$qp: decoded output changed with CI" >&2
      fail=$((fail + 1)); continue
    fi
    echo "PASS $fixture qp$qp ($csize B CI stream, decode identical)"
    pass=$((pass + 1))
  done
done
echo "gate_cli: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
