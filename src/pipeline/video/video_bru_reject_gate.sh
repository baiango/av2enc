#!/usr/bin/env bash
# Gate for the BRU rejection (§5.4.6 enable_bru / §5.18.2 use_bru).
#
# 1. Builds the BRU component gate against the already-built core and asserts
#    that the pinned dav2d stops at the BRU frame header (the stream is valid;
#    the decoder does not parse BRU yet).
# 2. Asserts the product CLI rejects --video-bru by name (exit 2, message names
#    the flag), so the feature can never be enabled silently.
#
# Usage:
#   src/pipeline/video/video_bru_reject_gate.sh --bin build/av2enc \
#       --core build/libav2enc_core.a [--dav2d /path/to/dav2d]
#
# Exit codes: 0 pass, 1 failure, 2 usage, 77 missing tool/decoder.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "$0")/../../.." && pwd)
BIN=""
CORE=""
DAV2D="${DAV2D:-}"
while [ $# -gt 0 ]; do
  case "$1" in
    --bin) BIN=$2; shift 2 ;;
    --core) CORE=$2; shift 2 ;;
    --dav2d) DAV2D=$2; shift 2 ;;
    -h|--help) sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "video_bru_reject_gate: unknown argument $1" >&2; exit 2 ;;
  esac
done
[ -n "$BIN" ] && [ -x "$BIN" ] || { echo "need --bin <av2enc>" >&2; exit 2; }
[ -n "$CORE" ] && [ -f "$CORE" ] || { echo "need --core <libav2enc_core.a>" >&2; exit 2; }
if [ -z "$DAV2D" ]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [ -x "$candidate" ]; then DAV2D=$candidate; break; fi
  done
fi
[ -n "$DAV2D" ] && [ -x "$DAV2D" ] || {
  echo "video_bru_reject_gate: dav2d not found (set DAV2D)" >&2; exit 77; }

OUT="${OUT:-${TMPDIR:-/tmp}/video_bru_reject_gate}"
mkdir -p "$OUT"
FAIL=0

echo "== build the BRU component gate against the core"
if ! g++ -std=c++20 -O2 -I "$ROOT/src" -Wno-missing-braces \
    "$ROOT/src/codec/inter/bru/bru_gate.cpp" \
    "$ROOT/src/codec/inter/bru/bru_writer.cpp" \
    "$ROOT/src/codec/inter/bru/bru_reader.cpp" \
    "$ROOT/src/codec/inter/bru/bru_region.cpp" \
    "$CORE" -o "$OUT/bru_gate" >"$OUT/build.log" 2>&1; then
  echo "FAIL: BRU gate link failed"; tail -5 "$OUT/build.log"; exit 1
fi

echo "== pinned dav2d must reject the BRU frame header"
set +e
DAV2D="$DAV2D" "$OUT/bru_gate" --out "$OUT/bru" >"$OUT/gate.log" 2>&1
GATE_RC=$?
set -e
cat "$OUT/gate.log"
if ! grep -q "dav2d rejects BRU" "$OUT/gate.log"; then
  echo "FAIL: the component gate no longer reports the pinned-decoder rejection"
  FAIL=1
fi
if ! grep -q "Error parsing frame header" "$OUT/bru/bru_gate.ivf.dav2d.log"; then
  echo "FAIL: pinned dav2d did not report the frame-header parse error"
  FAIL=1
fi
if [ "$GATE_RC" -eq 0 ] && [ "$FAIL" -eq 0 ]; then
  echo "FAIL: unexpected BRU gate success (decoder support changed?)"; FAIL=1
fi

echo "== CLI must reject --video-bru by name"
set +e
"$BIN" /dev/null -o "$OUT/bru_cli.ivf" --video --frames 2 --video-bru \
  >"$OUT/cli.log" 2>&1
CLI_RC=$?
set -e
if [ "$CLI_RC" -ne 2 ]; then
  echo "FAIL: --video-bru exit $CLI_RC (expected 2)"; FAIL=1
fi
if ! grep -q -- "--video-bru is not decoder-gated" "$OUT/cli.log"; then
  echo "FAIL: --video-bru rejection does not name the flag"; FAIL=1
fi
cat "$OUT/cli.log"

if [ "$FAIL" -ne 0 ]; then
  echo "video_bru_reject_gate: FAIL"; exit 1
fi
echo "video_bru_reject_gate: PASS (pinned dav2d rejects BRU; CLI rejects by name)"
