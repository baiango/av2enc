#!/usr/bin/env bash
# Smoke test for the default-off bit census / decision dump.
#
# Verifies that
#   * the diagnostics do not change the IVF bytes,
#   * AV2ENC_BIT_CENSUS and AV2ENC_DECISION_DUMP produce parseable records,
#   * the census accounts for the tile payload (only the §8.2 trailer is left),
#   * tools/bit_census.py reports an identical run as identical.
#
# Usage: tools/test_bit_census.sh [--bin build-release/av2enc]
set -uo pipefail

ROOT=$(cd -- "$(dirname -- "$0")/.." && pwd)
BIN="$ROOT/build-release/av2enc"
while [ $# -gt 0 ]; do
    case "$1" in
        --bin) BIN=${2:-}; shift 2 ;;
        -h|--help) sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "test_bit_census: unknown argument $1" >&2; exit 2 ;;
    esac
done
[ -x "$BIN" ] || { echo "test_bit_census: no binary at $BIN" >&2; exit 2; }

CLIP="$ROOT/tests/data/basic_64x64_1f.y4m"
[ -f "$CLIP" ] || { echo "test_bit_census: missing $CLIP" >&2; exit 77; }

TMP=$(mktemp -d "${TMPDIR:-/tmp}/av2enc-census.XXXXXX")
trap 'rm -rf "$TMP"' EXIT
FAILED=0
fail() { echo "FAIL: $*" >&2; FAILED=1; }

COMMON=(--video --frames 1 --qp 189)

"$BIN" "$CLIP" -o "$TMP/plain.ivf" "${COMMON[@]}" >/dev/null 2>&1 ||
    fail "plain encode failed"
AV2ENC_BIT_CENSUS="$TMP/census.jsonl" \
AV2ENC_DECISION_DUMP="$TMP/decisions.jsonl" \
    "$BIN" "$CLIP" -o "$TMP/diag.ivf" "${COMMON[@]}" >/dev/null 2>&1 ||
    fail "diagnostic encode failed"

if [ -f "$TMP/plain.ivf" ] && [ -f "$TMP/diag.ivf" ]; then
    cmp -s "$TMP/plain.ivf" "$TMP/diag.ivf" ||
        fail "the diagnostics changed the IVF bytes"
fi

[ -s "$TMP/census.jsonl" ] || fail "no census records"
[ -s "$TMP/decisions.jsonl" ] || fail "no decision records"

python3 - "$TMP/census.jsonl" <<'PY' || FAILED=1
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
if not rows:
    print("FAIL: empty census", file=sys.stderr)
    sys.exit(1)
total_info = sum(float(r["info_bits"]) for r in rows)
total_payload = sum(int(r["payload_bytes"]) for r in rows)
gap = total_payload * 8 - total_info
if gap < 0:
    print(f"FAIL: census exceeds payload by {-gap:.1f} bits", file=sys.stderr)
    sys.exit(1)
if gap > 64 * len(rows):
    print(f"FAIL: {gap:.1f} bits unaccounted for {len(rows)} tile(s)",
          file=sys.stderr)
    sys.exit(1)
print(f"census: {len(rows)} tile(s), {total_info:.0f} info bits, "
      f"{gap:.0f} trailer bits")
PY

if ! python3 "$ROOT/tools/bit_census.py" --base "$TMP/census.jsonl" \
        --cand "$TMP/census.jsonl" \
        --decisions "$TMP/decisions.jsonl" "$TMP/decisions.jsonl" \
        >"$TMP/report.txt" 2>&1; then
    fail "tools/bit_census.py failed"
fi
grep -q "decision fields: identical" "$TMP/report.txt" ||
    fail "self-comparison did not report identical decisions"
grep -q "delta +0 (" "$TMP/report.txt" ||
    fail "self-comparison did not report a zero bit delta"

if [ "$FAILED" -ne 0 ]; then
    cat "$TMP/report.txt" >&2 2>/dev/null || true
    exit 1
fi
echo "bit census: PASS"
