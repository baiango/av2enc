#!/usr/bin/env bash
# Decoder gate for the AV2 v1.0.0 content interpretation OBU.
#
# The gate:
#   1. builds the module and runs its unit tests (pinned bit goldens,
#      round-trips, invalid-input rejections, per-CVS constancy);
#   2. encodes (or takes) a verified M1/M2 stream and inserts spec-conformant
#      CI OBUs at the §7.3.3 position (first OBU of the coded output frame
#      unit: after the sequence header, before the CLK/first frame);
#   3. scans the result and requires a byte-exact self round-trip and
#      before-first-frame ordering;
#   4. decodes the unmodified and patched streams with the pinned
#      `dav2d --strict 1 --threads 1` and requires byte-identical output.
#
# The pinned dav2d parses the CI OBU grammar (it reports "Overrun in OBU bit
# buffer" for a truncated CI payload, while an unknown OBU type is reported as
# "Unknown OBU type"). It does not enforce the §6.14 chroma-position <= 5
# conformance rule. This script probes both behaviours and reports the verdict
# explicitly; either way the gate requires decode-invariance plus the
# self-round-trip.
#
# Usage:
#   src/frame/metadata/content_interpretation/run_ci_gate.sh \
#       --bin build-m1/av2enc [--dav2d <path>] [--stream existing.ivf]
#
# Exit: 0 pass, 1 failure, 2 usage, 77 skip (missing decoder/fixture/tool).
set -euo pipefail

CI_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
BIN=""
DAV2D="${DAV2D:-}"
STREAM=""
FIXTURE="$ROOT/tests/data/basic_64x64_1f.y4m"

while [ $# -gt 0 ]; do
  case "$1" in
    --bin) BIN=$2; shift 2 ;;
    --dav2d) DAV2D=$2; shift 2 ;;
    --stream) STREAM=$2; shift 2 ;;
    --fixture) FIXTURE=$2; shift 2 ;;
    -h|--help)
      sed -n '2,28p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "ci_gate: unknown argument $1" >&2; exit 2 ;;
  esac
done

if [ -z "$DAV2D" ]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [ -x "$candidate" ]; then DAV2D=$candidate; break; fi
  done
fi
if [ -z "$DAV2D" ] || [ ! -x "$DAV2D" ]; then
  echo "SKIP: no dav2d binary (set DAV2D=...)" >&2
  exit 77
fi
if [ -z "$STREAM" ] && { [ -z "$BIN" ] || [ ! -x "$BIN" ]; }; then
  echo "ci_gate: --bin must point at an av2enc executable" >&2
  exit 2
fi
if [ -z "$STREAM" ] && [ ! -f "$FIXTURE" ]; then
  echo "SKIP: fixture not found: $FIXTURE" >&2
  exit 77
fi
if [ -n "$STREAM" ] && [ ! -f "$STREAM" ]; then
  echo "ci_gate: --stream not found: $STREAM" >&2
  exit 2
fi

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

failures=0
pass=0

fail() {
  failures=$((failures + 1))
  echo "FAIL: $1" >&2
}

ok() {
  pass=$((pass + 1))
  echo "ok:   $1"
}

echo "== build and unit tests =="
if ! "$CI_DIR/build.sh" "$TMP/build" >"$TMP/build.log" 2>&1; then
  cat "$TMP/build.log" >&2
  echo "FAIL: build" >&2
  exit 1
fi
CI_APPEND="$TMP/build/append_content_interpretation"
CI_TEST="$TMP/build/content_interpretation_test"
if "$CI_TEST"; then
  ok "content_interpretation_test"
else
  fail "content_interpretation_test"
fi

echo
echo "== source stream =="
BASE="$TMP/base.ivf"
if [ -n "$STREAM" ]; then
  cp "$STREAM" "$BASE"
  echo "source: $STREAM (external verified stream)"
else
  if ! "$BIN" "$FIXTURE" -o "$BASE" --still >"$TMP/encode.log" 2>&1; then
    cat "$TMP/encode.log" >&2
    echo "FAIL: M1 encode" >&2
    exit 1
  fi
  echo "source: $FIXTURE encoded with --still"
fi

echo "decoder: $("$DAV2D" --version 2>&1 | tail -1)"

decode() {
  local input=$1
  local output=$2
  "$DAV2D" -q -i "$input" -o "$output" --threads 1 --strict 1
}

if ! decode "$BASE" "$TMP/base.y4m" >"$TMP/base.decode.log" 2>&1; then
  cat "$TMP/base.decode.log" >&2
  echo "FAIL: unmodified stream does not decode" >&2
  exit 1
fi
ok "unmodified stream decodes with --strict 1"

# insert_variant <name> <expected-params-description> <tool args...>
insert_variant() {
  local name=$1
  local description=$2
  shift 2
  local ivf="$TMP/$name.ivf"
  local scan="$TMP/$name.scan"
  local decoded="$TMP/$name.y4m"

  if ! "$CI_APPEND" "$BASE" -o "$ivf" "$@" >"$TMP/$name.insert.log" 2>&1; then
    cat "$TMP/$name.insert.log" >&2
    fail "$name: insert"
    return
  fi

  if ! "$CI_APPEND" "$ivf" --scan >"$scan" 2>&1; then
    cat "$scan" >&2
    fail "$name: scan"
    return
  fi
  if ! grep -q "before first frame: ok" "$scan"; then
    cat "$scan" >&2
    fail "$name: CI OBU is not before the first frame"
    return
  fi
  if ! grep -q "round-trip: ok" "$scan"; then
    cat "$scan" >&2
    fail "$name: byte-exact self round-trip"
    return
  fi
  if ! grep -q "invalid: 0" "$scan"; then
    cat "$scan" >&2
    fail "$name: scanner reported invalid CI data"
    return
  fi

  if ! decode "$ivf" "$decoded" >"$TMP/$name.decode.log" 2>&1; then
    cat "$TMP/$name.decode.log" >&2
    fail "$name: dav2d --strict 1 rejected the stream"
    return
  fi
  if ! cmp -s "$TMP/base.y4m" "$decoded"; then
    fail "$name: decoded pixels changed"
    return
  fi

  local before after
  before=$(stat -f%z "$BASE" 2>/dev/null || stat -c%s "$BASE")
  after=$(stat -f%z "$ivf" 2>/dev/null || stat -c%s "$ivf")
  echo "ok:   $name ($description), ${before} -> ${after} bytes, ordering + pixels ok"
  pass=$((pass + 1))
}

echo
echo "== spec-conformant CI OBUs =="
insert_variant rich \
  "interlace-field, BT.2100 PQ, chroma 0/5, explicit 16:11 SAR, 1001/30000 timing" \
  --scan-type 2 --color-idc 2 --chroma-top 0 --chroma-bottom 5 \
  --sar 16x11 --num-units 1001 --time-scale 30000 --equal-interval \
  --ticks-minus-1 0
insert_variant progressive \
  "progressive, BT.709 SDR, 4:2:0 CSP_CENTER, 1:1 SAR, no timing" \
  --scan-type 1 --color-idc 1 --chroma-top 1 --aspect-idc 1 --no-timing
insert_variant explicit_color \
  "complementary field pair, explicit primaries/transfer/matrix, 4:3 SAR" \
  --scan-type 3 --primaries 9 --transfer 16 --matrix 9 --full-range \
  --chroma-top 2 --chroma-bottom 3 --aspect-idc 14 --no-timing
insert_variant minimal \
  "all presence flags clear" \
  --scan-type 0 --no-color --no-chroma --no-aspect --no-timing
insert_variant four_two_two \
  "progressive 4:2:2 table, 160:99 SAR" \
  --scan-type 1 --color-idc 5 --chroma-top 1 --chroma-format 422 \
  --sar 160x99 --no-timing

echo
echo "== decoder CI parsing probe =="
parses_ci=0
if "$CI_APPEND" "$BASE" -o "$TMP/truncated.ivf" --probe-truncated \
     >"$TMP/truncated.insert.log" 2>&1; then
  if decode "$TMP/truncated.ivf" "$TMP/truncated.y4m" \
       >"$TMP/truncated.decode.log" 2>&1; then
    echo "note: dav2d accepted an empty-payload CI OBU; it skips the CI payload"
    parses_ci=0
  else
    if grep -qi "overrun in OBU bit buffer" "$TMP/truncated.decode.log"; then
      echo "note: dav2d parses content_interpretation_obu() (truncated payload"
      echo "      reports 'Overrun in OBU bit buffer' while an unknown OBU type"
      echo "      reports 'Unknown OBU type')"
    else
      echo "note: dav2d rejected the truncated CI OBU:"
      sed 's/^/      /' "$TMP/truncated.decode.log"
    fi
    parses_ci=1
  fi
  pass=$((pass + 1))
else
  cat "$TMP/truncated.insert.log" >&2
  fail "truncated CI probe insertion"
fi

echo
echo "== decoder CI conformance probe (§6.14 chroma <= 5) =="
if "$CI_APPEND" "$BASE" -o "$TMP/invalid_chroma.ivf" --probe-invalid-chroma \
     >"$TMP/invalid_chroma.insert.log" 2>&1; then
  if decode "$TMP/invalid_chroma.ivf" "$TMP/invalid_chroma.y4m" \
       >"$TMP/invalid_chroma.decode.log" 2>&1; then
    echo "note: dav2d does not reject ci_chroma_sample_position_top = 6 under"
    echo "      --strict 1; CI conformance checking is not implemented."
    inval_pixels=0
    if ! cmp -s "$TMP/base.y4m" "$TMP/invalid_chroma.y4m"; then
      inval_pixels=1
    fi
    if [ "$inval_pixels" -eq 0 ]; then
      echo "note: decode output is unchanged (pixels identical)"
    else
      fail "non-conforming CI probe changed the decoded pixels"
    fi
    pass=$((pass + 1))
  else
    echo "note: dav2d rejects ci_chroma_sample_position_top = 6; the decoder"
    echo "      enforces the §6.14 chroma position conformance rule."
    pass=$((pass + 1))
  fi
else
  cat "$TMP/invalid_chroma.insert.log" >&2
  fail "invalid-chroma probe insertion"
fi

echo
echo "== scan of the gated stream =="
"$CI_APPEND" "$TMP/rich.ivf" --scan | sed 's/^/  /'

echo
if [ "$parses_ci" -eq 1 ]; then
  echo "verdict: pinned dav2d parses the CI OBU grammar and the CI OBU is"
  echo "         accepted in the §7.3.3 position without changing pixels."
else
  echo "verdict: pinned dav2d does not consume CI payload; the gate relies on"
  echo "         decode-invariance plus the byte-exact self round-trip."
fi
echo "summary: $pass passed, $failures failed"
if [ "$failures" -ne 0 ]; then
  exit 1
fi
echo "GATE PASS"
