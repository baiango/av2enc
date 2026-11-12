#!/usr/bin/env bash
# End-to-end gate for the integrated neutral delta-Q / segmentation syntax.
#
# The reference still encoder keeps one quantizer for the whole frame, so the
# opt-in syntax signals an all-disabled 8-segment table (segment id 0
# everywhere) and a zero reduced delta at every superblock.  That changes the
# header and the tile symbols but not the reconstruction: the decoded output
# must match the plain `--still` reconstruction, and the default `--still`
# stream must stay byte-identical to its pinned MD5.
#
# Usage:
#   src/codec/entropy/deltaq/still/gate_integration.sh --bin build-int/av2enc
#   ... [--fixture tests/data/basic_640x360_1f.y4m] [--qp 90,141]
#
# Exit 77 when no pinned dav2d is available, like the other gates.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "$0")/../../../../.." && pwd)
BIN=""
FIXTURE="tests/data/basic_64x64_1f.y4m"
QPS="90,141"
DAV2D="${DAV2D:-}"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

while [ $# -gt 0 ]; do
  case "$1" in
    --bin) BIN=$2; shift 2 ;;
    --fixture) FIXTURE=$2; shift 2 ;;
    --qp) QPS=$2; shift 2 ;;
    --dav2d) DAV2D=$2; shift 2 ;;
    -h|--help) sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "gate_integration: unknown argument $1" >&2; exit 2 ;;
  esac
done

[ -n "$BIN" ] && [ -x "$BIN" ] || {
  echo "gate_integration: --bin must point at an av2enc executable" >&2
  exit 2
}
if [ -z "$DAV2D" ]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [ -x "$candidate" ]; then DAV2D=$candidate; break; fi
  done
fi
[ -n "$DAV2D" ] && [ -x "$DAV2D" ] || {
  echo "gate_integration: no dav2d binary (set DAV2D=...)" >&2
  exit 77
}
[ -f "$ROOT/$FIXTURE" ] || {
  echo "gate_integration: missing fixture $FIXTURE" >&2
  exit 77
}

md5_of() { md5 -q "$1" 2>/dev/null || md5sum "$1" | awk '{print $1}'; }

recon_md5() {
  "$DAV2D" --strict 1 --threads 1 -q -i "$1" -o "$2" >/dev/null 2>&1 || return 1
  md5_of "$2"
}

fail=0
pass=0
for qp in ${QPS//,/ }; do
  base="$TMP/base_$qp.ivf"
  "$BIN" "$ROOT/$FIXTURE" -o "$base" --frames 1 --still --qp "$qp" \
    >/dev/null 2>&1
  ref_recon=$(recon_md5 "$base" "$TMP/base_$qp.y4m") || {
    echo "FAIL qp$qp: baseline did not decode" >&2; fail=$((fail + 1)); continue;
  }
  for args in "--still-segmentation" "--still-delta-q" \
              "--still-segmentation --still-delta-q --still-delta-q-res 3" \
              "--still-segmentation --still-delta-q --adaptive-cdf"; do
    tag=$(echo "$args" | tr ' -' '__')
    ivf="$TMP/q${qp}$tag.ivf"
    if ! "$BIN" "$ROOT/$FIXTURE" -o "$ivf" --frames 1 --still --qp "$qp" \
        $args >"$TMP/log" 2>&1; then
      echo "FAIL qp$qp [$args]: encode failed" >&2
      cat "$TMP/log" >&2
      fail=$((fail + 1))
      continue
    fi
    got=$(recon_md5 "$ivf" "$TMP/q${qp}$tag.y4m") || {
      echo "FAIL qp$qp [$args]: dav2d rejected the stream" >&2
      fail=$((fail + 1)); continue;
    }
    if [ "$got" != "$ref_recon" ]; then
      echo "FAIL qp$qp [$args]: decoded output differs from --still recon" >&2
      fail=$((fail + 1)); continue
    fi
    echo "PASS qp$qp [$args] $(stat -c%s "$ivf" 2>/dev/null || stat -f%z "$ivf") B $(md5_of "$ivf")"
    pass=$((pass + 1))
  done
done

echo "gate_integration: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
