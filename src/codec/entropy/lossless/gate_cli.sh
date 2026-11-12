#!/usr/bin/env bash
# End-to-end gate for the CLI-integrated lossless still path (--lossless).
#
# Encodes the repository fixtures with the opt-in lossless module through the
# real `av2enc` binary, decodes with the pinned dav2d, and requires the decoded
# Y/U/V bytes to equal the source frame exactly (lossless = 0 LSB error).
#
# Usage:
#   src/codec/entropy/lossless/gate_cli.sh --bin build-int/av2enc
#   ... [--dav2d tools/bin/dav2d-macos]
#
# Exit 77 when no pinned decoder is available, like the other gates.
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
    -h|--help) sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
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
  [ -f "$ROOT/$fixture" ] || { echo "SKIP missing $fixture"; continue; }
  for variant in "" "--lossless-fsc" "--adaptive-cdf" \
                 "--lossless-fsc --adaptive-cdf"; do
    tag=$(basename "$fixture" .y4m)$(echo "$variant" | tr ' -' '__')
    ivf="$TMP/$tag.ivf"
    y4m="$TMP/$tag.y4m"
    # shellcheck disable=SC2086
    if ! "$BIN" "$ROOT/$fixture" -o "$ivf" --frames 1 --lossless \
        $variant >"$TMP/log" 2>&1; then
      echo "FAIL $fixture [$variant]: encode failed" >&2
      cat "$TMP/log" >&2
      fail=$((fail + 1))
      continue
    fi
    if ! "$DAV2D" --strict 1 --threads 1 -q -i "$ivf" -o "$y4m" \
        >/dev/null 2>&1; then
      echo "FAIL $fixture [$variant]: dav2d rejected the stream" >&2
      fail=$((fail + 1))
      continue
    fi
    if ! python3 - "$ROOT/$fixture" "$y4m" <<'PY'
import sys

def first_frame(path):
    data = open(path, "rb").read()
    nl = data.index(b"\n")
    header = data[:nl].decode()
    fields = dict((t[0], t[1:]) for t in header.split()[1:] if t[0] in "WH")
    w, h = int(fields["W"]), int(fields["H"])
    pos = nl + 1
    if data[pos:pos + 5] != b"FRAME":
        raise SystemExit("missing FRAME")
    pos = data.index(b"\n", pos) + 1
    return data[pos:pos + w * h * 3 // 2]

src = first_frame(sys.argv[1])
dec = first_frame(sys.argv[2])
if len(dec) != len(src):
    raise SystemExit(f"size {len(dec)} != {len(src)}")
diffs = sum(1 for a, b in zip(src, dec) if a != b)
if diffs:
    raise SystemExit(f"{diffs} differing samples")
PY
    then
      echo "FAIL $fixture [$variant]: decoded output is not lossless" >&2
      fail=$((fail + 1))
      continue
    fi
    bytes=$(stat -c%s "$ivf" 2>/dev/null || stat -f%z "$ivf")
    md5=$(md5 -q "$ivf" 2>/dev/null || md5sum "$ivf" | awk '{print $1}')
    echo "PASS $fixture [$variant] $bytes B $md5"
    pass=$((pass + 1))
  done
done

echo "gate_cli: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
