#!/usr/bin/env bash
# Decoder gate: emit CfL with a non-zero cfl_ds_filter_index, decode with the
# pinned dav2d (`--strict 1 --threads 1`) and require 0 reconstruction diffs.
#
# For every index in 0 1 2:
#   1. encode testdata/shine_256x256_1f.y4m with the instrumented M1 build;
#   2. assert parse_seq_index.py reads that exact index from the sequence
#      header (i.e. the stream is not silently still index 0);
#   3. dump the encoder reconstruction with the same instrumented build;
#   4. decode with tools/bin/dav2d[-macos] --strict 1 --threads 1;
#   5. compare encoder recon vs decoder recon with tools/still_verify/verify.py.
#
# Usage:
#   ./src/codec/intra/cfl/ds_filters/gate/check_gate.sh
#   SKIP_BUILD=1 ./.../check_gate.sh        (reuse gate/build binaries)
#
# Exit: 0 only if every index is bit-exact.  Artifacts stay in gate/build/gate.

set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
OUT="$HERE/build"
WORK="$OUT/gate"
SRC="$ROOT/testdata/shine_256x256_1f.y4m"

if [ "${SKIP_BUILD:-0}" != "1" ]; then
  "$HERE/build_gate.sh"
fi
[ -x "$OUT/av2enc_ds" ] || { echo "check_gate: run build_gate.sh first" >&2; exit 2; }
[ -x "$OUT/still_dump_ds" ] || { echo "check_gate: run build_gate.sh first" >&2; exit 2; }
[ -f "$SRC" ] || { echo "check_gate: missing $SRC" >&2; exit 2; }

DAV2D="${DAV2D:-}"
if [ -z "$DAV2D" ]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [ -x "$candidate" ]; then DAV2D="$candidate"; break; fi
  done
fi
[ -n "$DAV2D" ] || { echo "check_gate: dav2d not found (set DAV2D)" >&2; exit 2; }

mkdir -p "$WORK"
# verify.py invokes the decoder itself.  Wrap the pinned binary so that its
# decode also uses the gate's required `--strict 1 --threads 1`.
DECODER_WRAPPER="$WORK/dav2d_strict1_threads1.sh"
cat > "$DECODER_WRAPPER" <<EOF
#!/bin/sh
exec "$DAV2D" --strict 1 --threads 1 "\$@"
EOF
chmod +x "$DECODER_WRAPPER"

printf 'index  parsed  Y/U/V diffs            result\n'
printf '%s\n' '---------------------------------------------------------'

failures=0
for index in 0 1 2; do
  ivf="$WORK/filter${index}.ivf"
  dump="$WORK/filter${index}.dump"
  dec="$WORK/filter${index}_dec.y4m"

  AV2ENC_CFL_DS_FILTER="$index" "$OUT/av2enc_ds" "$SRC" -o "$ivf" --still >/dev/null
  parsed=$(python3 "$HERE/parse_seq_index.py" "$ivf")
  AV2ENC_CFL_DS_FILTER="$index" "$OUT/still_dump_ds" -i "$SRC" -o "$dump" >/dev/null

  "$DAV2D" -q -i "$ivf" -o "$dec" --strict 1 --threads 1
  set +e
  report=$(python3 "$ROOT/tools/still_verify/verify.py" --ivf "$ivf" --dump "$dump" \
             --decoder "$DECODER_WRAPPER" --keep-dec "$dec" 2>&1)
  status=$?
  set -e
  summary=$(printf '%s\n' "$report" | grep -E '^  plane|^RESULT:' | tr '\n' ' ')
  if [ "$parsed" != "$index" ] || [ "$status" -ne 0 ]; then
    failures=$((failures + 1))
    printf '%-6s %-7s %-24s FAIL\n' "$index" "$parsed" "$summary"
    printf '%s\n' "$report" >&2
  else
    printf '%-6s %-7s %-24s BIT-EXACT\n' "$index" "$parsed" "$summary"
  fi
done

if [ "$failures" -ne 0 ]; then
  echo "check_gate: $failures index(es) failed" >&2
  exit 1
fi
echo "check_gate: all three indices bit-exact"
