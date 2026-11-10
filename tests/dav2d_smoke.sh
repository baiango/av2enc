#!/usr/bin/env bash
# CTest wrapper for the pinned dav2d validation gate.
#
# Runs the known-good sample through tools/validate_dav2d.sh so the gate itself
# (decoder binary + validator + ffprobe) is exercised. The encoder's real output
# is validated separately with the same script. Exit 77 = CTest skip.

set -euo pipefail

ROOT=${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
SAMPLE="$ROOT/testdata/sample_480x270_130f.ivf"
VALIDATOR="$ROOT/tools/validate_dav2d.sh"

DECODER=""
if [[ -n "${DAV2D:-}" && -x "${DAV2D}" ]]; then
  DECODER="${DAV2D}"
elif [[ -x "$ROOT/tools/bin/dav2d" ]]; then
  DECODER="$ROOT/tools/bin/dav2d"
elif [[ -x "$ROOT/tools/bin/dav2d-macos" ]]; then
  DECODER="$ROOT/tools/bin/dav2d-macos"
else
  echo "SKIP: dav2d binary not found (checked DAV2D, tools/bin/dav2d, tools/bin/dav2d-macos)" >&2
  exit 77
fi

if [[ ! -f "$SAMPLE" ]]; then
  echo "SKIP: known-good sample not found: $SAMPLE" >&2
  exit 77
fi

if [[ ! -x "$VALIDATOR" ]]; then
  echo "SKIP: validator script not found or not executable: $VALIDATOR" >&2
  exit 77
fi

echo "dav2d_smoke: decoder: $("$DECODER" --version 2>&1 | tail -1)"
echo "dav2d_smoke: sample:  $SAMPLE"
DAV2D="$DECODER" "$VALIDATOR" "$SAMPLE"
