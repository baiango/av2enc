#!/usr/bin/env bash
# tools/tool_enum.sh - exhaustive enumeration of the six AV2 intra tool switches
#
#   tools/tool_enum.sh <src.y4m> <outdir> <qp>
#
# Encodes all 2^6 subsets of {dip, ibp, ist, idtx, mrl, sdp} at a fixed QP,
# decodes each with avmdec, and prints (bytes, cvvdp) so each subset can be
# compared at its OWN bytes against a baseline curve.  This is the space that
# a singleton sweep cannot bound (proofs/rdproof/Lack.lean).
#
# sdp is included so its rejection is recorded as a data point rather than
# silently skipped.
set -euo pipefail
if [ "$#" -lt 3 ]; then sed -n '2,11p' "$0" >&2; exit 2; fi
SRC=$1; OUTDIR=$2; QP=$3
BIN=${BIN:-build-release/av2enc}
mkdir -p "$OUTDIR"
FLAGS=(dip ibp ist idtx mrl sdp)
for mask in $(seq 0 63); do
  args=()
  name=""
  for i in 0 1 2 3 4 5; do
    if [ $(( (mask >> i) & 1 )) -eq 1 ]; then
      args+=("--video-${FLAGS[$i]}")
      name="${name}${FLAGS[$i]:+_${FLAGS[$i]}}"
    fi
  done
  [ -z "$name" ] && name="none"
  out="$OUTDIR/m${mask}${name}.ivf"
  if "$BIN" "$SRC" -o "$out" --video --frames 1 --qp "$QP" --cpu-threads 2 \
        --nightly "${args[@]}" >"$out.log" 2>&1 && [ -s "$out" ]; then
    echo "OK $mask $name $(wc -c <"$out" | tr -d ' ')"
  else
    echo "REJECTED $mask $name $(head -1 "$out.log" | cut -c1-80)"
    rm -f "$out"
  fi
done
