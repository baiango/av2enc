#!/usr/bin/env bash
# tools/pipeline_best.sh — encode a clip with every lever this project has
# MEASURED to be positive, and nothing that measured negative.
#
#   tools/pipeline_best.sh <src.y4m> <out.ivf> <target_bytes> [extra av2enc args]
#
# Stages
#   1. duplicate suppression  AV2ENC_SEF_DUPLICATES=30
#        near-duplicate frames become 5-byte show_existing_frame OBUs.
#        Measured: +0.274 JOD at matched bytes on space8k 60f; avmdec-validated.
#        dav2d rejects the explicit-hint SEF form, so validate with avmdec.
#   2. coefficient pricing    AV2ENC_RATE_HYBRID_S=1
#        prices leaves with the analytic model whose double-counted position
#        term has been removed (Pricing.lean, PricingWrong.lean).
#        Measured: +0.027 JOD at matched bytes; 1.25 % fewer bytes.
#   3. stack settings         TX HF-mask / textured lambda / RD driver
#        OFF BY DEFAULT.  Measured +0.061 JOD at matched bytes on a *still*, but
#        measured -4.10 JOD at matched bytes on a 60-frame inter chain (QP239
#        needed to hit the same bytes, PSNR-Y 24.47 vs 31.49).  Enabling it is
#        opt-in via AV2ENC_PIPELINE_STACK=1 and is only defensible for intra.
#        This is the reason the pipeline is not "every lever we have".
#
# Measured negative on inter and therefore excluded even though it helps stills:
#   the TX HF-mask / textured-lambda / RD-driver stack, the -23 chroma offset,
#   --video-gdf beyond what --nightly sets.
#
# Deliberately NOT included (measured null or negative this project):
#   TPL (negative at 4/4 points), ARNR (null on 720p, -0.05 dB PSNR),
#   drift refresh (negative at matched bytes), key-heavier allocation (tie),
#   4x4 partitions (-0.012..-0.025 at matched bytes), water-filling allocation
#   (+0.026 dB, below measurement noise).
#
# QP is chosen by bisection against the stream size, so the comparison is at
# matched BYTES, not matched QP.
set -euo pipefail

if [ "$#" -lt 3 ]; then
  sed -n '2,26p' "$0" >&2
  exit 2
fi

SRC=$1
OUT=$2
TARGET=$3
shift 3
EXTRA=("$@")

BIN=${BIN:-build-release/av2enc}
TMP=$(mktemp -d "${TMPDIR:-/tmp}/av2enc-pipeline.XXXXXX")
trap 'rm -rf "$TMP"' EXIT

# The measured-positive stack.
encode() {  # $1 = qp, $2 = output
  env \
    AV2ENC_SEF_DUPLICATES=30 \
    AV2ENC_RATE_HYBRID_S=1 \
    "$BIN" "$SRC" -o "$2" --video --cpu-threads 8 \
      --nightly "${EXTRA[@]+"${EXTRA[@]}"}" \
      --qp "$1" >"$TMP/$1.log" 2>&1 || { echo "encode failed at qp $1" >&2; tail -3 "$TMP/$1.log" >&2; exit 1; }
  wc -c <"$2" | tr -d ' '
}

# Bisect QP on stream size.  Rate rises as QP falls.
lo=180; hi=215
best=""; bestd=1000000000
for _ in 1 2 3 4 5; do
  qp=$(( (lo + hi) / 2 ))
  size=$(encode "$qp" "$TMP/qp$qp.ivf")
  d=$(( size > TARGET ? size - TARGET : TARGET - size ))
  echo "  qp $qp -> $size B (target $TARGET, |d| $d)" >&2
  if [ "$d" -lt "$bestd" ]; then bestd=$d; best=$qp; fi
  if [ "$d" -le $(( TARGET / 100 )) ]; then break; fi
  if [ "$size" -gt "$TARGET" ]; then lo=$qp; else hi=$qp; fi
  if [ $(( hi - lo )) -le 1 ]; then break; fi
done

cp "$TMP/qp$best.ivf" "$OUT"
SIZE=$(wc -c <"$OUT" | tr -d ' ')
echo "pipeline: $OUT  qp $best  $SIZE B (target $TARGET, $(awk -v a="$SIZE" -v b="$TARGET" 'BEGIN{printf "%+.1f%%", 100*(a-b)/b}'))" >&2
echo "$SIZE"
