#!/usr/bin/env bash
# AV2ENC_RD_JOINT_P lane ladder (docs/rd-joint-plan.md).
#
# Arms:
#   base   --video-rd-preset best with AV2ENC_RD_JOINT_P=0 (86.33 lead stack)
#   joint  + AV2ENC_RD_JOINT_P=1 (measured P joint partition + TX/levels solve)
#
# Every point is strict-decoded, validated, per-frame PSNR'd, and its
# --video-recon payload compared with the decoder output.  Matched K is
# tools/oracle.py --target-kbps 100.
#
#   tools/lane-joint-plan/ladder.sh /tmp/rd-joint "114 116 118 119 120"
set -u
ROOT=$(cd -- "$(dirname -- "$0")/../.." && pwd)
BIN=${BIN:-$ROOT/build-release/av2enc}
REF=${REF:-$ROOT/testdata/nasa_480x240_30fps_60f.y4m}
OUT=${1:-/tmp/rd-joint-ladder}
QPS=${2:-"114 116 118 119 120"}
ARMS=${ARMS:-"base joint"}
DAV2D=${DAV2D:-$ROOT/tools/bin/dav2d-macos}
COMMON=(--video --frames 60 --cpu-threads 8 --video-rd-preset best)
TARGET_KBPS=${TARGET_KBPS:-100}
mkdir -p "$OUT"
fail=0
arm_env() {
  case "$1" in
    base) echo "AV2ENC_RD_JOINT_P=0" ;;
    joint) echo "AV2ENC_RD_JOINT_P=1" ;;
    joint120) echo "AV2ENC_RD_JOINT_P=1 AV2ENC_RD_JOINT_P_LAMBDA=${JLAMBDA:-1.0}" ;;
    *) echo "" ;;
  esac
}
read -r -a arm_list <<< "$ARMS"
narms=${#arm_list[@]}
for qp in $QPS; do
  off=$(( qp % narms ))
  rotated=()
  for i in $(seq 0 $((narms - 1))); do
    rotated+=("${arm_list[$(( (i + off) % narms ))]}")
  done
  for arm in "${rotated[@]}"; do
    tag="${arm}_q${qp}"
    ivf="$OUT/$tag.ivf"
    if [ "${REUSE:-0}" = 1 ] && [ -f "$ivf" ] && [ -s "$OUT/$tag.dec.y4m" ]; then
      echo "cached $tag"
      continue
    fi
    extra=$(arm_env "$arm")
    # shellcheck disable=SC2086
    if ! env $extra "$BIN" "$REF" -o "$ivf" --qp "$qp" "${COMMON[@]}" \
         --video-recon "$OUT/$tag.recon.y4m" > "$OUT/$tag.log" 2>&1; then
      echo "encode failed: $tag" >&2
      fail=1
      continue
    fi
    "$DAV2D" --strict 1 -i "$ivf" -o "$OUT/$tag.dec.y4m" \
        > "$OUT/$tag.dec.log" 2>&1 ||
        { echo "strict decode failed: $tag" >&2; fail=1; }
    "$ROOT/tools/validate_dav2d.sh" "$ivf" "$REF" > "$OUT/$tag.val.log" 2>&1 ||
        { echo "validate failed: $tag" >&2; fail=1; }
    python3 "$ROOT/tools/perframe_psnr.py" "$REF" "$OUT/$tag.dec.y4m" "$ivf" \
        --frames 60 > "$OUT/$tag.psnr" 2>&1 ||
        { echo "psnr failed: $tag" >&2; fail=1; }
    python3 - "$OUT/$tag.recon.y4m" "$OUT/$tag.dec.y4m" "$OUT/$tag.recon.ok" <<'PY'
import sys
def payload(p):
    data = open(p, 'rb').read()
    return data[data.find(b'FRAME'):]
ok = payload(sys.argv[1]) == payload(sys.argv[2])
open(sys.argv[3], 'w').write('ok\n' if ok else 'MISMATCH\n')
sys.exit(0 if ok else 1)
PY
    [ $? -eq 0 ] || { echo "recon mismatch: $tag" >&2; fail=1; }
    echo "done $tag $(stat -f %z "$ivf" 2>/dev/null || stat -c %s "$ivf") B"
  done
done

for arm in "${arm_list[@]}"; do
  args=()
  for qp in $QPS; do
    ivf="$OUT/${arm}_q${qp}.ivf"
    [ -s "$ivf" ] && args+=(--ivf "${arm}@${qp}=${ivf}")
  done
  [ ${#args[@]} -ge 2 ] || continue
  python3 "$ROOT/tools/oracle.py" --ref "$REF" "${args[@]}" \
      --target-kbps "$TARGET_KBPS" --json "$OUT/${arm}_oracle.json" \
      > "$OUT/${arm}_oracle.log" 2>&1 ||
      { echo "oracle failed: $arm" >&2; fail=1; }
  python3 "$ROOT/tools/lane-pdriver/kfit.py" \
      --oracle "$OUT/${arm}_oracle.json" --target-kbps "$TARGET_KBPS" \
      --json "$OUT/${arm}_kfit.json" > "$OUT/${arm}_kfit.log" 2>&1 ||
      { echo "kfit failed: $arm" >&2; fail=1; }
done
# Pairwise verdict once both arms scored (the first arm is the reference).
if [ -f "$OUT/base_kfit.json" ] && [ -f "$OUT/joint_kfit.json" ]; then
  python3 - "$OUT/base_kfit.json" "$OUT/joint_kfit.json" <<'PY'
import json, sys
base = json.load(open(sys.argv[1]))
joint = json.load(open(sys.argv[2]))
bk = base.get('base', {})
jk = joint.get('joint', {})
print(f"base  bracket {bk.get('k_bracket'):.3f}  fit {bk.get('k_fit'):.3f}")
print(f"joint bracket {jk.get('k_bracket'):.3f}  fit {jk.get('k_fit'):.3f}")
if bk.get('k_bracket') is not None and jk.get('k_bracket') is not None:
    print(f"delta bracket {jk['k_bracket'] - bk['k_bracket']:+.3f}")
if bk.get('k_fit') is not None and jk.get('k_fit') is not None:
    print(f"delta fit     {jk['k_fit'] - bk['k_fit']:+.3f}")
PY
fi
exit "$fail"
