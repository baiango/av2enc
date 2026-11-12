#!/usr/bin/env bash
# M2 P-frame rate-aware decision experiment.
#
# Builds two host-only CPU gates from the same sources:
#   * baseline: the repository src/pipeline/video/video_frame.cpp
#   * rdo:      a generated copy with the M2 RDO dispatch hook
#
# Every stream is decoded with `dav2d --strict 1` and the gate compares the
# decoded pictures against the encoder reconstruction sample by sample.  The
# script then runs tools/quality.py for PSNR/CVVDP and writes results.md.
#
# Cases: 2-frame and 8-frame 1280x720, single 64x64 leaf and full tree.
# Variants: baseline default, baseline --inter-residual, rate-aware full,
# rate-aware with the baseline partition, rate-aware without residual coding,
# and rate-aware with distortion-only candidate selection.
#
# Usage:
#   ./run_video_rdo_experiment.sh
#   ./run_video_rdo_experiment.sh --input clip.y4m --out /tmp/m2-rdo
#   ./run_video_rdo_experiment.sh --cases 2f_sb,8f_tree
#   ./run_video_rdo_experiment.sh --no-quality     # decode/recon only
#   ./run_video_rdo_experiment.sh --rebuild        # force a full rebuild
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
CXX="${CXX:-c++}"
INPUT="${INPUT:-$ROOT/testdata/shine_1280x720_60f.y4m}"
OUT="${OUT:-${TMPDIR:-/tmp}/video_rdo_experiment}"
CASES="2f_sb,2f_tree,8f_sb,8f_tree"
RUN_QUALITY=1
REBUILD=0

while [ $# -gt 0 ]; do
  case "$1" in
    --input|-i) INPUT="$2"; shift ;;
    --out) OUT="$2"; shift ;;
    --cases) CASES="$2"; shift ;;
    --no-quality) RUN_QUALITY=0 ;;
    --rebuild) REBUILD=1 ;;
    -h|--help)
      sed -n '2,22p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *)
      echo "run_video_rdo_experiment: unknown argument: $1" >&2
      exit 2
      ;;
  esac
  shift
done

[ -f "$INPUT" ] || {
  echo "run_video_rdo_experiment: missing input $INPUT" >&2
  exit 2
}

if [ -z "${REAL_DAV2D:-}" ]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos" \
                   "$(command -v dav2d || true)"; do
    if [ -n "$candidate" ] && [ -x "$candidate" ]; then
      REAL_DAV2D="$candidate"
      break
    fi
  done
fi
[ -n "${REAL_DAV2D:-}" ] || {
  echo "run_video_rdo_experiment: dav2d not found; set REAL_DAV2D" >&2
  exit 2
}
REAL_DAV2D="$(cd -- "$(dirname -- "$REAL_DAV2D")" && pwd)/$(basename -- "$REAL_DAV2D")"
WRAPPER="$HERE/dav2d_strict.sh"
chmod +x "$WRAPPER"

mkdir -p "$OUT/obj" "$OUT/results"
OBJ="$OUT/obj"
BIN_BASE="$OUT/video_gate_base"
BIN_RDO="$OUT/video_gate_rdo"
PATCHED="$OUT/video_frame_rdo.cpp"
METRICS="$OUT/results/metrics.tsv"
: > "$METRICS"
printf 'case\tvariant\tbytes\tpsnr_y\tpsnr_avg\tcvvdp\tivf\n' >> "$METRICS"

FLAGS=(-std=c++20 -O2 -I"$ROOT/src" -Wall -Wextra -Wpedantic -Wconversion
       -Wshadow -Wno-missing-braces -Wno-sign-conversion)

SHARED=(
  "$ROOT/src/core/io/y4m.cpp"
  "$ROOT/src/core/io/ivf.cpp"
  "$ROOT/src/core/bitstream/bitwriter.cpp"
  "$ROOT/src/core/bitstream/obu.cpp"
  "$ROOT/src/core/bitstream/basic_headers.cpp"
  "$ROOT/src/codec/partition/tree/partition.cpp"
  "$ROOT/src/codec/intra/modes/intra_modes.cpp"
  "$ROOT/src/codec/transform/variants/tx_variants.cpp"
  "$ROOT/src/codec/transform/variants/tx_select.cpp"
  "$ROOT/src/codec/transform/tx_type/tx_type.cpp"
  "$ROOT/src/codec/rate/rdo/rdo.cpp"
  "$ROOT/src/codec/rate/rdo/intra_decision.cpp"
  "$ROOT/src/codec/rate/rdoq/rdoq.cpp"
  "$ROOT/src/codec/rate/rdoq/rdoq_av2_model.cpp"
  "$ROOT/src/codec/entropy/av2/av2_entropy.cpp"
  "$ROOT/src/codec/inter/me/me.cpp"
  "$ROOT/src/codec/inter/mc/mc.cpp"
  "$ROOT/src/codec/inter/merge/merge.cpp"
  "$ROOT/src/pipeline/video/video_headers.cpp"
  "$ROOT/src/pipeline/video/video_syntax.cpp"
  "$ROOT/src/codec/transform/inter_residual/inter_tx_type.cpp"
  "$ROOT/src/codec/transform/inter_residual/inter_coeff_syntax.cpp"
  "$ROOT/src/codec/transform/inter_residual/inter_residual.cpp"
  "$ROOT/src/codec/rate/decide/config.cpp"
  "$ROOT/src/codec/rate/decide/transform_stream.cpp"
  "$ROOT/src/codec/rate/decide/intra_rdo.cpp"
  "$ROOT/src/codec/rate/decide/chroma_rdo.cpp"
  "$ROOT/src/codec/rate/decide/partition_rdo.cpp"
  "$ROOT/src/codec/rate/rdo/video/config.cpp"
  "$ROOT/src/codec/rate/rdo/video/context.cpp"
  "$ROOT/src/codec/rate/rdo/video/rate_model.cpp"
  "$ROOT/src/codec/rate/rdo/video/residual_decision.cpp"
  "$ROOT/src/codec/rate/rdo/video/inter_decision.cpp"
  "$ROOT/src/codec/rate/rdo/video/intra_fallback.cpp"
  "$ROOT/src/codec/rate/rdo/video/partition_decision.cpp"
)

object_name() {
  printf '%s' "$1" | sed 's|/|_|g; s|\.cpp$|.o|'
}

build() {
  local objects=()
  echo "== build =="
  for source in "${SHARED[@]}"; do
    local obj="$OBJ/$(object_name "$source")"
    if [ "$REBUILD" = 1 ] || [ ! -f "$obj" ] || [ "$source" -nt "$obj" ]; then
      "$CXX" "${FLAGS[@]}" -c "$source" -o "$obj"
    fi
    objects+=("$obj")
  done
  local gate_obj="$OBJ/video_gate.o"
  if [ "$REBUILD" = 1 ] || [ ! -f "$gate_obj" ] ||
     [ "$ROOT/src/pipeline/video/video_gate.cpp" -nt "$gate_obj" ]; then
    "$CXX" "${FLAGS[@]}" -c "$ROOT/src/pipeline/video/video_gate.cpp" -o "$gate_obj"
  fi

  local base_obj="$OBJ/video_frame_base.o"
  if [ "$REBUILD" = 1 ] || [ ! -f "$base_obj" ] ||
     [ "$ROOT/src/pipeline/video/video_frame.cpp" -nt "$base_obj" ]; then
    "$CXX" "${FLAGS[@]}" -c "$ROOT/src/pipeline/video/video_frame.cpp" \
      -o "$base_obj"
  fi

  python3 "$HERE/patch_video_rdo.py" -o "$PATCHED"
  local rdo_obj="$OBJ/video_frame_rdo.o"
  if [ "$REBUILD" = 1 ] || [ ! -f "$rdo_obj" ] ||
     [ "$PATCHED" -nt "$rdo_obj" ]; then
    "$CXX" "${FLAGS[@]}" -c "$PATCHED" -o "$rdo_obj"
  fi

  "$CXX" "$gate_obj" "$base_obj" "${objects[@]}" -o "$BIN_BASE"
  "$CXX" "$gate_obj" "$rdo_obj" "${objects[@]}" -o "$BIN_RDO"
}

run_gate() {
  local bin=$1 case_name=$2 variant=$3 out=$4 log=$5
  local -a env_vars=()
  local -a gate_opts=()
  case "$variant" in
    base_default) ;;
    base_residual) gate_opts+=(--inter-residual) ;;
    rdo_full) env_vars+=(AV2ENC_VIDEO_RDO=1) ;;
    rdo_nopart) env_vars+=(AV2ENC_VIDEO_RDO=1 AV2ENC_VIDEO_RDO_NO_PART=1) ;;
    rdo_no_residual)
      env_vars+=(AV2ENC_VIDEO_RDO=1 AV2ENC_VIDEO_RDO_NO_RESIDUAL=1)
      ;;
    rdo_sse_only)
      env_vars+=(AV2ENC_VIDEO_RDO=1 AV2ENC_VIDEO_RDO_SSE_ONLY=1)
      ;;
    *) echo "unknown variant $variant" >&2; return 2 ;;
  esac
  case "$case_name" in
    2f_sb) gate_opts+=(--frames 2 --single-block) ;;
    2f_tree) gate_opts+=(--frames 2) ;;
    8f_sb) gate_opts+=(--frames 8 --single-block) ;;
    8f_tree) gate_opts+=(--frames 8) ;;
    *) echo "unknown case $case_name" >&2; return 2 ;;
  esac

  if [ ${#env_vars[@]} -eq 0 ]; then
    env DAV2D="$WRAPPER" REAL_DAV2D="$REAL_DAV2D" \
      "$bin" --input "$INPUT" "${gate_opts[@]}" --out "$out" \
      > "$log" 2>&1
  else
    env "${env_vars[@]}" DAV2D="$WRAPPER" REAL_DAV2D="$REAL_DAV2D" \
      "$bin" --input "$INPUT" "${gate_opts[@]}" --out "$out" \
      > "$log" 2>&1
  fi
}

metric() {
  local json=$1 field=$2
  python3 - "$json" "$field" <<'PY'
import json
import sys
with open(sys.argv[1]) as handle:
    report = json.load(handle)
field = sys.argv[2]
if field == "psnr_y":
    print(f"{report['psnr']['psnr_y']:.4f}")
elif field == "psnr_avg":
    print(f"{report['psnr']['psnr_avg']:.4f}")
elif field == "cvvdp":
    print(f"{report['cvvdp_jod']:.4f}")
else:
    raise SystemExit(f"unknown metric {field}")
PY
}

build

echo "== patch neutrality =="
NEUTRAL="$OUT/results/neutral"
mkdir -p "$NEUTRAL"
env DAV2D="$WRAPPER" REAL_DAV2D="$REAL_DAV2D" \
  "$BIN_BASE" --input "$INPUT" --frames 2 --out "$NEUTRAL/base.ivf" \
  > "$NEUTRAL/base.log" 2>&1
env DAV2D="$WRAPPER" REAL_DAV2D="$REAL_DAV2D" \
  "$BIN_RDO" --input "$INPUT" --frames 2 --out "$NEUTRAL/rdo_off.ivf" \
  > "$NEUTRAL/rdo_off.log" 2>&1
if cmp -s "$NEUTRAL/base.ivf" "$NEUTRAL/rdo_off.ivf"; then
  echo "  patch neutrality: RDO binary with the hook off is byte-identical"
else
  echo "  PATCH NEUTRALITY FAILED" >&2
  exit 1
fi

IFS=',' read -r -a CASE_LIST <<< "$CASES"
for case_name in "${CASE_LIST[@]}"; do
  for variant in base_default base_residual rdo_full rdo_nopart \
                 rdo_no_residual rdo_sse_only; do
    out_ivf="$OUT/results/$case_name.$variant.ivf"
    log="$OUT/results/$case_name.$variant.log"
    echo "== $case_name / $variant =="
    if [ "$variant" = "base_default" ] || [ "$variant" = "base_residual" ]; then
      run_gate "$BIN_BASE" "$case_name" "$variant" "$out_ivf" "$log"
    else
      run_gate "$BIN_RDO" "$case_name" "$variant" "$out_ivf" "$log"
    fi
    if ! grep -q "M2 GATE PASS" "$log"; then
      cat "$log" >&2
      echo "gate failed: $case_name / $variant" >&2
      exit 1
    fi
    if ! grep -q "recon exact" "$log"; then
      cat "$log" >&2
      echo "recon comparison did not pass: $case_name / $variant" >&2
      exit 1
    fi
    bytes=$(grep -o 'total_bytes=[0-9]*' "$log" | tail -1 | cut -d= -f2)
    case "$case_name" in
      8f_*) quality_ivf="$out_ivf.clip.ivf" ;;
      *) quality_ivf="$out_ivf" ;;
    esac
    if [ "$RUN_QUALITY" = "1" ]; then
      json="$OUT/results/$case_name.$variant.json"
      env DAV2D="$WRAPPER" REAL_DAV2D="$REAL_DAV2D" \
        python3 "$ROOT/tools/quality.py" --ref "$INPUT" \
        --ivf "$quality_ivf" --json "$json" \
        > "$OUT/results/$case_name.$variant.quality.log" 2>&1 || {
          cat "$OUT/results/$case_name.$variant.quality.log" >&2
          echo "quality failed: $case_name / $variant" >&2
          exit 1
        }
      psnr_y=$(metric "$json" psnr_y)
      psnr_avg=$(metric "$json" psnr_avg)
      cvvdp=$(metric "$json" cvvdp)
    else
      psnr_y=$(grep -o 'psnr_y=[0-9.]*' "$log" | tail -1 | cut -d= -f2)
      psnr_avg=$(grep -o 'psnr_yuv=[0-9.]*' "$log" | tail -1 | cut -d= -f2)
      cvvdp="nan"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$case_name" "$variant" "$bytes" "$psnr_y" "$psnr_avg" "$cvvdp" \
      "$quality_ivf" >> "$METRICS"
  done
done

python3 "$HERE/summarize_results.py" "$METRICS" "$OUT/results.md" \
  "$OUT/results.json"
echo
cat "$OUT/results.md"
echo
echo "results: $OUT/results.md"
