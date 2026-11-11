#!/usr/bin/env bash
# Self-test for verify_m3_gpu.sh verdict logic, using a mock encoder.
#
# This does NOT test any GPU code.  It proves that the gate classifies the
# failure modes correctly without ever returning a false PASS:
#
#   reject   GPU rejects a feature flag            -> SKIP-REJECTED (3)
#   staged   GPU emits CPU-reference bytes         -> SKIP-CPU-STAGED (3)
#   ignore   GPU accepts but drops a flag          -> IGNORED (1)
#   noeffect feature does not change CPU bytes     -> SKIP-NO-EFFECT (3)
#   noattest identical bytes, no nsys              -> SKIP-UNATTESTED (3)
#   pass     identical bytes + nsys kernels        -> PASS (0)
#   stats    --attest stats with a GPU stats file  -> PASS (0)
#   nondet   second GPU encode differs             -> NONDETERMINISTIC (1)
#   san      memcheck errors                       -> SANITIZER (1)
#   matrix   readiness table splits READY/NOT-READY
#
# Usage:
#   tools/m3_gpu/selftest.sh [--encoder build-m1/av2enc]
#
# Requires: an av2enc CPU build, dav2d, ffmpeg (all local).  No CUDA needed.
set -uo pipefail

SELF=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$SELF/../.." && pwd)
ENCODER=${AV2ENC:-$ROOT/build-m1/av2enc}
DECODER=${DAV2D:-}

while [ $# -gt 0 ]; do
    case "$1" in
        --encoder) ENCODER=$2; shift 2 ;;
        --decoder) DECODER=$2; shift 2 ;;
        -h|--help) sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "selftest: unknown argument $1" >&2; exit 2 ;;
    esac
done

[ -x "$ENCODER" ] || { echo "selftest: no encoder at $ENCODER" >&2; exit 2; }
if [ -z "$DECODER" ]; then
    for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
        [ -x "$candidate" ] && { DECODER=$candidate; break; }
    done
fi
[ -n "$DECODER" ] && [ -x "$DECODER" ] || { echo "selftest: no dav2d" >&2; exit 2; }
command -v ffmpeg >/dev/null 2>&1 || { echo "selftest: no ffmpeg" >&2; exit 2; }

TMP=$(mktemp -d "${TMPDIR:-/tmp}/m3_gpu_selftest.XXXXXX")
trap 'rm -rf "$TMP"' EXIT

cat > "$TMP/av2enc" <<'MOCK'
#!/usr/bin/env bash
set -uo pipefail
REAL=${MOCK_REAL:?}
scenario=${MOCK_SCENARIO:-ok}
args=("$@")
gpu=0; feats=(); out=""; prev=0
for a in "${args[@]}"; do
  [ "$a" = "--m1-gpu" ] && gpu=1
  case "$a" in --m1-cdef|--m1-ccso|--m1-rdoq|--m1-tx-type|--m1-rdo-intra|--m1-uv-full) feats+=("$a") ;; esac
  if [ "$prev" = 1 ]; then out=$a; prev=0; fi
  [ "$a" = "-o" ] && prev=1
done
has() { local f; for f in ${feats[@]+"${feats[@]}"}; do [ "$f" = "$1" ] && return 0; done; return 1; }
strip_features() {
  local a keep=()
  for a in "${args[@]}"; do
    case "$a" in --m1-cdef|--m1-ccso|--m1-rdoq|--m1-tx-type|--m1-rdo-intra|--m1-uv-full|--m1-cdef-strengths|--m1-ccso-bands) continue ;; esac
    keep+=("$a")
  done
  args=(${keep[@]+"${keep[@]}"})
}
if [ "$gpu" = 0 ]; then
  [ "$scenario" = noeffect ] && strip_features
  exec "$REAL" ${args[@]+"${args[@]}"}
fi
[ "$scenario" = reject ] && has --m1-cdef && {
  echo "av2enc: error: M1 GPU: --m1-cdef/--m1-rdoq/--m1-tx-type/--m1-rdo-intra are not ported to the GPU pipeline yet" >&2
  exit 2
}
[ "$scenario" = ignore ] && has --m1-ccso && strip_features
cpuargs=()
for a in "${args[@]}"; do
  [ "$a" = "--m1-gpu" ] && { cpuargs+=(--m1); continue; }
  cpuargs+=("$a")
done
[ "$scenario" = nondet ] && case "$out" in *gpu-run2*) cpuargs+=(--adaptive-cdf) ;; esac
"$REAL" "${cpuargs[@]}" || exit $?
[ "$scenario" = staged ] && echo "av2enc: note: frame_bytes_from_cpu=1; frame bytes came from the CPU M1 reference (linked stage hooks: 0)" >&2
if [ -n "${AV2ENC_GPU_STATS:-}" ] && [ "${MOCK_NO_STATS:-0}" = 0 ]; then
  if [ "$scenario" = staged ]; then
    printf '{"backend":"cpu-reference","frame_bytes_from_cpu":true}\n' >"$AV2ENC_GPU_STATS"
  else
    printf '{"backend":"gpu","frame_bytes_from_cpu":false,"kernels":42}\n' >"$AV2ENC_GPU_STATS"
  fi
fi
echo "m1gpu: backend=gpu stages: plan:gpu search:gpu tx:gpu" >&2
exit 0
MOCK

cat > "$TMP/nsys" <<'MOCK'
#!/usr/bin/env bash
set -uo pipefail
cmd=$1; shift
if [ "$cmd" = stats ]; then
  echo "Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name"
  echo "100.0,1000,7,100,100,100,100,0.0,m1_kernel"
  exit 0
fi
if [ "$cmd" = profile ]; then
  prefix=""; rest=()
  while [ $# -gt 0 ]; do
    case "$1" in
      -o) prefix=$2; shift 2 ;;
      --force-overwrite|--trace=*|--sample=*|--cpuctxsw=*) shift ;;
      true|false) shift ;;
      *) rest=("$@"); break ;;
    esac
  done
  "${rest[@]}"; rc=$?
  [ -n "$prefix" ] && printf 'fake\n' > "$prefix.nsys-rep"
  exit $rc
fi
exit 1
MOCK

cat > "$TMP/compute-sanitizer" <<'MOCK'
#!/usr/bin/env bash
set -uo pipefail
cmd=()
while [ $# -gt 0 ]; do
  case "$1" in
    --tool|--error-exitcode|--target-processes) shift 2 ;;
    *) cmd=("$@"); break ;;
  esac
done
"${cmd[@]}"; rc=$?
n=${MOCK_SAN_ERRORS:-0}
printf '========= ERROR SUMMARY: %s errors\n' "$n"
[ "$n" -gt 0 ] && exit 99
exit $rc
MOCK

chmod +x "$TMP/av2enc" "$TMP/nsys" "$TMP/compute-sanitizer"

export MOCK_REAL=$ENCODER
export NSYS="$TMP/nsys"
export COMPUTE_SANITIZER="$TMP/compute-sanitizer"

pass=0; fail=0

check() {
    local name=$1 want_rc=$2 want_class=$3 scen=$4
    shift 4
    MOCK_SCENARIO=$scen "$SELF/verify_m3_gpu.sh" \
        --encoder "$TMP/av2enc" --decoder "$DECODER" \
        --size 64x64 --workdir "$TMP/$name" "$@" \
        >"$TMP/$name.log" 2>&1
    local rc=$?
    local cls
    cls=$(grep -E '^CLASS: ' "$TMP/$name.log" | tail -1 | sed 's/^CLASS: //')
    if [ "$rc" = "$want_rc" ] && [ "$cls" = "$want_class" ]; then
        printf 'PASS  %-22s rc=%s class=%s\n' "$name" "$rc" "$cls"
        pass=$((pass + 1))
    else
        printf 'FAIL  %-22s rc=%s want=%s class=%s want=%s\n' \
            "$name" "$rc" "$want_rc" "$cls" "$want_class"
        tail -5 "$TMP/$name.log" | sed 's/^/      /'
        fail=$((fail + 1))
    fi
}

check reject      3 "SKIP-REJECTED"    reject   --features cdef
check staged      3 "SKIP-CPU-STAGED"  staged   --features cdef
check ignore      1 "IGNORED"          ignore   --features ccso
check noeffect    3 "SKIP-NO-EFFECT"   noeffect --features cdef
export MOCK_NO_STATS=1
check noattest    3 "SKIP-UNATTESTED"  ok       --features uv-full --no-nsys
unset MOCK_NO_STATS
check pass        0 "PASS"             ok       --features uv-full
check stats       0 "PASS"             ok       --features uv-full --attest stats
check nondet      1 "NONDETERMINISTIC" nondet   --features uv-full --determinism
export MOCK_SAN_ERRORS=3
check san         1 "SANITIZER"        ok       --features uv-full --sanitize --attest stats
unset MOCK_SAN_ERRORS

MOCK_SAN_ERRORS=0 MOCK_SCENARIO=ok "$SELF/verify_m3_gpu.sh" \
    --encoder "$TMP/av2enc" --decoder "$DECODER" --size 64x64 \
    --features uv-full --sanitize --workdir "$TMP/sanok" \
    >"$TMP/sanok.log" 2>&1
if [ $? -eq 0 ] && grep -q '^CLASS: PASS' "$TMP/sanok.log"; then
    printf 'PASS  %-22s rc=0 class=PASS\n' "sanok"
    pass=$((pass + 1))
else
    printf 'FAIL  %-22s\n' "sanok"
    tail -5 "$TMP/sanok.log" | sed 's/^/      /'
    fail=$((fail + 1))
fi

MOCK_SCENARIO=reject NSYS="$TMP/nsys" "$SELF/verify_m3_gpu.sh" --matrix \
    --encoder "$TMP/av2enc" --decoder "$DECODER" --sizes 64x64 \
    --modes baseline,cdef --matrix-budget 0 --workdir "$TMP/matrix" \
    >"$TMP/matrix.log" 2>&1
if grep -q 'GPU-READY FEATURES:     (none)' "$TMP/matrix.log" &&
   grep -q 'cdef .*NOT-READY.*SKIP-REJECTED' "$TMP/matrix.log"; then
    printf 'PASS  %-22s readiness split\n' "matrix"
    pass=$((pass + 1))
else
    printf 'FAIL  %-22s readiness split\n' "matrix"
    grep -E 'NOT-READY|GPU-READY' "$TMP/matrix.log" | sed 's/^/      /'
    fail=$((fail + 1))
fi

echo ""
echo "selftest: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
