#!/usr/bin/env bash
# Run every CUDA module build/test script and print a PASS/FAIL table.
#
# Each suite is a self-contained `build*_test.sh` / `build_and_test.sh` /
# `build_*_gate.sh` under src/cuda/.  The script discovers them from the tree,
# so a new module is picked up without editing this file (no gate inventory).
#
# Suites that invoke nvcc are labelled "gpu"; the rest are host-emulation
# suites and need no device.  Both kinds are run; a missing nvcc/device shows up
# as FAIL or SKIP from the suite itself, not from a hidden filter here.
#
# Suites in the same directory share a default BUILD_DIR, so they are pinned to
# one worker each; different directories run in parallel.
#
# Usage:
#   tools/cuda_module_suites.sh                     # run everything, N workers (N=CPU count, max 16)
#   tools/cuda_module_suites.sh --list              # print the discovered suites
#   tools/cuda_module_suites.sh -f transform        # only paths matching regex
#   tools/cuda_module_suites.sh --kind host         # host-emulation suites only
#   tools/cuda_module_suites.sh --kind gpu          # nvcc suites only
#   tools/cuda_module_suites.sh -j 8 --timeout 600
#   tools/cuda_module_suites.sh --logs /tmp/cuda-suites
#
# Exit status: 0 when every selected suite passed (or was skipped), 1 when any
# suite failed or timed out, 2 on usage error.
set -uo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Default worker count is derived from the host after argument parsing; a
# compile-heavy suite is ~single-threaded, so 4 workers left most of a shared
# 28-core box idle for the whole run.
JOBS=""
TIMEOUT=900
LOGS="build-cuda-suites/logs"
FILTER=""
KIND="all"
LIST_ONLY=0
QUIET=0
NO_COLOR="${NO_COLOR:-}"

usage() {
  sed -n '2,27p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -l|--list) LIST_ONLY=1; shift ;;
    -f|--filter) FILTER="${2:-}"; shift 2 ;;
    -j|--jobs) JOBS="${2:-}"; shift 2 ;;
    -t|--timeout) TIMEOUT="${2:-}"; shift 2 ;;
    --logs) LOGS="${2:-}"; shift 2 ;;
    --kind) KIND="${2:-}"; shift 2 ;;
    -q|--quiet) QUIET=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "cuda_module_suites: unknown option '$1'" >&2; exit 2 ;;
  esac
done

case "$KIND" in
  all|host|gpu) ;;
  *) echo "cuda_module_suites: --kind must be all, host or gpu" >&2; exit 2 ;;
esac
if [[ -z "$JOBS" ]]; then
  if command -v nproc >/dev/null 2>&1; then
    JOBS="$(nproc)"
  elif command -v sysctl >/dev/null 2>&1; then
    JOBS="$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"
  else
    JOBS=4
  fi
  # Compilation is memory-hungry; 16 concurrent c++ processes is plenty.
  (( JOBS > 16 )) && JOBS=16
fi
case "$JOBS" in
  ''|*[!0-9]*|0) echo "cuda_module_suites: --jobs needs a positive integer" >&2; exit 2 ;;
esac
case "$TIMEOUT" in
  ''|*[!0-9]*|0) echo "cuda_module_suites: --timeout needs a positive integer" >&2; exit 2 ;;
esac

# Discover suites.  Only build/test entry points; helper scripts such as
# build_plan_dump.sh are not suites.
scripts_file="$(mktemp)"
find src/cuda -name 'build*.sh' -type f -print 2>/dev/null | sort |
  while IFS= read -r path; do
    base="$(basename "$path")"
    case "$base" in
      build_and_test.sh|*_test.sh|*_gate.sh) printf '%s\n' "$path" ;;
    esac
  done >"$scripts_file"

# nvcc-backed vs host-emulation.
kind_of() {
  if grep -qE '(^|[^[:alnum:]_])nvcc([[:space:]]|$)' "$1"; then
    echo gpu
  else
    echo host
  fi
}

selected_file="$(mktemp)"
while IFS= read -r path; do
  [[ -n "$FILTER" ]] && ! printf '%s\n' "$path" | grep -Eq -- "$FILTER" && continue
  k="$(kind_of "$path")"
  [[ "$KIND" != all && "$k" != "$KIND" ]] && continue
  printf '%s\t%s\n' "$k" "$path" >>"$selected_file"
done <"$scripts_file"

total="$(wc -l <"$selected_file" | tr -d ' ')"
if [[ "$total" -eq 0 ]]; then
  echo "cuda_module_suites: no suites matched" >&2
  rm -f "$scripts_file" "$selected_file"
  exit 2
fi

if [[ "$LIST_ONLY" -eq 1 ]]; then
  while IFS=$'\t' read -r k path; do printf '%-4s %s\n' "$k" "$path"; done <"$selected_file"
  echo "-- $total suite(s)"
  rm -f "$scripts_file" "$selected_file"
  exit 0
fi

mkdir -p "$LOGS"
rm -f "$LOGS"/*.log "$LOGS"/*.status 2>/dev/null

# Group suites by directory so same-dir suites share a worker (shared BUILD_DIR).
workers_file="$(mktemp)"
awk -F'\t' '{ d=$2; sub(/\/[^\/]*$/, "", d); print d "\t" $1 "\t" $2 }' "$selected_file" |
  sort >"$workers_file"

run_one() {
  local kind="$1" path="$2"
  local slug log start end status rc
  slug="$(printf '%s' "$path" | tr '/' '_' | sed 's/\.sh$//')"
  log="$LOGS/$slug.log"
  start="$(date +%s)"
  # Absolute path: suites compute their repo root from "$0" and some cd twice.
  # stdin is /dev/null so a suite can never eat the worker's job list.
  ( cd "$ROOT" && timeout "$TIMEOUT" bash "$ROOT/$path" ) \
    </dev/null >"$log" 2>&1
  rc=$?
  end="$(date +%s)"
  case "$rc" in
    0) status=PASS ;;
    77) status=SKIP ;;
    124) status=TIMEOUT ;;
    *) status=FAIL ;;
  esac
  # Exit 77 is the documented "CUDA/inputs unavailable" code in this tree; a
  # suite that prints a SKIP note but finishes its checks still exits 0.
  printf '%s\t%s\t%s\t%s\t%s\n' "$status" "$((end - start))" "$kind" "$path" "$log" \
    >"$LOGS/$slug.status"
  if [[ "$QUIET" -eq 0 ]]; then
    printf '%-7s %4ss  %-4s %s\n' "$status" "$((end - start))" "$kind" "$path" >&2
  fi
}

# Distribute directory groups round-robin; each worker runs its groups serially.
worker_main() {
  local worker_id="$1"
  local idx=0 dir kind path
  local cur_dir=""
  while IFS=$'\t' read -r dir kind path; do
    if [[ "$dir" != "$cur_dir" ]]; then
      idx=$((idx + 1))
      cur_dir="$dir"
    fi
    if (( (idx - 1) % JOBS != worker_id )); then
      continue
    fi
    run_one "$kind" "$path"
  done <"$workers_file"
}

if [[ "$JOBS" -gt 1 ]]; then
  pids=()
  for ((w = 0; w < JOBS; w++)); do
    worker_main "$w" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
else
  worker_main 0
fi

echo
printf '%-7s %6s  %-4s %s\n' STATUS TIME KIND SUITE
printf '%-7s %6s  %-4s %s\n' ------- ------ ---- -----
pass=0 fail=0 skip=0 timed=0
fail_lines=""
while IFS=$'\t' read -r status secs kind path log; do
  printf '%-7s %5ss  %-4s %s\n' "$status" "$secs" "$kind" "$path"
  case "$status" in
    PASS) pass=$((pass + 1)) ;;
    SKIP) skip=$((skip + 1)) ;;
    TIMEOUT) timed=$((timed + 1)); fail_lines+="  TIMEOUT $path  ($log)"$'\n' ;;
    *) fail=$((fail + 1)); fail_lines+="  FAIL    $path  ($log)"$'\n' ;;
  esac
done < <(sort "$LOGS"/*.status)

echo
echo "total $total: pass $pass, fail $fail, skip $skip, timeout $timed"
echo "logs: ${LOGS#"$ROOT/"}"
if [[ -n "$fail_lines" ]]; then
  printf '%s' "$fail_lines"
fi

rm -f "$scripts_file" "$selected_file" "$workers_file"
if [[ "$fail" -ne 0 || "$timed" -ne 0 ]]; then
  exit 1
fi
exit 0
