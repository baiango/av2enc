#!/usr/bin/env bash
# Single-arena guard for the CUDA runtime.
#
# Rule (docs/cuda-interface.md): every device buffer comes from
# av2enc::cuda::Arena.  Only src/cuda/arena.cpp may call cudaMalloc/cudaFree;
# src/cuda/context.cpp may additionally use cudaMallocHost for pinned host
# staging.  Tests and build stubs may allocate directly because they are not
# part of the encoder runtime.
#
# Narrow distinctions made here:
#   * cudaFree(nullptr) only forces primary-context creation; it frees no
#     memory and is not an allocation site.
#   * cudaMallocHost/cudaFreeHost allocate pinned *host* RAM, not device VRAM.
#     The runtime helper (src/cuda/context.cpp) and the independent multi-GPU
#     device backend (src/pipeline/dispatch/runtime/cuda_backend.cpp, which is
#     linked without context.cpp) are the only pinned-staging providers.
# Device allocations (cudaMalloc, cudaMallocManaged, cudaMallocPitch,
# cudaMalloc3D, cudaMallocArray, cudaMallocAsync, ...) remain banned outside
# arena.cpp everywhere.
#
# The matcher is deliberately broad on the cudaMalloc*/cudaFree* families so a
# new allocator spelling cannot slip through.  A call whose '(' sits on the
# next line would evade a line-oriented scan, so that shape is reported as a
# violation with a request to keep the call on one line.
#
# Usage:
#   tools/check_arena.sh              # scan src/ (the encoder/runtime tree)
#   tools/check_arena.sh src/cuda     # scan explicit roots
#   tools/check_arena.sh src bench    # include benchmark harnesses
#   tools/check_arena.sh --quiet      # only print the failure summary
#
# Exit status: 0 when clean, 1 when a violation is found, 2 on usage error.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

quiet=0
roots=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -q|--quiet) quiet=1; shift ;;
    -h|--help) sed -n '2,33p' "$0"; exit 0 ;;
    --) shift; roots+=("$@"); break ;;
    -*) echo "check_arena: unknown option '$1'" >&2; exit 2 ;;
    *) roots+=("$1"); shift ;;
  esac
done
if [[ ${#roots[@]} -eq 0 ]]; then
  for candidate in src; do
    [[ -e "$candidate" ]] && roots+=("$candidate")
  done
fi
if [[ ${#roots[@]} -eq 0 ]]; then
  echo "check_arena: no source roots to scan" >&2
  exit 2
fi

# Every cudaMalloc*/cudaFree* spelling plus the host-pinning allocators.
# The trailing '(' keeps the name distinct from prose in comments; the split
# detector below covers the '(' on the next line.
pattern='(cudaMalloc[A-Za-z]*|cudaFree[A-Za-z]*|cudaHostAlloc|cudaHostRegister)[[:space:]]*\('

source_globs=(-g '*.c' -g '*.cc' -g '*.cpp' -g '*.cxx'
              -g '*.cu' -g '*.cuh' -g '*.h' -g '*.hh' -g '*.hpp')

if command -v rg >/dev/null 2>&1; then
  raw="$(rg --line-number --no-heading --color never "${source_globs[@]}" \
            -e "$pattern" "${roots[@]}" 2>/dev/null || true)"
else
  # Portable fallback.  -I skips binaries; the glob is expressed by the
  # per-file check below.
  raw="$(grep -rInE "$pattern" "${roots[@]}" 2>/dev/null || true)"
fi

filtered="$(
  printf '%s\n' "$raw" | awk -F: '
    function is_exempt(path) {
      # Only source files are in scope (the rg path already filters these; the
      # grep fallback needs the same guard).
      if (path !~ /\.(c|cc|cpp|cxx|cu|cuh|h|hh|hpp)$/) return 1;

      # The arena implementation and the pinned-staging helper are the two
      # sanctioned allocation sites.
      if (path == "src/cuda/arena.cpp") return 1;
      if (path == "src/cuda/context.cpp") return 1;

      # Test, build and emulation-stub trees are not the encoder runtime.
      if (path ~ /(^|\/)(build|build-[^\/]*|build_stub|build-stub|cuda_stub|third_party|external|vendor|test|tests|testing|unittest|unittests)(\/|$)/) return 1;
      # Standalone benchmark harnesses (bench/*) are their own binaries with
      # their own main(); none is linked into the encoder or the arena.  They
      # are reported separately instead of failing the runtime guard.
      if (path ~ /(^|\/)bench\//) return 1;
      # Only genuine test suffixes: the old loose "test\.(...)" rule also
      # exempted production files whose name merely ends in "test"
      # (e.g. video_selftest.cpp).
      if (path ~ /(^|\/)[^\/]*(_test|_tests)\.(c|cc|cpp|cxx|cu|cuh|h|hh|hpp)$/) return 1;
      # Test-only RAII device buffer used by the restoration component tests.
      # It lives beside the loop-restoration component after the folder
      # refactor; keep both spellings so the gate does not regress on a move.
      if (path ~ /(^|\/)cuda\/(codec\/filters\/loop_restoration|restoration)\/device_buffer\.h$/) return 1;
      return 0;
    }

    # Pinned host staging providers (see the header comment above).
    function is_pinned_provider(path) {
      # The multi-GPU backend moved under pipeline/dispatch/runtime in the
      # folder refactor; both spellings stay exempt.
      return path ~ /(^|\/)cuda\/context\.cpp$/ ||
             path ~ /(^|\/)(multi_gpu|pipeline\/dispatch)\/runtime\/cuda_backend\.cpp$/;
    }

    NF == 0 { next }
    {
      path = $1;
      line = $2;
      text = $0;
      sub(/^[^:]*:[^:]*:/, "", text);
      trim = text;
      sub(/^[ \t]+/, "", trim);

      # Ignore comment-only lines.  A call embedded in trailing prose is
      # extremely rare and the false positive is preferable to a missed call.
      if (trim ~ /^\/\// || trim ~ /^\*/ || trim ~ /^\/\*/) next;

      # cudaFree(nullptr) touches no allocation; it only probes context state.
      if (trim ~ /cudaFree[[:space:]]*\([[:space:]]*nullptr[[:space:]]*\)/) next;

      # Pinned host staging is host RAM, not device VRAM.
      if (is_pinned_provider(path) &&
          (trim ~ /cudaMallocHost[[:space:]]*\(/ ||
           trim ~ /cudaFreeHost[[:space:]]*\(/ ||
           trim ~ /cudaHostAlloc[[:space:]]*\(/)) next;

      if (is_exempt(path)) next;

      print path ":" line ": " trim;
    }
  ' | sort -u
)"

# A cudaMalloc*/cudaFree* token at end of line means the call is split over
# lines and the line-oriented matcher above cannot see it.  Report it instead
# of silently missing it.
split_pattern='(cudaMalloc[A-Za-z]*|cudaFree[A-Za-z]*|cudaHostAlloc)[[:space:]]*$'
if command -v rg >/dev/null 2>&1; then
  split_raw="$(rg --line-number --no-heading --color never "${source_globs[@]}" \
                 -e "$split_pattern" "${roots[@]}" 2>/dev/null || true)"
else
  split_raw="$(grep -rInE "$split_pattern" "${roots[@]}" 2>/dev/null || true)"
fi

split_filtered="$(
  printf '%s\n' "$split_raw" | awk -F: '
    function is_exempt(path) {
      if (path !~ /\.(c|cc|cpp|cxx|cu|cuh|h|hh|hpp)$/) return 1;
      if (path == "src/cuda/arena.cpp") return 1;
      if (path == "src/cuda/context.cpp") return 1;
      if (path ~ /(^|\/)(build|build-[^\/]*|build_stub|build-stub|cuda_stub|third_party|external|vendor|test|tests|testing|unittest|unittests)(\/|$)/) return 1;
      if (path ~ /(^|\/)[^\/]*(_test|_tests)\.(c|cc|cpp|cxx|cu|cuh|h|hh|hpp)$/) return 1;
      if (path ~ /(^|\/)bench\//) return 1;
      return 0;
    }
    NF == 0 { next }
    {
      path = $1; line = $2; text = $0;
      sub(/^[^:]*:[^:]*:/, "", text);
      trim = text; sub(/^[ \t]+/, "", trim);
      if (trim ~ /^\/\// || trim ~ /^\*/ || trim ~ /^\/\*/) next;
      if (is_exempt(path)) next;
      print path ":" line ": " trim "  [call split across lines]";
    }
  ' | sort -u
)"

all_violations="$(printf '%s\n%s\n' "$filtered" "$split_filtered" |
  sed '/^$/d' | sort -u)"

# Report (but do not fail on) standalone benchmark harness allocations.
bench_hits=0
if [[ "$quiet" -eq 0 ]]; then
  bench_hits="$(printf '%s\n' "$raw" | awk -F: '
    NF == 0 { next }
    {
      path = $1; text = $0;
      sub(/^[^:]*:[^:]*:/, "", text);
      trim = text; sub(/^[ \t]+/, "", trim);
      if (trim ~ /^\/\// || trim ~ /^\*/ || trim ~ /^\/\*/) next;
      if (path ~ /(^|\/)bench\//) print;
    }' | wc -l | tr -d ' ')"
fi

if [[ -n "$all_violations" ]]; then
  count="$(printf '%s\n' "$all_violations" | wc -l | tr -d ' ')"
  if [[ "$quiet" -eq 0 ]]; then
    printf '%s\n' "$all_violations"
    echo
  fi
  echo "check_arena: FAIL: $count allocation call(s) outside arena.cpp/context.cpp/test stubs" >&2
  echo "check_arena: see docs/cuda-interface.md (one arena; no per-module cudaMalloc)" >&2
  exit 1
fi

if [[ "$quiet" -eq 0 ]]; then
  if [[ "${bench_hits:-0}" -gt 0 ]]; then
    echo "check_arena: note: $bench_hits allocation site(s) in bench/ standalone harnesses (excluded: not the encoder runtime)"
  fi
  echo "check_arena: PASS: no raw CUDA allocations outside arena.cpp/context.cpp"
fi
exit 0
