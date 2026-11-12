#!/usr/bin/env bash
# Device-visibility audit for the shared chroma loop-restoration header.
#
# Checks that `lr_chroma_device.cuh` contains no host-only function reachable
# from a kernel, no mutable global state, and no namespace-scope table that
# would need host storage in the device pass.  When nvcc is available it also
# compiles a probe kernel that calls every entry point, which fails at compile
# time if any symbol is not device-callable.
set -euo pipefail
cd "$(dirname "$0")"

FILE="lr_chroma_device.cuh"
FAIL=0

say() { printf 'audit: %s\n' "$*"; }
bad() {
  printf 'AUDIT FAIL: %s\n' "$*" >&2
  FAIL=1
}

# 1. Every lrc_* function definition is marked AV2ENC_LRC_HD.  Our style puts
#    the return type on the line after the macro, so an unmarked definition is
#    a line at column 0 that names lrc_* and opens a parameter list.
unmarked="$(grep -nE '^[A-Za-z_].*[[:space:]]lrc_[a-z0-9_]+[[:space:]]*\(' "$FILE" \
             | grep -v '^[0-9]*:AV2ENC_LRC_HD' || true)"
if [ -n "$unmarked" ]; then
  bad "unmarked device function definition(s):"
  printf '%s\n' "$unmarked" >&2
fi
marked="$(grep -c '^AV2ENC_LRC_HD' "$FILE" || true)"
say "AV2ENC_LRC_HD definitions: $marked"

# 2. Template definitions must be followed by the marker.
if grep -nE '^template <' "$FILE" | while IFS=: read -r line _; do
     next=$((line + 1))
     sed -n "${next}p" "$FILE" | grep -q '^AV2ENC_LRC_HD' || exit 1
   done; then
  say "template definitions marked: yes"
else
  bad "template definition without AV2ENC_LRC_HD"
fi

# 3. No device-hostile constructs.
if grep -nE '__constant__|__shared__|cudaMalloc|cudaFree|cudaMemcpy' "$FILE"; then
  bad "allocation or CUDA-runtime call in the shared device header"
fi
if grep -nE 'std::(vector|string|array|map|algorithm)' "$FILE"; then
  bad "host-only std:: container/call in the shared device header"
fi
if grep -nE '^(static|const|inline constexpr)[[:space:]]+[A-Za-z_].*=' "$FILE"; then
  bad "namespace-scope object in the shared device header"
fi

# 4. The only tables are the AV2ENC_LR_TABLE ones from restoration_common.h.
tables="$(grep -c 'kWienerNsConfigUv\|kWienerFilters420' "$FILE" || true)"
say "normative table references (device tables via AV2ENC_LR_TABLE): $tables"

# 5. Compile probe when nvcc exists.
if command -v nvcc >/dev/null 2>&1; then
  ARCH="${ARCH:-sm_86}"
  TMP="$(mktemp -d "${TMPDIR:-/tmp}/lr_chroma_audit.XXXXXX")"
  trap 'rm -rf "$TMP"' EXIT
  cat > "$TMP/probe.cu" <<'PROBE'
#include "lr_chroma_device.cuh"

using namespace av2enc::cuda::lr_chroma;

__global__ void lrc_visibility_probe(
    LrChromaCore core, const LrChromaUnit* units, int count,
    std::int16_t* resolved, std::uint8_t* active, LrChromaBankState* bank) {
  LrChromaBlock block;
  const bool ctx = lrc_block_context(core, 0, 0, 0, &block);
  (void)lrc_count_units(64, 64);
  (void)lrc_plane_width(8, 1);
  (void)lrc_plane_height(8, 1);
  (void)lrc_present_coeffs(2);
  (void)lrc_read_sample<unsigned char>(core.chroma[0].cdef, 64, 0, 0);
  (void)lrc_read_sample<unsigned short>(core.chroma[0].cdef, 64, 0, 0);
  if (ctx) {
    (void)lrc_source_sample<unsigned char>(core, block, 0, 0, 0);
    (void)lrc_source_sample<unsigned short>(core, block, 0, 0, 0);
    (void)lrc_luma_source_sample<unsigned char>(core, block, 0, 0);
    (void)lrc_luma_sample<unsigned short>(core, block, 0, 0);
    lrc_copy_block<unsigned char>(core, 0, block.x, block.y, block.w, block.h);
    lrc_filter_block<unsigned short>(core, block, 0, resolved);
  }
  lrc_process_block<unsigned char>(core, 0, 0, 0, resolved, active);
  lrc_process_block<unsigned short>(core, 1, 0, 0, resolved, active);
  lrc_resolve_plane(units, count, resolved, active, bank);
}
PROBE
  if nvcc -std=c++20 -arch="$ARCH" --expt-relaxed-constexpr -c \
      -I"$PWD" -I"$PWD/../.." "$TMP/probe.cu" -o "$TMP/probe.o" \
      >"$TMP/probe.log" 2>&1; then
    say "nvcc device-visibility probe: PASS ($ARCH)"
  else
    bad "nvcc device-visibility probe failed:"
    cat "$TMP/probe.log" >&2
  fi
else
  say "nvcc not found: textual audit only"
fi

if [ "$FAIL" -ne 0 ]; then
  exit 1
fi
say "PASS"
