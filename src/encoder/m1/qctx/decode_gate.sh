#!/usr/bin/env bash
# End-to-end coefficient-CDF qctx decode gate.
#
# Encodes the same still picture at base_q_idx values that select every
# COEFF_CDF_Q_CTXS slice (0..3) and requires the pinned dav2d decoder to
# decode each stream.  A wrong slice makes the coefficient CDFs diverge from
# the decoder's, so the stream fails to decode; this is the real gate, not a
# model check.
#
# This script does not build or modify the encoder.  It uses an already-built
# `av2enc` binary:
#   AV2ENC=/path/to/av2enc  (default: first build*/av2enc found)
#
# Exit codes: 0 pass, 1 failure, 77 skip (no encoder, decoder or test media).
#
# The current M1 CPU path accepts base_q_idx 1..255 and derives the slice in
# m1_syntax.cpp from config_.base_q_idx.  If that integration is absent, the
# encoded streams will still decode at QP 90 and fail at QP 91, which is
# exactly what this gate reports.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../../.." && pwd)"
SOURCE="${QCTX_Y4M:-${ROOT}/tests/data/m0_64x64_1f.y4m}"

if [[ -z "${AV2ENC:-}" ]]; then
  # Prefer the canonical CPU M1 build dirs before older experiment builds.
  for candidate in "${ROOT}/build-m1/av2enc" "${ROOT}/build/av2enc"; do
    if [[ -x "${candidate}" ]]; then
      AV2ENC="${candidate}"
      break
    fi
  done
fi
if [[ -z "${AV2ENC:-}" ]]; then
  for candidate in "${ROOT}"/build*/av2enc; do
    if [[ -x "${candidate}" ]]; then
      AV2ENC="${candidate}"
      break
    fi
  done
fi

if [[ -z "${AV2ENC:-}" || ! -x "${AV2ENC}" ]]; then
  echo "SKIP: no av2enc binary (set AV2ENC=/path/to/av2enc)" >&2
  exit 77
fi
echo "qctx_decode_gate: encoder ${AV2ENC}"
if [[ ! -f "${SOURCE}" ]]; then
  echo "SKIP: no test media at ${SOURCE} (set QCTX_Y4M=...)" >&2
  exit 77
fi

DAV2D_BIN="${DAV2D:-}"
if [[ -z "${DAV2D_BIN}" ]]; then
  for candidate in "${ROOT}/tools/bin/dav2d" "${ROOT}/tools/bin/dav2d-macos"; do
    if [[ -x "${candidate}" ]]; then
      DAV2D_BIN="${candidate}"
      break
    fi
  done
fi
if [[ -z "${DAV2D_BIN}" || ! -x "${DAV2D_BIN}" ]]; then
  echo "SKIP: no dav2d decoder (set DAV2D=/path/to/dav2d)" >&2
  exit 77
fi

TMP="$(mktemp -d "${TMPDIR:-/tmp}/qctx-decode.XXXXXX")"
trap 'rm -rf "${TMP}"' EXIT

# One QP per slice plus the exact boundaries.  10-bit MaxQ is 303; 255 is the
# 8-bit maximum this test media exercises.
QPS=(90 91 140 141 190 191 255)

slice_for_qp() {
  local q=$1
  if (( q <= 90 )); then echo 0
  elif (( q <= 140 )); then echo 1
  elif (( q <= 190 )); then echo 2
  else echo 3
  fi
}

declare -a CHECKSUMS=()
for qp in "${QPS[@]}"; do
  ivf="${TMP}/qp${qp}.ivf"
  log="${TMP}/qp${qp}.log"
  if ! "${AV2ENC}" "${SOURCE}" -o "${ivf}" --m1 --qp "${qp}" \
      --frames 1 >"${log}" 2>&1; then
    echo "FAIL: encode qp=${qp} (slice $(slice_for_qp "${qp}"))" >&2
    tail -8 "${log}" >&2
    exit 1
  fi
  if ! DAV2D="${DAV2D_BIN}" "${ROOT}/tools/validate_dav2d.sh" "${ivf}" \
      >>"${log}" 2>&1; then
    echo "FAIL: dav2d rejected qp=${qp} (slice $(slice_for_qp "${qp}"))" >&2
    tail -20 "${log}" >&2
    exit 1
  fi
  CHECKSUMS+=("$(cksum "${ivf}" | awk '{print $1}')")
  echo "qp=${qp} slice=$(slice_for_qp "${qp}") decode OK ($(wc -c <"${ivf}" | tr -d ' ') bytes)"
done

# A QP clamp to 90 would make these streams identical; require every
# representative QP to produce a distinct bitstream.
unique_count=$(printf '%s\n' "${CHECKSUMS[@]}" | sort -u | wc -l | tr -d ' ')
if [[ "${unique_count}" -ne "${#CHECKSUMS[@]}" ]]; then
  echo "FAIL: ${unique_count} unique streams for ${#CHECKSUMS[@]} QPs;" \
       "the encoder appears to clamp base_q_idx" >&2
  exit 1
fi

echo "qctx_decode_gate: PASS (${#QPS[@]} streams, slices 0..3)"
