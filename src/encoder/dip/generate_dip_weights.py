#!/usr/bin/env python3
"""Regenerate the AV2 DIP weight table from the normative spec PDF.

The HTML version of the AV2 v1.0.0 specification omits numeric tables; the PDF
prints `Dip_Weights[6][64][11]` on pages 1095-1100.  This script extracts that
one table, validates its dimensions and a fixed content checksum, and writes:

  src/encoder/dip/dip_weights_data.inc
  src/encoder/dip/dip_weights.h

The .inc file is included by both the CPU reference (`kDipWeights`) and the
CUDA device header (`kDipWeightsDevice`), so the two copies cannot drift.

Usage:
  python3 src/encoder/dip/generate_dip_weights.py
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PDF = ROOT / "spec" / "AV2_Spec_v1.0.0.pdf"
OUT_INC = Path(__file__).resolve().parent / "dip_weights_data.inc"
OUT_H = Path(__file__).resolve().parent / "dip_weights.h"

# SHA-256 of the comma-separated 4224 values in mode-major raster order.
EXPECTED_SHA256 = "e5deedf1ed42fb944e233733c735593b04a3ae7116b51045bc692a11f53d04f7"
EXPECTED_SUM = 393140


def extract_values() -> list[int]:
    if not PDF.exists():
        raise SystemExit(f"missing spec PDF: {PDF}")
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("pypdf is required: python3 -m pip install pypdf") from exc

    reader = PdfReader(PDF)
    text = "\n".join((reader.pages[i].extract_text() or "") for i in range(1094, 1101))
    text = re.sub(r"AV2 Specification Page \d+ of \d+", "", text)
    text = re.sub(r"↑ Back to Table of Contents", "", text)
    idx = text.find("Dip_Weights")
    if idx < 0:
        raise SystemExit("Dip_Weights table not found in PDF pages 1095-1100")
    segment = text[idx:]
    start = segment.find("{")
    depth = 0
    end = None
    for pos, char in enumerate(segment[start:], start):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = pos + 1
                break
    if end is None:
        raise SystemExit("unterminated Dip_Weights initializer")
    values = [int(v) for v in re.findall(r"-?\d+", segment[start:end])]
    if len(values) != 6 * 64 * 11:
        raise SystemExit(
            f"Dip_Weights value count {len(values)} != {6 * 64 * 11}; "
            "refusing to guess a partial table"
        )
    digest = hashlib.sha256(",".join(map(str, values)).encode()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise SystemExit(
            f"Dip_Weights checksum {digest} != {EXPECTED_SHA256}; "
            "the PDF table changed or was parsed incorrectly"
        )
    if sum(values) != EXPECTED_SUM:
        raise SystemExit("Dip_Weights sum check failed")
    if min(values) < -32768 or max(values) > 32767:
        raise SystemExit("Dip_Weights values do not fit int16")
    return values


def write_inc(values: list[int]) -> None:
    lines: list[str] = []
    for mode in range(6):
        lines.append(f"    // dip_mode = {mode}")
        lines.append("    {")
        for row in range(64):
            base = mode * 704 + row * 11
            chunk = ", ".join(str(v) for v in values[base : base + 11])
            lines.append(f"        {{{chunk}}},")
        lines.append("    },")
    OUT_INC.write_text("\n".join(lines) + "\n")


def write_header() -> None:
    OUT_H.write_text(
        """// AV2 v1.0.0 DIP linear model weights.
//
// Normative source: AV2 Bitstream & Decoding Process Specification v1.0.0
// (2026-05-28, commit 38f28e7), §9.3 default CDF tables, table
// `Dip_Weights[6][64][11]` (PDF pages 1095-1100). The PDF is the only
// artefact that prints the numbers; the HTML specification omits them.
//
// Generated from spec/AV2_Spec_v1.0.0.pdf by
// src/encoder/dip/generate_dip_weights.py; do not hand-edit.
//
// Checksum (SHA-256 of the comma-separated values in raster order, mode
// major, 64 output positions, 11 features):
//   e5deedf1ed42fb944e233733c735593b04a3ae7116b51045bc692a11f53d04f7

#pragma once

#include <cstdint>

namespace av2enc::dip {

// Dip_Weights[mode][i * 8 + j][feature]. Values are signed Q10 model
// coefficients: the predictor computes (sum + 512) >> 10.  The initializer
// body is shared verbatim with the CUDA device table in
// src/cuda/dip/dip_device.cuh so the CPU and GPU cannot drift.
inline constexpr std::int16_t kDipWeights[6][64][11] = {
#include "encoder/dip/dip_weights_data.inc"
};

}  // namespace av2enc::dip
"""
    )


def main() -> int:
    values = extract_values()
    write_inc(values)
    write_header()
    print(f"wrote {OUT_INC} and {OUT_H} ({len(values)} weights)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
