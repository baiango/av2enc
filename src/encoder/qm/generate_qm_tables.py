#!/usr/bin/env python3
"""Generate the normative AV2 §9.4 quantizer-matrix tables.

Source: ``spec/AV2_Spec_v1.0.0.pdf`` §9.4.3 (the HTML draft only says the
arrays live in ``qm_offset.h`` / ``quantizer_matrix.h``; the PDF renders the
numbers).

The generator is deliberately strict.  It refuses to emit anything unless:

  * the PDF SHA-256 matches the pinned AV2 v1.0.0 file,
  * ``Qm_Offset`` has 25 entries and ``Quantizer_Matrix`` is [15][2][3600],
  * every 8x8/8x4/4x8-style packed block fits exactly inside 3600 entries and
    the 25 coded TX sizes map to a block of ``min(32,w) * min(32,h)`` entries,
  * the informative §9.4.2 subsampling derivation reproduces every packed
    matrix for all 15 levels and both plane types.

Outputs (all under this directory, so no other agent's files are touched):

  * ``qm_tables.h``  - inline constexpr C++ tables
  * ``qm_tables.json`` - audit copy with the same numbers

Usage:
    python3 generate_qm_tables.py [--check]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PDF = REPO / "spec" / "AV2_Spec_v1.0.0.pdf"
PDF_SHA256 = "e9916f091e4e83446aad6b4601641c5b292e569c144c4163b26a4497573b533f"

sys.path.insert(0, str(REPO / "tools" / "spec_extract"))
from specpdf import find_array, pdf_text  # noqa: E402

# AV2 TxSize enum order 0..24 (§6.19.6.1 / constants.json).
TX_SIZES = [
    (4, 4), (8, 8), (16, 16), (32, 32), (64, 64),
    (4, 8), (8, 4), (8, 16), (16, 8), (16, 32), (32, 16),
    (32, 64), (64, 32), (4, 16), (16, 4), (8, 32), (32, 8),
    (16, 64), (64, 16), (4, 32), (32, 4), (8, 64), (64, 8),
    (4, 64), (64, 4),
]

NUM_LEVELS = 15
NUM_PLANE_TYPES = 2
TABLE_ENTRIES = 3600


def tx_dims(tx_size: int) -> tuple[int, int]:
    return TX_SIZES[tx_size]


def adjusted_dims(tx_size: int) -> tuple[int, int]:
    w, h = TX_SIZES[tx_size]
    return min(32, w), min(32, h)


def fail(message: str) -> None:
    raise SystemExit(f"generate_qm_tables: {message}")


def extract() -> tuple[list[int], list[int]]:
    if not PDF.exists():
        fail(f"missing spec PDF: {PDF}")
    digest = hashlib.sha256(PDF.read_bytes()).hexdigest()
    if digest != PDF_SHA256:
        fail(f"PDF sha256 {digest} != pinned {PDF_SHA256}")

    text = pdf_text(PDF)

    offsets = find_array(text, "Qm_Offset")
    if offsets is None:
        fail("Qm_Offset not found in the spec PDF")
    matrix = find_array(text, "Quantizer_Matrix")
    if matrix is None:
        fail("Quantizer_Matrix not found in the spec PDF")

    if offsets.shape != [len(TX_SIZES)]:
        fail(f"Qm_Offset shape {offsets.shape}, expected [25]")
    if matrix.shape != [NUM_LEVELS, NUM_PLANE_TYPES, TABLE_ENTRIES]:
        fail(f"Quantizer_Matrix shape {matrix.shape}, expected [15, 2, 3600]")

    off = [int(v) for v in offsets.values]
    values = [int(v) for v in matrix.values]
    if any(v < 0 or v > 4096 for v in off):
        fail("Qm_Offset value out of range")
    if any(v < 0 or v > 255 for v in values):
        fail("Quantizer_Matrix value outside 0..255")
    return off, values


def validate(off: list[int], qm: list[int]) -> None:
    # 1. Every offset must start a block that fits and uses the adjusted size.
    used = sorted(set(off))
    for tx_size, start in enumerate(off):
        tw, th = adjusted_dims(tx_size)
        end = start + tw * th
        if start < 0 or end > TABLE_ENTRIES:
            fail(f"tx size {tx_size} block [{start},{end}) leaves the table")
        # A block must be wholly contiguous up to the next distinct offset, or
        # the end of the table for the final block.
        pos = used.index(start)
        limit = used[pos + 1] if pos + 1 < len(used) else TABLE_ENTRIES
        if end != limit:
            fail(
                f"tx size {tx_size}: block end {end} != next offset {limit} "
                f"(adjusted {tw}x{th})"
            )

    # 2. Shared offsets may only be shared by equal adjusted sizes.
    first: dict[int, int] = {}
    for tx_size, start in enumerate(off):
        if start in first:
            if adjusted_dims(first[start]) != adjusted_dims(tx_size):
                fail(
                    f"offset {start} shared by differently sized tx "
                    f"{first[start]} and {tx_size}"
                )
        else:
            first[start] = tx_size

    # 3. §9.4.2 informative derivation must reproduce every packed matrix.
    fundamental_offsets = {
        "square": off[3],      # 32x32
        "wide": off[10],       # 32x16
        "high": off[9],        # 16x32
    }
    for level in range(NUM_LEVELS):
        for plane_type in range(NUM_PLANE_TYPES):
            base = (level * NUM_PLANE_TYPES + plane_type) * TABLE_ENTRIES
            packed = qm[base:base + TABLE_ENTRIES]
            for tx_size in range(len(TX_SIZES)):
                tw, th = adjusted_dims(tx_size)
                if tw == th:
                    fw, fh = 32, 32
                    kind = "square"
                elif tw > th:
                    fw, fh = 32, 16
                    kind = "wide"
                else:
                    fw, fh = 16, 32
                    kind = "high"
                fo = fundamental_offsets[kind]
                fundamental = packed[fo:fo + fw * fh]
                rw, rh = fw // tw, fh // th
                pw, ph = (rw + 1) // 2 - 1, (rh + 1) // 2 - 1
                derived = [
                    fundamental[(rh * i + ph) * fw + rw * j + pw]
                    for i in range(th)
                    for j in range(tw)
                ]
                got = packed[off[tx_size]:off[tx_size] + tw * th]
                if derived != got:
                    fail(
                        f"§9.4.2 derivation mismatch: level {level} "
                        f"plane_type {plane_type} tx_size {tx_size}"
                    )


def format_array(values: list[int], per_line: int, indent: str) -> str:
    lines = []
    for i in range(0, len(values), per_line):
        chunk = ", ".join(str(v) for v in values[i:i + per_line])
        lines.append(f"{indent}{chunk},")
    return "\n".join(lines)


def render_constants(off: list[int]) -> str:
    out = []
    out.append("// Generated by src/encoder/qm/generate_qm_tables.py. Do not edit.")
    out.append("//")
    out.append("// Normative AV2 v1.0.0 §9.4.3 Qm_Offset and level constants.  This")
    out.append("// header is small so CPU/GPU code that only needs the rules does not")
    out.append("// pull in the 108,000-entry Quantizer_Matrix table.")
    out.append("#pragma once")
    out.append("")
    out.append("#include <cstdint>")
    out.append("")
    out.append("namespace av2enc::qm {")
    out.append("")
    out.append(f"inline constexpr int kNumQmLevels = {NUM_LEVELS};")
    out.append("inline constexpr int kQmPlaneTypes = 2;")
    out.append(f"inline constexpr int kQmTableEntries = {TABLE_ENTRIES};")
    out.append(f"inline constexpr int kNumTxSizes = {len(TX_SIZES)};")
    out.append("// Level 15 is the §7.14.4 sentinel: no quantizer matrix is applied.")
    out.append("inline constexpr int kNoQmLevel = 15;")
    out.append("")
    out.append("// Indexed by the TxSize enum (0..24).")
    out.append("inline constexpr std::uint16_t kQmOffset[kNumTxSizes] = {")
    out.append(format_array(off, 8, "    "))
    out.append("};")
    out.append("")
    out.append("}  // namespace av2enc::qm")
    out.append("")
    return "\n".join(out)


def render_table(qm: list[int]) -> str:
    out = []
    out.append("// Generated by src/encoder/qm/generate_qm_tables.py. Do not edit.")
    out.append("//")
    out.append("// Normative AV2 v1.0.0 §9.4.3 Quantizer_Matrix, extracted from")
    out.append("// spec/AV2_Spec_v1.0.0.pdf (sha256 " + PDF_SHA256 + ").")
    out.append("// [level 0..14][plane type: 0 = luma, 1 = chroma][3600].")
    out.append("#pragma once")
    out.append("")
    out.append("#include <cstdint>")
    out.append("")
    out.append('#include "encoder/qm/qm_constants.h"')
    out.append("")
    out.append("namespace av2enc::qm {")
    out.append("")
    out.append("inline constexpr std::uint8_t")
    out.append("    kQuantizerMatrix[kNumQmLevels][kQmPlaneTypes][kQmTableEntries] = {")
    for level in range(NUM_LEVELS):
        out.append(f"  {{  // level {level}")
        for plane_type in range(NUM_PLANE_TYPES):
            base = (level * NUM_PLANE_TYPES + plane_type) * TABLE_ENTRIES
            out.append(
                f"    {{  // {'luma' if plane_type == 0 else 'chroma'}"
            )
            out.append(format_array(qm[base:base + TABLE_ENTRIES], 24, "      "))
            out.append("    },")
        out.append("  },")
    out.append("};")
    out.append("")
    out.append("}  // namespace av2enc::qm")
    out.append("")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="validate and compare, do not rewrite files")
    args = parser.parse_args()

    off, qm = extract()
    validate(off, qm)

    constants = render_constants(off)
    table = render_table(qm)
    json_text = json.dumps(
        {
            "clause": "9.4.3",
            "source": "spec/AV2_Spec_v1.0.0.pdf",
            "pdf_sha256": PDF_SHA256,
            "tx_sizes": TX_SIZES,
            "Qm_Offset": off,
            "Quantizer_Matrix": [qm[i:i + TABLE_ENTRIES]
                                 for i in range(0, len(qm), TABLE_ENTRIES)],
        },
        separators=(",", ":"),
    ) + "\n"

    constants_path = HERE / "qm_constants.h"
    table_path = HERE / "qm_tables.h"
    json_path = HERE / "qm_tables.json"
    if args.check:
        if not constants_path.exists() or constants_path.read_text() != constants:
            fail(f"{constants_path} is stale; rerun without --check")
        if not table_path.exists() or table_path.read_text() != table:
            fail(f"{table_path} is stale; rerun without --check")
        if not json_path.exists() or json_path.read_text() != json_text:
            fail(f"{json_path} is stale; rerun without --check")
        print("qm tables: up to date")
        return 0

    constants_path.write_text(constants)
    table_path.write_text(table)
    json_path.write_text(json_text)
    print(f"wrote {constants_path.relative_to(REPO)} ({len(constants)} bytes)")
    print(f"wrote {table_path.relative_to(REPO)} ({len(table)} bytes)")
    print(f"wrote {json_path.relative_to(REPO)} ({len(json_text)} bytes)")
    print("validated: 15 levels x 2 plane types x 25 TX sizes against §9.4.2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
