#!/usr/bin/env python3
"""Generate the M0 machine-readable tables under spec/tables/ from the spec PDF.

Every table is parsed from the pinned AV2 v1.0.0 PDF (spec/AV2_Spec_v1.0.0.pdf).
Nothing is hand-typed: if a table cannot be found the script fails loudly with
the spec section it looked in.

Usage:
    python3 tools/spec_extract/extract_tables.py [--out spec/tables]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from specpdf import CArray, SPEC_PDF, check, dump_json, find_array, find_arrays, pdf_text, section

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "spec" / "tables"

# ---------------------------------------------------------------------------
# clause references (for the generated headers and for failure messages)
# ---------------------------------------------------------------------------
CLAUSE = {
    "Ac_Qlookup": "7.14.2 (Dequantization functions)",
    "Dct_Kernel32": "9.6 (1d transform tables)",
    "Default_Coeff_Base_Cdf": "9.3 (Default CDF tables)",
}

BLOCK_SIZE_ENUM = [
    "BLOCK_4X4", "BLOCK_4X8", "BLOCK_8X4", "BLOCK_8X8", "BLOCK_8X16",
    "BLOCK_16X8", "BLOCK_16X16", "BLOCK_16X32", "BLOCK_32X16", "BLOCK_32X32",
    "BLOCK_32X64", "BLOCK_64X32", "BLOCK_64X64", "BLOCK_64X128", "BLOCK_128X64",
    "BLOCK_128X128", "BLOCK_128X256", "BLOCK_256X128", "BLOCK_256X256",
    "BLOCK_4X16", "BLOCK_16X4", "BLOCK_8X32", "BLOCK_32X8", "BLOCK_16X64",
    "BLOCK_64X16", "BLOCK_4X32", "BLOCK_32X4", "BLOCK_8X64", "BLOCK_64X8",
]
TX_SIZE_ENUM = [
    "TX_4X4", "TX_8X8", "TX_16X16", "TX_32X32", "TX_64X64", "TX_4X8",
    "TX_8X4", "TX_8X16", "TX_16X8", "TX_16X32", "TX_32X16", "TX_32X64",
    "TX_64X32", "TX_4X16", "TX_16X4", "TX_8X32", "TX_32X8", "TX_16X64",
    "TX_64X16", "TX_4X32", "TX_32X4", "TX_8X64", "TX_64X8", "TX_4X64",
    "TX_64X4",
]

ENUMS = {
    "BLOCK_": BLOCK_SIZE_ENUM,
    "TX_": TX_SIZE_ENUM,
}
BLOCK_INVALID = 255

# Tables we want as numbers (identifiers resolved through ENUMS).
NUMERIC_TABLES = [
    "Mi_Width_Log2", "Mi_Height_Log2", "Num_4x4_Blocks_Wide",
    "Num_4x4_Blocks_High", "Size_Group", "Max_Tx_Size_Rect",
    "Partition_Subsize", "Subsampled_Size", "Tx_Size_Sqr", "Tx_Size_Sqr_Up",
    "Tx_Width", "Tx_Height", "Tx_Width_Log2", "Tx_Height_Log2",
    "Adjusted_Tx_Size", "Partition_Size_Adjust",
    "Partition_Size_Adjust_Rect_Type", "Size_To_Tx_Part_Group_Lookup",
    "Size_To_Tx_Type_Group_Vert_And_Horz",
    "Size_To_Tx_Type_Group_Vert_Or_Horz", "H_Partition_Midsize",
    "Size_Class", "Sig_Ref_Diff_Offset", "Mode_To_Angle",
]

# Tables that stay symbolic (mode names, etc.).
SYMBOLIC_TABLES = [
    "Reordered_Y_Mode", "Default_Mode_List_Y", "Default_Mode_List_Uv",
]

# CDF arrays the M0 syntax path actually reads.  Everything else in 9.3 is
# extracted too, but only these are emitted as a typed C++ header.
M0_CDFS = [
    "Default_Do_Split_Cdf",
    "Default_Do_Square_Split_Cdf",
    "Default_Y_Mode_Set_Cdf",
    "Default_Y_Mode_Index_Cdf",
    "Default_Y_Mode_Offset_Cdf",
    "Default_Uv_Mode_Cfl_Not_Allowed_Cdf",
    "Default_Txb_Skip_Cdf",
    "Default_Eob_Pt_256_Cdf",
    "Default_Eob_Pt_1024_Cdf",
    "Default_Eob_Extra_Cdf",
    "Default_Coeff_Base_Cdf",
    "Default_Coeff_Base_Eob_Cdf",
    "Default_Coeff_Br_Cdf",
    "Default_Coeff_Base_Uv_Cdf",
    "Default_Coeff_Base_Eob_Uv_Cdf",
    "Default_Coeff_Br_Uv_Cdf",
    "Default_Dc_Sign_Cdf",
]


def resolve_enum(name: str, table: str) -> int:
    if name == "BLOCK_INVALID":
        return BLOCK_INVALID
    if name == "TX_INVALID":
        return 255
    for prefix, names in ENUMS.items():
        if name.startswith(prefix) and name in names:
            return names.index(name)
    raise SystemExit(f"{table}: cannot resolve enum {name!r}")


def to_numbers(arr: CArray) -> list | int:
    def conv(v):
        if isinstance(v, list):
            return [conv(x) for x in v]
        if isinstance(v, str):
            return resolve_enum(v, arr.name)
        return v

    return conv(arr.data)


def get(text: str, name: str) -> CArray:
    arr = find_array(text, name)
    if arr is None:
        clause = CLAUSE.get(name, "9.2 Conversion tables")
        raise SystemExit(f"table {name} not found in the spec PDF (see clause {clause})")
    return arr


def gen_block_tables(text: str, out: Path) -> dict:
    tables = {}
    for name in NUMERIC_TABLES:
        arr = get(text, name)
        data = to_numbers(arr)
        shape = []
        probe = data
        while isinstance(probe, list):
            shape.append(len(probe))
            probe = probe[0] if probe else None
        tables[name] = {"clause": "9.2", "shape": shape, "data": data}

    for name in SYMBOLIC_TABLES:
        arr = get(text, name)
        tables[name] = {"clause": "5.20.5 / 5.20.5.6", "shape": arr.shape,
                        "data": arr.data}

    tables["enum_block_size"] = {"clause": "6.19.3.1 Table 6.22", "data": BLOCK_SIZE_ENUM}
    tables["enum_tx_size"] = {"clause": "6.19.6.1", "data": TX_SIZE_ENUM}
    dump_json(out / "block_tables.json", tables)
    return tables


def qlookup(q: int, ac: list[int]) -> int:
    if q < 25:
        return ac[q]
    return ac[((q - 1) % 24) + 1] << ((q - 1) // 24)


def gen_quant(text: str, out: Path, constants: dict[str, int]) -> dict:
    ac = get(text, "Ac_Qlookup")
    check(ac.shape == [25], f"Ac_Qlookup shape {ac.shape} != [25]")
    ac = ac.data
    maxq8 = constants.get("MAXQ_8_BITS")
    check(isinstance(maxq8, int), "MAXQ_8_BITS missing from constants table")
    steps = [qlookup(q, ac) for q in range(maxq8 + 1)]
    obj = {
        "clause": "7.14.2",
        "Ac_Qlookup": ac,
        "maxq_8bit": maxq8,
        "qlookup_q0_255": steps,
        "note": "qlookup(q) = Ac_Qlookup[q] for q<25 else Ac_Qlookup[((q-1)%24)+1] << ((q-1)/24)",
    }
    dump_json(out / "quant.json", obj)
    return obj


def gen_transforms(text: str, out: Path) -> dict:
    kernels = {}
    for n in (4, 8, 16, 32):
        arr = get(text, f"Dct_Kernel{n}")
        check(arr.shape == [n, n], f"Dct_Kernel{n} shape {arr.shape}")
        kernels[f"Dct_Kernel{n}"] = arr.data
    dump_json(out / "dct_kernels.json", {"clause": "9.6", "kernels": kernels})

    k32 = kernels["Dct_Kernel32"]
    lines = [
        "// Generated by tools/spec_extract/extract_tables.py from",
        "// spec/AV2_Spec_v1.0.0.pdf clause 9.6. Do not edit.",
        "#pragma once",
        "#include <cstdint>",
        "",
        "namespace av2spec {",
        "// Dct_Kernel32[j][i] (row-major, 32x32, Q6).",
        "inline constexpr int16_t kDctKernel32[32][32] = {",
    ]
    for row in k32:
        lines.append("    {" + ", ".join(f"{v:4d}" for v in row) + "},")
    lines += ["};", "", "}  // namespace av2spec", ""]
    (out / "dct32.h").write_text("\n".join(lines))
    return kernels


def gen_scan(out: Path) -> dict:
    """Diagonal scan from 5.20.7.30 get_scan(), implemented literally."""
    scans = {}
    for n in (4, 8, 16, 32):
        w = h = n
        out_pos = []
        x = y = 0
        for _ in range(w * h):
            out_pos.append(y * w + x)
            x += 1
            y -= 1
            if y < 0 or x >= w:
                x += 1
                s = min(x, h - 1 - y)
                x -= s
                y += s
        scans[str(n)] = out_pos
    obj = {
        "clause": "5.20.7.30",
        "note": "TX_CLASS_2D diagonal scan; index c -> raster position pos = row*Min(W,32)+col",
        "scans": scans,
    }
    dump_json(out / "scan.json", obj)
    return obj


def gen_cdfs(text: str, out: Path) -> tuple[dict, dict]:
    sec = section(text, "9.3. Default CDF tables", "9.4. Quantizer matrix tables")
    arrays = find_arrays(sec, "Default_")
    check(len(arrays) > 150, f"expected ~180 default CDFs, found {len(arrays)}")

    all_cdfs = {}
    for arr in arrays:
        vals = arr.values
        check(all(isinstance(v, int) for v in vals), f"{arr.name}: non-numeric entry")
        all_cdfs[arr.name] = {
            "shape": arr.shape,
            "dims_decl": arr.dims_decl,
            "data": vals,
        }
    dump_json(out / "default_cdf.json", {"clause": "9.3", "arrays": all_cdfs})

    m0 = {}
    for name in M0_CDFS:
        check(name in all_cdfs, f"M0 CDF {name} missing from 9.3")
        m0[name] = all_cdfs[name]
    dump_json(out / "m0_cdf.json", {"clause": "9.3", "arrays": m0})
    return all_cdfs, m0


def gen_m0_cdf_header(m0: dict, out: Path) -> None:
    lines = [
        "// Generated by tools/spec_extract/extract_tables.py from",
        "// spec/AV2_Spec_v1.0.0.pdf clause 9.3. Do not edit.",
        "//",
        "// Each CDF is N+1 entries: N-1 cumulative values, then the rate,",
        "// then the adaptation count (see clause 8.2.6).",
        "#pragma once",
        "#include <cstdint>",
        "",
        "namespace av2spec {",
        "",
    ]
    for name, cdf in m0.items():
        cname = "k" + name[len("Default_"):-len("_Cdf")]
        shape = cdf["shape"]
        dims = "".join(f"[{d}]" for d in shape)
        flat = cdf["data"]
        lines.append(f"// shape {shape}")
        lines.append(f"inline constexpr uint16_t {cname}{dims} = {{")
        if len(shape) == 1:
            lines.append("    " + ", ".join(str(v) for v in flat) + ",")
        else:
            # emit row-per-line for readability
            stride = 1
            for d in shape[1:]:
                stride *= d
            for i in range(shape[0]):
                row = flat[i * stride:(i + 1) * stride]
                lines.append("    {" + ", ".join(str(v) for v in row) + "},")
        lines.append("};")
        lines.append("")
    lines += ["}  // namespace av2spec", ""]
    (out / "m0_cdf.h").write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--pdf", type=Path, default=SPEC_PDF)
    ap.add_argument("--constants", type=Path, default=DEFAULT_OUT / "constants.json",
                    help="constants.json produced by extract_constants.py")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    text = pdf_text(args.pdf)

    constants = {}
    if args.constants.exists():
        constants = json.loads(args.constants.read_text())
    else:
        print(f"warning: {args.constants} missing; MAXQ falls back to scan",
              file=sys.stderr)
        constants = {"MAXQ_8_BITS": 255}

    gen_block_tables(text, args.out)
    gen_quant(text, args.out, constants)
    gen_transforms(text, args.out)
    gen_scan(args.out)
    _, m0 = gen_cdfs(text, args.out)
    gen_m0_cdf_header(m0, args.out)
    print(f"wrote tables to {args.out}")


if __name__ == "__main__":
    main()
