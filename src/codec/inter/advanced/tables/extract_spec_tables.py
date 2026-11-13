#!/usr/bin/env python3
"""Regenerate the AV2 inter tables in this directory from the spec PDF.

The AV2 v1.0.0 HTML rendering replaces numeric tables with "defined in the
<name>.h header file". The PDF (`spec/AV2_Spec_v1.0.0.pdf`) contains the numbers,
so this script extracts them and emits C++ headers:

    warp_filters.h   §9.5  Warped_Filters[449][8], Ext_Warped_Filters[65][6]
    wedge_tables.h   §7.13.3.27 wedge LUTs + codebook, §9.2 Wedge_Bits,
                     §7.13.3.29 Ii_Weights_1d, §5.20.7.18 angle->index map

Usage:
    python3 extract_spec_tables.py            # rewrite the headers
    python3 extract_spec_tables.py --check    # verify headers are current

Requires PyMuPDF (`pip install pymupdf`). This is a development tool; it is not
part of the encoder build.
"""

import argparse
import pathlib
import re
import sys

import pymupdf

HERE = pathlib.Path(__file__).resolve().parent
PDF = HERE.parents[4] / "spec" / "AV2_Spec_v1.0.0.pdf"

HEADER_BANNER = (
    "// Generated from spec/AV2_Spec_v1.0.0.pdf by\n"
    "// src/codec/inter/advanced/tables/extract_spec_tables.py.\n"
    "// Do not hand-edit; rerun the script instead.\n"
)

# Host and device passes need different storage classes for a namespace-scope
# table: CUDA cannot odr-use a host `constexpr` array from device code.  The
# device pass of nvcc (__CUDA_ARCH__ defined) gets `__device__ constexpr`
# storage; every host pass keeps a plain `constexpr` copy.
TABLE_STORAGE_MACRO = (
    "// Host and device passes need different storage classes for a\n"
    "// namespace-scope table: CUDA cannot odr-use a host `constexpr` array from\n"
    "// device code (\"identifier is undefined in device code\").  Same pattern as\n"
    "// deblock_core.h / mc_subpel_tables.h.\n"
    "#if defined(__CUDA_ARCH__)\n"
    "#define AV2ENC_ADV_TABLE static __device__ constexpr\n"
    "#else\n"
    "#define AV2ENC_ADV_TABLE static constexpr\n"
    "#endif"
)


def page_text(doc, pages):
    text = "".join(doc[i].get_text() for i in pages)
    return re.sub(r"AV2 Specification\s*Page \d+ of \d+", "", text)


def brace_rows(text):
    rows = []
    for group in re.findall(r"\{([^{}]*)\}", text):
        rows.append([int(v.replace(" ", "")) for v in re.findall(r"-?\s*\d+", group)])
    return rows


def rows_between(text, start_marker, end_marker):
    begin = text.index(start_marker)
    begin = text.index("=", begin) + 1
    end = text.index(end_marker, begin)
    return brace_rows(text[begin:end])


def emit_array(name, elem_type, values, width, columns, indent="    "):
    out = []
    total = len(values)
    assert total % columns == 0, (name, total, columns)
    rows = total // columns
    out.append(f"AV2ENC_ADV_TABLE {elem_type} {name}[{rows}][{columns}] = {{")
    for r in range(rows):
        chunk = values[r * columns:(r + 1) * columns]
        out.append(indent + "{" + ", ".join(f"{v:5d}" for v in chunk) + "},")
    out.append("};")
    return out


def emit_flat(name, elem_type, values, columns, indent="    "):
    out = [f"AV2ENC_ADV_TABLE {elem_type} {name}[{len(values)}] = {{"]
    for i in range(0, len(values), columns):
        chunk = values[i:i + columns]
        out.append(indent + ", ".join(f"{v:4d}" for v in chunk) + ",")
    out.append("};")
    return out


def make_warp_headers(doc):
    text = page_text(doc, range(840, 845))
    warp = rows_between(text, "Warped_Filters[", "Ext_Warped_Filters")
    ext = rows_between(text, "Ext_Warped_Filters[", "9.6.")
    assert len(warp) == 449, len(warp)
    assert all(len(r) == 8 and sum(r) == 128 for r in warp), "Warped_Filters malformed"
    assert len(ext) == 65, len(ext)
    assert all(len(r) == 6 and sum(r) == 128 for r in ext), "Ext_Warped_Filters malformed"
    # §9.5's table is centred at index 3*64: phase 0 is an exact identity.
    assert warp[3 * 64] == [0, 0, 0, 127, 1, 0, 0, 0], warp[192]

    lines = [
        "// AV2 v1.0.0 §9.5 warp filter tables.",
        "//",
        "// Warped_Filters[WARPEDPIXEL_PREC_SHIFTS*7 + 1][8]   (449 x 8, 8-tap)",
        "// Ext_Warped_Filters[EXT_WARP_PHASES + 1][EXT_WARP_TAPS] (65 x 6, 6-tap)",
        "//",
        "// Index 3*WARPEDPIXEL_PREC_SHIFTS (192) is the zero-phase row. Every row",
        "// sums to 128, i.e. 7 bits of filter precision (§9.5, FILTER_BITS = 7).",
        "#ifndef AV2ENC_INTER_ADVANCED_TABLES_WARP_FILTERS_H_",
        "#define AV2ENC_INTER_ADVANCED_TABLES_WARP_FILTERS_H_",
        "",
        "#include <cstdint>",
        "",
        TABLE_STORAGE_MACRO,
        "",
        "namespace av2enc::inter::adv::tables {",
        "",
        "inline constexpr int kWarpedFilterRows = 449;",
        "inline constexpr int kWarpedFilterTaps = 8;",
        "inline constexpr int kExtWarpedFilterRows = 65;",
        "inline constexpr int kExtWarpedFilterTaps = 6;",
        "",
    ]
    lines += emit_array("kWarpedFilters", "std::int16_t",
                        [v for row in warp for v in row], 8, 8)
    lines.append("")
    lines += emit_array("kExtWarpedFilters", "std::int16_t",
                        [v for row in ext for v in row], 6, 6)
    lines += [
        "",
        "}  // namespace av2enc::inter::adv::tables",
        "",
        "#undef AV2ENC_ADV_TABLE",
        "",
        "#endif  // AV2ENC_INTER_ADVANCED_TABLES_WARP_FILTERS_H_",
        "",
    ]
    return HEADER_BANNER + "\n".join(lines)


WEDGE_ANGLE_NAMES = [
    "WEDGE_0", "WEDGE_14", "WEDGE_27", "WEDGE_45", "WEDGE_63", "WEDGE_90",
    "WEDGE_117", "WEDGE_135", "WEDGE_153", "WEDGE_166", "WEDGE_180",
    "WEDGE_194", "WEDGE_207", "WEDGE_225", "WEDGE_243", "WEDGE_270",
    "WEDGE_297", "WEDGE_315", "WEDGE_333", "WEDGE_346",
]


def make_wedge_header(doc):
    # §7.13.3.27 (pages ~503-505): wedge codebook and distance weight LUTs.
    block = page_text(doc, range(503, 506))
    cos_lut = [int(v) for v in re.findall(r"-?\d+",
               block.split("Wedge_Cos_Lut[WEDGE_ANGLES] = {")[1]
               .split("}")[0])]
    sin_lut = [int(v) for v in re.findall(r"-?\d+",
               block.split("Wedge_Sin_Lut[WEDGE_ANGLES] = {")[1]
               .split("}")[0])]
    pos_lut = [int(v) for v in re.findall(r"\d+",
               block.split("Pos_Dist_2_Bld_Weight[WEDGE_BLD_LUT_SIZE] = {")[1]
               .split("}")[0])]
    neg_lut = [int(v) for v in re.findall(r"\d+",
               block.split("Neg_Dist_2_Bld_Weight[WEDGE_BLD_LUT_SIZE] = {")[1]
               .split("}")[0])]
    codebook_text = block.split("Wedge_Codebook[WEDGE_TYPES][3] = {")[1].split("}")[0]
    codebook = []
    for angle, xoff, yoff in re.findall(r"\{\s*([A-Z_0-9]+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\}",
                                        block.split("Wedge_Codebook[WEDGE_TYPES][3] = {")[1]):
        codebook.append((WEDGE_ANGLE_NAMES.index(angle), int(xoff), int(yoff)))

    assert len(cos_lut) == 20 and len(sin_lut) == 20, (len(cos_lut), len(sin_lut))
    assert len(pos_lut) == 32 and len(neg_lut) == 32, (len(pos_lut), len(neg_lut))
    assert len(codebook) == 68, len(codebook)
    assert all(0 <= a < 20 and 1 <= x <= 7 and 1 <= y <= 7 for a, x, y in codebook)

    # §5.20.7.18 (page ~207-208): Wedge_Angle_Dist_2_Index. The table is written
    # as 20 rows of 4 against the row count, so parse by value rather than brace.
    table = []
    text = page_text(doc, range(209, 212))
    num = [int(v) for v in re.findall(r"-?\d+",
           text.split("Wedge_Angle_Dist_2_Index[ WEDGE_ANGLES ][ NUM_WEDGE_DIST ] = {")[1]
           .split("}\n}")[0])]
    assert len(num) == 80, len(num)
    for r in range(20):
        table.append(num[r * 4:(r + 1) * 4])
    assert table[0] == [-1, 0, 1, 2] and table[19] == [-1, 65, 66, 67], table

    # §7.13.3.29 (page ~505): Ii_Weights_1d[128].
    mask_block = page_text(doc, range(504, 507))
    ii = [int(v) for v in re.findall(r"\d+",
          mask_block.split("Ii_Weights_1d[ 128 ] = {")[1].split("}")[0])]
    assert len(ii) == 128 and all(1 <= v <= 60 for v in ii), (len(ii), ii[:4])
    assert ii[0] == 60 and ii[-1] == 1 and ii[-2] == 1, (ii[0], ii[-2:])

    # §9.2 (page ~614): Wedge_Bits[BLOCK_SIZES].
    s92 = page_text(doc, range(614, 616))
    wedge_bits = [int(v) for v in re.findall(r"\d+",
                  s92.split("Wedge_Bits[ BLOCK_SIZES ] =")[1].split("}")[0])]
    assert len(wedge_bits) == 29, len(wedge_bits)
    assert wedge_bits[3] == 4 and wedge_bits[0] == 0, wedge_bits

    import json
    blocks = json.loads((HERE.parents[4] / "spec" / "tables" /
                         "block_tables.json").read_text())
    widths = [4 * v for v in blocks["Num_4x4_Blocks_Wide"]["data"]]
    heights = [4 * v for v in blocks["Num_4x4_Blocks_High"]["data"]]
    names = blocks["enum_block_size"]["data"]
    assert len(widths) == 29 and len(names) == 29, (len(widths), len(names))
    assert names[3] == "BLOCK_8X8" and names[6] == "BLOCK_16X16", names[:7]

    lines = [
        "// AV2 v1.0.0 wedge and inter-intra tables.",
        "//",
        "// §5.20.7.18  Wedge_Angle_Dist_2_Index[20][4]",
        "// §7.13.3.27  Wedge_Cos_Lut / Wedge_Sin_Lut / distance weights / codebook",
        "// §7.13.3.29  Ii_Weights_1d[128]",
        "// §9.2        Wedge_Bits[BLOCK_SIZES] (0 = block has no wedge modes)",
        "#ifndef AV2ENC_INTER_ADVANCED_TABLES_WEDGE_TABLES_H_",
        "#define AV2ENC_INTER_ADVANCED_TABLES_WEDGE_TABLES_H_",
        "",
        "#include <cstdint>",
        "",
        TABLE_STORAGE_MACRO,
        "",
        "namespace av2enc::inter::adv::tables {",
        "",
        "inline constexpr int kWedgeAngles = 20;",
        "inline constexpr int kWedgeTypes = 68;",
        "inline constexpr int kNumWedgeDist = 4;",
        "inline constexpr int kWedgeBldLutSize = 32;",
        "inline constexpr int kWedgeBoundaryTypes = 2;",
        "inline constexpr int kBlockSizes = 29;",
        "",
    ]
    lines += emit_flat("kWedgeCosLut", "std::int8_t", cos_lut, 20)
    lines.append("")
    lines += emit_flat("kWedgeSinLut", "std::int8_t", sin_lut, 20)
    lines.append("")
    lines += emit_flat("kPosDist2BldWeight", "std::int8_t", pos_lut, 16)
    lines.append("")
    lines += emit_flat("kNegDist2BldWeight", "std::int8_t", neg_lut, 16)
    lines.append("")
    lines += emit_array("kWedgeAngleDist2Index", "std::int8_t",
                        [v for row in table for v in row], 1, 4)
    lines.append("")
    lines += emit_flat("kWedgeBits", "std::int8_t", wedge_bits, 12)
    lines.append("")
    lines += emit_flat("kWedgeBlockWidth", "std::int16_t", widths, 10)
    lines.append("")
    lines += emit_flat("kWedgeBlockHeight", "std::int16_t", heights, 10)
    lines.append("")
    lines.append("// §9.2 enum_block_size order (BLOCK_4X4 = 0, BLOCK_8X8 = 3, ...).")
    lines.append("inline constexpr int kBlock8x8 = 3;")
    lines.append("inline constexpr int kBlock16x16 = 6;")
    lines.append("inline constexpr int kBlock64x64 = 12;")
    lines.append("")
    lines += emit_flat("kIiWeights1d", "std::int8_t", ii, 12)
    lines += [
        "",
        "// §7.13.3.27 Wedge_Codebook[WEDGE_TYPES][3] = { { angle, xoff, yoff } }.",
        "// `angle` indexes kWedgeCosLut / kWedgeSinLut.",
        "AV2ENC_ADV_TABLE std::int8_t kWedgeCodebook[kWedgeTypes][3] = {",
    ]
    for a, x, y in codebook:
        lines.append(f"    {{{a:2d}, {x:2d}, {y:2d}}},")
    lines += [
        "};",
        "",
        "}  // namespace av2enc::inter::adv::tables",
        "",
        "#undef AV2ENC_ADV_TABLE",
        "",
        "#endif  // AV2ENC_INTER_ADVANCED_TABLES_WEDGE_TABLES_H_",
        "",
    ]
    return HEADER_BANNER + "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="fail if the checked-in headers differ")
    args = parser.parse_args()

    doc = pymupdf.open(PDF)
    outputs = {
        HERE / "warp_filters.h": make_warp_headers(doc),
        HERE / "wedge_tables.h": make_wedge_header(doc),
    }
    if args.check:
        stale = [str(p) for p, text in outputs.items()
                 if not p.exists() or p.read_text() != text]
        if stale:
            print("stale generated headers: " + ", ".join(stale), file=sys.stderr)
            return 1
        print("generated headers are current")
        return 0
    for path, text in outputs.items():
        path.write_text(text)
        print(f"wrote {path} ({len(text)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
