#!/usr/bin/env python3
"""Generate tx_type_tables.h from the AV2 spec tables.

Source of truth (all committed except the cached HTML):

  spec/tables/block_tables.json   §9.2 Size_Class, Tx_Size_Sqr, Tx_Size_Sqr_Up,
                                  Md_Idx_To_Type
  spec/tables/default_cdf.json    §9.3 default intra TX-type CDFs
  spec/tables/scan.json           §5.20.7.30 TX_CLASS_2D diagonal scans
  spec/tables/_cache/spec.html    §5.20.7.29 Tx_Type_In_Set_Intra (the cached
                                  HTML is git-ignored; regenerate it with
                                  tools/spec_extract/build_all.py if missing)

The generated header is committed, so normal builds and tests do not need the
PDF or the HTML cache.

Usage:
    python3 generate_tx_type_tables.py [--check]

--check regenerates into memory and diffs against the committed header.
"""
from __future__ import annotations

import html as html_mod
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
TABLES = REPO / "spec" / "tables"
OUT = HERE / "tx_type_tables.h"

TX_SIZES_ALL = 25
TX_TYPES = 16
SET_TYPES_INTRA = 7
INTRA_MODES = 13


def load_json(path: pathlib.Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def table_data(tables: dict, name: str):
    entry = tables[name]
    return entry["data"], entry.get("shape")


def strip_html(chunk: str) -> str:
    text = re.sub(r"<[^>]+>", " ", chunk)
    text = html_mod.unescape(text)
    return text


def parse_tx_type_in_set_intra() -> list[list[int]]:
    path = TABLES / "_cache" / "spec.html"
    if not path.exists():
        raise SystemExit(
            f"missing {path}\n"
            "Tx_Type_In_Set_Intra is only in the cached HTML. Run\n"
            "  python3 tools/spec_extract/build_all.py\n"
            "to regenerate the cache, or keep the committed header."
        )
    raw = path.read_text(encoding="utf-8", errors="replace")
    key = "Tx_Type_In_Set_Intra"
    start = -1
    pos = 0
    while True:
        pos = raw.find(key, pos)
        if pos < 0:
            break
        if "TX_SET_TYPES_INTRA" in raw[pos:pos + 400]:
            start = pos  # keep the last match: the table definition
        pos += len(key)
    if start < 0:
        raise SystemExit(f"{key} table not found in {path}")
    tail = raw[start:]
    end = tail.find("Tx_Type_In_Set_Inter")
    if end > 0:
        tail = tail[:end]
    text = strip_html(tail)
    # Skip the declaration "[ TX_SET_TYPES_INTRA ][ TX_TYPES ]".
    if "=" not in text:
        raise SystemExit(f"malformed {key} table in {path}")
    text = text.split("=", 1)[1]
    numbers = [int(n) for n in re.findall(r"\d+", text)]
    expected = SET_TYPES_INTRA * TX_TYPES
    if len(numbers) < expected:
        raise SystemExit(
            f"expected {expected} entries for {key}, found {len(numbers)}")
    rows = [numbers[i * TX_TYPES:(i + 1) * TX_TYPES]
            for i in range(SET_TYPES_INTRA)]
    for row in rows:
        if any(v not in (0, 1) for v in row):
            raise SystemExit(f"{key} is not 0/1: {row}")
    return rows


def fmt_rows(rows, width=16, indent="    ") -> str:
    out = []
    for row in rows:
        vals = ", ".join(str(v) for v in row)
        out.append(f"{indent}{{ {vals} }},")
    return "\n".join(out)


def main() -> int:
    tables = load_json(TABLES / "block_tables.json")
    cdfs = load_json(TABLES / "default_cdf.json")["arrays"]
    scans = load_json(TABLES / "scan.json")["scans"]

    size_class, sc_shape = table_data(tables, "Size_Class")
    tx_sqr, sqr_shape = table_data(tables, "Tx_Size_Sqr")
    tx_sqr_up, sqr_up_shape = table_data(tables, "Tx_Size_Sqr_Up")
    md_idx, md_shape = table_data(tables, "Md_Idx_To_Type")

    assert sc_shape == [TX_SIZES_ALL], sc_shape
    assert sqr_shape == [TX_SIZES_ALL], sqr_shape
    assert sqr_up_shape == [TX_SIZES_ALL], sqr_up_shape
    assert md_shape == [4, INTRA_MODES, 7], md_shape

    set1 = cdfs["Default_Intra_Tx_Type_Set1_Cdf"]
    set2 = cdfs["Default_Intra_Tx_Type_Set2_Cdf"]
    long_cdf = cdfs["Default_Intra_Tx_Type_Long_Cdf"]
    is_long = cdfs["Default_Is_Long_Side_Dct_Cdf"]
    assert set1["shape"] == [3, 8], set1["shape"]
    assert set2["shape"] == [3, 3], set2["shape"]
    assert long_cdf["shape"] == [4, 5], long_cdf["shape"]
    assert is_long["shape"] == [2, 3], is_long["shape"]

    tx_type_in_set = parse_tx_type_in_set_intra()

    # Sanity: the membership rows must be a superset of what the coding tables
    # can signal for each named set (see tx_type.cpp for the derivation).
    # Set 0 = DCTONLY, 1 = WIDE_64, 2 = HIGH_64, 3 = WIDE_32, 4 = HIGH_32,
    # 5 = INTRA_1, 6 = INTRA_2 (§6.19.8.2).
    inv_long = [
        [[10, 12, 14, 9], [11, 13, 15, 9]],       # flag 0
        [[0, 1, 4, 11], [0, 2, 5, 10]],           # flag 1
    ]
    reachable = {
        0: {0},
        1: set(inv_long[1][0]),
        2: set(inv_long[1][1]),
        3: set(inv_long[0][0]) | set(inv_long[1][0]),
        4: set(inv_long[0][1]) | set(inv_long[1][1]),
        5: {t for sc in md_idx for row in sc for t in row},
        6: {t for sc in md_idx for row in sc for t in row[:2]},
    }
    for set_id, types in reachable.items():
        row = tx_type_in_set[set_id]
        missing = sorted(t for t in types if row[t] != 1)
        if missing:
            raise SystemExit(
                f"set {set_id}: types {missing} are signalled but not in "
                f"Tx_Type_In_Set_Intra")

    body = []
    body.append("// Generated by src/encoder/tx_type/generate_tx_type_tables.py.")
    body.append("// Sources: AV2 v1.0.0 §5.20.7.29, §9.2, §9.3, §5.20.7.30")
    body.append("// (commit 38f28e7). Do not edit by hand.")
    body.append("#pragma once")
    body.append("")
    body.append("#include <cstdint>")
    body.append("")
    body.append("namespace av2enc::txtype {")
    body.append("")

    def emit_array(decl, values, comment, per_line=16):
        body.append(f"// {comment}")
        body.append(decl + " = {")
        for i in range(0, len(values), per_line):
            chunk = values[i:i + per_line]
            body.append("    " + ", ".join(str(v) for v in chunk) + ",")
        body.append("};")
        body.append("")

    emit_array(
        "inline constexpr std::uint8_t kSizeClass[25]",
        size_class,
        "Size_Class[TX_SIZES_ALL], §9.2.")
    emit_array(
        "inline constexpr std::uint8_t kTxSizeSqr[25]",
        tx_sqr,
        "Tx_Size_Sqr[TX_SIZES_ALL], §9.2 (0..4 = 4x4..64x64).")
    emit_array(
        "inline constexpr std::uint8_t kTxSizeSqrUp[25]",
        tx_sqr_up,
        "Tx_Size_Sqr_Up[TX_SIZES_ALL], §9.2 (0..4 = 4x4..64x64).")

    body.append("// Md_Idx_To_Type[Size_Class][YMode][intra_tx_type], §9.2.")
    body.append("// The last dimension is INTRA_TX_TYPES = 7 symbol indices.")
    body.append("inline constexpr std::uint8_t kMdIdxToType[4][13][7] = {")
    for sc in range(4):
        body.append("    {")
        for mode in range(INTRA_MODES):
            row = ", ".join(str(v) for v in md_idx[sc][mode])
            body.append(f"        {{ {row} }},")
        body.append("    },")
    body.append("};")
    body.append("")

    body.append("// Tx_Type_Inv_Long[is_long_side_dct][orientation][symbol],")
    body.append("// §5.20.8.2. orientation 0 = WIDE (width > height),")
    body.append("// 1 = HIGH (height > width).")
    body.append("inline constexpr std::uint8_t kTxTypeInvLong[2][2][4] = {")
    for flag in range(2):
        body.append("    {")
        for orient in range(2):
            row = ", ".join(str(v) for v in inv_long[flag][orient])
            body.append(f"        {{ {row} }},")
        body.append("    },")
    body.append("};")
    body.append("")

    body.append("// Tx_Type_In_Set_Intra[set][TX_TYPE], §5.20.7.29.")
    body.append("// Row order is the §6.19.8.2 set value: 0 DCTONLY, 1 WIDE_64,")
    body.append("// 2 HIGH_64, 3 WIDE_32, 4 HIGH_32, 5 INTRA_1, 6 INTRA_2.")
    body.append("inline constexpr std::uint8_t kTxTypeInSetIntra[7][16] = {")
    body.append(fmt_rows(tx_type_in_set))
    body.append("};")
    body.append("")

    body.append("// Default_Intra_Tx_Type_Set1_Cdf[Tx_Size_Sqr][8] (§9.3):")
    body.append("// 7 symbols + adaptation rate/counter.")
    body.append("inline constexpr std::uint16_t kIntraTxTypeSet1Cdf[3][8] = {")
    d = set1["data"]
    for i in range(3):
        body.append("    { " + ", ".join(str(v) for v in d[i * 8:(i + 1) * 8]) + " },")
    body.append("};")
    body.append("")
    body.append("// Default_Intra_Tx_Type_Set2_Cdf[Tx_Size_Sqr][3] (§9.3): 2 symbols.")
    body.append("inline constexpr std::uint16_t kIntraTxTypeSet2Cdf[3][3] = {")
    d = set2["data"]
    for i in range(3):
        body.append("    { " + ", ".join(str(v) for v in d[i * 3:(i + 1) * 3]) + " },")
    body.append("};")
    body.append("")
    body.append("// Default_Intra_Tx_Type_Long_Cdf[Tx_Size_Sqr][5] (§9.3): 4 symbols.")
    body.append("inline constexpr std::uint16_t kIntraTxTypeLongCdf[4][5] = {")
    d = long_cdf["data"]
    for i in range(4):
        body.append("    { " + ", ".join(str(v) for v in d[i * 5:(i + 1) * 5]) + " },")
    body.append("};")
    body.append("")
    body.append("// Default_Is_Long_Side_Dct_Cdf[is_inter][3] (§9.3): 2 symbols.")
    body.append("inline constexpr std::uint16_t kIsLongSideDctCdf[2][3] = {")
    d = is_long["data"]
    for i in range(2):
        body.append("    { " + ", ".join(str(v) for v in d[i * 3:(i + 1) * 3]) + " },")
    body.append("};")
    body.append("")

    body.append("// TX_CLASS_2D diagonal scans (§5.20.7.30), square sizes only;")
    body.append("// rectangles are generated by the same algorithm in tx_type.cpp.")
    for n in (4, 8, 16, 32):
        vals = scans[str(n)]
        assert len(vals) == n * n, (n, len(vals))
        body.append(f"inline constexpr std::uint16_t kScan{n}[{n * n}] = {{")
        for i in range(0, len(vals), 16):
            body.append("    " + ", ".join(str(v) for v in vals[i:i + 16]) + ",")
        body.append("};")
        body.append("")

    body.append("}  // namespace av2enc::txtype")
    body.append("")

    text = "\n".join(body)
    if "--check" in sys.argv:
        current = OUT.read_text(encoding="utf-8")
        if current != text:
            print(f"{OUT} is stale; rerun {pathlib.Path(__file__).name}",
                  file=sys.stderr)
            return 1
        print(f"{OUT} up to date")
        return 0

    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT}")
    print(f"  sizes={TX_SIZES_ALL} modes={INTRA_MODES} sets={SET_TYPES_INTRA} "
          f"types={TX_TYPES}")
    print(f"  Md_Idx_To_Type rows={len(md_idx)}x{len(md_idx[0])}")
    print(f"  Tx_Type_In_Set_Intra rows={len(tx_type_in_set)}")
    for i, row in enumerate(tx_type_in_set):
        print(f"    set {i}: {sum(row)} types")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
