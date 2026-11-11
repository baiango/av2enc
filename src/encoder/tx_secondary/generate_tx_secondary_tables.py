#!/usr/bin/env python3
"""Generate the AV2 secondary-transform tables.

Sources (AV2 v1.0.0, commit 38f28e7):

  §9.7 attachments from https://av2.aomedia.org/v1.0.0/attachments/
    ist_4x4_kernel.h   sha256 9fafc6bc5086c992dabc1b26816713f9a56e5ccdba386130034165d6df985404
    ist_8x8_kernel.h   sha256 65d17c0e9a34843da3f5fe48262c7170383b59c15e477ab55733796de871fcc4
    stx_scan_map.h     sha256 8825c77db8b394c9274df8cb368c14271d12eed218eb228aae133956c4ec7146

  §9.2  spec/tables/block_tables.json + constants.json
  §9.3  spec/tables/default_cdf.json
  §7.15.3 mappings and Stx_Scan_Order arrays: spec/tables/_cache/spec.html

The generated headers are committed, so a normal build does not need the PDF,
the HTML cache, or network access.

Usage:
    python3 generate_tx_secondary_tables.py
    python3 generate_tx_secondary_tables.py --attachments /path/to/headers
    python3 generate_tx_secondary_tables.py --check

`--attachments` points at a directory containing the three §9.7 headers.  With
no option the script downloads them into $TMPDIR/av2enc_ist_attachments.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
TABLES = REPO / "spec" / "tables"
SPEC_HTML = TABLES / "_cache" / "spec.html"
BASE_URL = "https://av2.aomedia.org/v1.0.0/attachments/"
ATTACHMENTS = {
    "ist_4x4_kernel.h": "9fafc6bc5086c992dabc1b26816713f9a56e5ccdba386130034165d6df985404",
    "ist_8x8_kernel.h": "65d17c0e9a34843da3f5fe48262c7170383b59c15e477ab55733796de871fcc4",
    "stx_scan_map.h": "8825c77db8b394c9274df8cb368c14271d12eed218eb228aae133956c4ec7146",
}


def clean_html(text: str) -> str:
    text = re.sub(r"<script.*?</script>", " ", text, flags=re.S)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    return html.unescape(re.sub(r"\s+", " ", text))


def brace_block(text: str, marker: str) -> str:
    start = text.index(marker)
    start = text.index("{", start)
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i]
    raise ValueError(f"unterminated brace block after {marker!r}")


def ints(text: str) -> list[int]:
    return [int(x) for x in re.findall(r"-?\d+", text)]


def load_attachment(path: Path) -> list[int]:
    raw = path.read_text()
    body = raw.split("=", 1)[1]
    body = re.sub(r"//.*", "", body)
    return [int(x) for x in re.findall(r"-?\d+", body)]


def fetch_attachments(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, digest in ATTACHMENTS.items():
        path = directory / name
        if not path.exists():
            print(f"download {BASE_URL + name}", file=sys.stderr)
            with urllib.request.urlopen(BASE_URL + name, timeout=60) as response:
                path.write_bytes(response.read())
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            raise SystemExit(
                f"{name}: sha256 {actual} != expected {digest}; refusing to guess"
            )


def shape_dims(shape: str) -> list[int]:
    return [int(x) for x in re.findall(r"\[(\d+)\]", shape)]


def emit_nested(values: list[int], dims: list[int], level: int = 0) -> str:
    pad = "    " * (level + 1)
    if len(dims) == 1:
        return "{ " + ", ".join(f"{v:4d}" for v in values) + " }"
    stride = 1
    for d in dims[1:]:
        stride *= d
    parts = []
    for i in range(dims[0]):
        parts.append(
            emit_nested(values[i * stride : (i + 1) * stride], dims[1:], level + 1)
        )
    return "{\n" + pad + (",\n" + pad).join(parts) + "\n" + "    " * level + "}"


def emit_array(name: str, ctype: str, shape: str, values: list[int], per_line: int = 16) -> str:
    del per_line
    dims = shape_dims(shape)
    count = 1
    for d in dims:
        count *= d
    assert count == len(values), (name, count, len(values))
    return f"inline constexpr {ctype} {name}{shape} = {emit_nested(values, dims)};\n"


def emit_device_array(name: str, ctype: str, shape: str, values: list[int], per_line: int = 16) -> str:
    del per_line
    dims = shape_dims(shape)
    count = 1
    for d in dims:
        count *= d
    assert count == len(values), (name, count, len(values))
    return f"__device__ const {ctype} {name}{shape} = {emit_nested(values, dims)};\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attachments", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    directory = args.attachments
    if directory is None:
        directory = Path(os.environ.get("TMPDIR", "/tmp")) / "av2enc_ist_attachments"
        fetch_attachments(directory)
    else:
        for name, digest in ATTACHMENTS.items():
            path = directory / name
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != digest:
                raise SystemExit(f"{name}: sha256 {actual} != expected {digest}")

    ist4 = load_attachment(directory / "ist_4x4_kernel.h")
    ist8 = load_attachment(directory / "ist_8x8_kernel.h")
    scan_map = load_attachment(directory / "stx_scan_map.h")
    assert len(ist4) == 14 * 3 * 8 * 16, len(ist4)
    assert len(ist8) == 11 * 3 * 32 * 48, len(ist8)
    assert len(scan_map) == 11 * 3 * 48, len(scan_map)
    assert min(ist4) >= -128 and max(ist4) <= 127
    assert min(ist8) >= -128 and max(ist8) <= 127
    assert max(scan_map) <= 63

    constants = json.loads((TABLES / "constants.json").read_text())
    block = json.loads((TABLES / "block_tables.json").read_text())
    cdf = json.loads((TABLES / "default_cdf.json").read_text())["arrays"]

    tx_width = block["Tx_Width"]["data"]
    tx_height = block["Tx_Height"]["data"]
    tx_width_log2 = block["Tx_Width_Log2"]["data"]
    tx_height_log2 = block["Tx_Height_Log2"]["data"]
    tx_size_sqr = block["Tx_Size_Sqr"]["data"]
    mode_to_angle = block["Mode_To_Angle"]["data"]

    sec_cdf = cdf["Default_Sec_Tx_Type_Cdf"]["data"]
    mps_cdf = cdf["Default_Most_Probable_Stx_Set_Cdf"]["data"]
    mps_adst_cdf = cdf["Default_Most_Probable_Stx_Set_Adst_Cdf"]["data"]
    assert len(sec_cdf) == 2 * 5 * 5
    assert len(mps_cdf) == 8
    assert len(mps_adst_cdf) == 5

    if not SPEC_HTML.exists():
        raise SystemExit(
            f"{SPEC_HTML} is required to regenerate the §7.15.3 mappings/scans; "
            "the committed generated headers already contain them"
        )
    text = clean_html(SPEC_HTML.read_text(encoding="utf-8", errors="replace"))

    mapping = ints(
        brace_block(text, "Inv_Most_Probable_Stx_Mapping [ INTRA_MODES - 1 ][ IST_DIR_SIZE ] =")
    )
    mapping_adst = ints(
        brace_block(
            text,
            "Inv_Most_Probable_Stx_Mapping_Adst [ INTRA_MODES - 1 ] [ IST_REDUCE_SET_SIZE_ADST_ADST ] =",
        )
    )
    scan_order_4 = ints(brace_block(text, "Stx_Scan_Order_4x4 [ IST_4X4_WIDTH ] ="))
    scan_order_8 = ints(brace_block(text, "Stx_Scan_Order_8x8 [ 64 ] ="))
    assert len(mapping) == 12 * 7, len(mapping)
    assert len(mapping_adst) == 12 * 4, len(mapping_adst)
    assert len(scan_order_4) == 16
    assert len(scan_order_8) == 64
    assert sorted(scan_order_4) == list(range(16))
    assert sorted(scan_order_8) == list(range(64))
    assert sorted(mapping[:7]) == list(range(7))
    assert sorted(mapping_adst[:4]) == list(range(4))

    # Reference values from the §9.7 attachment headers / spec text.  These
    # catch a parser that silently reads the wrong array.
    assert ist4[:8] == [102, -45, -53, -5, -13, -3, -3, 19]
    assert ist8[:8] == [103, -44, -48, -9, -12, -6, -4, 19]
    assert scan_map[:8] == [0, 1, 2, 3, 4, 5, 6, 7]
    assert mapping[:7] == [6, 1, 0, 5, 4, 3, 2]
    assert mapping_adst[:4] == [3, 1, 0, 2]
    assert scan_order_4[:8] == [0, 1, 4, 8, 5, 2, 3, 6]
    assert scan_order_8[:8] == [0, 1, 8, 16, 9, 2, 3, 10]

    header = []
    header.append(
        "// Generated by src/encoder/tx_secondary/generate_tx_secondary_tables.py.\n"
        "// Sources: AV2 v1.0.0 §7.15.3, §9.2, §9.3, §9.7 (commit 38f28e7).\n"
        "// Do not edit by hand.\n"
        "#pragma once\n\n#include <cstdint>\n\nnamespace av2enc::txsec {\n\n"
    )
    header.append(f"inline constexpr int kStxTypes = {constants['STX_TYPES']};\n")
    header.append(f"inline constexpr int kIstDirSize = {constants['IST_DIR_SIZE']};\n")
    header.append(
        f"inline constexpr int kIstReduceSetSizeAdstAdst = "
        f"{constants['IST_REDUCE_SET_SIZE_ADST_ADST']};\n"
    )
    header.append(f"inline constexpr int kIst4x4Width = {constants['IST_4X4_WIDTH']};\n")
    header.append(f"inline constexpr int kIst4x4Height = {constants['IST_4X4_HEIGHT']};\n")
    header.append(f"inline constexpr int kIst8x8Width = {constants['IST_8X8_WIDTH']};\n")
    header.append(f"inline constexpr int kIst8x8Height = {constants['IST_8X8_HEIGHT']};\n")
    header.append(
        f"inline constexpr int kIst8x8HeightRed = {constants['IST_8X8_HEIGHT_RED']};\n"
    )
    header.append(f"inline constexpr int kIstSetSize4x4 = {constants['IST_SET_SIZE_4X4']};\n")
    header.append(f"inline constexpr int kIstSetSize8x8 = {constants['IST_SET_SIZE_8X8']};\n")
    header.append(f"inline constexpr int kTxSizes = {constants['TX_SIZES']};\n\n")

    header.append("// §9.2 tables, indexed by TX_SIZE 0..24.\n")
    header.append(emit_array("kTxWidth", "std::uint8_t", "[25]", tx_width))
    header.append(emit_array("kTxHeight", "std::uint8_t", "[25]", tx_height))
    header.append(emit_array("kTxWidthLog2", "std::uint8_t", "[25]", tx_width_log2))
    header.append(emit_array("kTxHeightLog2", "std::uint8_t", "[25]", tx_height_log2))
    header.append(emit_array("kTxSizeSqr", "std::uint8_t", "[25]", tx_size_sqr))
    header.append(emit_array("kModeToAngle", "std::uint8_t", "[13]", mode_to_angle))

    header.append("// §7.15.3 Inv_Most_Probable_Stx_Mapping[YMode][most_probable_stx_set].\n")
    header.append(emit_array("kInvMostProbableStxMapping", "std::uint8_t", "[12][7]", mapping))
    header.append(
        "// §7.15.3 Inv_Most_Probable_Stx_Mapping_Adst[YMode][most_probable_stx_set].\n"
    )
    header.append(
        emit_array("kInvMostProbableStxMappingAdst", "std::uint8_t", "[12][4]", mapping_adst)
    )

    header.append("// §7.15.3 Stx_Scan_Order_4x4 / Stx_Scan_Order_8x8.\n")
    header.append(emit_array("kStxScanOrder4x4", "std::uint8_t", "[16]", scan_order_4))
    header.append(emit_array("kStxScanOrder8x8", "std::uint8_t", "[64]", scan_order_8))

    header.append("// §9.3 Default_Sec_Tx_Type_Cdf[is_inter][Tx_Size_Sqr][STX_TYPES+1].\n")
    header.append(emit_array("kSecTxTypeCdf", "std::uint16_t", "[2][5][5]", sec_cdf, per_line=5))
    header.append("// §9.3 Default_Most_Probable_Stx_Set_Cdf[IST_DIR_SIZE+1].\n")
    header.append(emit_array("kMostProbableStxSetCdf", "std::uint16_t", "[8]", mps_cdf, per_line=8))
    header.append(
        "// §9.3 Default_Most_Probable_Stx_Set_Adst_Cdf"
        "[IST_REDUCE_SET_SIZE_ADST_ADST+1].\n"
    )
    header.append(
        emit_array("kMostProbableStxSetAdstCdf", "std::uint16_t", "[5]", mps_adst_cdf, per_line=5)
    )

    header.append(
        "// §9.7 Ist_4x4_Kernel[IST_SET_SIZE_4X4][STX_TYPES-1]"
        "[IST_4X4_HEIGHT][IST_4X4_WIDTH].\n"
        "// Values fit int8; the spec header declares int16.  The arithmetic\n"
        "// sign-extends before the int64 accumulation, so this is exact.\n"
    )
    header.append(emit_array("kIst4x4Kernel", "std::int8_t", "[14][3][8][16]", ist4))
    header.append(
        "// §9.7 Ist_8x8_Kernel[IST_SET_SIZE_8X8][STX_TYPES-1]"
        "[IST_8X8_HEIGHT][IST_8X8_WIDTH].\n"
    )
    header.append(emit_array("kIst8x8Kernel", "std::int8_t", "[11][3][32][48]", ist8, per_line=24))
    header.append(
        "// §9.7 Stx_Scan_Map[IST_SET_SIZE_8X8][STX_TYPES-1][IST_8X8_WIDTH].\n"
    )
    header.append(emit_array("kStxScanMap", "std::uint8_t", "[11][3][48]", scan_map, per_line=24))
    header.append("\n}  // namespace av2enc::txsec\n")

    device = []
    device.append(
        "// Generated by src/encoder/tx_secondary/generate_tx_secondary_tables.py.\n"
        "// Device copies of the §9.7 tables; include only from CUDA translation units.\n"
        "// Do not edit by hand.\n"
        "#pragma once\n\n#include <cstdint>\n\n"
        "namespace av2enc::txsec::dev {\n\n"
    )
    device.append(emit_device_array("kIst4x4Kernel", "std::int8_t", "[14][3][8][16]", ist4))
    device.append(
        emit_device_array("kIst8x8Kernel", "std::int8_t", "[11][3][32][48]", ist8, per_line=24)
    )
    device.append(
        emit_device_array("kStxScanMap", "std::uint8_t", "[11][3][48]", scan_map, per_line=24)
    )
    device.append(emit_device_array("kTxWidth", "std::uint8_t", "[25]", tx_width))
    device.append(emit_device_array("kTxHeight", "std::uint8_t", "[25]", tx_height))
    device.append(emit_device_array("kTxWidthLog2", "std::uint8_t", "[25]", tx_width_log2))
    device.append(emit_device_array("kStxScanOrder4x4", "std::uint8_t", "[16]", scan_order_4))
    device.append(emit_device_array("kStxScanOrder8x8", "std::uint8_t", "[64]", scan_order_8))
    device.append("\n}  // namespace av2enc::txsec::dev\n")

    outputs = {
        HERE / "tx_secondary_tables.h": "".join(header),
        HERE / "tx_secondary_tables_dev.cuh": "".join(device),
    }
    failed = False
    for path, content in outputs.items():
        if args.check:
            if not path.exists() or path.read_text() != content:
                print(f"stale: {path}", file=sys.stderr)
                failed = True
        else:
            path.write_text(content)
            print(f"wrote {path} ({len(content)} bytes)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
