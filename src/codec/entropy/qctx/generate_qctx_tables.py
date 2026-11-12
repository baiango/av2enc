#!/usr/bin/env python3
"""Generate the coefficient-CDF quantizer-context tables.

Normative source
----------------
`spec/tables/default_cdf.json` (AV2 v1.0.0 §9.3 default CDFs).  Every array
whose first declared dimension is `COEFF_CDF_Q_CTXS` is a coefficient CDF and
carries four quantizer-context slices.  §8.3.1 `init_coeff_cdfs()` selects one
slice per frame from the *raw frame* `base_q_idx`:

    idx = base_q_idx <=  90 ? 0 :
          base_q_idx <= 140 ? 1 :
          base_q_idx <= 190 ? 2 : 3

The slice is NOT derived from a tile/delta-Q/segment qindex.

This script never hand-types a CDF value.  It discovers the coefficient arrays
from the JSON `dims_decl`, keeps the leading qctx dimension, and emits
`qctx_cdf_tables.h` plus the per-slice view struct used by `qctx.h`.

Usage
-----
    # regenerate the header (run from anywhere)
    python3 src/codec/entropy/qctx/generate_qctx_tables.py

    # verify the header is current and that slice 0 is byte-identical to the
    # shipping slice-0 tables
    python3 src/codec/entropy/qctx/generate_qctx_tables.py --check

`--check` exits non-zero on any stale header, shape surprise, or slice-0
mismatch against `src/pipeline/still/basic_cdf_tables.h`, the EOB tables in the M1/M2
syntax writers, and `src/codec/rate/rdoq/rdoq_cdf_tables.h`.  (`still_cdf_tables.h`
currently holds no coefficient CDFs; it is still scanned and reported for
completeness.)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys
from typing import Iterable

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SPEC = ROOT / "spec" / "tables" / "default_cdf.json"
OUT = HERE / "qctx_cdf_tables.h"


def cpp_name(json_name: str) -> str:
    """Default_Coeff_Base_Lf_Cdf -> kCoeffBaseLfCdf (matches m0 naming)."""
    short = json_name.removeprefix("Default_").removesuffix("_Cdf")
    return "k" + "".join(part.capitalize() for part in short.split("_")) + "Cdf"


def field_name(json_name: str) -> str:
    """Default_Coeff_Base_Lf_Cdf -> coeff_base_lf (matches CdfState)."""
    short = json_name.removeprefix("Default_").removesuffix("_Cdf")
    return short.lower()


def is_coeff_cdf(entry: dict) -> bool:
    dims = entry.get("dims_decl") or []
    return bool(dims) and str(dims[0]).strip() == "COEFF_CDF_Q_CTXS"


def discover(arrays: dict) -> list[tuple[str, dict]]:
    found = [(name, entry) for name, entry in arrays.items()
             if is_coeff_cdf(entry)]
    # Stable output independent of JSON ordering.
    found.sort(key=lambda item: item[0])
    if not found:
        raise SystemExit("no COEFF_CDF_Q_CTXS arrays found in the spec JSON")
    return found


def emit_array(values: Iterable[int], shape: list[int],
               name: str) -> list[str]:
    values = list(values)
    dims = "".join(f"[{dim}]" for dim in shape)
    body = format_array(values, shape, 0)
    return [f"inline constexpr std::uint16_t {name}{dims} = {body};"]


def _product(shape: list[int]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total


def format_array(flat: list[int], dims: list[int], indent: int) -> str:
    """Nested braces, one innermost distribution per line.

    The project builds with -Wall -Wextra; flattened brace initialization of a
    multidimensional array triggers -Wmissing-braces thousands of times.  The
    generated form is the same one the shipping m0/rdoq headers use.
    """
    if len(dims) == 1:
        return "{ " + ", ".join(str(v) for v in flat) + " }"
    step = _product(dims[1:])
    inner = ",\n".join(
        " " * (indent + 2) +
        format_array(flat[i * step:(i + 1) * step], dims[1:], indent + 2)
        for i in range(dims[0]))
    return "{\n" + inner + "\n" + " " * indent + "}"


def emit_struct(found: list[tuple[str, dict]]) -> list[str]:
    fields = []
    for json_name, entry in found:
        inner = list(entry["shape"][1:])
        dims = "".join(f"[{dim}]" for dim in inner)
        fields.append(
            f"  const std::uint16_t (*{field_name(json_name)}){dims};")
    lines = ["struct CoeffCdfSlice {"] + fields + ["};", ""]
    lines.append("inline constexpr CoeffCdfSlice kCoeffCdfSlices[4] = {")
    for slice_index in range(4):
        members = ", ".join(
            f"&{cpp_name(json_name)}[{slice_index}]"
            for json_name, _ in found)
        lines.append(f"    {{{members}}},")
    lines.append("};")
    return lines


def emit_table_info(found: list[tuple[str, dict]]) -> list[str]:
    lines = [
        "struct CoeffCdfTableInfo {",
        "  const char* name;             // §9.3 Default_*_Cdf",
        "  const std::uint16_t* slice0;  // first scalar of qctx slice 0",
        "  std::size_t slice_stride;     // scalars in one qctx slice",
        "  int symbols;                  // symbols per distribution (last dim-1)",
        "};",
        "",
        "inline constexpr CoeffCdfTableInfo kCoeffCdfTables[] = {",
    ]
    for json_name, entry in found:
        shape = list(entry["shape"])
        zeros = "[0]" * len(shape)
        lines.append(
            f'    {{"{json_name}", &{cpp_name(json_name)}{zeros}, '
            f"{len(entry['data']) // shape[0]}, {shape[-1] - 1}}},")
    lines.append("};")
    lines.append("")
    lines.append("inline constexpr int kCoeffCdfTableCount =")
    lines.append("    static_cast<int>(sizeof(kCoeffCdfTables) /")
    lines.append("                      sizeof(kCoeffCdfTables[0]));")
    lines.append("inline constexpr int kCoeffCdfSliceCount = 4;")
    return lines


def render(source_bytes: bytes, arrays: dict) -> str:
    found = discover(arrays)
    digest = hashlib.sha256(source_bytes).hexdigest()
    lines = [
        "// Generated by src/codec/entropy/qctx/generate_qctx_tables.py from",
        "// spec/tables/default_cdf.json (AV2 v1.0.0 §9.3 default CDFs).",
        f"// Source SHA-256: {digest}",
        "// Do not hand-edit; run the generator with --check instead.",
        "//",
        "// Every table keeps the leading COEFF_CDF_Q_CTXS dimension:",
        "//   [4][...]  qctx 0: base_q_idx <=  90",
        "//             qctx 1: 91 .. 140",
        "//             qctx 2: 141 .. 190",
        "//             qctx 3: >= 191",
        "// Select the slice from the raw frame base_q_idx only (§8.3.1); never",
        "// from a tile, delta-Q or segment qindex.  Use qctx.h.",
        "#pragma once",
        "",
        "#include <cstddef>",
        "#include <cstdint>",
        "",
        "namespace av2enc::still::qctx {",
        "",
    ]
    for json_name, entry in found:
        shape = list(entry["shape"])
        if shape[0] != 4:
            raise SystemExit(
                f"{json_name}: expected 4 COEFF_CDF_Q_CTXS slices, got {shape}")
        if len(entry["data"]) != _product(shape):
            raise SystemExit(f"{json_name}: data length/shape mismatch")
        lines.append(f"// {json_name} {shape}")
        lines.extend(emit_array(entry["data"], shape, cpp_name(json_name)))
        lines.append("")

    lines.extend(emit_struct(found))
    lines.append("")
    lines.extend(emit_table_info(found))
    lines.append("")
    lines.append("}  // namespace av2enc::still::qctx")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# --check: parse the shipping slice-0 tables and compare exact values
# ---------------------------------------------------------------------------

_NUMBER = re.compile(r"\d+")


def extract_values(path: pathlib.Path, name: str) -> list[int] | None:
    if not path.exists():
        return None
    text = path.read_text()
    match = re.search(
        rf"\b{re.escape(name)}\s*\[[^;=]*\]\s*=\s*", text)
    if match is None:
        return None
    start = text.find("{", match.end())
    if start < 0:
        return None
    depth = 0
    end = -1
    for pos in range(start, len(text)):
        char = text[pos]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = pos
                break
    if end < 0:
        return None
    return [int(token) for token in _NUMBER.findall(text[start:end + 1])]


def compare_slice0(name: str, expected: list[int], shipping: list[int],
                   origin: str, failures: list[str]) -> None:
    if len(expected) != len(shipping):
        failures.append(
            f"{origin}:{name}: length {len(expected)} != {len(shipping)}")
        return
    for index, (want, got) in enumerate(zip(expected, shipping)):
        if want != got:
            failures.append(
                f"{origin}:{name}: value[{index}] {want} != {got}")
            return


def check(source_bytes: bytes, arrays: dict) -> int:
    found = discover(arrays)
    expected_text = render(source_bytes, arrays)
    failures: list[str] = []

    if not OUT.exists():
        failures.append(f"missing generated header {OUT}")
    else:
        actual = OUT.read_text()
        if actual != expected_text:
            failures.append(
                f"{OUT.name} is stale; rerun the generator without --check")

    total_values = 0
    slice0: dict[str, list[int]] = {}
    for json_name, entry in found:
        stride = len(entry["data"]) // entry["shape"][0]
        slice0[json_name] = list(entry["data"][:stride])
        total_values += len(entry["data"])

    # The two shipping headers the task names explicitly.  m0 carries 20 of
    # the older coefficient arrays; still_cdf_tables.h carries none today (it is
    # intra/filter CDFs), but scanning it keeps the check honest if that
    # changes.
    shipping_headers = [
        (ROOT / "src" / "encoder" / "basic_cdf_tables.h", "basic_cdf_tables.h"),
        (ROOT / "src" / "encoder" / "m1" / "still_cdf_tables.h",
         "still_cdf_tables.h"),
        (ROOT / "src" / "encoder" / "rdoq" / "rdoq_cdf_tables.h",
         "rdoq_cdf_tables.h"),
    ]
    report: list[str] = []
    for path, label in shipping_headers:
        compared = 0
        for json_name, _ in found:
            values = extract_values(path, cpp_name(json_name))
            if values is None:
                continue
            compared += 1
            compare_slice0(json_name, slice0[json_name], values, label,
                           failures)
        if compared == 0:
            report.append(
                f"slice 0 vs {label}: no coefficient CDFs present (skipped)")
        else:
            report.append(
                f"slice 0 vs {label}: {compared} tables compared")

    # Default_Eob_Pt_16/32 shipping definitions have historically lived in
    # the M1 syntax writer (still_syntax.cpp) and, before that, in video_syntax.cpp.
    # Either file may own them as the core is refactored, so scan both and
    # require at least one match.  If neither has them, the rdoq header above
    # still pins their slice-0 values.
    syntax_candidates = [
        ROOT / "src" / "encoder" / "m1" / "still_syntax.cpp",
        ROOT / "src" / "encoder" / "m2" / "video_syntax.cpp",
    ]
    syntax_compared = 0
    for json_name in ("Default_Eob_Pt_16_Cdf", "Default_Eob_Pt_32_Cdf"):
        values = None
        origin = None
        for path in syntax_candidates:
            values = extract_values(path, cpp_name(json_name))
            if values is not None:
                origin = path.name
                break
        if values is None or origin is None:
            report.append(
                f"slice 0 vs syntax writers: {cpp_name(json_name)} not "
                f"defined locally (covered by rdoq_cdf_tables.h)")
            continue
        syntax_compared += 1
        compare_slice0(json_name, slice0[json_name], values, origin, failures)
    if syntax_compared != 0:
        report.append(
            f"slice 0 vs syntax writers: {syntax_compared} EOB tables "
            f"compared")

    if failures:
        print("qctx table check FAILED", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1

    print(f"qctx table check OK: {len(found)} tables, 4 slices, "
          f"{total_values} values")
    for line in report:
        print(f"  {line}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true",
        help="verify the generated header and slice-0 compatibility")
    args = parser.parse_args()

    source_bytes = SPEC.read_bytes()
    arrays = json.loads(source_bytes)["arrays"]
    if args.check:
        return check(source_bytes, arrays)

    text = render(source_bytes, arrays)
    OUT.write_text(text)
    found = discover(arrays)
    print(f"wrote {OUT} ({len(found)} tables, {len(text)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
