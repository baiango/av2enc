#!/usr/bin/env python3
"""Cross-check the vendored attachments against the pinned PDF.

AV2 v1.0.0's HTML rendering of §9.8/§9.9 only says "defined in the <name>.h
header file", but the pinned PDF (`spec/AV2_Spec_v1.0.0.pdf`, SHA-256
`e9916f09…b533f`) prints the full initializer data in §9.8 and §9.9.  This
script parses that printed data and compares it, number for number, with the
verbatim attachments in `raw/` (which are also what generated the C++ headers).

It is optional: the PDF is not redistributed, and `pypdf` may not be
installed.  Without either, it prints SKIP and exits 0 unless `--require` is
given.

Usage:
    python3 verify_pdf_tables.py
    python3 verify_pdf_tables.py --check      # same; nonzero on mismatch
    python3 verify_pdf_tables.py --require    # missing PDF/pypdf is an error
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
PDF = REPO_ROOT / "spec" / "AV2_Spec_v1.0.0.pdf"
RAW = HERE / "raw"

# (spec name, raw file)
TABLES = [
    ("Gdf_Alpha", "gdf_alpha.h"),
    ("Gdf_Weight", "gdf_weight.h"),
    ("Gdf_Bias", "gdf_bias.h"),
    ("Gdf_Intra_Error", "gdf_intra_error.h"),
    ("Gdf_Inter_Error", "gdf_inter_error.h"),
    ("Pc_Wiener_Sub_Classify2", "pc_wiener_sub_classify2.h"),
    ("Pc_Wiener_Lut_To_Class", "pc_wiener_lut_to_class.h"),
    ("Pc_Wiener_Sub_Classify", "pc_wiener_sub_classify.h"),
    ("Pc_Wiener_Filters", "pc_wiener_filters.h"),
    ("Gaussian_Sequence", "gaussian_sequence.h"),
]


def normalize(text: str) -> str:
    text = re.sub(r"AV2 Specification Page \d+ of \d+", " ", text)
    return re.sub(r"\s+", " ", text)


def declaration_re(name: str) -> re.Pattern[str]:
    return re.compile(re.escape(name) + r"\s*(?:\[[^\]]*\])+\s*=")


def parse_initializer(text: str, start: int) -> list[int]:
    eq = text.find("=", start)
    if eq < 0:
        raise ValueError("no initializer")
    values: list[int] = []
    depth = 0
    started = False
    i = eq + 1
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
            started = True
            i += 1
            continue
        if ch == "}":
            depth -= 1
            i += 1
            if started and depth == 0:
                return values
            continue
        if ch == "-" or ch.isdigit():
            m = re.match(r"-?\s*\d+", text[i:])
            if m:
                values.append(int(m.group(0).replace(" ", "")))
                i += len(m.group(0))
                continue
        i += 1
    raise ValueError("unterminated initializer")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="nonzero on mismatch")
    ap.add_argument("--require", action="store_true",
                    help="fail if the PDF or pypdf is unavailable")
    args = ap.parse_args(argv)

    if not PDF.is_file():
        msg = f"SKIP: {PDF} not present (not redistributed)"
        print(msg, file=sys.stderr if args.require else sys.stdout)
        return 1 if args.require else 0
    try:
        import pypdf  # type: ignore
    except Exception:
        msg = "SKIP: pypdf not installed (pip install pypdf)"
        print(msg, file=sys.stderr if args.require else sys.stdout)
        return 1 if args.require else 0

    reader = pypdf.PdfReader(str(PDF))
    pages = [normalize(page.extract_text() or "") for page in reader.pages]
    text = " ".join(pages)

    failed = False
    for name, filename in TABLES:
        pattern = declaration_re(name)
        match = pattern.search(text)
        if not match:
            print(f"MISSING {name} in PDF", file=sys.stderr)
            failed = True
            continue
        try:
            pdf_values = parse_initializer(text, match.start())
        except ValueError as exc:
            print(f"PARSE   {name}: {exc}", file=sys.stderr)
            failed = True
            continue

        body = (RAW / filename).read_text(encoding="utf-8").split("=", 1)[1]
        raw_values = [int(x) for x in re.findall(r"-?\d+", body)]
        if pdf_values != raw_values:
            print(f"MISMATCH {name}: pdf {len(pdf_values)} values, "
                  f"raw {len(raw_values)}", file=sys.stderr)
            for i, (a, b) in enumerate(zip(pdf_values, raw_values)):
                if a != b:
                    print(f"  first diff at index {i}: pdf {a}, raw {b}",
                          file=sys.stderr)
                    break
            failed = True
            continue

        page = next((i + 1 for i, p in enumerate(pages) if pattern.search(p)), 0)
        print(f"ok      {name:24s} {len(raw_values):6d} values (PDF p{page})")

    if failed:
        return 1
    print(f"all {len(TABLES)} tables match the pinned PDF")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
