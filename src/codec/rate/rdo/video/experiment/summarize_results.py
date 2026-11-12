#!/usr/bin/env python3
"""Format the M2 RDO experiment TSV into Markdown + JSON.

Input TSV columns:
    case  variant  bytes  psnr_y  psnr_avg  cvvdp  ivf

Delta columns are relative to `base_default` in the same case.  This script
makes no speed claim and performs no encoding; it only formats measurements.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("tsv", type=Path)
    parser.add_argument("markdown", type=Path)
    parser.add_argument("json_out", type=Path)
    args = parser.parse_args()

    rows = []
    with args.tsv.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            rows.append(
                {
                    "case": row["case"],
                    "variant": row["variant"],
                    "bytes": int(row["bytes"]),
                    "psnr_y": float(row["psnr_y"]),
                    "psnr_avg": float(row["psnr_avg"]),
                    "cvvdp": float(row["cvvdp"]),
                    "ivf": row["ivf"],
                }
            )

    by_case: dict[str, list[dict]] = {}
    for row in rows:
        by_case.setdefault(row["case"], []).append(row)

    lines = [
        "# M2 P-frame rate-aware decision experiment",
        "",
        "All streams decoded with `dav2d --strict 1`; the M2 gate compares the",
        "decoded pictures against the encoder reconstruction sample by sample.",
        "PSNR/CVVDP are from `tools/quality.py` on the decoded stream.",
        "No speed claim: the columns are bytes and quality only.",
        "",
    ]
    summary: dict[str, object] = {}
    for case, entries in by_case.items():
        baseline = next(
            (entry for entry in entries if entry["variant"] == "base_default"),
            None,
        )
        lines.append(f"## {case}")
        lines.append("")
        lines.append(
            "| variant | bytes | Δbytes | PSNR-Y | ΔPSNR-Y | PSNR-avg | "
            "CVVDP | ΔCVVDP |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        case_summary = {}
        for entry in entries:
            if baseline is None:
                delta_bytes = ""
                delta_psnr = ""
                delta_cvvdp = ""
            else:
                delta_bytes = (
                    f"{100.0 * (entry['bytes'] - baseline['bytes']) / baseline['bytes']:+.2f}%"
                )
                delta_psnr = f"{entry['psnr_y'] - baseline['psnr_y']:+.3f}"
                delta_cvvdp = f"{entry['cvvdp'] - baseline['cvvdp']:+.4f}"
            lines.append(
                f"| {entry['variant']} | {entry['bytes']} | {delta_bytes} | "
                f"{entry['psnr_y']:.3f} | {delta_psnr} | "
                f"{entry['psnr_avg']:.3f} | {entry['cvvdp']:.4f} | "
                f"{delta_cvvdp} |"
            )
            case_summary[entry["variant"]] = entry
        lines.append("")
        summary[case] = case_summary

    args.markdown.write_text("\n".join(lines) + "\n")
    args.json_out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
