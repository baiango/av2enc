#!/usr/bin/env python3
"""Ladder summary for docs/rd-tail-floor.md.

Reads the ladder directory written by tools/lane-tail-floor/ladder.sh:
`<arm>_q<QP>.ivf`, `.psnr` (perframe_psnr.py rows), `.log`, and reports
bytes / kbps / mean PSNR-Y per arm and QP.  Streams are 60 frames at 30 fps,
so duration is 2.0 s.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

FPS = 30.0
FRAMES = 60


def read_psnr(path: Path) -> float:
    total = 0.0
    count = 0
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            try:
                total += float(parts[1])
                count += 1
            except ValueError:
                continue
    return total / count if count else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, type=Path)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--qps", default="")
    args = ap.parse_args()

    qps = [int(q) for q in args.qps.split()] if args.qps else None
    rows = []
    for ivf in sorted(args.dir.glob("*_q*.ivf")):
        match = re.match(r"(.+)_q(\d+)\.ivf$", ivf.name)
        if not match:
            continue
        arm, qp = match.group(1), int(match.group(2))
        if qps is not None and qp not in qps:
            continue
        psnr_path = args.dir / f"{arm}_q{qp}.psnr"
        bytes_ = ivf.stat().st_size
        rows.append({
            "arm": arm,
            "qp": qp,
            "bytes": bytes_,
            "kbps": bytes_ * 8.0 / (FRAMES / FPS) / 1000.0,
            "psnr_y": read_psnr(psnr_path) if psnr_path.exists() else None,
            "recon": (args.dir / f"{arm}_q{qp}.recon.ok").exists()
            and (args.dir / f"{arm}_q{qp}.recon.ok").read_text().strip(),
        })
    rows.sort(key=lambda r: (r["arm"], r["qp"]))
    print("| QP | arm | bytes | kbps | PSNR-Y |")
    print("|---:|---|---:|---:|---:|")
    for r in rows:
        print(f"| {r['qp']} | {r['arm']} | {r['bytes']:,} | {r['kbps']:.3f} | "
              f"{r['psnr_y']:.3f} |")
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
