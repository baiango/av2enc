#!/usr/bin/env python3
"""Compare two arms' K at matched bytes by interpolating their rate curves.

Every `*_q<qp>_quality.json` / `_extra.json` / `.ivf` triple in a directory is
one measured point.  For each byte target inside the overlap of the two curves,
interpolate every metric linearly in bytes between the bracketing points and
recompute K with `tools/kpi.py`'s formula.  This is the comparison that matters:
at equal QP an arm that spends more bytes always looks better.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path


def kpi(jod: float, ba: float, ss2: float) -> float:
    q_c = max(0.0, min(1.0, jod / 10.0))
    q_b = 1.0 / (1.0 + max(ba, 0.0) / 3.0)
    q_s = max(0.0, min(1.0, (ss2 + 50.0) / 150.0))
    return 100.0 * (0.50 * q_c + 0.25 * q_b + 0.25 * q_s)


def curve(directory: Path, label: str) -> list[tuple[int, dict]]:
    out = []
    for qj in glob.glob(str(directory / f"{label}_q*_quality.json")):
        qp = int(re.search(r"_q(\d+)_", qj).group(1))
        ivf = directory / f"{label}_q{qp}.ivf"
        ej = directory / f"{label}_q{qp}_extra.json"
        if not (ivf.exists() and ivf.stat().st_size > 0 and ej.exists()):
            continue
        q, e = json.loads(Path(qj).read_text()), json.loads(ej.read_text())
        out.append((ivf.stat().st_size,
                    {"qp": qp, "jod": q["cvvdp_jod"],
                     "ba": e["butteraugli_mean"],
                     "ss2": e["ssimulacra2_mean"],
                     "psnr_y": q["psnr"]["psnr_y"]}))
    return sorted(out)


def at(curve_: list[tuple[int, dict]], target: int) -> dict | None:
    lo = [p for p in curve_ if p[0] <= target]
    hi = [p for p in curve_ if p[0] >= target]
    if not lo or not hi:
        return None
    a, b = max(lo, key=lambda p: p[0]), min(hi, key=lambda p: p[0])
    t = 0.0 if a[0] == b[0] else (target - a[0]) / (b[0] - a[0])
    m = {k: a[1][k] + t * (b[1][k] - a[1][k])
         for k in ("jod", "ba", "ss2", "psnr_y")}
    return {"qp": (a[1]["qp"], b[1]["qp"]), "bytes": (a[0], b[0]),
            "gap": abs(b[0] - a[0]) / target, **m,
            "K": kpi(m["jod"], m["ba"], m["ss2"])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--targets", type=int, nargs="+", default=[24500, 25010, 25500])
    args = ap.parse_args()
    cb, ca = curve(args.dir, args.base), curve(args.dir, args.arm)
    print(f"base {args.base}: {[p[0] for p in cb]}")
    print(f"arm  {args.arm}: {[p[0] for p in ca]}")
    print(f"{'target':>7} | {'base K':>7} {'base qp':>12} | "
          f"{'arm K':>7} {'arm qp':>12} | {'dK':>6}")
    for t in args.targets:
        b, a = at(cb, t), at(ca, t)
        if b is None or a is None:
            print(f"{t:7d} | outside the overlap")
            continue
        print(f"{t:7d} | {b['K']:7.2f} {str(b['qp']):>12} | "
              f"{a['K']:7.2f} {str(a['qp']):>12} | {a['K'] - b['K']:+6.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
