#!/usr/bin/env python3
"""Interpolate one arm's K at a byte target from a hk_screen grid, and find
where its K crosses a threshold.

    tools/hk_curve.py "DIR" LABEL --targets 29000 29500 30000 --cross 50
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path


def kpi(j, b, s):
    return 100.0 * (0.50 * min(1.0, j / 10.0) + 0.25 / (1.0 + max(b, 0.0) / 3.0)
                    + 0.25 * max(0.0, min(1.0, (s + 50.0) / 150.0)))


def curve(directory: Path, label: str):
    pts = {}
    for f in glob.glob(str(directory / f"{label}-*" / f"{label}_q*_quality.json")):
        qp = int(f.split("_q")[1].split("_")[0])
        ivf = f.replace("_quality.json", ".ivf")
        ej = f.replace("_quality.json", "_extra.json")
        if not (os.path.exists(ivf) and os.path.getsize(ivf) > 0 and os.path.exists(ej)):
            continue
        q, e = json.loads(Path(f).read_text()), json.loads(Path(ej).read_text())
        j, b, s = q["cvvdp_jod"], e["butteraugli_mean"], e["ssimulacra2_mean"]
        pts[qp] = (os.path.getsize(ivf), qp, j, b, s, kpi(j, b, s))
    return sorted(pts.values())


def at(c, target):
    lo = [p for p in c if p[0] <= target]
    hi = [p for p in c if p[0] >= target]
    if not lo or not hi:
        return None
    a, b = max(lo, key=lambda p: p[0]), min(hi, key=lambda p: p[0])
    t = 0.0 if a[0] == b[0] else (target - a[0]) / (b[0] - a[0])
    return (a[5] + t * (b[5] - a[5]), a[1], b[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    ap.add_argument("label")
    ap.add_argument("--targets", type=int, nargs="*", default=[])
    ap.add_argument("--cross", type=float, default=None)
    a = ap.parse_args()
    c = curve(a.dir, a.label)
    print(f"=== {a.label}: {len(c)} points, {c[0][0]}..{c[-1][0]} B")
    for p in c:
        print(f"  QP{p[1]} {p[0]:7d} B  JOD {p[2]:.3f} BA {p[3]:5.2f} "
              f"SS2 {p[4]:6.2f}  K {p[5]:.2f}")
    for t in a.targets:
        r = at(c, t)
        print(f"  @{t} B -> K {r[0]:.3f} (QP{r[1]}..QP{r[2]})" if r
              else f"  @{t} B -> outside")
    if a.cross is not None:
        for A, B in zip(c, c[1:]):
            if A[5] < a.cross <= B[5]:
                tt = (a.cross - A[5]) / (B[5] - A[5])
                print(f"  K={a.cross} at {A[0] + tt * (B[0] - A[0]):.0f} B "
                      f"(QP{A[1]} {A[0]} B -> QP{B[1]} {B[0]} B)")
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
