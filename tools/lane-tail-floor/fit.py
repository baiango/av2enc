#!/usr/bin/env python3
"""Five-point fits for docs/rd-tail-floor.md.

* `--ladder` oracle JSON  -> matched K (bracket) and a five-point OLS fit of K
  against bitrate, evaluated at the target rate.
* `--single DIR` single-point oracle JSONs -> the per-QP source-replacement
  ceiling curve, each segment fitted against bitrate and evaluated at target.

Ordinary least squares, the same reading docs/rd-tail.md and docs/rd-engine0.md
use.  R^2 is reported so a noisy ceiling row is visible.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SEGMENTS = ("f0-32", "f33-43", "f44-59")


def ols(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx if sxx else 0.0
    intercept = my - slope * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 0.0
    return intercept, slope, r2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", type=Path)
    ap.add_argument("--single", type=Path)
    ap.add_argument("--target", type=float, default=100.0)
    args = ap.parse_args()

    if args.ladder:
        report = json.loads(args.ladder.read_text())
        print("## five-point K fit")
        print()
        print("| candidate | points | bracket | K@target | fit K | slope | R2 |")
        print("|---|---:|---|---:|---:|---:|---:|")
        for cand in report["candidates"]:
            pts = cand["points"]
            xs = [p["kbps"] for p in pts]
            ys = [p["kpi"] for p in pts]
            intercept, slope, r2 = ols(xs, ys)
            fit = intercept + slope * args.target
            matched = cand["matched"]
            print(f"| {cand['candidate']} | {len(pts)} | "
                  f"{'..'.join(matched['bracket'])} | {matched['kpi']:.3f} | "
                  f"{fit:.3f} | {slope:+.4f} | {r2:.3f} |")
        print()
        print("| candidate | QP | kbps | K |")
        print("|---|---:|---:|---:|")
        for cand in report["candidates"]:
            for p in sorted(cand["points"], key=lambda p: p["kbps"]):
                print(f"| {cand['candidate']} | {p['point']} | {p['kbps']:.3f} | "
                      f"{p['kpi']:.3f} |")
        print()

    if args.single:
        rows: dict[str, list[dict]] = {}
        krows: dict[str, list[dict]] = {}
        for path in sorted(args.single.glob("*_q*.json")):
            report = json.loads(path.read_text())
            name = path.stem  # fbase_q113
            arm, qp = name.rsplit("_q", 1)
            for cand in report["candidates"]:
                for point in cand.get("points", []):
                    krows.setdefault(arm, []).append({
                        "qp": int(qp),
                        "kbps": point["kbps"],
                        "kpi": point["kpi"],
                    })
                for seg in cand.get("segments", []):
                    if seg["segment"] not in SEGMENTS:
                        continue
                    if seg["replacement"] != "source":
                        continue
                    rows.setdefault((arm, seg["segment"]), []).append({
                        "qp": int(qp),
                        "kbps": report["candidates"][0]["points"][0]["kbps"],
                        "kpi": seg["kpi"],
                    })
        print("## source-replacement ceiling curve (single-point oracles)")
        print()
        print("| arm | segment | QP | kbps | ceiling K |")
        print("|---|---|---:|---:|---:|")
        for (arm, seg), pts in sorted(rows.items()):
            for p in sorted(pts, key=lambda p: p["qp"]):
                print(f"| {arm} | {seg} | {p['qp']} | {p['kbps']:.3f} | "
                      f"{p['kpi']:.3f} |")
        print()
        print("## ceiling fits at target")
        print()
        print("| arm | segment | points | fit K | slope | R2 |")
        print("|---|---|---:|---:|---:|---:|")
        for (arm, seg), pts in sorted(rows.items()):
            xs = [p["kbps"] for p in pts]
            ys = [p["kpi"] for p in pts]
            if len(xs) < 2:
                print(f"| {arm} | {seg} | {len(xs)} | {ys[0]:.3f} | - | - |")
                continue
            intercept, slope, r2 = ols(xs, ys)
            print(f"| {arm} | {seg} | {len(xs)} | "
                  f"{intercept + slope * args.target:.3f} | {slope:+.4f} | "
                  f"{r2:.3f} |")
        print()
        print("## K fit from the single-point oracles")
        print()
        print("| arm | points | fit K | slope | R2 |")
        print("|---|---:|---:|---:|---:|")
        for arm, pts in sorted(krows.items()):
            xs = [p["kbps"] for p in pts]
            ys = [p["kpi"] for p in pts]
            if len(xs) < 2:
                continue
            intercept, slope, r2 = ols(xs, ys)
            print(f"| {arm} | {len(xs)} | "
                  f"{intercept + slope * args.target:.3f} | {slope:+.4f} | "
                  f"{r2:.3f} |")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
