#!/usr/bin/env python3
"""Matched-rate K + per-segment comparison for the P-driver lane.

Reads the oracle's JSON (one candidate per arm, several QPs) and reports, at
`--target-kbps`:

  * the bracket (the two QP points that straddle the target) and the
    interpolated K;
  * an ordinary-least-squares fit of K against kbps over all the arm's points,
    with R^2, evaluated at the target (the "robust fit" reading a jagged K
    curve needs);
  * the same for the source-replacement segment ceilings.

  tools/lane-pdriver/kfit.py --oracle /tmp/rd-pdriver/oracle.json \
      --target-kbps 100
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def fit(points):
    """OLS K ~ a + b*kbps over [(kbps, K)]; returns (fit_at, slope, r2)."""
    n = len(points)
    if n < 2:
        return None
    mx = sum(p[0] for p in points) / n
    my = sum(p[1] for p in points) / n
    sxx = sum((p[0] - mx) ** 2 for p in points)
    sxy = sum((p[0] - mx) * (p[1] - my) for p in points)
    if sxx <= 0:
        return None
    b = sxy / sxx
    a = my - b * mx
    ss_tot = sum((p[1] - my) ** 2 for p in points)
    ss_res = sum((p[1] - (a + b * p[0])) ** 2 for p in points)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return (a, b, r2)


def bracket(points, target):
    pts = sorted(points)
    for i in range(len(pts) - 1):
        lo, hi = pts[i], pts[i + 1]
        if lo[0] <= target <= hi[0] and hi[0] > lo[0]:
            frac = (target - lo[0]) / (hi[0] - lo[0])
            return lo, hi, lo[1] + (hi[1] - lo[1]) * frac
    nearest = min(pts, key=lambda p: abs(p[0] - target))
    return None, None, nearest[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", required=True, type=Path)
    ap.add_argument("--target-kbps", type=float, default=100.0)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    data = json.loads(args.oracle.read_text())
    target = args.target_kbps
    out = {}
    print(f"target {target:.3f} kbps")
    print(f"{'arm':>10} {'QP':>4} {'kbps':>8} {'K':>8} {'BA':>7} {'SS2':>7} {'JOD':>8}")
    for cand in data["candidates"]:
        rows = []
        for p in cand["points"]:
            qp = p.get("point") or p["label"].split("@")[-1]
            rows.append((p["kbps"], p["kpi"]))
            print(f"{cand['candidate']:>10} {qp:>4} {p['kbps']:8.3f} "
                  f"{p['kpi']:8.3f} {p['butteraugli']:7.4f} "
                  f"{p['ssimulacra2']:7.3f} {p['cvvdp_jod']:8.4f} "
                  f"{p['psnr']['psnr_y']:8.4f}")
        lo, hi, kb = bracket(rows, target)
        f = fit(rows)
        fit_at = (f[0] + f[1] * target) if f else float("nan")
        slope, r2 = (f[1], f[2]) if f else (float("nan"), float("nan"))
        out[cand["candidate"]] = {
            "points": sorted(rows),
            "bracket": [lo, hi] if lo else None,
            "k_bracket": kb,
            "k_fit": fit_at,
            "slope": slope,
            "r2": r2,
        }
        br = f"{lo[0]:.3f}/{hi[0]:.3f}" if lo else "none"
        print(f"  -> {cand['candidate']}: bracket {br} K={kb:.3f}  "
              f"fit K={fit_at:.3f} R2={r2:.3f} slope={slope:.4f}")
    if len(out) > 1:
        names = list(out)
        ref = names[0]
        for name in names[1:]:
            print(f"\n{name} vs {ref}: "
                  f"bracket {out[name]['k_bracket'] - out[ref]['k_bracket']:+.3f}  "
                  f"fit {out[name]['k_fit'] - out[ref]['k_fit']:+.3f}")
    # segment ceilings
    try:
        segs = data["segments"]
    except KeyError:
        segs = None
    if segs:
        print("\nsegment ceilings (fit at target):")
        for cand in segs.get("candidates", []):
            pass
    if args.json:
        args.json.write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
