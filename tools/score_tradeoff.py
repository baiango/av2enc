#!/usr/bin/env python3
"""QBS: Quality-Bitrate-Speed trade-off score for an encoder.

Objectives:
    maximize CVVDP (JOD)
    minimize bitrate (kbps)
    maximize encode speed (realtime x)

A point A dominates B if A has >= CVVDP, <= bitrate, and >= speed, with at
least one strict improvement. The Pareto front is the set of non-dominated
operating points.

Budgeted score (the headline number):
    QBS = max CVVDP among points with bitrate <= R_max and speed >= S_min.

A separate budget can flip the question:
    min bitrate among points with CVVDP >= Q_min and speed >= S_min.
    max speed among points with CVVDP >= Q_min and bitrate <= R_max.

QBS is intentionally constrained. It does not let an encoder buy speed by
spending rate or sacrificing quality; those are hard budgets.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_speed import Point, cpus_for, load_points  # noqa: E402


def dominates(a: Point, b: Point) -> bool:
    return (
        a.c >= b.c
        and a.b <= b.b
        and a.s >= b.s
        and (a.c > b.c or a.b < b.b or a.s > b.s)
    )


def pareto(points: list[Point]) -> list[Point]:
    return [p for p in points if not any(dominates(q, p) for q in points if q is not p)]


def fmt(x: float | None, nd: int = 4) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def best(points: list[Point], key: str, reverse: bool) -> Point | None:
    if not points:
        return None
    return max(points, key=lambda p: getattr(p, key)) if reverse else min(
        points, key=lambda p: getattr(p, key)
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="+", type=Path)
    ap.add_argument("--reference", default="AVM")
    ap.add_argument("--q-min", type=float, default=9.0,
                    help="minimum CVVDP JOD (default 9.0)")
    ap.add_argument("--r-max", type=float, required=True,
                    help="maximum bitrate kbps")
    ap.add_argument("--s-min", type=float,
                    help="minimum encode speed, realtime x")
    ap.add_argument("--min-encode-fps", type=float,
                    help="alternative to --s-min: minimum encoded frames/second")
    ap.add_argument("--source-fps", type=float, default=60.0,
                    help="source frame rate for --min-encode-fps (default 60)")
    ap.add_argument("--allow-mixed-hardware", action="store_true")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    if args.s_min is None:
        if args.min_encode_fps is None:
            raise SystemExit("pass --s-min or --min-encode-fps")
        if args.source_fps <= 0:
            raise SystemExit("--source-fps must be positive")
        args.s_min = args.min_encode_fps / args.source_fps

    by_enc = load_points(args.reports, None)
    if not by_enc:
        raise SystemExit("no reports")
    encs = sorted(by_enc)

    all_cpus = sorted({p.cpu for pts in by_enc.values() for p in pts})
    if len(all_cpus) > 1 and not args.allow_mixed_hardware:
        print("error: reports come from different CPU models:", file=sys.stderr)
        for e in encs:
            print(f"  {e}: {', '.join(cpus_for(by_enc[e]))}", file=sys.stderr)
        print("pass --allow-mixed-hardware for an explicitly unfair comparison",
              file=sys.stderr)
        return 2

    print(f"## Pareto front (CVVDP up, bitrate down, speed up)")
    print("| encoder | QP | CVVDP | kbps | speed (x realtime) |")
    print("|---|---:|---:|---:|---:|")
    for e in encs:
        for p in sorted(pareto(by_enc[e]), key=lambda x: x.c, reverse=True):
            print(f"| {e} | {p.qp or '-'} | {p.c:.4f} | {p.b:.1f} | {p.s:.4f} |")

    print()
    print(f"## Budget: CVVDP >= {args.q_min:.3f} JOD, "
          f"bitrate <= {args.r_max:.1f} kbps, speed >= {args.s_min:.4f}x "
          f"(>= {args.s_min * args.source_fps:.2f} fps at {args.source_fps:g} fps source)")
    print("| encoder | feasible | QBS (max CVVDP) | min kbps at Q/S | max speed at Q/R |")
    print("|---|---:|---:|---:|---:|")

    out = {
        "budget": {
            "q_min": args.q_min,
            "r_max_kbps": args.r_max,
            "s_min_realtime_x": args.s_min,
            "min_encode_fps": args.min_encode_fps,
            "source_fps": args.source_fps,
        },
        "encoders": {},
    }
    for e in encs:
        pts = by_enc[e]
        feasible = [p for p in pts
                    if p.c >= args.q_min and p.b <= args.r_max and p.s >= args.s_min]
        qbs = best(feasible, "c", reverse=True)
        min_rate = best([p for p in pts
                         if p.c >= args.q_min and p.s >= args.s_min], "b", reverse=False)
        max_speed = best([p for p in pts
                          if p.c >= args.q_min and p.b <= args.r_max], "s", reverse=True)
        print(
            f"| {e} | {'yes' if qbs else 'no'} | "
            f"{fmt(qbs.c if qbs else None)} | "
            f"{fmt(min_rate.b if min_rate else None, 1)} | "
            f"{fmt(max_speed.s if max_speed else None)} |"
        )
        out["encoders"][e] = {
            "feasible": bool(qbs),
            "qbs_cvvdp": qbs.c if qbs else None,
            "qbs_kbps": qbs.b if qbs else None,
            "qbs_realtime_x": qbs.s if qbs else None,
            "min_kbps_at_q_s": min_rate.b if min_rate else None,
            "max_realtime_x_at_q_r": max_speed.s if max_speed else None,
            "pareto": [
                {"qp": p.qp, "cvvdp": p.c, "kbps": p.b, "realtime_x": p.s}
                for p in sorted(pareto(pts), key=lambda x: x.c, reverse=True)
            ],
            "cpu": cpus_for(pts),
        }

    if args.json:
        args.json.write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
