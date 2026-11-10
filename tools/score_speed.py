#!/usr/bin/env python3
"""Score encoding speed as a function of CVVDP quality.

For each encoder, operating points are (CVVDP JOD, encode speed). Speed is
expressed as a realtime factor:

    realtime_x = video_seconds / encode_seconds

We interpolate speed S(C) and bitrate B(C) over CVVDP C, then report:

    CSS(C)   = S(C) at a target CVVDP                  [realtime x]
    CSS-AUC  = mean of S(C) over the common CVVDP range [realtime x]
    speedup  = S_encoder(C) / S_reference(C)
    rate     = B_encoder(C) / B_reference(C)

CSS-AUC is the single summary score. Higher is better. Always read it with the
iso-quality bitrate: a fast encoder that spends much more rate is not better.

Inputs are JSON reports from tools/quality.py. Each report should contain:
  cvvdp_jod, bitrate_kbps, fps, frames, and either realtime_x or encode_seconds.
The encoder name comes from the "encoder" field, --encoder, or the parent dir.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Point:
    c: float          # CVVDP JOD
    s: float          # realtime factor
    b: float          # kbps
    qp: str = ""
    cpu: str = "unknown"
    host: dict | None = None


def load_points(paths: list[Path], fallback_encoder: str | None) -> dict[str, list[Point]]:
    by_enc: dict[str, list[Point]] = {}
    for p in paths:
        d = json.loads(p.read_text())
        if "cvvdp_jod" not in d or "bitrate_kbps" not in d:
            print(f"skip {p}: not a quality report", file=sys.stderr)
            continue
        enc = d.get("encoder") or fallback_encoder or p.parent.name
        c = float(d["cvvdp_jod"])
        b = float(d["bitrate_kbps"])
        host = d.get("host") or {}
        cpu = host.get("cpu_model") or d.get("cpu_model") or "unknown"

        if "realtime_x" in d:
            s = float(d["realtime_x"])
        elif "encode_seconds" in d and d["encode_seconds"] > 0:
            frames = int(d.get("frames", 0))
            fps = d.get("fps", "60/1")
            num, den = (int(x) for x in fps.split("/"))
            video_seconds = frames * den / num
            s = video_seconds / float(d["encode_seconds"])
        else:
            raise SystemExit(
                f"{p}: need realtime_x or encode_seconds "
                f"(add --encode-seconds when running tools/quality.py)"
            )

        by_enc.setdefault(enc, []).append(
            Point(c, s, b, str(d.get("qp", "")), cpu, host or None)
        )

    for enc, pts in by_enc.items():
        pts.sort(key=lambda x: x.c)
    return by_enc


def cpus_for(points: list[Point]) -> list[str]:
    return sorted({p.cpu for p in points})


def interp(points: list[Point], c: float, attr: str) -> float | None:
    if not points or c < points[0].c or c > points[-1].c:
        return None
    for a, b in zip(points, points[1:]):
        if a.c <= c <= b.c:
            if b.c == a.c:
                return getattr(a, attr)
            t = (c - a.c) / (b.c - a.c)
            return getattr(a, attr) + (getattr(b, attr) - getattr(a, attr)) * t
    return getattr(points[-1], attr)


def auc(points: list[Point], lo: float, hi: float) -> float | None:
    if hi <= lo:
        return None
    xs = [lo] + sorted(p.c for p in points if lo < p.c < hi) + [hi]
    total = 0.0
    for x0, x1 in zip(xs, xs[1:]):
        s0, s1 = interp(points, x0, "s"), interp(points, x1, "s")
        if s0 is None or s1 is None:
            return None
        total += (s0 + s1) * 0.5 * (x1 - x0)
    return total / (hi - lo)


def fmt(x: float | None, nd: int = 4) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="+", type=Path)
    ap.add_argument("--encoder", help="fallback encoder name")
    ap.add_argument("--reference", default="AVM")
    ap.add_argument(
        "--targets",
        default="8.0,8.5,9.0,9.5",
        help="comma-separated CVVDP targets (default 8.0,8.5,9.0,9.5)",
    )
    ap.add_argument("--range", nargs=2, type=float, metavar=("LO", "HI"),
                    help="CVVDP range for CSS-AUC (default: common range)")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--plot", type=Path, help="write a PNG plot")
    ap.add_argument("--allow-mixed-hardware", action="store_true",
                    help="allow comparing reports from different CPU models")
    args = ap.parse_args()

    by_enc = load_points(args.reports, args.encoder)
    if args.reference not in by_enc:
        raise SystemExit(f"reference encoder {args.reference!r} not in reports")

    encs = sorted(by_enc)
    hardware = {e: cpus_for(by_enc[e]) for e in encs}
    all_cpus = sorted({c for cpus in hardware.values() for c in cpus})
    if len(all_cpus) > 1 and not args.allow_mixed_hardware:
        print("error: reports come from different CPU models:", file=sys.stderr)
        for e in encs:
            print(f"  {e}: {', '.join(hardware[e])}", file=sys.stderr)
        print(
            "CSS is hardware-specific. Run the reference encoder on the same "
            "CPU, or pass --allow-mixed-hardware for an explicitly unfair "
            "comparison.",
            file=sys.stderr,
        )
        return 2
    lo = max(min(p.c for p in by_enc[e]) for e in encs)
    hi = min(max(p.c for p in by_enc[e]) for e in encs)
    if args.range:
        lo, hi = args.range
    if hi <= lo:
        raise SystemExit(f"no common CVVDP range (common: {lo:.4f}..{hi:.4f})")

    targets = [float(x) for x in args.targets.split(",") if x.strip()]
    targets = [t for t in targets if lo <= t <= hi]
    if not targets:
        targets = [lo, (lo + hi) / 2, hi]

    ref = by_enc[args.reference]
    rows = []
    for t in targets:
        rs = interp(ref, t, "s")
        rb = interp(ref, t, "b")
        row = {"cvvdp": t, "reference_speed": rs, "reference_kbps": rb, "encoders": {}}
        for e in encs:
            s = interp(by_enc[e], t, "s")
            b = interp(by_enc[e], t, "b")
            row["encoders"][e] = {
                "realtime_x": s,
                "kbps": b,
                "speedup_vs_ref": (s / rs) if (s is not None and rs) else None,
                "rate_ratio_vs_ref": (b / rb) if (b is not None and rb) else None,
            }
        rows.append(row)

    summary = {}
    for e in encs:
        a = auc(by_enc[e], lo, hi)
        summary[e] = {"css_auc_realtime_x": a}

    header = ["CVVDP", "encoder", "speed (x realtime)", "kbps", "speedup", "rate ratio"]
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---:"] * len(header)) + "|")
    for row in rows:
        for e in encs:
            v = row["encoders"][e]
            print(
                f"| {row['cvvdp']:.2f} | {e} | {fmt(v['realtime_x'])} | "
                f"{fmt(v['kbps'], 1)} | {fmt(v['speedup_vs_ref'], 3)} | "
                f"{fmt(v['rate_ratio_vs_ref'], 3)} |"
            )

    print()
    print(f"CSS-AUC over CVVDP [{lo:.3f}, {hi:.3f}] (speed in x realtime):")
    for e in encs:
        print(f"  {e:12s} {fmt(summary[e]['css_auc_realtime_x'])}")

    print()
    print("Hardware:")
    for e in encs:
        print(f"  {e}: {', '.join(hardware[e])}")

    out = {
        "reference": args.reference,
        "range": [lo, hi],
        "targets": rows,
        "summary": summary,
        "hardware": hardware,
        "mixed_hardware": len(all_cpus) > 1,
        "definition": (
            "S(C)=video_seconds/encode_seconds at iso-CVVDP; "
            "CSS-AUC=mean S(C) over the common CVVDP range"
        ),
    }
    if args.json:
        args.json.write_text(json.dumps(out, indent=2) + "\n")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
            for e in encs:
                pts = by_enc[e]
                ax1.plot([p.c for p in pts], [p.s for p in pts], "o-", label=e)
                ax2.plot([p.c for p in pts], [p.b for p in pts], "o-", label=e)
            ax1.set_ylabel("encode speed (x realtime)")
            ax1.set_yscale("log")
            ax1.grid(True, which="both", alpha=0.3)
            ax1.legend()
            ax2.set_xlabel("CVVDP (JOD)")
            ax2.set_ylabel("bitrate (kbps)")
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            fig.tight_layout()
            fig.savefig(args.plot, dpi=150)
            print(f"plot: {args.plot}")
        except Exception as exc:
            print(f"warning: could not plot: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
