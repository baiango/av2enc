#!/usr/bin/env python3
"""Rank every screened arm by its K at matched bytes.

Reads one or more `screen.jsonl` files written by `tools/hk_screen.py`, and for
each label finds the QP pair that brackets the target in BYTES, interpolating
every metric (never in QP order - the harness trap in
docs/rd-key-partition-floor.md §6.1).  Prints the matched-byte K so arms with
different rate spend can be compared at all, which is the only comparison that
means anything on a picture where K moves ~1.0 per QP.

    tools/hk_rank.py --target 25010 DIR [DIR ...]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def kpi(jod: float, ba: float, ss2: float) -> float:
    return 100.0 * (0.50 * max(0.0, min(1.0, jod / 10.0))
                    + 0.25 / (1.0 + max(ba, 0.0) / 3.0)
                    + 0.25 * max(0.0, min(1.0, (ss2 + 50.0) / 150.0)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, required=True)
    ap.add_argument("dirs", nargs="+", type=Path)
    args = ap.parse_args()

    # Key on the arm's stream identity (label + tag), never the label alone: a
    # label reused with a different env in another screen is a different arm.
    pts: dict[tuple[str, str], dict[int, dict]] = defaultdict(dict)
    meta: dict[tuple[str, str], dict] = {}
    for d in args.dirs:
        for line in (d / "screen.jsonl").read_text().splitlines():
            r = json.loads(line)
            key = (r["label"], r.get("tag", ""))
            pts[key][r["qp"]] = r
            meta[key] = r

    rows = []
    for key, by_qp in pts.items():
        label = key[0]
        curve = sorted(((r["bytes"], r) for r in by_qp.values()))
        lo = [c for c in curve if c[0] <= args.target]
        hi = [c for c in curve if c[0] >= args.target]
        if lo and hi:
            a = max(lo, key=lambda c: c[0])[1]
            b = min(hi, key=lambda c: c[0])[1]
            t = 0.0 if a["bytes"] == b["bytes"] else \
                (args.target - a["bytes"]) / (b["bytes"] - a["bytes"])
            m = {k: a[k] + t * (b[k] - a[k])
                 for k in ("jod", "ba", "ss2", "psnr_y")}
            rows.append({"label": label, "K": kpi(m["jod"], m["ba"], m["ss2"]),
                         "qp": (a["qp"], b["qp"]),
                         "bytes": (a["bytes"], b["bytes"]), "m": m,
                         "bracketed": True})
        else:
            near = min(curve, key=lambda c: abs(c[0] - args.target))[1]
            rows.append({"label": label, "K": near["K"], "qp": (near["qp"],),
                         "bytes": (near["bytes"], near["bytes"]),
                         "m": {"jod": near["jod"], "ba": near["ba"],
                               "ss2": near["ss2"], "psnr_y": near["psnr_y"]},
                         "bracketed": False})
    best = max((r["K"] for r in rows if r["bracketed"]), default=0.0)
    print(f"target {args.target} B   (dK vs best bracketed arm)")
    for r in sorted(rows, key=lambda r: -r["K"]):
        flag = "" if r["bracketed"] else "  NOT BRACKETED"
        print(f"  {r['label']:14s} K {r['K']:6.2f}  ({r['K'] - best:+5.2f})  "
              f"qp {str(r['qp']):12s} {r['bytes'][0]:6d}..{r['bytes'][1]:6d} B  "
              f"JOD {r['m']['jod']:.3f} BA {r['m']['ba']:5.2f} "
              f"SS2 {r['m']['ss2']:6.1f}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
