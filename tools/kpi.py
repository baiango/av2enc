#!/usr/bin/env python3
"""Composite KPI K over CVVDP, Butteraugli and SSIMULACRA2.

`K` maps each metric onto a common 0-100 quality axis with fixed, published
anchors (not min/max of the measured set, so scores are comparable across runs),
then takes a weighted mean:

    q_cvvdp = clamp(JOD / 10, 0, 1)           # JOD 10 is the top of the scale
    q_ba    = 1 / (1 + BA / 3)                # BA 3 is "clearly visible"
    q_ss2   = clamp((SS2 + 50) / 150, 0, 1)   # SS2 100 = perfect, -50 = floor

    K = 100 * (0.50*q_cvvdp + 0.25*q_ba + 0.25*q_ss2)

CVVDP carries half the weight because it is the project's tuned objective and its
display model (`standard_fhd`) matches the deployment case; the other two exist
to stop CVVDP's saturation near 10 JOD from hiding real quality differences, so
they are weighted equally at a quarter each.

Usage:
  tools/kpi.py --cvvdp 9.713 --butteraugli 4.14 --ssimulacra2 79.08
  tools/kpi.py --quality-json q.json --extra-json e.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def components(cvvdp: float, butteraugli: float, ssimulacra2: float):
    q_cvvdp = clamp(cvvdp / 10.0)
    q_ba = 1.0 / (1.0 + max(butteraugli, 0.0) / 3.0)
    q_ss2 = clamp((ssimulacra2 + 50.0) / 150.0)
    return q_cvvdp, q_ba, q_ss2


def composite(cvvdp: float, butteraugli: float, ssimulacra2: float) -> float:
    q_cvvdp, q_ba, q_ss2 = components(cvvdp, butteraugli, ssimulacra2)
    return 100.0 * (0.50 * q_cvvdp + 0.25 * q_ba + 0.25 * q_ss2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cvvdp", type=float)
    ap.add_argument("--butteraugli", type=float)
    ap.add_argument("--ssimulacra2", type=float)
    ap.add_argument("--quality-json", type=Path,
                    help="tools/quality.py --json output (CVVDP)")
    ap.add_argument("--extra-json", type=Path,
                    help="tools/extra_metrics.py --json output")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    c, b, s = args.cvvdp, args.butteraugli, args.ssimulacra2
    if args.quality_json:
        c = json.loads(args.quality_json.read_text())["cvvdp_jod"]
    if args.extra_json:
        d = json.loads(args.extra_json.read_text())
        b, s = d["butteraugli_mean"], d["ssimulacra2_mean"]
    if None in (c, b, s):
        ap.error("need --cvvdp/--butteraugli/--ssimulacra2 or the two json files")

    q_cvvdp, q_ba, q_ss2 = components(c, b, s)
    k = composite(c, b, s)
    print(f"cvvdp={c:.4f} butteraugli={b:.4f} ssimulacra2={s:.3f} "
          f"K={k:.2f} (q={q_cvvdp:.3f}/{q_ba:.3f}/{q_ss2:.3f})")
    if args.json:
        args.json.write_text(json.dumps({
            "cvvdp": c, "butteraugli": b, "ssimulacra2": s,
            "q_cvvdp": q_cvvdp, "q_butteraugli": q_ba, "q_ssimulacra2": q_ss2,
            "kpi": k,
        }, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
