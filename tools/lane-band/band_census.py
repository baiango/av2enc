#!/usr/bin/env python3
"""Week-1 band census: which error band does SS2 / BA charge for?

Decomposes each frame's luma error into the same nested box-difference bands
the AV2ENC_PIPE_BA mask uses (1 px, 2-4 px, 4-8 px, 8+ px), then asks whether
the band energies explain per-frame SSIMULACRA2 and Butteraugli *beyond the
source content* (the source's own band energies).  This is the kill gate for
the band-D project: if the error bands add no variance over content, stop.

  tools/lane-band/band_census.py \
      --ref testdata/nasa_480x240_30fps_60f.y4m \
      --test /tmp/base.dec.y4m --metrics /tmp/ba_corr.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent / "lane-refresh"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ba_proxy_corr import read_y4m  # noqa: E402


def bands(plane: np.ndarray) -> np.ndarray:
    """Four nested band-pass energies: 1, 2-4, 4-8, 8+ px."""
    p = plane.astype(np.float64)
    m1 = uniform_filter(p, 3, mode="nearest")
    m3 = uniform_filter(p, 7, mode="nearest")
    m7 = uniform_filter(p, 15, mode="nearest")
    return np.array([
        np.mean((p - m1) ** 2),
        np.mean((m1 - m3) ** 2),
        np.mean((m3 - m7) ** 2),
        np.mean(m7 ** 2),
    ])


def logs(x):
    return np.log(np.maximum(x, 1e-9))


def r2(y, X):
    y = np.asarray(y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    X = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"), beta


def partial_r(y, x, controls):
    """Correlation of y and x after regressing both on the controls."""
    def resid(v):
        A = np.column_stack([np.ones(len(v))] +
                            [np.asarray(c) for c in controls])
        beta, *_ = np.linalg.lstsq(A, v, rcond=None)
        return v - A @ beta
    ry = resid(np.asarray(y, dtype=np.float64))
    rx = resid(np.asarray(x, dtype=np.float64))
    if np.std(ry) == 0 or np.std(rx) == 0:
        return float("nan")
    return float(np.corrcoef(ry, rx)[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, type=Path)
    ap.add_argument("--test", required=True, type=Path)
    ap.add_argument("--metrics", type=Path,
                    help="ba_proxy_corr.py --json output (per-frame ba/ss2)")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    _, _, src = read_y4m(args.ref)
    _, _, rec = read_y4m(args.test)
    n = min(len(src), len(rec))

    metrics = json.loads(args.metrics.read_text())["rows"] if args.metrics else None
    if metrics is None:
        raise SystemExit("--metrics is required (run ba_proxy_corr.py first)")

    rows = []
    for i in range(n):
        e = src[i].astype(np.float64) - rec[i].astype(np.float64)
        eb = bands(e)
        sb = bands(src[i])
        rows.append({
            "frame": i,
            "ss2": metrics[i]["ss2"],
            "ba": metrics[i]["ba"],
            "err": eb,
            "src": sb,
        })

    segments = {
        "head 1-32": list(range(1, min(33, n))),
        "tail 33-59": list(range(33, min(60, n))),
        "all": list(range(n)),
    }
    report = {}
    for name, idx in segments.items():
        if len(idx) < 8:
            continue
        E = logs(np.array([rows[i]["err"] for i in idx]))
        A = logs(np.array([rows[i]["src"] for i in idx]))
        ss2 = np.array([rows[i]["ss2"] for i in idx])
        ba = np.array([rows[i]["ba"] for i in idx])
        out = {"n": len(idx), "bands": {}}
        for label, y in (("ss2", ss2), ("ba", ba)):
            r_a, _ = r2(y, A)
            r_ae, beta = r2(y, np.column_stack([A, E]))
            out[label] = {
                "r2_content": r_a,
                "r2_content_plus_error": r_ae,
                "delta_r2": r_ae - r_a,
                "betas_error": list(beta[1 + A.shape[1]:]),
            }
            for k in range(4):
                out["bands"].setdefault(f"band{k + 1}", {})[label] = {
                    "r_raw": float(np.corrcoef(ss2 if label == "ss2" else ba, E[:, k])[0, 1]),
                    "r_partial": partial_r(y, E[:, k], [A[:, j] for j in range(4)]),
                }
        report[name] = out

    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")

    for name, out in report.items():
        print(f"\n{name}  n={out['n']}")
        for label in ("ss2", "ba"):
            v = out[label]
            print(f"  {label:3s} R2 content={v['r2_content']:+.3f}  "
                  f"+error={v['r2_content_plus_error']:+.3f}  "
                  f"dR2={v['delta_r2']:+.3f}")
        for band, v in out["bands"].items():
            print(f"  {band}: ss2 r={v['ss2']['r_raw']:+.3f} "
                  f"partial={v['ss2']['r_partial']:+.3f} | "
                  f"ba r={v['ba']['r_raw']:+.3f} "
                  f"partial={v['ba']['r_partial']:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
