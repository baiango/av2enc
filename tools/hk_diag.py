#!/usr/bin/env python3
"""Where the rendered error of one stream actually sits: plane, frequency band
and block-activity bucket.  This directs a residual-accuracy change: it answers
whether the deficit is luma or chroma, coarse or fine structure.

    tools/hk_diag.py --ref r.y4m --test t.y4m [--name label]

Bands are nested box differences (1 px, 1-2 px, 2-4 px, 4-8 px) on the luma
plane, the same decomposition docs/band-d.md used for the SSIMULACRA2 charge.
"""
from __future__ import annotations

import argparse
import numpy as np


def planes(path: str):
    d = open(path, "rb").read()
    if d.startswith(b"YUV4MPEG2"):
        d = d[d.index(b"\n") + 1:]
        while d.startswith(b"FRAME\n"):
            d = d[6:]
    w, h = 1280, 720
    y = np.frombuffer(d[: w * h], np.uint8).reshape(h, w).astype(np.float64)
    i = w * h
    cw, ch = w // 2, h // 2
    u = np.frombuffer(d[i:i + cw * ch], np.uint8).reshape(ch, cw).astype(np.float64)
    i += cw * ch
    v = np.frombuffer(d[i:i + cw * ch], np.uint8).reshape(ch, cw).astype(np.float64)
    return y, u, v


def box(x: np.ndarray, n: int) -> np.ndarray:
    k = np.ones(n) / n
    t = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 1, x)
    return np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 0, t)


def bands(lum: np.ndarray) -> dict[str, float]:
    """Error energy per nested band, from 1 px (finest) upward."""
    lo1 = box(lum, 3)
    lo2 = box(lo1, 5)
    lo4 = box(lo2, 9)
    lo8 = box(lo4, 17)
    # differences isolate each scale: e^2 of (x - blurred) carries that band.
    out = {}
    out["1px"] = float(((lum - lo1) ** 2).sum())
    out["1-2px"] = float(((lo1 - lo2) ** 2).sum())
    out["2-4px"] = float(((lo2 - lo4) ** 2).sum())
    out["4-8px"] = float(((lo4 - lo8) ** 2).sum())
    out["8px+"] = float(((lo8 - box(lo8, 33)) ** 2).sum())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--name", default="stream")
    a = ap.parse_args()

    rY, rU, rV = planes(a.ref)
    tY, tU, tV = planes(a.test)

    eY, eU, eV = rY - tY, rU - tU, rV - tV
    # Each chroma sample covers 4 luma samples at 4:2:0, so its squared error
    # counts 4x when comparing per-picture energy.
    ey = float((eY ** 2).sum())
    eu = float((eU ** 2).sum()) * 4
    ev = float((eV ** 2).sum()) * 4
    tot = ey + eu + ev
    print(f"=== {a.name}")
    print(f"  luma   rmse {np.sqrt((eY ** 2).mean()):6.2f}  share {100 * ey / tot:5.1f}%")
    print(f"  chroma U rmse {np.sqrt((eU ** 2).mean()):5.2f}  share {100 * eu / tot:5.1f}%")
    print(f"  chroma V rmse {np.sqrt((eV ** 2).mean()):5.2f}  share {100 * ev / tot:5.1f}%")

    # luma error by frequency band
    eb = bands(eY)
    sb = bands(rY)
    tt = sum(eb.values())
    print("  luma error energy by band:")
    for k in ("1px", "1-2px", "2-4px", "4-8px", "8px+"):
        if tt > 0:
            print(f"    {k:6s} {100 * eb[k] / tt:5.1f}% of error   "
                  f"(source carries {100 * sb[k] / sum(sb.values()):5.1f}%)")

    # block buckets by source activity
    src = rY
    act = box(src, 3)
    hp = (src - act) ** 2
    s16 = hp[:720, :1280].reshape(720 // 16, 16, 1280 // 16, 16).mean((1, 3))
    e16 = (eY ** 2).reshape(720 // 16, 16, 1280 // 16, 16).mean((1, 3))
    thr_hi, thr_lo = np.percentile(s16, 66), np.percentile(s16, 33)
    m_hi, m_lo = s16 >= thr_hi, s16 <= thr_lo
    print(f"  by source activity (16px blocks):"
          f"  textured {np.sqrt(e16[m_hi].mean()):5.2f}"
          f"  mid {np.sqrt(e16[~m_hi & ~m_lo].mean()):5.2f}"
          f"  flat {np.sqrt(e16[m_lo].mean()):5.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())