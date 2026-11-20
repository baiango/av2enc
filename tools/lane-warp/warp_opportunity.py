#!/usr/bin/env python3
"""Local-warp opportunity: affine vs translational prediction, per 16x16 block.

The core project's first question is whether a per-block affine predictor has
anything to buy on this clip.  The decoder's local warp is constrained (neighbour
MVs, 4x4 grid, warp filters), so this optical-flow affine is an optimistic upper
bound: a small gain closes the idea before any code is written.

  tools/lane-warp/warp_opportunity.py --ref REF --test decoded.y4m
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "tools" / "lane-refresh") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools" / "lane-refresh"))

from ba_proxy_corr import read_y4m  # noqa: E402


def block_sse(a: np.ndarray, b: np.ndarray) -> float:
    d = a.astype(np.float64) - b.astype(np.float64)
    return float(np.mean(d * d))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, type=Path)
    ap.add_argument("--test", required=True, type=Path)
    ap.add_argument("--frames", default="10,30,50")
    ap.add_argument("--block", type=int, default=16)
    args = ap.parse_args()

    _, _, src = read_y4m(args.ref)
    _, _, dec = read_y4m(args.test)
    n = min(len(src), len(dec))
    frames = [int(x) for x in args.frames.split(",") if int(x) < n]
    bs = args.block
    h, w = src[0].shape

    tr_tot = 0.0
    af_tot = 0.0
    blocks = 0
    better = 0
    gain_sum = 0.0
    for fi in frames:
        s = src[fi].astype(np.float32)
        sp = src[fi - 1].astype(np.float32)
        dp = dec[fi - 1].astype(np.float32)
        flow = cv2.calcOpticalFlowFarneback(
            sp, s, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        for y in range(0, h - bs + 1, bs):
            for x in range(0, w - bs + 1, bs):
                yy, xx = np.mgrid[y:y + bs, x:x + bs]
                pts = np.stack([xx.ravel(), yy.ravel()], 1).astype(np.float32)
                dst = pts + flow[y:y + bs, x:x + bs].reshape(-1, 2)
                m, _ = cv2.estimateAffine2D(pts, dst, method=cv2.RANSAC,
                                            ransacReprojThreshold=2.0)
                sse_tr = None
                if m is not None:
                    inv = cv2.invertAffineTransform(m)
                    # m maps global prev -> global current; warpAffine samples
                    # the block-local reference, so rebase the map to the
                    # block origin.
                    origin = np.array([x, y], dtype=np.float64)
                    inv_local = inv.copy()
                    inv_local[:, 2] += inv[:, :2] @ origin - origin
                    pred = cv2.warpAffine(
                        dp[y:y + bs, x:x + bs], inv_local, (bs, bs),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE)
                    sse_af = block_sse(s[y:y + bs, x:x + bs], pred)
                else:
                    sse_af = None
                # Translation baseline: rounded median flow in the block.
                med = np.median(flow[y:y + bs, x:x + bs].reshape(-1, 2), 0)
                dx, dy = int(round(med[0])), int(round(med[1]))
                ys = np.clip(np.arange(y, y + bs) + dy, 0, h - 1)
                xs = np.clip(np.arange(x, x + bs) + dx, 0, w - 1)
                pred_tr = dp[np.ix_(ys, xs)]
                sse_tr = block_sse(s[y:y + bs, x:x + bs], pred_tr)
                if sse_af is None:
                    continue
                tr_tot += sse_tr
                af_tot += sse_af
                blocks += 1
                if sse_af < sse_tr * 0.95:
                    better += 1
                    gain_sum += (sse_tr - sse_af) / max(sse_tr, 1e-9)

    red = (tr_tot - af_tot) / max(tr_tot, 1e-9) * 100.0
    print(f"frames={frames} blocks={blocks}")
    print(f"translation SSE={tr_tot:.1f}  affine SSE={af_tot:.1f}  "
          f"reduction={red:+.2f}%")
    print(f"blocks with >5% affine gain: {better}/{blocks} "
          f"({100.0 * better / max(blocks, 1):.1f}%)  "
          f"mean gain among them={100.0 * gain_sum / max(better, 1):.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
