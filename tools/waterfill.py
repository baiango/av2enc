#!/usr/bin/env python3
"""Water-filling allocation over measured per-frame R-D points.

Given the same clip encoded at several fixed QPs, each frame i has points
(bits_i(q), sse_i(q)).  We minimise total SSE subject to a total-bit budget by
greedy marginal allocation over each frame's upper convex hull -- the
constructive form of the water-filling theorem proved in
proofs/rdproof/RdAlloc.lean (greedy2_optimal).

Reports the constant-QP plan against the optimal plan at the same total bits.
"""
import re
import sys
from fractions import Fraction

import numpy as np


def read_y4m(path):
    with open(path, "rb") as f:
        header = f.readline().decode().strip()
        w = int(re.search(r"W(\d+)", header).group(1))
        h = int(re.search(r"H(\d+)", header).group(1))
        frames = []
        ysize = w * h
        fsize = ysize * 3 // 2
        while True:
            line = f.readline()
            if not line:
                break
            if not line.startswith(b"FRAME"):
                break
            buf = f.read(fsize)
            if len(buf) < fsize:
                break
            frames.append(np.frombuffer(buf[:ysize], dtype=np.uint8))
    return w, h, frames


def frame_bits(log_path):
    with open(log_path) as f:
        for line in f:
            if "frame bits:" in line:
                return [int(x) for x in line.split(":", 1)[1].split()]
    raise SystemExit(f"no 'frame bits' line in {log_path}")


def hull(points):
    """Upper convex hull of (bits, sse) with sse decreasing: returns the
    upgrade segments (dbits, dsse) with sse reduction per extra bit sorted by
    slope, using rationals so the comparison is exact."""
    # points sorted by bits ascending; keep only decreasing sse
    pts = sorted(set(points))
    pruned = []
    for b, s in pts:
        if pruned and s >= pruned[-1][1]:
            continue  # no gain
        pruned.append((b, s))
    # convex hull in (bits, sse): maintain decreasing |slope|
    hs = []
    for p in pruned:
        while len(hs) >= 2:
            (b0, s0), (b1, s1) = hs[-2], hs[-1]
            slope_prev = Fraction(s1 - s0, b1 - b0)
            slope_new = Fraction(p[1] - s0, p[0] - b0)
            if abs(slope_new) >= abs(slope_prev):
                hs.pop()
            else:
                break
        hs.append(p)
    segs = []
    for k in range(len(hs) - 1):
        b0, s0 = hs[k]
        b1, s1 = hs[k + 1]
        segs.append((b1 - b0, s0 - s1))  # (extra bits, sse saved)
    return segs


def main():
    ref_path = sys.argv[1]
    specs = sys.argv[2:]  # each: qp:ivf_log:decoded_y4m
    _, _, ref = read_y4m(ref_path)
    qps, bits, sses = [], [], []
    for spec in specs:
        qp, log, dec = spec.split(":")
        qps.append(int(qp))
        bits.append(frame_bits(log))
        _, _, decf = read_y4m(dec)
        sses.append([int(np.sum((ref[i].astype(np.int64) - decf[i].astype(np.int64)) ** 2))
                     for i in range(len(ref))])
    qps.sort()
    order = sorted(range(len(qps)), key=lambda k: qps[k])
    qps = [qps[k] for k in order]
    bits = [bits[k] for k in order]
    sses = [sses[k] for k in order]
    n = len(ref)
    print(f"frames={n}  QPs={qps}")
    total_px_all = n * 1440 * 720
    for k, qp in enumerate(qps):
        p = 10 * np.log10(255.0**2 / (sum(sses[k]) / total_px_all))
        print(f"  QP{qp}: total {sum(bits[k])/8:.0f} B  SSE {sum(sses[k])/1e9:.4f}e9  "
              f"PSNR {p:.3f} dB")

    # per-frame points
    pts = [[(bits[k][i], sses[k][i]) for k in range(len(qps))] for i in range(n)]

    # constant-QP reference: the middle QP
    mid = len(qps) // 2
    ref_bits = sum(bits[mid])
    ref_sse = sum(sses[mid])
    print(f"\nconstant QP{qps[mid]}: {ref_bits/8:.0f} B, SSE {ref_sse/1e9:.4f}e9")

    # greedy water-filling from the coarsest QP
    cur_qp = [len(qps) - 1] * n          # index of chosen point per frame
    cur_bits = sum(pts[i][cur_qp[i]][0] for i in range(n))
    cur_sse = sum(pts[i][cur_qp[i]][1] for i in range(n))
    total_px = n * 1440 * 720

    segs = [hull(pts[i]) for i in range(n)]
    cursor = [0] * n
    while True:
        best = None
        for i in range(n):
            if cursor[i] >= len(segs[i]):
                continue
            db, ds = segs[i][cursor[i]]
            if db <= 0 or ds <= 0:
                continue
            if cur_bits + db > ref_bits:
                continue
            score = Fraction(ds, db)
            if best is None or score > best[0]:
                best = (score, i, db, ds)
        if best is None:
            break
        _, i, db, ds = best
        cur_bits += db
        cur_sse -= ds
        cursor[i] += 1

    print(f"water-filled  : {cur_bits/8:.0f} B, SSE {cur_sse/1e9:.4f}e9")
    dsse = ref_sse - cur_sse
    print(f"\nSSE reduction at matched bits: {dsse/1e9:.4f}e9 "
          f"({100*dsse/ref_sse:+.2f}%)")
    psnr_ref = 10 * np.log10(255.0**2 / (ref_sse / total_px))
    psnr_opt = 10 * np.log10(255.0**2 / (cur_sse / total_px))
    print(f"PSNR: constant {psnr_ref:.3f} dB -> water-filled {psnr_opt:.3f} dB "
          f"({psnr_opt - psnr_ref:+.3f} dB)")
    moved = sum(1 for i in range(n) if cursor[i] > 0)
    print(f"frames upgraded: {moved}/{n}")


if __name__ == "__main__":
    main()
