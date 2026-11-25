#!/usr/bin/env python3
"""Apply one fixed intra predictor set to two reconstructions and report the
pred / recon / keeps decomposition (the 3.6ae/3.6af quantities) on a common
yardstick.

The point of a SINGLE yardstick is that the optimism factor -- how much better
a signalled encoder search could do than this probe -- is the same on both
sides, so the RATIO between them is meaningful even though the absolute values
are optimistic.

Predictor set: the 13 AV2 base modes (DC, V, H, D45, D135, D113, D157, D203,
D67, SMOOTH, SMOOTH_V, SMOOTH_H, PAETH), each plus the +/-3 x 3-degree angle
deltas on the 8 directional modes. SSE-best per 8x8 block, no signalling cost.

Usage: predprobe2.py <recon.y4m> <recon.y4m>
"""
import sys
import numpy as np

W, H = 1280, 720
B = 8


def read_y(p):
    d = open(p, 'rb').read()
    i = d.index(b'\n') + 1
    if d[i:i + 6] == b'FRAME\n':
        i += 6
    return np.frombuffer(d[i:i + W * H], np.uint8).reshape(H, W).astype(np.float64)


# --- reference sample access (with edge clamping, as a decoder does) ---------
def above(rec, x0, y0, w):
    if y0 == 0:
        return None
    row = rec[y0 - 1, max(x0 - 1, 0):min(x0 + w + 1, W)]
    return row


def left(rec, x0, y0, h):
    if x0 == 0:
        return None
    col = rec[max(y0 - 1, 0):min(y0 + h, H), x0 - 1]
    return col


def dc_pred(rec, x0, y0, w, h):
    a = above(rec, x0, y0, w)
    l = left(rec, x0, y0, h)
    vals = []
    if a is not None:
        vals.append(a.mean())
    if l is not None:
        vals.append(l.mean())
    return np.full((h, w), float(np.mean(vals)) if vals else 128.0)


def v_pred(rec, x0, y0, w, h):
    if y0 == 0:
        return dc_pred(rec, x0, y0, w, h)
    return np.tile(rec[y0 - 1, x0:x0 + w], (h, 1))


def h_pred(rec, x0, y0, w, h):
    if x0 == 0:
        return dc_pred(rec, x0, y0, w, h)
    return np.tile(rec[y0:y0 + h, x0 - 1][:, None], (1, w))


def smooth_pred(rec, x0, y0, w, h):
    a = above(rec, x0, y0, w)
    l = left(rec, x0, y0, h)
    if a is None or l is None:
        return dc_pred(rec, x0, y0, w, h)
    # AV1/AV2 SMOOTH weights
    ww = np.array([32, 32, 32, 32, 32, 32, 32, 32,
                   28, 24, 20, 16, 12, 8, 4, 0], float)
    ws = ww[:w] if w <= 16 else np.interp(np.arange(w), np.linspace(0, 15, 16), ww)
    out = np.empty((h, w))
    for y in range(h):
        for x in range(w):
            wy = ws[y] if y < len(ws) else ws[-1]
            wx = ws[x] if x < len(ws) else ws[-1]
            tot = wy + wx
            if tot == 0:
                out[y, x] = 128.0
            else:
                out[y, x] = (wy * a[x] + wx * l[y]) / tot
    return out


def smooth_v(rec, x0, y0, w, h):
    a = above(rec, x0, y0, w)
    if a is None:
        return dc_pred(rec, x0, y0, w, h)
    ww = np.array([32, 28, 24, 20, 16, 12, 8, 4], float)[:h]
    out = np.empty((h, w))
    for y in range(h):
        out[y, :] = (ww[y] * a[:w] + (32 - ww[y]) * (rec[y0 + y, x0] if w > 0 else 128)) / 32.0
    return out


def smooth_h(rec, x0, y0, w, h):
    l = left(rec, x0, y0, h)
    if l is None:
        return dc_pred(rec, x0, y0, w, h)
    ww = np.array([32, 28, 24, 20, 16, 12, 8, 4], float)[:w]
    out = np.empty((h, w))
    for x in range(w):
        out[:, x] = (ww[x] * l[:h] + (32 - ww[x]) * (rec[y0, x0 + x] if h > 0 else 128)) / 32.0
    return out


def paeth_pred(rec, x0, y0, w, h):
    a = above(rec, x0, y0, w)
    l = left(rec, x0, y0, h)
    if a is None or l is None:
        return dc_pred(rec, x0, y0, w, h)
    tl = rec[y0 - 1, x0 - 1] if (x0 > 0 and y0 > 0) else 128.0
    out = np.empty((h, w))
    for y in range(h):
        for x in range(w):
            p = a[x] + l[y] - tl
            pa, pb, pc = abs(p - a[x]), abs(p - l[y]), abs(p - tl)
            out[y, x] = a[x] if (pa <= pb and pa <= pc) else (l[y] if pb <= pc else tl)
    return out


# --- directional: nearest-sample along the ray, angle in degrees ------------
def angle_pred(rec, x0, y0, w, h, deg):
    a = above(rec, x0, y0, w + h + 2)
    l = left(rec, x0, y0, h + w + 2)
    if a is None and l is None:
        return np.full((h, w), 128.0)
    if a is None:
        return h_pred(rec, x0, y0, w, h)
    if l is None:
        return v_pred(rec, x0, y0, w, h)
    # extend above to the right, left downward (repeat last)
    ax = np.concatenate([a, np.full(w + h + 2, a[-1])])[:w + h + 2]
    ly = np.concatenate([np.full(w + h + 2, l[0]), l])[:w + h + 2]
    rad = np.deg2rad(deg)
    dx, dy = np.cos(rad), -np.sin(rad)
    out = np.empty((h, w))
    for y in range(h):
        for x in range(w):
            # march back toward the reference edge
            t = 0.0
            px, py = float(x + 1.0), float(y + 1.0)
            # step until we leave the block upward or leftward
            if abs(dy) > abs(dx):
                t = (y + 1.0) / max(abs(dy), 1e-6)
                sx = (x + 1.0) - t * dx
                sy = 0.0
            else:
                t = (x + 1.0) / max(abs(dx), 1e-6)
                sy = (y + 1.0) - t * dy
                sx = 0.0
            if sy <= 0.5:
                k = int(round(sx - 1))
                k = min(max(k, 0), len(ax) - 1)
                out[y, x] = ax[k]
            else:
                k = int(round(sy - 1))
                k = min(max(k, 0), len(ly) - 1)
                out[y, x] = ly[k]
    return out


BASE_MODES = [
    ("DC", lambda r, x, y, w, h: dc_pred(r, x, y, w, h)),
    ("V", lambda r, x, y, w, h: v_pred(r, x, y, w, h)),
    ("H", lambda r, x, y, w, h: h_pred(r, x, y, w, h)),
    ("D45", lambda r, x, y, w, h: angle_pred(r, x, y, w, h, 45)),
    ("D67", lambda r, x, y, w, h: angle_pred(r, x, y, w, h, 67)),
    ("D113", lambda r, x, y, w, h: angle_pred(r, x, y, w, h, 113)),
    ("D135", lambda r, x, y, w, h: angle_pred(r, x, y, w, h, 135)),
    ("D157", lambda r, x, y, w, h: angle_pred(r, x, y, w, h, 157)),
    ("D203", lambda r, x, y, w, h: angle_pred(r, x, y, w, h, 203)),
    ("SMOOTH", lambda r, x, y, w, h: smooth_pred(r, x, y, w, h)),
    ("PAETH", lambda r, x, y, w, h: paeth_pred(r, x, y, w, h)),
]

DIR_DEG = {"V": 90, "H": 180, "D45": 45, "D67": 67, "D113": 113,
           "D135": 135, "D157": 157, "D203": 203}


def preds():
    """All base modes plus +/-3..3 angle deltas on the 8 directional modes."""
    out = list(BASE_MODES)
    for name, deg in DIR_DEG.items():
        for d in (-3, -2, -1, 1, 2, 3):
            a = deg + 3 * d
            if a <= 0 or a >= 270:
                continue
            out.append((f"{name}{d:+d}",
                        (lambda r, x, y, w, h, aa=a: angle_pred(r, x, y, w, h, aa))))
    return out


def measure(src, rec):
    P = preds()
    tot = 0.0
    for y0 in range(0, H, B):
        for x0 in range(0, W, B):
            s = src[y0:y0 + B, x0:x0 + B]
            best = None
            for _, f in P:
                p = f(rec, x0, y0, B, B)
                e = float(((s - p) ** 2).sum())
                if best is None or e < best:
                    best = e
            tot += best
    return float(np.sqrt(tot / (W * H)))


def main():
    src = read_y('src720_1f.y4m')
    for path in sys.argv[1:]:
        rec = read_y(path)
        pred = measure(src, rec)
        recon = float(np.sqrt(((src - rec) ** 2).mean()))
        keeps = (recon / pred) ** 2
        print(f"  {path:38s} pred {pred:8.4f}  recon {recon:8.4f}  keeps {keeps*100:6.2f}%")


if __name__ == '__main__':
    main()
