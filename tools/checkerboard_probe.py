#!/usr/bin/env python3
"""Prototype: does suppressing the 1-px ("checkerboard") band cost K?

Two experiments, both on an already-decoded stream, so no encoder change:

  A. Global band suppression: remove a fraction `s` of the 1-px alternating
     basis from the whole luma plane, re-score.
  B. Exchange experiment: suppressed stream is SMALLER in the encoder; the
     bytes saved are not modelled here, so we additionally report how much
     K we would get if those bytes were spent on the ~25 KB budget.

The point of the measurement: if suppression is ~free in K, we can throttle
1-px detail in the quantiser and bank the saved bytes into coefficients.
"""
import numpy as np
import subprocess
import sys
import os

W, H = 1280, 720


def read_yuv(p):
    d = open(p, 'rb').read()
    i = d.index(b'\n') + 1
    if d[i:i + 6] == b'FRAME\n':
        i += 6
    y = np.frombuffer(d[i:i + W * H], np.uint8).reshape(H, W)
    j = i + W * H
    cw, ch = W // 2, H // 2
    u = np.frombuffer(d[j:j + cw * ch], np.uint8).reshape(ch, cw)
    v = np.frombuffer(d[j + cw * ch:j + 2 * cw * ch], np.uint8).reshape(ch, cw)
    return y, u, v


def write_yuv(p, y, u, v):
    with open(p, 'wb') as f:
        f.write(b'YUV4MPEG2 W1280 H720 F25:1 Ip A1:1 C420jpeg\nFRAME\n')
        f.write(y.tobytes())
        f.write(u.tobytes())
        f.write(v.tobytes())


def cb_basis(b=8):
    yy, xx = np.mgrid[0:b, 0:b]
    return (((yy + xx) % 2) * 2.0 - 1.0)


def suppress(y, s, b=8):
    """Remove fraction s of the per-block 1-px alternating component."""
    hb, wb = H // b, W // b
    cb = cb_basis(b)
    out = y.astype(np.float64).copy()
    blk = out[:hb * b, :wb * b].reshape(hb, b, wb, b)
    amp = (blk * cb[None, :, None, :]).mean((1, 3))
    blk -= s * amp[:, None, :, None] * cb[None, :, None, :]
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def score(ref, test):
    q = '/tmp/pq.json'
    e = '/tmp/pe.json'
    subprocess.run([sys.executable, '/Users/arin/av2enc-publish/tools/quality.py',
                    '--ref', ref, '--test', test, '--json', q],
                   capture_output=True)
    subprocess.run([sys.executable, '/Users/arin/av2enc-publish/tools/extra_metrics.py',
                    '--ref', ref, '--test', test, '--json', e],
                   capture_output=True)
    r = subprocess.run([sys.executable, '/Users/arin/av2enc-publish/tools/kpi.py',
                        '--quality-json', q, '--extra-json', e],
                       capture_output=True, text=True)
    return r.stdout.strip()


def main():
    ref = sys.argv[1]
    src = sys.argv[2]
    y, u, v = read_yuv(src)
    print(f"{'suppression':>12}  score")
    for s in (0.0, 0.25, 0.5, 0.75, 1.0):
        out = f'/tmp/sup_{s:.2f}.y4m'
        write_yuv(out, suppress(y, s), u, v)
        print(f"{s*100:11.0f}%  {score(ref, out)}")


if __name__ == '__main__':
    main()
