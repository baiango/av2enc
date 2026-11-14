#!/usr/bin/env python3
"""Per-frame PSNR-Y / bytes of an AV2 IVF against a reference Y4M.

Usage: perframe_psnr.py REF.ivf-decoded.y4m TEST.y4m [IVF] [--frames N]

Measurement helper for the drift-refresh experiment; not part of the encoder.
"""
import argparse
import math
import struct

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


def read_y4m(path, limit):
    frames = []
    w = h = 0
    with open(path, "rb") as f:
        line = f.readline()
        if not line.startswith(b"YUV4MPEG2"):
            raise SystemExit(f"{path}: not a Y4M file")
        toks = line.split()
        for t in toks:
            if t.startswith(b"W"):
                w = int(t[1:])
            if t.startswith(b"H"):
                h = int(t[1:])
        while len(frames) < limit:
            head = f.readline()
            if not head or not head.startswith(b"FRAME"):
                break
            y = f.read(w * h)
            f.read(w * h // 2)
            frames.append(y)
    return w, h, frames


def ivf_sizes(path):
    sizes = []
    with open(path, "rb") as f:
        header = f.read(32)
        if len(header) < 32 or header[:4] != b"DKIF":
            raise SystemExit(f"{path}: not an IVF file")
        while True:
            fh = f.read(12)
            if len(fh) < 12:
                break
            size = struct.unpack("<I", fh[:4])[0]
            f.seek(size, 1)
            sizes.append(size)
    return sizes


def psnr_y(a, b):
    if np is not None:
        aa = np.frombuffer(a, dtype=np.uint8).astype(np.int32)
        bb = np.frombuffer(b, dtype=np.uint8).astype(np.int32)
        d = aa - bb
        mse = float(np.dot(d, d)) / d.size
    else:
        sse = 0
        for x, y in zip(a, b):
            d = x - y
            sse += d * d
        mse = sse / len(a)
    return 99.0 if mse == 0 else 10.0 * math.log10(255.0 * 255.0 / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("test")
    ap.add_argument("ivf", nargs="?")
    ap.add_argument("--frames", type=int, default=10 ** 9)
    args = ap.parse_args()
    _, _, ref = read_y4m(args.ref, args.frames)
    _, _, test = read_y4m(args.test, args.frames)
    sizes = ivf_sizes(args.ivf) if args.ivf else []
    n = min(len(ref), len(test))
    total = 0.0
    for i in range(n):
        p = psnr_y(ref[i], test[i])
        total += p
        extra = f" {sizes[i]}" if i < len(sizes) else ""
        print(f"{i} {p:.3f}{extra}")
    print(f"mean {total / n:.4f}")


if __name__ == "__main__":
    main()
