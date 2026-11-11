#!/usr/bin/env python3
"""Build the DDT gate clip from a pinned test frame.

The gate needs at least two frames with real motion so the inter-residual path
chooses non-DCT transform types on 8x8/16x16 blocks.  The repository's
shine_256x256_1f.y4m is one frame; this script writes a deterministic 3-frame
4:2:0 clip by shifting it horizontally by 3/6 luma samples (1/2 chroma
samples).  No random data, no external tools.

Usage: make_motion_clip.py <shine_256x256_1f.y4m> <out.y4m>
"""
import sys

import numpy as np


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 2
    source, target = argv
    data = open(source, "rb").read()
    header_end = data.index(b"\n")
    header = data[:header_end]
    pos = header_end + 1
    if data[pos : pos + 6] != b"FRAME\n":
        raise SystemExit("expected a single raw FRAME")
    pos += 6
    width = height = 256
    cw = ch = 128
    y = np.frombuffer(data[pos : pos + width * height], dtype=np.uint8)
    pos += width * height
    u = np.frombuffer(data[pos : pos + cw * ch], dtype=np.uint8)
    pos += cw * ch
    v = np.frombuffer(data[pos : pos + cw * ch], dtype=np.uint8)
    y = y.reshape(height, width)
    u = u.reshape(ch, cw)
    v = v.reshape(ch, cw)

    with open(target, "wb") as out:
        out.write(header + b"\n")
        for frame in range(3):
            sx = 3 * frame
            out.write(b"FRAME\n")
            out.write(np.roll(y, (0, -sx), axis=(0, 1)).tobytes())
            out.write(np.roll(u, (0, -(sx // 2)), axis=(0, 1)).tobytes())
            out.write(np.roll(v, (0, -(sx // 2)), axis=(0, 1)).tobytes())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
