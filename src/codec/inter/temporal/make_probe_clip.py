#!/usr/bin/env python3
"""Generate the small 4:2:0 pan clip used by the dav2d TIP probe.

Usage:
  python3 make_probe_clip.py /tmp/pan128.y4m [frames]

The clip is intentionally synthetic: an 8-frame 128x128 moving ramp with
chroma motion.  It exists only to give the AVM encoder enough temporal
structure to emit a TIP frame, so `probe_dav2d_tip.py` can ask dav2d whether it
accepts one.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    out = Path(sys.argv[1])
    frames = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    width = height = 128
    with out.open("wb") as f:
        f.write(f"YUV4MPEG2 W{width} H{height} F30:1 Ip A1:1 C420jpeg\n".encode())
        for t in range(frames):
            y = bytearray(width * height)
            u = bytearray(width * height // 4)
            v = bytearray(width * height // 4)
            for r in range(height):
                for c in range(width):
                    y[r * width + c] = (
                        c * 2 + r + t * 7 + ((c + t * 3) // 8) * 17
                    ) & 255
            for r in range(height // 2):
                for c in range(width // 2):
                    u[r * (width // 2) + c] = (128 + (c - r) // 2 + t * 2) & 255
                    v[r * (width // 2) + c] = (128 + (r - c) // 2 - t * 2) & 255
            f.write(b"FRAME\n")
            f.write(y)
            f.write(u)
            f.write(v)
    print(f"wrote {out} ({frames} frames, {width}x{height})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
