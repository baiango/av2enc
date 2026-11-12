#!/usr/bin/env python3
"""Plane PSNR / SSE between two 8-bit 4:2:0 Y4M files (first frame).

Usage: y4m_psnr.py source.y4m decoded.y4m
Prints one JSON object:
  {"width":W,"height":H,"y":{"sse":..,"psnr":..}, "u":..., "v":...,
   "weighted_psnr":..}
"""

from __future__ import annotations

import json
import math
import sys


def read_first(path: str) -> tuple[int, int, list[bytes]]:
    with open(path, "rb") as f:
        header = f.readline()
        if not header.startswith(b"YUV4MPEG2"):
            raise SystemExit(f"{path}: not a Y4M file")
        width = height = 0
        for token in header.split():
            if token.startswith(b"W"):
                width = int(token[1:])
            elif token.startswith(b"H"):
                height = int(token[1:])
        frame_header = f.readline()
        if not frame_header.startswith(b"FRAME"):
            raise SystemExit(f"{path}: missing FRAME header")
        cw, ch = (width + 1) // 2, (height + 1) // 2
        sizes = [width * height, cw * ch, cw * ch]
        planes = []
        for size in sizes:
            data = f.read(size)
            if len(data) != size:
                raise SystemExit(f"{path}: truncated frame")
            planes.append(data)
        return width, height, planes


def sse_plane(a: bytes, b: bytes) -> int:
    return sum((x - y) * (x - y) for x, y in zip(a, b))


def psnr(sse: int, samples: int) -> float:
    if sse == 0:
        return float("inf")
    mse = sse / samples
    return 10.0 * math.log10((255.0 * 255.0) / mse)


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: y4m_psnr.py source.y4m decoded.y4m", file=sys.stderr)
        return 2
    sw, sh, src = read_first(sys.argv[1])
    dw, dh, dec = read_first(sys.argv[2])
    if (sw, sh) != (dw, dh):
        raise SystemExit(f"size mismatch {sw}x{sh} vs {dw}x{dh}")
    cw, ch = (sw + 1) // 2, (sh + 1) // 2
    samples = [sw * sh, cw * ch, cw * ch]
    out: dict[str, object] = {"width": sw, "height": sh}
    weighted_sse = 0.0
    weighted_samples = 0.0
    for name, a, b, n in zip(("y", "u", "v"), src, dec, samples):
        sse = sse_plane(a, b)
        out[name] = {"sse": sse, "psnr": psnr(sse, n)}
        weight = 4.0 if name == "y" else 1.0
        weighted_sse += weight * sse / n
        weighted_samples += weight
    mse = weighted_sse / weighted_samples
    out["weighted_psnr"] = (
        float("inf") if mse == 0 else 10.0 * math.log10((255.0 * 255.0) / mse)
    )
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
