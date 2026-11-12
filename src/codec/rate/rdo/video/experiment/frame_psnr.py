#!/usr/bin/env python3
"""Per-frame luma PSNR between a reference Y4M and a decoded Y4M.

The M2 RDO experiment reports stream-level PSNR/CVVDP.  This helper exists to
show where the luma difference lives when a P-frame decision changes: the
keyframe is identical across variants, so the total PSNR is dominated by it.
It performs no encoding and makes no speed claim.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path


def read_header(handle) -> tuple[int, int, str]:
    header = handle.readline().decode("ascii").rstrip("\n")
    match = re.search(r"W(\d+)\s+H(\d+)", header)
    if not match:
        raise SystemExit(f"bad Y4M header: {header}")
    width = int(match.group(1))
    height = int(match.group(2))
    chroma = "420"
    cm = re.search(r"C([0-9A-Za-z]+)", header)
    if cm:
        chroma = cm.group(1)
    return width, height, chroma


def frame_size(width: int, height: int, chroma: str) -> int:
    if chroma.startswith("420"):
        cw, ch = (width + 1) // 2, (height + 1) // 2
    elif chroma.startswith("422"):
        cw, ch = (width + 1) // 2, height
    else:
        cw, ch = width, height
    return width * height + 2 * cw * ch


def read_frames(path: Path):
    with path.open("rb") as handle:
        width, height, chroma = read_header(handle)
        size = frame_size(width, height, chroma)
        while True:
            marker = handle.readline()
            if not marker:
                break
            if marker.strip() != b"FRAME":
                raise SystemExit(f"bad frame marker in {path}")
            payload = handle.read(size)
            if len(payload) != size:
                break
            yield width, height, payload


def psnr_y(a: bytes, b: bytes) -> float:
    sse = 0
    for x, y in zip(a, b):
        d = x - y
        sse += d * d
    if sse == 0:
        return 99.0
    mse = sse / len(a)
    return 10.0 * math.log10((255.0 * 255.0) / mse)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", required=True, type=Path)
    parser.add_argument("--decoded", required=True, type=Path)
    parser.add_argument("--label", default="")
    parser.add_argument("--frames", type=int, default=0)
    args = parser.parse_args()

    ref_frames = list(read_frames(args.ref))
    dec_frames = list(read_frames(args.decoded))
    if args.frames > 0:
        ref_frames = ref_frames[: args.frames]
        dec_frames = dec_frames[: args.frames]
    if len(ref_frames) != len(dec_frames):
        raise SystemExit(
            f"frame count differs: ref={len(ref_frames)} dec={len(dec_frames)}"
        )
    print("frame\tpsnr_y")
    for index, (ref, dec) in enumerate(zip(ref_frames, dec_frames)):
        width, height, ref_payload = ref
        _, _, dec_payload = dec
        luma = width * height
        value = psnr_y(ref_payload[:luma], dec_payload[:luma])
        suffix = f" {args.label}" if args.label else ""
        print(f"{index}\t{value:.3f}{suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
