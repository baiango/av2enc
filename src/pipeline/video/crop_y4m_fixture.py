#!/usr/bin/env python3
"""Crop the top-left WxH window of a Y4M clip into a small 4:2:0 fixture.

The reference-management gates need an 8-picture 64x64 flat-P fixture at a
size where the product encoder makes inter decisions.  testdata only ships
larger clips, so the gate scripts crop one at run time instead of committing
binary media (AGENTS.md: never commit test media).

Usage:
  crop_y4m_fixture.py --input clip.y4m --width 64 --height 64 \
      --frames 8 --out fixture.y4m
"""
import argparse
import math
import sys


def parse_header(line):
    if not line.startswith(b"YUV4MPEG2"):
        raise SystemExit("not a Y4M stream")
    fields = line.decode("ascii").split()
    width = height = None
    for field in fields[1:]:
        if field.startswith("W"):
            width = int(field[1:])
        elif field.startswith("H"):
            height = int(field[1:])
    if width is None or height is None:
        raise SystemExit("Y4M header has no W/H")
    return width, height


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="")
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--frames", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--offset-x", type=int, default=0)
    ap.add_argument("--offset-y", type=int, default=0)
    ap.add_argument("--pan-x", type=int, default=0)
    ap.add_argument("--pan-y", type=int, default=0)
    ap.add_argument("--synthetic-moving", action="store_true")
    args = ap.parse_args()
    w, h = args.width, args.height
    if w <= 0 or h <= 0 or w % 2 or h % 2:
        raise SystemExit("crop dimensions must be positive and even")
    if args.synthetic_moving:
        with open(args.out, "wb") as dst:
            dst.write(
                f"YUV4MPEG2 W{w} H{h} F30:1 Ip A1:1 C420jpeg\n".encode("ascii")
            )
            for t in range(args.frames):
                dst.write(b"FRAME\n")
                luma = bytearray()
                for j in range(h):
                    for i in range(w):
                        value = (
                            100.0
                            + 70.0 * math.sin((i + 2 * t) / 7.0)
                            + 50.0 * math.cos((j + t) / 5.0)
                            + 30.0 * math.sin((i + j + 3 * t) / 11.0)
                        )
                        luma.append(max(0, min(255, int(value))))
                dst.write(luma)
                for _ in range(2):
                    dst.write(bytes([128] * ((w // 2) * (h // 2))))
        return 0
    if not args.input:
        raise SystemExit("--input is required unless --synthetic-moving is set")
    with open(args.input, "rb") as src, open(args.out, "wb") as dst:
        sw, sh = parse_header(src.readline())
        dst.write(
            f"YUV4MPEG2 W{w} H{h} F30:1 Ip A1:1 C420jpeg\n".encode("ascii")
        )
        cw, ch = w // 2, h // 2
        scw = (sw + 1) // 2
        written = 0
        while written < args.frames:
            ox = args.offset_x + written * args.pan_x
            oy = args.offset_y + written * args.pan_y
            if ox < 0 or oy < 0 or ox + w > sw or oy + h > sh:
                raise SystemExit("crop window leaves the input frame")
            marker = src.readline()
            if not marker:
                break
            y = src.read(sw * sh)
            u = src.read(scw * ((sh + 1) // 2))
            v = src.read(scw * ((sh + 1) // 2))
            if len(y) != sw * sh:
                break
            dst.write(b"FRAME\n")
            for row in range(h):
                start = (oy + row) * sw + ox
                dst.write(y[start:start + w])
            for row in range(ch):
                start = ((oy // 2) + row) * scw + (ox // 2)
                dst.write(u[start:start + cw])
            for row in range(ch):
                start = ((oy // 2) + row) * scw + (ox // 2)
                dst.write(v[start:start + cw])
            written += 1
        if written != args.frames:
            raise SystemExit(
                f"input clip has {written} usable frames, need {args.frames}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
