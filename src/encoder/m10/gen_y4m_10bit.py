#!/usr/bin/env python3
"""Generate a deterministic C420p10 Y4M frame for the M10 gate.

    python3 gen_y4m_10bit.py 64 64 out.y4m
    python3 gen_y4m_10bit.py 640 360 out.y4m

The pattern uses the full 10-bit range (including the low two bits that an
8-bit source cannot produce) so the 10-bit transform/clip path is exercised.
"""
import struct
import sys


def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__, file=sys.stderr)
        return 2
    width, height, path = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    cw, ch = (width + 1) // 2, (height + 1) // 2
    state = 0x9E3779B9

    def rnd():
        nonlocal state
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        return state

    y = bytearray()
    for j in range(height):
        for i in range(width):
            v = (i * 5 + j * 3 + ((i ^ j) << 1) + (rnd() & 63)) & 1023
            y += struct.pack("<H", v)
    u = bytearray()
    v = bytearray()
    for j in range(ch):
        for i in range(cw):
            u += struct.pack("<H", (i * 11 + j * 7 + (rnd() & 31)) & 1023)
            v += struct.pack("<H", (i * 3 + j * 13 + (rnd() & 31)) & 1023)
    with open(path, "wb") as f:
        f.write(
            f"YUV4MPEG2 W{width} H{height} F1:1 Ip A0:0 C420p10\n".encode())
        f.write(b"FRAME\n")
        f.write(y)
        f.write(u)
        f.write(v)
    print(f"wrote {path}: {width}x{height} C420p10")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
