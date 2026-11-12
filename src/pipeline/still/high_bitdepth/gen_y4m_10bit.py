#!/usr/bin/env python3
"""Generate a deterministic 10-bit Y4M frame for the M10/hbd gates.

    python3 gen_y4m_10bit.py 64 64 out.y4m [420|422|444|400]
    python3 gen_y4m_10bit.py 640 360 out.y4m

The pattern uses the full 10-bit range (including the low two bits that an
8-bit source cannot produce) so the 10-bit transform/clip path is exercised.
The optional fourth argument selects the chroma format; it defaults to 420 so
every existing caller keeps producing the same C420p10 stream.
"""
import struct
import sys


def main() -> int:
    if len(sys.argv) not in (4, 5):
        print(__doc__, file=sys.stderr)
        return 2
    width, height, path = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    chroma = sys.argv[4] if len(sys.argv) == 5 else "420"
    if chroma == "420":
        sub_x, sub_y, tag = 1, 1, "C420p10"
    elif chroma == "422":
        sub_x, sub_y, tag = 1, 0, "C422p10"
    elif chroma == "444":
        sub_x, sub_y, tag = 0, 0, "C444p10"
    elif chroma == "400":
        sub_x, sub_y, tag = 1, 1, "Cmono10"
    else:
        print(f"unsupported chroma {chroma}", file=sys.stderr)
        return 2

    cw = (width + (1 << sub_x) - 1) >> sub_x
    ch = (height + (1 << sub_y) - 1) >> sub_y
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
    planes = [y]
    if chroma != "400":
        u = bytearray()
        v = bytearray()
        for j in range(ch):
            for i in range(cw):
                u += struct.pack("<H", (i * 11 + j * 7 + (rnd() & 31)) & 1023)
                v += struct.pack("<H", (i * 3 + j * 13 + (rnd() & 31)) & 1023)
        planes.append(u)
        planes.append(v)
    with open(path, "wb") as f:
        f.write(
            f"YUV4MPEG2 W{width} H{height} F1:1 Ip A0:0 {tag}\n".encode())
        f.write(b"FRAME\n")
        for plane in planes:
            f.write(plane)
    print(f"wrote {path}: {width}x{height} {tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
