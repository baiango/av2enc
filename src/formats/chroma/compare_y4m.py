#!/usr/bin/env python3
"""Compare two 8-bit Y4M files sample-by-sample.

Usage: compare_y4m.py <recon.y4m> <decoded.y4m>

Supports C420/C422/C444/Cmono/C400.  Exits 0 only when every plane of every
frame is byte-identical.  Used by gate.sh for the "0 reconstruction diffs"
requirement; it does not average or use PSNR.
"""
import sys


def read_y4m(path):
    with open(path, "rb") as f:
        data = f.read()
    newline = data.find(b"\n")
    if newline < 0:
        raise SystemExit(f"{path}: no Y4M header")
    header = data[:newline].decode("latin1")
    tokens = header.split()
    if not tokens or tokens[0] != "YUV4MPEG2":
        raise SystemExit(f"{path}: not a Y4M file")
    width = int(next(t[1:] for t in tokens if t.startswith("W")))
    height = int(next(t[1:] for t in tokens if t.startswith("H")))
    colorspace = next((t for t in tokens if t.startswith("C")), "C420")
    position = newline + 1
    frames = []
    while position < len(data):
        end = data.find(b"\n", position)
        if end < 0:
            break
        marker = data[position:end]
        position = end + 1
        if not marker.startswith(b"FRAME"):
            break
        if colorspace in ("C420", "C420jpeg", "C420mpeg2", "C420paldv"):
            size = width * height + 2 * ((width + 1) // 2) * ((height + 1) // 2)
        elif colorspace == "C422":
            size = width * height + 2 * ((width + 1) // 2) * height
        elif colorspace == "C444":
            size = width * height * 3
        elif colorspace in ("Cmono", "C400", "Cmonochrome"):
            size = width * height
        else:
            raise SystemExit(f"{path}: unsupported colorspace {colorspace}")
        if position + size > len(data):
            raise SystemExit(f"{path}: truncated frame")
        frames.append(data[position:position + size])
        position += size
    return width, height, colorspace, frames


def plane_sizes(width, height, colorspace):
    if colorspace == "C422":
        return [("Y", width * height),
                ("U", ((width + 1) // 2) * height),
                ("V", ((width + 1) // 2) * height)]
    if colorspace == "C444":
        return [("Y", width * height), ("U", width * height),
                ("V", width * height)]
    if colorspace in ("Cmono", "C400", "Cmonochrome"):
        return [("Y", width * height)]
    return [("Y", width * height),
            ("U", ((width + 1) // 2) * ((height + 1) // 2)),
            ("V", ((width + 1) // 2) * ((height + 1) // 2))]


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: compare_y4m.py <recon.y4m> <decoded.y4m>")
    aw, ah, acs, aframes = read_y4m(sys.argv[1])
    bw, bh, bcs, bframes = read_y4m(sys.argv[2])
    print(f"recon:   {aw}x{ah} {acs} {len(aframes)} frame(s)")
    print(f"decoded: {bw}x{bh} {bcs} {len(bframes)} frame(s)")
    if (aw, ah) != (bw, bh):
        print("FAIL: dimensions differ")
        return 1
    if len(aframes) != len(bframes):
        print("FAIL: frame count differs")
        return 1
    failed = False
    for frame_index, (recon, decoded) in enumerate(zip(aframes, bframes)):
        offset = 0
        for name, size in plane_sizes(aw, ah, acs):
            a = recon[offset:offset + size]
            b = decoded[offset:offset + size]
            offset += size
            if a == b:
                print(f"frame {frame_index} plane {name}: exact ({size} samples)")
            else:
                diffs = sum(1 for x, y in zip(a, b) if x != y)
                maximum = max((abs(x - y) for x, y in zip(a, b)), default=0)
                print(f"frame {frame_index} plane {name}: FAIL "
                      f"({diffs} diffs, max {maximum})")
                failed = True
    if failed:
        return 1
    print("OK: reconstruction is bit-exact (0 diffs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
