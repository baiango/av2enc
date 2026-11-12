#!/usr/bin/env python3
"""Compare two Y4M files sample-by-sample.

Usage: compare_y4m.py <recon.y4m> <decoded.y4m>

Supports C420/C422/C444/Cmono/C400 and their 10-bit *p10 variants.  Exits 0
only when every plane of every frame is byte-identical.  Used by the format
gates for the "0 reconstruction diffs" requirement; it does not average or use
PSNR.  For 10-bit streams the comparison is still exact because both files use
little-endian 16-bit LSB-aligned samples.
"""
import sys


def coloursapce_layout(colorspace):
    """Return (chroma, sample_bytes) or raise SystemExit."""
    ten_bit = colorspace.endswith("p10")
    base = colorspace[:-3] if ten_bit else colorspace
    sample_bytes = 2 if ten_bit else 1
    if base in ("420", "420jpeg", "420mpeg2", "420paldv"):
        return "420", sample_bytes
    if base == "422":
        return "422", sample_bytes
    if base == "444":
        return "444", sample_bytes
    if base in ("mono", "400", "monochrome"):
        return "400", sample_bytes
    # Monochrome 10-bit is spelled "mono10" (not "monop10").
    if base in ("mono10", "40010", "monochrome10"):
        return "400", 2
    raise SystemExit(f"unsupported colorspace {colorspace}")


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
    colorspace = next((t[1:] for t in tokens if t.startswith("C")), "420")
    chroma, sample_bytes = coloursapce_layout(colorspace)
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
        size = plane_sizes(width, height, chroma)
        frame_size = sum(s for _, s in size) * sample_bytes
        if position + frame_size > len(data):
            raise SystemExit(f"{path}: truncated frame")
        frames.append(data[position:position + frame_size])
        position += frame_size
    return width, height, colorspace, chroma, sample_bytes, frames


def plane_sizes(width, height, chroma):
    """Plane sizes in samples (multiply by sample_bytes for the frame size)."""
    if chroma == "422":
        return [("Y", width * height),
                ("U", ((width + 1) // 2) * height),
                ("V", ((width + 1) // 2) * height)]
    if chroma == "444":
        return [("Y", width * height), ("U", width * height),
                ("V", width * height)]
    if chroma == "400":
        return [("Y", width * height)]
    return [("Y", width * height),
            ("U", ((width + 1) // 2) * ((height + 1) // 2)),
            ("V", ((width + 1) // 2) * ((height + 1) // 2))]


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: compare_y4m.py <recon.y4m> <decoded.y4m>")
    aw, ah, acs, achroma, abytes, aframes = read_y4m(sys.argv[1])
    bw, bh, bcs, bchroma, bbytes, bframes = read_y4m(sys.argv[2])
    print(f"recon:   {aw}x{ah} C{acs} {len(aframes)} frame(s)")
    print(f"decoded: {bw}x{bh} C{bcs} {len(bframes)} frame(s)")
    if (aw, ah) != (bw, bh):
        print("FAIL: dimensions differ")
        return 1
    if (achroma, abytes) != (bchroma, bbytes):
        print("FAIL: chroma format/bit depth differ")
        return 1
    if len(aframes) != len(bframes):
        print("FAIL: frame count differs")
        return 1
    failed = False
    for frame_index, (recon, decoded) in enumerate(zip(aframes, bframes)):
        offset = 0
        for name, size in plane_sizes(aw, ah, achroma):
            byte_size = size * abytes
            a = recon[offset:offset + byte_size]
            b = decoded[offset:offset + byte_size]
            offset += byte_size
            if a == b:
                print(f"frame {frame_index} plane {name}: exact ({size} samples)")
            else:
                diffs = sum(1 for x, y in zip(a, b) if x != y)
                maximum = max((abs(x - y) for x, y in zip(a, b)), default=0)
                print(f"frame {frame_index} plane {name}: FAIL "
                      f"({diffs} byte diffs, max {maximum})")
                failed = True
    if failed:
        return 1
    print("OK: reconstruction is bit-exact (0 diffs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
