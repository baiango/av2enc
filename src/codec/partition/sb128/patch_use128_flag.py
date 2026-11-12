#!/usr/bin/env python3
"""Clear use_128x128_superblock in an sb128 IVF stream.

Negative control for the gate: the tile payload is unchanged, so a decoder
that reads the header as a 64x64-SB stream must desynchronise.  This proves the
positive gate is actually exercising the 128-SB path rather than a
64-SB-compatible interpretation.

usage: patch_use128_flag.py input.ivf output.ivf
"""

import sys


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: patch_use128_flag.py input.ivf output.ivf", file=sys.stderr)
        return 2
    data = bytearray(open(sys.argv[1], "rb").read())
    if len(data) < 44 or data[:4] != b"DKIF":
        print("not an IVF file", file=sys.stderr)
        return 2

    pos = 32 + 12  # IVF file header + one frame header
    sequence = None
    while pos < len(data):
        size = 0
        shift = 0
        while True:
            byte = data[pos]
            pos += 1
            size |= (byte & 0x7F) << shift
            if not byte & 0x80:
                break
            shift += 7
        start = pos
        obu_type = (data[start] >> 2) & 0x1F
        if obu_type == 1:
            sequence = (start, size)
            break
        pos += size
    if sequence is None:
        print("no sequence header OBU", file=sys.stderr)
        return 2

    start, size = sequence
    bits = []
    for byte in data[start + 1 : start + size]:
        for shift in range(7, -1, -1):
            bits.append((byte >> shift) & 1)
    offset = 0

    def read_bits(count: int) -> int:
        nonlocal offset
        value = 0
        for _ in range(count):
            value = (value << 1) | bits[offset]
            offset += 1
        return value

    def uvlc() -> int:
        nonlocal offset
        zeros = 0
        while bits[offset] == 0:
            zeros += 1
            offset += 1
        offset += 1
        value = (1 << zeros) - 1
        if zeros:
            value += read_bits(zeros)
        return value

    uvlc()  # seq_header_id
    read_bits(5)  # seq_profile_idc
    read_bits(1)  # single_picture_header_flag
    read_bits(5)  # seq_level_idx
    uvlc()  # chroma_format_idc
    uvlc()  # bit_depth_idc
    width_bits = read_bits(4) + 1
    height_bits = read_bits(4) + 1
    read_bits(width_bits)
    read_bits(height_bits)
    read_bits(1)  # seq_cropping_window_present_flag
    read_bits(1)  # use_256x256_superblock
    use_128_bit = offset
    if bits[use_128_bit] != 1:
        print("use_128x128_superblock is already 0", file=sys.stderr)
        return 2

    byte_index = start + 1 + use_128_bit // 8
    bit_in_byte = 7 - (use_128_bit % 8)
    data[byte_index] &= ~(1 << bit_in_byte)
    open(sys.argv[2], "wb").write(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
