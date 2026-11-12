#!/usr/bin/env python3
"""Read cfl_ds_filter_index from the first sequence header of an IVF stream.

This is intentionally a small independent parser (not the encoder's writer) so
the decoder gate proves that the non-zero index is actually on the wire.  It
parses only as far as §5.4.5 sequence_intra_config(); every preceding field is
read with the same widths the M1 writer uses.

Usage: parse_seq_index.py stream.ivf
Exit:  0 with the index on stdout, 2 on malformed input.
"""

from __future__ import annotations

import struct
import sys


class Bits:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0

    def bit(self) -> int:
        if self.pos >= len(self.data) * 8:
            raise ValueError("out of bits")
        byte = self.data[self.pos >> 3]
        value = (byte >> (7 - (self.pos & 7))) & 1
        self.pos += 1
        return value

    def bits(self, n: int) -> int:
        value = 0
        for _ in range(n):
            value = (value << 1) | self.bit()
        return value

    def uvlc(self) -> int:
        zeros = 0
        while self.bit() == 0:
            zeros += 1
            if zeros > 31:
                raise ValueError("uvlc too long")
        if zeros == 0:
            return 0
        return ((1 << zeros) - 1) + self.bits(zeros)


def leb128(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        if offset >= len(data):
            raise ValueError("truncated leb128")
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return value, offset
        shift += 7


def parse_sequence_header(payload: bytes) -> int:
    b = Bits(payload)
    b.uvlc()                 # seq_header_id
    b.bits(5)                # seq_profile_idc
    single_picture = b.bit()  # single_picture_header_flag
    b.bits(5)                # seq_level_idx
    if not single_picture:
        b.bit()              # seq_tier
    b.uvlc()                 # chroma_format_idc
    b.uvlc()                 # bit_depth_idc
    width_bits = b.bits(4) + 1
    height_bits = b.bits(4) + 1
    b.bits(width_bits)       # max_frame_width_minus_1
    b.bits(height_bits)      # max_frame_height_minus_1
    b.bit()                  # seq_cropping_window_present_flag
    b.bits(5)                # sequence_partition_config flags
    b.bits(2)                # sequence_segment_config flags
    b.bit()                  # enable_dip
    b.bit()                  # enable_intra_edge_filter
    b.bit()                  # enable_mrls
    b.bit()                  # enable_cfl_intra
    return b.bits(2)         # cfl_ds_filter_index


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: parse_seq_index.py stream.ivf", file=sys.stderr)
        return 2
    data = open(sys.argv[1], "rb").read()
    if data[:4] != b"DKIF":
        raise SystemExit("not an IVF file")
    header_len = struct.unpack_from("<H", data, 6)[0]
    frame_count = struct.unpack_from("<I", data, 24)[0]
    if frame_count < 1:
        raise SystemExit("no frames")
    payload_size = struct.unpack_from("<I", data, header_len)[0]
    payload = data[header_len + 4 + 8 : header_len + 4 + 8 + payload_size]
    offset = 0
    while offset < len(payload):
        obu_size, offset = leb128(payload, offset)
        end = offset + obu_size
        if end > len(payload):
            raise SystemExit("truncated OBU")
        header = payload[offset]
        obu_type = (header >> 2) & 0x1F
        extension = (header >> 7) & 1
        header_bytes = 1 + (5 if extension else 0)
        body = payload[offset + header_bytes : end]
        if obu_type == 1:  # OBU_SEQUENCE_HEADER
            print(parse_sequence_header(body))
            return 0
        offset = end
    raise SystemExit("no sequence header OBU")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ValueError as exc:
        print(f"parse_seq_index: {exc}", file=sys.stderr)
        sys.exit(2)
