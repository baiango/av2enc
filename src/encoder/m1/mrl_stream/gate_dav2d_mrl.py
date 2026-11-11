#!/usr/bin/env python3
"""dav2d gate for the minimal M1 + MRL stream writer.

mrl_gate writes, per case, ``<case>.ivf`` and ``<case>.y`` (the exact expected
luma plane produced by the MRL prediction core; residuals and loop filters are
off).  This script:

  * parses the sequence header and requires ``enable_mrls == 1``;
  * decodes the IVF with the pinned dav2d;
  * compares the Y4M luma plane byte-for-byte with ``<case>.y``.

A pixels mismatch means the MRL index / secondary-line syntax or the MRL
prediction core disagrees with the reference decoder.

Usage: python3 gate_dav2d_mrl.py --dir DIR
Exit codes: 0 pass, 1 gate failure, 2 setup error.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import struct
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[4]


class BitReader:
    def __init__(self, data: bytes) -> None:
        self.bits = []
        for byte in data:
            for shift in range(7, -1, -1):
                self.bits.append((byte >> shift) & 1)
        self.pos = 0

    def bit(self) -> int:
        if self.pos >= len(self.bits):
            raise ValueError("bitstream exhausted")
        value = self.bits[self.pos]
        self.pos += 1
        return value

    def bits_n(self, n: int) -> int:
        value = 0
        for _ in range(n):
            value = (value << 1) | self.bit()
        return value

    def uvlc(self) -> int:
        leading = 0
        while self.bit() == 0:
            leading += 1
            if leading > 32:
                raise ValueError("uvlc too long")
        if leading == 0:
            return 0
        return (1 << leading) - 1 + self.bits_n(leading)


def read_leb128(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return value, offset
        shift += 7


def split_obus(payload: bytes) -> list[tuple[int, bytes, int]]:
    obus = []
    offset = 0
    while offset < len(payload):
        size, after_size = read_leb128(payload, offset)
        end = after_size + size
        if end > len(payload):
            raise ValueError("truncated OBU")
        header = payload[after_size]
        obu_type = (header >> 2) & 0x1F
        obus.append((obu_type, payload[offset:end], after_size - offset))
        offset = end
    return obus


def sequence_header_flags(obu_bytes: bytes, size_prefix: int) -> dict[str, int]:
    header = obu_bytes[size_prefix]
    if (header >> 2) & 0x1F != 1:
        raise ValueError("not a sequence header OBU")
    reader = BitReader(obu_bytes[size_prefix + 1 :])
    reader.uvlc()                            # seq_header_id
    reader.bits_n(5)                         # seq_profile_idc
    single_picture = reader.bit()            # single_picture_header_flag
    reader.bits_n(5)                         # seq_level_idx
    chroma_format = reader.uvlc()
    reader.uvlc()                            # bit_depth_idc
    width_bits = reader.bits_n(4) + 1
    height_bits = reader.bits_n(4) + 1
    reader.bits_n(width_bits)                # max_frame_width_minus_1
    reader.bits_n(height_bits)               # max_frame_height_minus_1
    reader.bit()                             # seq_cropping_window_present_flag
    use_256 = reader.bit()
    if not use_256:
        reader.bit()                         # use_128x128_superblock
    if chroma_format != 1:                   # CHROMA_FORMAT_400 has no SDP
        enable_sdp = reader.bit()
        if enable_sdp and not single_picture:
            reader.bit()                     # enable_extended_sdp
    enable_ext_partitions = reader.bit()
    enable_ext_seg = reader.bit()
    seq_seg_info = reader.bit()
    enable_dip = reader.bit()
    enable_edge = reader.bit()
    enable_mrls = reader.bit()
    return {
        "chroma_format": chroma_format,
        "enable_ext_partitions": enable_ext_partitions,
        "enable_ext_seg": enable_ext_seg,
        "seq_seg_info": seq_seg_info,
        "enable_dip": enable_dip,
        "enable_intra_edge_filter": enable_edge,
        "enable_mrls": enable_mrls,
    }


def ivf_payload(path: pathlib.Path) -> bytes:
    data = path.read_bytes()
    if data[:4] != b"DKIF":
        raise ValueError(f"{path}: not an IVF file")
    frames = struct.unpack_from("<I", data, 24)[0]
    if frames != 1:
        raise ValueError(f"{path}: expected 1 frame, got {frames}")
    size = struct.unpack_from("<I", data, 32)[0]
    payload = data[44 : 44 + size]
    if len(payload) != size:
        raise ValueError(f"{path}: truncated IVF frame")
    return payload


def y4m_luma(path: pathlib.Path, width: int, height: int) -> bytes:
    data = path.read_bytes()
    marker = data.find(b"FRAME")
    if marker < 0:
        raise ValueError(f"{path}: no FRAME marker")
    start = data.find(b"\n", marker)
    if start < 0:
        raise ValueError(f"{path}: truncated FRAME marker")
    plane = data[start + 1 : start + 1 + width * height]
    if len(plane) != width * height:
        raise ValueError(f"{path}: truncated luma plane")
    return plane


def decode(decoder: str, ivf: pathlib.Path, out_y4m: pathlib.Path) -> bool:
    result = subprocess.run(
        [decoder, "-q", "-i", str(ivf), "-o", str(out_y4m), "--threads", "1"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        return False
    return out_y4m.is_file() and out_y4m.stat().st_size > 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--decoder", default="")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.dir)
    decoder = args.decoder
    if not decoder:
        for candidate in ("tools/bin/dav2d", "tools/bin/dav2d-macos"):
            path = REPO / candidate
            if path.is_file() and os.access(path, os.X_OK):
                decoder = str(path)
                break
    if not decoder or not os.access(decoder, os.X_OK):
        print("error: dav2d not found (--decoder)", file=sys.stderr)
        return 2

    cases = [
        ("mrl_dc_64x64", 64, 64),
        ("mrl_dir_64x64_32", 64, 64),
        ("mrl_dir_64x64_16", 64, 64),
        ("mrl_dir_64x64_8", 64, 64),
        ("mrl_dir_64x64_4", 64, 64),
        ("mrl_dir_64x64_16_adaptive", 64, 64),
        ("mrl_dir_128x128_sb", 128, 128),
        ("mrl_origin_64x64", 64, 64),
    ]

    failures = 0
    for name, width, height in cases:
        ivf = out_dir / f"{name}.ivf"
        expected = out_dir / f"{name}.y"
        if not ivf.is_file() or not expected.is_file():
            print(f"error: missing {name}.ivf/.y", file=sys.stderr)
            return 2

        try:
            flags = {}
            for obu_type, obu_bytes, size_prefix in split_obus(
                ivf_payload(ivf)
            ):
                if obu_type == 1:
                    flags = sequence_header_flags(obu_bytes, size_prefix)
            if flags.get("enable_mrls") != 1:
                print(f"FAIL[{name}]: enable_mrls is not 1", file=sys.stderr)
                failures += 1
                continue
        except ValueError as error:
            print(f"FAIL[{name}]: {error}", file=sys.stderr)
            failures += 1
            continue

        y4m = out_dir / f"{name}.y4m"
        if not decode(decoder, ivf, y4m):
            print(f"FAIL[{name}]: dav2d rejected the stream", file=sys.stderr)
            failures += 1
            continue
        try:
            got = y4m_luma(y4m, width, height)
        except ValueError as error:
            print(f"FAIL[{name}]: {error}", file=sys.stderr)
            failures += 1
            continue
        want = expected.read_bytes()
        if got != want:
            first = next(
                (i for i, (a, b) in enumerate(zip(got, want)) if a != b), -1
            )
            print(
                f"FAIL[{name}]: luma mismatch at sample {first} "
                f"(got {got[first] if first >= 0 else -1}, "
                f"want {want[first] if first >= 0 else -1})",
                file=sys.stderr,
            )
            failures += 1
            continue
        print(f"PASS[{name}]: {len(want)} luma samples match the MRL core")

    if failures:
        return 1
    print("PASS: minimal M1 + MRL dav2d gate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
