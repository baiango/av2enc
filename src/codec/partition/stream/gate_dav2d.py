#!/usr/bin/env python3
"""Block-level dav2d gate for the M1 extended-partition adapter.

gate_minimal writes four 64x64 IVF files (baseline NONE and HORZ_3, each with
frozen and adaptive CDFs).  The M1 header writer cannot yet set
``enable_ext_partitions``, so this script inserts the two §5.4.3 flag bits into
the ext_* sequence headers and decodes:

  baseline_<mode>.ivf  vs  ext_<mode>.ivf

Requirements for a pass:
  * dav2d decodes both streams without an error;
  * both Y4M frames are byte-identical (same pixels);
  * every sample is 128, the DC prediction of an all-skip 64x64 frame.

A failure means either the pinned decoder rejects the extended partition
syntax or our minimal leaf writer / symbol plan desynchronised.  Both are hard
integration failures.

Usage:
    python3 gate_dav2d.py --dir /tmp/partition_stream_test_gate_out
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


def write_leb128(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            byte |= 0x80
        out.append(byte)
        if not value:
            return bytes(out)


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


def rewrite_sequence_header(obu_bytes: bytes, size_prefix: int) -> bytes:
    header = obu_bytes[size_prefix]
    if (header >> 2) & 0x1F != 1:
        raise ValueError("not a sequence header OBU")
    payload = obu_bytes[size_prefix + 1 :]
    reader = BitReader(payload)
    reader.uvlc()                         # seq_header_id
    reader.bits_n(5)                      # seq_profile_idc
    single_picture = reader.bit()         # single_picture_header_flag
    reader.bits_n(5)                      # seq_level_idx
    chroma_format = reader.uvlc()
    reader.uvlc()                         # bit_depth_idc
    width_bits = reader.bits_n(4) + 1
    height_bits = reader.bits_n(4) + 1
    reader.bits_n(width_bits)             # max_frame_width_minus_1
    reader.bits_n(height_bits)            # max_frame_height_minus_1
    reader.bit()                          # seq_cropping_window_present_flag
    use_256 = reader.bit()
    if not use_256:
        reader.bit()                      # use_128x128_superblock
    if chroma_format != 1:                # CHROMA_FORMAT_400 has no SDP
        enable_sdp = reader.bit()
        if enable_sdp and not single_picture:
            reader.bit()                  # enable_extended_sdp
    flag_pos = reader.pos
    if reader.bit() != 0:
        raise ValueError("expected enable_ext_partitions = 0")

    bits = reader.bits
    if bits[flag_pos] != 0:
        raise ValueError("unexpected flag bit")
    new_bits = bits[:flag_pos] + [1, 1] + bits[flag_pos + 1 :]
    new_payload = bytearray()
    for i in range(0, len(new_bits), 8):
        chunk = new_bits[i : i + 8]
        chunk += [0] * (8 - len(chunk))
        byte = 0
        for bit in chunk:
            byte = (byte << 1) | bit
        new_payload.append(byte)

    out = bytearray()
    out += write_leb128(1 + len(new_payload))
    out.append(header)
    out += new_payload
    return bytes(out)


def patch_ivf(data: bytes) -> bytes:
    if data[:4] != b"DKIF":
        raise ValueError("not an IVF file")
    header = bytearray(data[:32])
    frames = struct.unpack_from("<I", header, 24)[0]
    if frames != 1:
        raise ValueError(f"expected 1 frame, got {frames}")
    offset = 32
    size = struct.unpack_from("<I", data, offset)[0]
    timestamp = data[offset + 4 : offset + 12]
    payload = data[offset + 12 : offset + 12 + size]
    if len(payload) != size:
        raise ValueError("truncated IVF frame")

    out_payload = bytearray()
    patched = False
    for obu_type, obu_bytes, size_prefix in split_obus(payload):
        if obu_type == 1:
            out_payload += rewrite_sequence_header(obu_bytes, size_prefix)
            patched = True
        else:
            out_payload += obu_bytes
    if not patched:
        raise ValueError("no sequence header OBU")
    out = bytearray(header)
    out += struct.pack("<I", len(out_payload))
    out += timestamp
    out += out_payload
    return bytes(out)


def y4m_pixels(path: pathlib.Path) -> bytes:
    data = path.read_bytes()
    marker = data.find(b"FRAME")
    if marker < 0:
        raise ValueError(f"{path}: no FRAME marker")
    start = data.find(b"\n", marker)
    if start < 0:
        raise ValueError(f"{path}: truncated FRAME marker")
    return data[start + 1 :]


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

    version = subprocess.run(
        [decoder, "--version"], capture_output=True, text=True
    )
    text = (version.stdout or version.stderr).strip()
    if text:
        print("decoder:", text.splitlines()[-1])

    failures = 0
    for mode in ("frozen", "adaptive"):
        base = out_dir / f"baseline_{mode}.ivf"
        if not base.is_file():
            print(f"error: missing baseline_{mode}.ivf", file=sys.stderr)
            return 2
        base_y4m = out_dir / f"baseline_{mode}.y4m"
        if not decode(decoder, base, base_y4m):
            print(f"FAIL[{mode}]: baseline failed to decode", file=sys.stderr)
            failures += 1
            continue
        base_pixels = y4m_pixels(base_y4m)
        if any(sample != 128 for sample in base_pixels):
            print(f"FAIL[{mode}]: baseline pixels are not all 128",
                  file=sys.stderr)
            failures += 1
            continue
        for case in ("ext", "ext4", "ext4b"):
            ext_raw = out_dir / f"{case}_{mode}.ivf"
            if not ext_raw.is_file():
                print(f"error: missing {case}_{mode}.ivf", file=sys.stderr)
                return 2
            ext = out_dir / f"{case}_{mode}_extflag.ivf"
            ext.write_bytes(patch_ivf(ext_raw.read_bytes()))
            ext_y4m = out_dir / f"{case}_{mode}.y4m"
            if not decode(decoder, ext, ext_y4m):
                print(f"FAIL[{case}/{mode}]: dav2d rejected the stream",
                      file=sys.stderr)
                failures += 1
                continue
            ext_pixels = y4m_pixels(ext_y4m)
            if base_pixels != ext_pixels:
                print(f"FAIL[{case}/{mode}]: pixels differ from baseline",
                      file=sys.stderr)
                failures += 1
                continue
            if any(sample != 128 for sample in ext_pixels):
                print(f"FAIL[{case}/{mode}]: pixels are not all 128",
                      file=sys.stderr)
                failures += 1
                continue
            print(f"PASS[{case}/{mode}]: {len(ext_pixels)} identical "
                  f"128-valued pixels")

    if failures:
        return 1
    print("PASS: block-level extended-partition dav2d gate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
