#!/usr/bin/env python3
"""Gate the AV2 sequence-header extended-partition flags with dav2d.

The M1 writer hardcodes ``enable_ext_partitions = 0`` (m1_headers.cpp) and its
partition model has no extended rows, so this component cannot emit a stream
that *uses* HORZ_3 / uneven 4-way without editing another agent's files.  What
it can do is prove that the pinned decoder accepts the tools at sequence-header
level:

  1. encode a 64x64 still with the existing M1 binary and ``--m1-depth 0`` so
     the superblock root is PARTITION_NONE (do_split = 0, no partition syntax
     after it);
  2. rewrite the sequence header to ``enable_ext_partitions = 1`` and
     ``enable_uneven_4way_partitions = 1``;
  3. decode both streams with dav2d and require byte-identical pixels.

Because a NONE root never reads rect_type / do_ext_partition, a decoder that
supports the flags must produce exactly the original frame.  A mismatch or an
error means the pinned decoder does not accept the tool set (or the patch is
wrong); either way it is a hard failure for integration.

Usage:
    python3 src/encoder/partition_ext/gate_dav2d.py \
        --encoder build-m1/av2enc --decoder tools/bin/dav2d-macos

Exit codes: 0 pass, 1 fail, 2 usage/setup error.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import struct
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parent.parent.parent.parent


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


def split_obus(payload: bytes) -> list[tuple[int, bytes]]:
    """Return (obu_type, full_obu_bytes) for one temporal unit."""
    obus = []
    offset = 0
    while offset < len(payload):
        size, after_size = read_leb128(payload, offset)
        end = after_size + size
        if end > len(payload):
            raise ValueError("truncated OBU")
        header = payload[after_size]
        obu_type = (header >> 2) & 0x1F
        obus.append((obu_type, payload[offset:end], after_size - offset, size))
        offset = end
    return obus


def rewrite_sequence_header(obu_bytes: bytes, size_prefix: int) -> bytes:
    header = obu_bytes[size_prefix]
    if (header >> 2) & 0x1F != 1:
        raise ValueError("not a sequence header OBU")
    payload = obu_bytes[size_prefix + 1 :]
    reader = BitReader(payload)
    seq_header_id = reader.uvlc()
    seq_profile_idc = reader.bits_n(5)
    single_picture_header_flag = reader.bit()
    seq_level_idx = reader.bits_n(5)
    chroma_format_idc = reader.uvlc()
    bit_depth_idc = reader.uvlc()
    width_bits_minus_1 = reader.bits_n(4)
    height_bits_minus_1 = reader.bits_n(4)
    width_bits = width_bits_minus_1 + 1
    height_bits = height_bits_minus_1 + 1
    max_width_minus_1 = reader.bits_n(width_bits)
    max_height_minus_1 = reader.bits_n(height_bits)
    cropping = reader.bit()
    use_256 = reader.bit()
    use_128 = 0 if use_256 else reader.bit()
    # Table 6.2: chroma_format_idc 1 is CHROMA_FORMAT_400 (Monochrome);
    # 0 is 4:2:0, 2 is 4:4:4, 3 is 4:2:2.
    enable_sdp = 0 if chroma_format_idc == 1 else reader.bit()
    if enable_sdp and not single_picture_header_flag:
        reader.bit()  # enable_extended_sdp
    flag_pos = reader.pos
    old_flag = reader.bit()
    if old_flag != 0:
        raise ValueError("expected enable_ext_partitions = 0 in the source stream")

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

    new_size = 1 + len(new_payload)
    out = bytearray()
    out += write_leb128(new_size)
    out.append(header)
    out += new_payload
    print(
        "  sequence header: "
        f"seq_header_id={seq_header_id} profile={seq_profile_idc} "
        f"level={seq_level_idx} chroma={chroma_format_idc} "
        f"bit_depth_idc={bit_depth_idc} {max_width_minus_1 + 1}x"
        f"{max_height_minus_1 + 1} 128SB={use_128} sdp={enable_sdp} "
        f"crop={cropping}"
    )
    print(
        f"  enable_ext_partitions bit at payload bit {flag_pos}; "
        f"payload {len(payload)} -> {len(new_payload)} bytes"
    )
    return bytes(out)


def patch_ivf(data: bytes) -> bytes:
    if data[:4] != b"DKIF":
        raise ValueError("not an IVF file")
    if len(data) < 32:
        raise ValueError("truncated IVF header")
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

    obus = split_obus(payload)
    print(f"  OBUs: {[t for t, *_ in obus]}")
    out_payload = bytearray()
    patched = False
    for obu_type, obu_bytes, size_prefix, _obu_size in obus:
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


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def decode(decoder: str, ivf: pathlib.Path, out_y4m: pathlib.Path) -> bool:
    result = run([decoder, "-q", "-i", str(ivf), "-o", str(out_y4m),
                  "--threads", "1"])
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        return False
    return out_y4m.is_file() and out_y4m.stat().st_size > 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", default=str(REPO / "build-m1" / "av2enc"))
    parser.add_argument("--decoder", default="")
    parser.add_argument("--source", default=str(REPO / "testdata" / "m0_64x64_1f.y4m"))
    args = parser.parse_args()

    encoder = pathlib.Path(args.encoder)
    if not encoder.is_file():
        # Try the default build directory names.
        for candidate in ("build/av2enc", "build-m0/av2enc"):
            alt = REPO / candidate
            if alt.is_file():
                encoder = alt
                break
    if not encoder.is_file():
        print(f"error: encoder not found: {args.encoder}", file=sys.stderr)
        return 2
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
    source = pathlib.Path(args.source)
    if not source.is_file():
        print(f"error: source not found: {source}", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix="av2ext_") as tmp:
        tmpdir = pathlib.Path(tmp)
        plain = tmpdir / "plain.ivf"
        patched = tmpdir / "patched.ivf"
        plain_y4m = tmpdir / "plain.y4m"
        patched_y4m = tmpdir / "patched.y4m"

        print("encode:", encoder)
        result = run([str(encoder), str(source), "-o", str(plain), "--m1",
                      "--m1-depth", "0"])
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr)
            print("error: encode failed", file=sys.stderr)
            return 2

        print("patch sequence header")
        patched.write_bytes(patch_ivf(plain.read_bytes()))

        version = run([decoder, "--version"])
        print("decoder:", (version.stdout or version.stderr).strip().splitlines()[-1]
              if (version.stdout or version.stderr).strip() else decoder)

        if not decode(decoder, plain, plain_y4m):
            print("error: the unpatched stream failed to decode", file=sys.stderr)
            return 2
        if not decode(decoder, patched, patched_y4m):
            print("FAIL: dav2d rejected enable_ext_partitions=1", file=sys.stderr)
            return 1

        a = plain_y4m.read_bytes()
        b = patched_y4m.read_bytes()
        if a != b:
            print("FAIL: decoded pixels differ after enabling the flags",
                  file=sys.stderr)
            return 1
        print(f"PASS: dav2d decoded the ext/uneven sequence flags; "
              f"{len(a)} identical Y4M bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
