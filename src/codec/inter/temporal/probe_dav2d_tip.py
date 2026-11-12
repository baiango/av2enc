#!/usr/bin/env python3
"""Print the OBU types in an AV2 IVF, flagging TIP OBUs.

This is the black-box probe used in README.md to answer "does dav2d f9dafbc
accept a stream that carries TIP / OPFL / refine-MV tools?"  It does not parse
frame headers; it only lists the OBU type sequence per IVF frame.

Usage:
  python3 probe_dav2d_tip.py stream.ivf
  python3 probe_dav2d_tip.py stream.ivf --decode /path/to/dav2d

OBU type 13 = OBU_LEADING_TIP, 14 = OBU_REGULAR_TIP (layered types 12/15 are
SEF/brt and are not TIP).
"""

from __future__ import annotations

import argparse
import struct
import subprocess
import sys
from pathlib import Path

TIP_OBU_TYPES = {13, 14}


def read_leb128(data: bytes, pos: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while pos < len(data):
        byte = data[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return value, pos
        shift += 7
    raise ValueError("truncated leb128")


def iter_frames(path: Path):
    data = path.read_bytes()
    if data[:4] != b"DKIF":
        raise ValueError("not an IVF file")
    pos = 32
    index = 0
    while pos + 12 <= len(data):
        size, _timestamp = struct.unpack_from("<IQ", data, pos)
        pos += 12
        payload = data[pos : pos + size]
        if len(payload) != size:
            raise ValueError(f"truncated frame {index}")
        pos += size
        obu_types = []
        inner = 0
        while inner < len(payload):
            obu_size, inner = read_leb128(payload, inner)
            obu = payload[inner : inner + obu_size]
            if not obu:
                break
            inner += obu_size
            obu_types.append((obu[0] >> 2) & 0x1F)
        yield index, size, obu_types
        index += 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ivf", type=Path)
    parser.add_argument("--decode", help="optional dav2d path to run the gate")
    args = parser.parse_args()

    tip_seen = False
    frames = 0
    for index, size, obu_types in iter_frames(args.ivf):
        frames += 1
        marked = [t for t in obu_types if t in TIP_OBU_TYPES]
        if marked:
            tip_seen = True
        suffix = "  <-- TIP OBU" if marked else ""
        print(f"unit  {index:3d} bytes={size:8d} obu_types={obu_types}{suffix}")

    print(f"\nIVF payloads: {frames}, TIP OBU present: {tip_seen}")
    if args.decode:
        out = args.ivf.with_suffix(".decoded.y4m")
        cmd = [args.decode, "-q", "-i", str(args.ivf), "-o", str(out),
               "--threads", "1"]
        print("run:", " ".join(cmd))
        result = subprocess.run(cmd, check=False)
        decoded = out.read_bytes().count(b"FRAME") if out.exists() else 0
        print(f"decode exit: {result.returncode}, output: {out}, "
              f"decoded display frames: {decoded}")
        return result.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
