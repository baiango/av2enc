#!/usr/bin/env python3
"""dav2d gate for the sub-8 minimal streams.

Runs the pinned decoder over every IVF written by gate_minimal and verifies:

  * the stream decodes (exit 0, one frame);
  * the frame size matches the IVF header;
  * every decoded luma and chroma sample is 128 (all-skip DC prediction);
  * all 8x8 streams decode to the same frame as the 8x8 baseline.

The last point is the real gate: a desynchronised sub-8 chroma ownership or
coefficient context changes the symbol stream and either fails to decode or
produces a different frame.

Usage:
  python3 gate_dav2d.py --dir build/sub8/gate [--dav2d tools/bin/dav2d-macos]

Exit codes: 0 ok, 1 failure, 127 decoder not found.
"""
import argparse
import hashlib
import os
import struct
import subprocess
import sys
import tempfile

EXPECTED = [
    "baseline_none",
    "vert_leaf4x8",
    "vert_first4x4",
    "vert_last4x4",
    "vert_all4x4",
    "horz_leaf8x4",
    "horz_first4x4",
    "horz_last4x4",
    "horz_all4x4",
    "edge16_all4x4",
    "multi_sb_128x64",
]

BASELINE_MD5 = "817dbf600caced6f1556c6690595f08f"  # 8x8, all samples 128
BASELINE_MD5_16 = "02b5d5d5ba2a5de00017b31c40c527bc"  # 16x16, all samples 128
BASELINE_MD5_128 = "1df78db99c089b5262bed91613ea71ce"  # 128x64, all samples 128


def find_decoder(explicit):
    if explicit:
        return explicit
    env = os.environ.get("DAV2D")
    if env:
        return env
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    for candidate in ("tools/bin/dav2d", "tools/bin/dav2d-macos"):
        path = os.path.join(root, candidate)
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


def ivf_header(path):
    with open(path, "rb") as handle:
        data = handle.read(32)
    if len(data) < 32 or data[:4] != b"DKIF":
        raise ValueError("not a DKIF file")
    width, height = struct.unpack_from("<HH", data, 12)
    count = struct.unpack_from("<I", data, 24)[0]
    return width, height, count


def parse_y4m(path):
    with open(path, "rb") as handle:
        data = handle.read()
    line_end = data.find(b"\n")
    if line_end < 0:
        raise ValueError("missing Y4M header")
    header = data[:line_end].decode("latin1")
    fields = header.split()
    if not fields or fields[0] != "YUV4MPEG2":
        raise ValueError("not a Y4M file")
    width = height = 0
    for field in fields[1:]:
        if field.startswith("W"):
            width = int(field[1:])
        elif field.startswith("H"):
            height = int(field[1:])
    frames = []
    pos = line_end + 1
    while pos < len(data):
        frame_end = data.find(b"\n", pos)
        if frame_end < 0:
            break
        tag = data[pos:frame_end]
        if not tag.startswith(b"FRAME"):
            raise ValueError("missing FRAME marker")
        pos = frame_end + 1
        # C420: 4:2:0 8-bit for the streams this gate writes.
        size = width * height + 2 * ((width + 1) // 2) * ((height + 1) // 2)
        if pos + size > len(data):
            raise ValueError("truncated frame")
        frames.append(data[pos:pos + size])
        pos += size
    return width, height, frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="gate_minimal output dir")
    parser.add_argument("--dav2d", default=None)
    args = parser.parse_args()

    decoder = find_decoder(args.dav2d)
    if decoder is None or not os.access(decoder, os.X_OK):
        print("gate_dav2d: dav2d not found (set DAV2D=...)", file=sys.stderr)
        return 127

    version = subprocess.run([decoder, "--version"], capture_output=True, text=True)
    print("decoder:", (version.stdout or version.stderr).strip().splitlines()[-1]
          if (version.stdout or version.stderr).strip() else decoder)

    failures = 0
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name in EXPECTED:
            ivf = os.path.join(args.dir, name + ".ivf")
            if not os.path.isfile(ivf):
                print(f"{name:16s} MISSING {ivf}")
                failures += 1
                continue
            try:
                width, height, count = ivf_header(ivf)
            except ValueError as error:
                print(f"{name:16s} BAD-IVF {error}")
                failures += 1
                continue
            out = os.path.join(tmp, name + ".y4m")
            run = subprocess.run(
                [decoder, "-q", "-i", ivf, "-o", out, "--threads", "1"],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            if run.returncode != 0 or not os.path.isfile(out):
                print(f"{name:16s} DECODE-FAIL rc={run.returncode} {run.stderr.strip()}")
                failures += 1
                continue
            try:
                ywidth, yheight, frames = parse_y4m(out)
            except ValueError as error:
                print(f"{name:16s} BAD-Y4M {error}")
                failures += 1
                continue
            if (ywidth, yheight) != (width, height):
                print(f"{name:16s} SIZE {ywidth}x{yheight} != {width}x{height}")
                failures += 1
                continue
            if count != 1 or len(frames) != 1:
                print(f"{name:16s} FRAME-COUNT ivf={count} y4m={len(frames)}")
                failures += 1
                continue
            payload = frames[0]
            if any(byte != 128 for byte in payload):
                print(f"{name:16s} PIXELS not all 128")
                failures += 1
                continue
            digest = hashlib.md5(payload).hexdigest()
            results[name] = digest
            print(f"{name:16s} OK  {width}x{height} md5={digest}")

    if failures:
        print(f"gate_dav2d: {failures} failure(s)", file=sys.stderr)
        return 1
    for name, digest in results.items():
        if name == "edge16_all4x4":
            expected = BASELINE_MD5_16
        elif name == "multi_sb_128x64":
            expected = BASELINE_MD5_128
        else:
            expected = BASELINE_MD5
        if digest != expected:
            print(f"gate_dav2d: {name} md5 {digest} != pinned {expected}",
                  file=sys.stderr)
            return 1
    print("gate_dav2d: all streams decode and match the pinned all-128 frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())
