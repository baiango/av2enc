#!/usr/bin/env python3
"""Bit-exact encoder-reconstruction vs dav2d comparison for M3 streams.

Decodes an IVF with the pinned ``dav2d --strict 1``, reads the first decoded
4:2:0 8-bit frame, and compares every sample against the encoder-side
reconstruction dumped by ``tools/still_verify/still_dump.cpp``.

The dump format is the documented still_dump format::

    "StillDUMP1\\n" + <one JSON header line> + Y + U + V

Exit codes:
    0  all planes bit-exact
    1  at least one differing sample
    2  usage / infrastructure error (missing files, decode failure, bad sizes)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

MAGIC = b"StillDUMP1\n"


def die(message: str) -> None:
    print(f"compare_recon: error: {message}", file=sys.stderr)
    raise SystemExit(2)


def find_decoder(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            die(f"decoder not found: {path}")
        return path
    if os.environ.get("DAV2D"):
        return find_decoder(os.environ["DAV2D"])
    root = Path(__file__).resolve().parents[2]
    for name in ("dav2d", "dav2d-macos"):
        candidate = root / "tools" / "bin" / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    die("dav2d not found; pass --decoder or set DAV2D")


def read_dump(path: Path) -> tuple[dict, dict[str, bytes]]:
    data = path.read_bytes()
    if not data.startswith(MAGIC):
        die(f"{path}: not an still_dump file (bad magic)")
    rest = data[len(MAGIC) :]
    newline = rest.find(b"\n")
    if newline < 0:
        die(f"{path}: truncated JSON header")
    try:
        header = json.loads(rest[:newline])
    except json.JSONDecodeError as exc:
        die(f"{path}: bad JSON header: {exc}")
    width = int(header["width"])
    height = int(header["height"])
    cw = int(header.get("chroma_width", (width + 1) // 2))
    ch = int(header.get("chroma_height", (height + 1) // 2))
    payload = rest[newline + 1 :]
    sizes = {"Y": width * height, "U": cw * ch, "V": cw * ch}
    expected = sum(sizes.values())
    if len(payload) != expected:
        die(
            f"{path}: payload is {len(payload)} bytes, expected {expected} "
            f"for {width}x{height} 4:2:0"
        )
    planes: dict[str, bytes] = {}
    offset = 0
    for name in ("Y", "U", "V"):
        planes[name] = payload[offset : offset + sizes[name]]
        offset += sizes[name]
    return header, planes


def read_y4m_first_frame(path: Path) -> tuple[int, int, dict[str, bytes]]:
    with path.open("rb") as f:
        header_line = f.readline()
        if not header_line.startswith(b"YUV4MPEG2"):
            die(f"{path}: not a Y4M file")
        width = height = 0
        chroma = ""
        for token in header_line.split()[1:]:
            if token[:1] == b"W":
                width = int(token[1:])
            elif token[:1] == b"H":
                height = int(token[1:])
            elif token[:1] == b"C":
                chroma = token[1:].decode("ascii", "replace")
        if width <= 0 or height <= 0:
            die(f"{path}: Y4M header has no dimensions")
        if chroma and not chroma.startswith("420"):
            die(f"{path}: expected 4:2:0, got C{chroma}")
        frame_line = f.readline()
        if not frame_line.startswith(b"FRAME"):
            die(f"{path}: missing first FRAME marker")
        cw, ch = (width + 1) // 2, (height + 1) // 2
        sizes = {"Y": width * height, "U": cw * ch, "V": cw * ch}
        planes: dict[str, bytes] = {}
        for name in ("Y", "U", "V"):
            data = f.read(sizes[name])
            if len(data) != sizes[name]:
                die(
                    f"{path}: truncated {name} plane "
                    f"({len(data)} of {sizes[name]} bytes)"
                )
            planes[name] = data
    return width, height, planes


def decode(decoder: Path, ivf: Path, out: Path) -> None:
    cmd = [
        str(decoder),
        "-q",
        "-i",
        str(ivf),
        "-o",
        str(out),
        "--threads",
        "1",
        "--strict",
        "1",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 or not out.is_file() or out.stat().st_size == 0:
        sys.stderr.write(proc.stderr.decode("utf-8", "replace"))
        die(f"dav2d --strict 1 failed ({proc.returncode}) on {ivf}")


def md5_bytes(parts: list[bytes]) -> str:
    digest = hashlib.md5()
    for part in parts:
        digest.update(part)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare still_dump encoder reconstruction with a "
        "dav2d --strict 1 decode, sample for sample."
    )
    parser.add_argument("--ivf", required=True)
    parser.add_argument("--dump", required=True)
    parser.add_argument("--decoder", default=None)
    parser.add_argument(
        "--keep-dec", default=None, help="copy the decoded Y4M here"
    )
    parser.add_argument(
        "--tsv",
        action="store_true",
        help="print one machine-readable line: "
        "STATUS dY dU dV maxdiff decoded_yuv_md5",
    )
    args = parser.parse_args()

    ivf = Path(args.ivf)
    dump_path = Path(args.dump)
    if not ivf.is_file():
        die(f"no such IVF: {ivf}")
    if not dump_path.is_file():
        die(f"no such dump: {dump_path}")

    header, enc = read_dump(dump_path)
    width = int(header["width"])
    height = int(header["height"])
    cw = int(header.get("chroma_width", (width + 1) // 2))
    ch = int(header.get("chroma_height", (height + 1) // 2))

    decoder = find_decoder(args.decoder)
    tmpdir = tempfile.mkdtemp(prefix="m3_compare_")
    dec_y4m = Path(tmpdir) / "decoded.y4m"
    try:
        decode(decoder, ivf, dec_y4m)
        if args.keep_dec:
            keep = Path(args.keep_dec)
            keep.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(dec_y4m, keep)
        dec_w, dec_h, dec = read_y4m_first_frame(dec_y4m)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    if (dec_w, dec_h) != (width, height):
        die(f"decoded frame is {dec_w}x{dec_h}, dump is {width}x{height}")

    sizes = {"Y": (width, height), "U": (cw, ch), "V": (cw, ch)}
    diffs = {}
    max_diff = 0
    first = None
    for name in ("Y", "U", "V"):
        a = enc[name]
        b = dec[name]
        if len(a) != len(b):
            die(f"plane {name}: dump {len(a)} bytes, decoded {len(b)} bytes")
        count = 0
        plane_first = None
        for idx, (x, y) in enumerate(zip(a, b)):
            if x == y:
                continue
            count += 1
            d = abs(x - y)
            if d > max_diff:
                max_diff = d
            if plane_first is None:
                plane_first = (idx, x, y)
        diffs[name] = count
        if first is None and plane_first is not None:
            w = sizes[name][0]
            first = (name, plane_first[0] % w, plane_first[0] // w,
                     plane_first[1], plane_first[2])

    total = sum(diffs.values())
    status = "BIT-EXACT" if total == 0 else "MISMATCH"
    # MD5 of the raw decoded YUV planes (not the Y4M container), so a pinned
    # value does not depend on the decoder's Y4M header text.
    dec_md5 = md5_bytes([dec["Y"], dec["U"], dec["V"]])

    if args.tsv:
        print(
            "\t".join(
                [
                    status,
                    str(diffs["Y"]),
                    str(diffs["U"]),
                    str(diffs["V"]),
                    str(max_diff),
                    dec_md5,
                ]
            )
        )
    else:
        print(f"ivf:      {ivf}")
        print(f"decoder:  {decoder}")
        print(
            f"dump:     {width}x{height} qp={header.get('base_q_idx')} "
            f"tile={header.get('tile_bytes')}B frame={header.get('frame_bytes')}B"
        )
        for name in ("Y", "U", "V"):
            w, h = sizes[name]
            print(
                f"plane {name}: {diffs[name]}/{w * h} differ, "
                f"decoded raw MD5 {dec_md5 if name == 'Y' else '(same frame)'}"
            )
        if first is not None:
            plane, x, y, a, b = first
            print(
                f"first diff: plane {plane} ({x}, {y}) enc={a} dec={b} d={a - b}"
            )
        print(f"RESULT:   {status} total_diffs={total} max|diff|={max_diff}")
        print(f"decoded_yuv_md5: {dec_md5}")

    return 0 if total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
