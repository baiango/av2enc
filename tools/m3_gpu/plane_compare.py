#!/usr/bin/env python3
"""Bit-exact plane comparison for M1 verification.

Compares the first frame of two sources, where each source is either:

* a Y4M file (8-bit 4:2:0), e.g. the output of the pinned ``dav2d``; or
* an ``StillDUMP1`` reconstruction dump (``tools/still_verify/still_dump.cpp`` and the
  GPU recon-dump contract documented in ``tools/still_gpu/README.md``).

The format is detected per file, so a decoded Y4M can be compared directly
against an encoder-side reconstruction dump. No tolerance is applied: any
sample difference fails. PSNR is never used as a gate.

Exit codes: 0 all planes bit-exact, 1 any differing sample, 2 setup/usage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path

DUMP_MAGIC = b"StillDUMP1\n"


def die(message: str) -> "NoReturn":  # noqa: F821 - string annotation
    print(f"plane_compare: error: {message}", file=sys.stderr)
    raise SystemExit(2)


@dataclass
class Frame:
    path: Path
    kind: str  # "y4m" | "dump"
    width: int
    height: int
    chroma_width: int
    chroma_height: int
    planes: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)


def md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def parse_y4m(path: Path) -> Frame:
    with path.open("rb") as handle:
        header = handle.readline()
        if not header.startswith(b"YUV4MPEG2"):
            die(f"{path}: not a Y4M file")
        width = height = 0
        chroma = "420"
        for token in header.split()[1:]:
            if token[:1] == b"W":
                width = int(token[1:])
            elif token[:1] == b"H":
                height = int(token[1:])
            elif token[:1] == b"C":
                chroma = token[1:].decode("latin-1")
        if width <= 0 or height <= 0:
            die(f"{path}: Y4M header has no dimensions")
        if not chroma.startswith("420"):
            die(f"{path}: only 8-bit 4:2:0 is supported (colorspace C{chroma})")
        if chroma.endswith(("p10", "p12", "p16")):
            die(f"{path}: only 8-bit 4:2:0 is supported (colorspace C{chroma})")
        line = handle.readline()
        while line and not line.startswith(b"FRAME"):
            line = handle.readline()
        if not line:
            die(f"{path}: no FRAME marker")
        cw, ch = (width + 1) // 2, (height + 1) // 2
        sizes = {"Y": width * height, "U": cw * ch, "V": cw * ch}
        planes = {}
        for name in ("Y", "U", "V"):
            data = handle.read(sizes[name])
            if len(data) != sizes[name]:
                die(f"{path}: truncated {name} plane "
                    f"({len(data)} of {sizes[name]} bytes)")
            planes[name] = data
    return Frame(path=path, kind="y4m", width=width, height=height,
                 chroma_width=cw, chroma_height=ch, planes=planes)


def parse_dump(path: Path) -> Frame:
    data = path.read_bytes()
    if not data.startswith(DUMP_MAGIC):
        die(f"{path}: not an StillDUMP1 file")
    rest = data[len(DUMP_MAGIC):]
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
    payload = rest[newline + 1:]
    sizes = {"Y": width * height, "U": cw * ch, "V": cw * ch}
    expected = sum(sizes.values())
    if len(payload) != expected:
        die(f"{path}: payload is {len(payload)} bytes, expected {expected} "
            f"for {width}x{height} 4:2:0")
    planes = {}
    offset = 0
    for name in ("Y", "U", "V"):
        planes[name] = payload[offset:offset + sizes[name]]
        offset += sizes[name]
    notes = []
    if "base_q_idx" in header:
        notes.append(f"qp={header.get('base_q_idx')}")
    if "max_partition_depth" in header:
        notes.append(f"depth={header.get('max_partition_depth')}")
    return Frame(path=path, kind="dump", width=width, height=height,
                 chroma_width=cw, chroma_height=ch, planes=planes,
                 notes=notes)


def load_frame(path: Path) -> Frame:
    if not path.is_file():
        die(f"no such file: {path}")
    with path.open("rb") as handle:
        head = handle.read(16)
    if head.startswith(DUMP_MAGIC):
        return parse_dump(path)
    if head.startswith(b"YUV4MPEG2"):
        return parse_y4m(path)
    die(f"{path}: unknown format (not Y4M or StillDUMP1)")


@dataclass
class PlaneStats:
    name: str
    width: int
    height: int
    total: int
    diffs: int = 0
    max_diff: int = 0
    first: tuple | None = None
    bbox: list = field(default_factory=lambda: [1 << 30, 1 << 30, -1, -1])
    histogram: dict = field(default_factory=dict)


def compare_plane(name: str, a: bytes, b: bytes, width: int, height: int) -> PlaneStats:
    stats = PlaneStats(name=name, width=width, height=height, total=width * height)
    if a == b:
        return stats
    for index, (left, right) in enumerate(zip(a, b)):
        if left == right:
            continue
        delta = left - right
        stats.diffs += 1
        stats.max_diff = max(stats.max_diff, abs(delta))
        stats.histogram[delta] = stats.histogram.get(delta, 0) + 1
        x, y = index % width, index // width
        if stats.first is None:
            stats.first = (x, y, left, right)
        stats.bbox[0] = min(stats.bbox[0], x)
        stats.bbox[1] = min(stats.bbox[1], y)
        stats.bbox[2] = max(stats.bbox[2], x)
        stats.bbox[3] = max(stats.bbox[3], y)
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bit-exact Y/U/V comparison of two Y4M or StillDUMP1 files.")
    parser.add_argument("cpu", type=Path, help="reference (CPU M1)")
    parser.add_argument("gpu", type=Path, help="under test (GPU M1)")
    parser.add_argument("--json", action="store_true",
                        help="print a machine-readable JSON result")
    parser.add_argument("--tsv", action="store_true",
                        help="print one machine-readable TSV line")
    args = parser.parse_args()

    ref = load_frame(args.cpu)
    test = load_frame(args.gpu)
    if (ref.width, ref.height) != (test.width, test.height):
        die(f"dimension mismatch: {ref.width}x{ref.height} vs "
            f"{test.width}x{test.height}")

    sizes = {
        "Y": (ref.width, ref.height),
        "U": (ref.chroma_width, ref.chroma_height),
        "V": (ref.chroma_width, ref.chroma_height),
    }
    stats = []
    for name in ("Y", "U", "V"):
        width, height = sizes[name]
        left = ref.planes[name]
        right = test.planes[name]
        if len(left) != width * height or len(right) != width * height:
            die(f"{name} plane size mismatch")
        stats.append(compare_plane(name, left, right, width, height))

    total_diffs = sum(item.diffs for item in stats)
    max_diff = max(item.max_diff for item in stats)
    first_bad = next((item for item in stats if item.diffs), None)

    if args.tsv:
        first_x = first_y = -1
        if first_bad is not None:
            first_x, first_y = first_bad.first[0], first_bad.first[1]
        print("\t".join([
            "BIT-EXACT" if total_diffs == 0 else "MISMATCH",
            str(stats[0].diffs), str(stats[1].diffs), str(stats[2].diffs),
            str(max_diff), str(first_x), str(first_y),
            first_bad.name if first_bad else "-",
        ]))
        return 0 if total_diffs == 0 else 1

    if args.json:
        print(json.dumps({
            "ok": total_diffs == 0,
            "cpu": {"path": str(ref.path), "kind": ref.kind},
            "gpu": {"path": str(test.path), "kind": test.kind},
            "width": ref.width,
            "height": ref.height,
            "planes": {
                item.name: {
                    "samples": item.total,
                    "diffs": item.diffs,
                    "max_abs_diff": item.max_diff,
                    "md5_cpu": md5(ref.planes[item.name]),
                    "md5_gpu": md5(test.planes[item.name]),
                }
                for item in stats
            },
            "total_diffs": total_diffs,
            "max_abs_diff": max_diff,
        }, indent=2, sort_keys=True))
        return 0 if total_diffs == 0 else 1

    print(f"cpu:  {ref.kind:4s} {ref.width}x{ref.height}  {ref.path}"
          + (f"  [{' '.join(ref.notes)}]" if ref.notes else ""))
    print(f"gpu:  {test.kind:4s} {test.width}x{test.height}  {test.path}"
          + (f"  [{' '.join(test.notes)}]" if test.notes else ""))
    for item in stats:
        if item.diffs == 0:
            print(f"  plane {item.name}: exact ({item.total} samples, "
                  f"md5 {md5(ref.planes[item.name])})")
            continue
        pct = 100.0 * item.diffs / item.total
        x, y, left, right = item.first
        print(f"  plane {item.name}: {item.diffs}/{item.total} differ "
              f"({pct:.3f}%), max|diff|={item.max_diff}")
        print(f"    first sample: ({x}, {y}) cpu={left} gpu={right} "
              f"d={left - right}")
        box = item.bbox
        print(f"    diff bbox:    x {box[0]}..{box[2]}, y {box[1]}..{box[3]}")
        print(f"    md5:          cpu={md5(ref.planes[item.name])} "
              f"gpu={md5(test.planes[item.name])}")
        hist = sorted(item.histogram.items(), key=lambda kv: (-kv[1], kv[0]))
        shown = ", ".join(f"{delta:+d}:{count}" for delta, count in hist[:8])
        if len(hist) > 8:
            shown += ", ..."
        print(f"    deltas:       {shown}")
    if total_diffs == 0:
        print("RESULT: PLANES BIT-EXACT (0 differing samples)")
    else:
        print(f"RESULT: PLANES MISMATCH ({total_diffs} differing samples, "
              f"Y {stats[0].diffs}, U {stats[1].diffs}, V {stats[2].diffs}, "
              f"max|diff|={max_diff})")
    return 0 if total_diffs == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
