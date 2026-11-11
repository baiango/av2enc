#!/usr/bin/env python3
"""Exact encoder-reconstruction vs decoder comparison for M1 full-intra.

The CLI writes only the bitstream; m1_dump writes M1Result.recon_y/u/v. This
tool decodes the IVF with the pinned dav2d and compares every sample of the
first frame. It reports per-plane counts, max |diff|, the first differing
4x4/8x8 block, a 64x64 superblock map and a delta histogram.

Exit codes:
  0  all planes bit-exact
  1  at least one differing sample
  2  usage / infrastructure error (missing files, decoder failure, size
     mismatch, expected-option mismatch)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

MAGIC = b"M1DUMP1\n"


def die(message: str) -> None:
    print(f"verify: error: {message}", file=sys.stderr)
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
    die("dav2d not found; set DAV2D=/path/to/dav2d")


@dataclass
class Dump:
    header: dict
    planes: dict[str, bytes]


def read_dump(path: Path) -> Dump:
    data = path.read_bytes()
    if not data.startswith(MAGIC):
        die(f"{path}: not a m1_dump file (bad magic)")
    rest = data[len(MAGIC):]
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
        die(f"{path}: payload is {len(payload)} bytes, expected {expected} "
            f"for {width}x{height} 4:2:0")
    planes = {}
    offset = 0
    for name in ("Y", "U", "V"):
        planes[name] = payload[offset : offset + sizes[name]]
        offset += sizes[name]
    return Dump(header=header, planes=planes)


def read_ivf_size(path: Path) -> tuple[int, int] | None:
    with path.open("rb") as f:
        head = f.read(32)
    if len(head) < 32 or head[:4] != b"DKIF":
        return None
    width, height = struct.unpack_from("<HH", head, 12)
    return width, height


def read_y4m_first_frame(path: Path) -> tuple[int, int, dict[str, bytes]]:
    with path.open("rb") as f:
        header_line = f.readline()
        if not header_line.startswith(b"YUV4MPEG2"):
            die(f"{path}: not a Y4M file")
        width = height = 0
        for token in header_line.split()[1:]:
            if token[:1] == b"W":
                width = int(token[1:])
            elif token[:1] == b"H":
                height = int(token[1:])
        if width <= 0 or height <= 0:
            die(f"{path}: Y4M header has no dimensions")
        frame_line = f.readline()
        if not frame_line.startswith(b"FRAME"):
            die(f"{path}: missing first FRAME marker")
        cw, ch = (width + 1) // 2, (height + 1) // 2
        sizes = {"Y": width * height, "U": cw * ch, "V": cw * ch}
        planes = {}
        for name in ("Y", "U", "V"):
            data = f.read(sizes[name])
            if len(data) != sizes[name]:
                die(f"{path}: truncated {name} plane "
                    f"({len(data)} of {sizes[name]} bytes)")
            planes[name] = data
    return width, height, planes


@dataclass
class PlaneStats:
    name: str
    width: int
    height: int
    sub_x: int
    sub_y: int
    total: int
    diffs: int = 0
    max_diff: int = 0
    first: tuple[int, int, int, int] | None = None  # x, y, enc, dec
    bbox: list[int] = field(default_factory=lambda: [1 << 30, 1 << 30, -1, -1])
    histogram: dict[int, int] = field(default_factory=dict)
    sb: dict[tuple[int, int], list[int]] = field(default_factory=dict)  # (sx,sy) -> [n, max]

    def note_diff(self, idx: int, d: int) -> None:
        x = idx % self.width
        y = idx // self.width
        self.diffs += 1
        absd = abs(d)
        if absd > self.max_diff:
            self.max_diff = absd
        self.histogram[d] = self.histogram.get(d, 0) + 1
        if self.first is None:
            self.first = (x, y, 0, 0)
        b = self.bbox
        if x < b[0]:
            b[0] = x
        if y < b[1]:
            b[1] = y
        if x > b[2]:
            b[2] = x
        if y > b[3]:
            b[3] = y
        sb_size_x = max(1, 64 >> self.sub_x)
        sb_size_y = max(1, 64 >> self.sub_y)
        key = (x // sb_size_x, y // sb_size_y)
        entry = self.sb.get(key)
        if entry is None:
            entry = [0, 0]
            self.sb[key] = entry
        entry[0] += 1
        if absd > entry[1]:
            entry[1] = absd

    def max_in_block(self, enc: bytes, dec: bytes, x0: int, y0: int,
                     size: int) -> int:
        best = 0
        x1 = min(x0 + size, self.width)
        y1 = min(y0 + size, self.height)
        for y in range(y0, y1):
            row = y * self.width
            for x in range(x0, x1):
                d = abs(enc[row + x] - dec[row + x])
                if d > best:
                    best = d
        return best


def compare_plane(name: str, enc: bytes, dec: bytes, width: int, height: int) -> PlaneStats:
    sub_x = 0 if name == "Y" else 1
    sub_y = 0 if name == "Y" else 1
    stats = PlaneStats(name=name, width=width, height=height, sub_x=sub_x,
                       sub_y=sub_y, total=width * height)
    # Fast path: identical planes need no scan.
    if enc == dec:
        return stats
    first_enc = first_dec = None
    for idx, (a, b) in enumerate(zip(enc, dec)):
        if a == b:
            continue
        stats.note_diff(idx, a - b)
        if first_enc is None:
            first_enc, first_dec = a, b
    assert stats.first is not None
    x, y = stats.first[0], stats.first[1]
    stats.first = (x, y, first_enc, first_dec)
    return stats


def luma_coords(stats: PlaneStats, x: int, y: int) -> tuple[int, int]:
    return x << stats.sub_x, y << stats.sub_y


def print_plane_report(stats: PlaneStats, enc: bytes, dec: bytes,
                       limit_sb: int) -> None:
    if stats.diffs == 0:
        print(f"  plane {stats.name}: exact ({stats.total} samples, max|diff|=0)")
        return
    pct = 100.0 * stats.diffs / stats.total
    print(f"  plane {stats.name}: {stats.diffs}/{stats.total} differ "
          f"({pct:.3f}%), max|diff|={stats.max_diff}")
    x, y, a, b = stats.first
    lx, ly = luma_coords(stats, x, y)
    suffix = "" if stats.name == "Y" else f" [luma {lx},{ly}]"
    print(f"    first sample:  ({x}, {y}) enc={a} dec={b} d={a - b}{suffix}")
    box = stats.bbox
    print(f"    diff bbox:     x {box[0]}..{box[2]}, y {box[1]}..{box[3]}")
    for size in (4, 8):
        bx, by = (x // size) * size, (y // size) * size
        block_max = stats.max_in_block(enc, dec, bx, by, size)
        blx, bly = luma_coords(stats, bx, by)
        extra = "" if stats.name == "Y" else f" [luma {blx},{bly}]"
        print(f"    first {size}x{size}:     block ({bx // size}, {by // size}) "
              f"at ({bx}, {by}) max|diff|={block_max}{extra}")
    items = sorted(stats.sb.items(), key=lambda kv: (-kv[1][0], kv[0][1], kv[0][0]))
    print(f"    64x64 superblocks with diffs: {len(items)}"
          + (f" (top {limit_sb})" if len(items) > limit_sb else ""))
    for (sx, sy), (count, maxd) in items[:limit_sb]:
        print(f"      sb ({sx}, {sy}): {count} diffs, max|diff|={maxd}")
    hist = sorted(stats.histogram.items(), key=lambda kv: (-kv[1], kv[0]))
    shown = ", ".join(f"{d:+d}:{n}" for d, n in hist[:8])
    if len(hist) > 8:
        shown += ", ..."
    print(f"    deltas:        {shown}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare M1 encoder reconstruction (m1_dump) with the "
                    "dav2d-decoded IVF, bit-exactly.")
    parser.add_argument("--ivf", required=True, help="IVF written by av2enc")
    parser.add_argument("--dump", required=True, help="dump written by m1_dump")
    parser.add_argument("--decoder", default=None, help="dav2d path")
    parser.add_argument("--keep-dec", metavar="PATH",
                        help="keep the decoded Y4M at PATH")
    parser.add_argument("--tsv", action="store_true",
                        help="print one machine-readable TSV line only")
    parser.add_argument("--limit-sb", type=int, default=8,
                        help="max superblocks to list per plane (default 8)")
    parser.add_argument("--expect-depth", type=int, default=None)
    parser.add_argument("--expect-deblock", type=int, choices=(0, 1), default=None)
    parser.add_argument("--expect-full-intra", type=int, choices=(0, 1), default=None)
    args = parser.parse_args()

    ivf = Path(args.ivf)
    dump_path = Path(args.dump)
    if not ivf.is_file():
        die(f"no such IVF: {ivf}")
    if not dump_path.is_file():
        die(f"no such dump: {dump_path}")

    dump = read_dump(dump_path)
    header = dump.header
    width = int(header["width"])
    height = int(header["height"])
    cw = int(header.get("chroma_width", (width + 1) // 2))
    ch = int(header.get("chroma_height", (height + 1) // 2))

    if args.expect_depth is not None and int(header.get("max_partition_depth", -1)) != args.expect_depth:
        die(f"dump depth is {header.get('max_partition_depth')}, expected {args.expect_depth}; "
            "the harness was run with different options than the CLI")
    if args.expect_deblock is not None and int(bool(header.get("enable_deblock"))) != args.expect_deblock:
        die(f"dump enable_deblock is {header.get('enable_deblock')}, expected {args.expect_deblock}")
    if args.expect_full_intra is not None and int(not header.get("luma_dc_only", True)) != args.expect_full_intra:
        die(f"dump luma_dc_only is {header.get('luma_dc_only')}, expected full_intra={args.expect_full_intra}")

    ivf_size = read_ivf_size(ivf)
    if ivf_size is not None and ivf_size != (width, height):
        die(f"IVF is {ivf_size[0]}x{ivf_size[1]} but dump is {width}x{height}")

    decoder = find_decoder(args.decoder)
    tmpdir = tempfile.mkdtemp(prefix="m1_verify_")
    dec_y4m = Path(tmpdir) / "decoded.y4m"
    try:
        cmd = [str(decoder), "-q", "-i", str(ivf), "-o", str(dec_y4m)]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0 or not dec_y4m.exists() or dec_y4m.stat().st_size == 0:
            sys.stderr.write(proc.stderr.decode("utf-8", "replace"))
            die(f"dav2d failed ({proc.returncode}) on {ivf}")
        if args.keep_dec:
            keep = Path(args.keep_dec)
            keep.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(dec_y4m, keep)
        dec_w, dec_h, dec_planes = read_y4m_first_frame(dec_y4m)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    if (dec_w, dec_h) != (width, height):
        die(f"decoded frame is {dec_w}x{dec_h}, dump is {width}x{height}")

    enc_sizes = {"Y": (width, height), "U": (cw, ch), "V": (cw, ch)}
    stats = []
    for name in ("Y", "U", "V"):
        w, h = enc_sizes[name]
        stats.append(compare_plane(name, dump.planes[name], dec_planes[name], w, h))

    total_diffs = sum(s.diffs for s in stats)
    max_diff = max(s.max_diff for s in stats)
    first_plane = next((s for s in stats if s.diffs), None)
    status = "BIT-EXACT" if total_diffs == 0 else "MISMATCH"

    if args.tsv:
        first_x = first_y = -1
        if first_plane is not None:
            first_x, first_y = luma_coords(first_plane, first_plane.first[0],
                                           first_plane.first[1])
        print("\t".join([
            status,
            str(stats[0].diffs), str(stats[1].diffs), str(stats[2].diffs),
            str(max_diff), str(first_x), str(first_y),
            first_plane.name if first_plane else "-",
        ]))
    else:
        opts = []
        opts.append(f"qp={header.get('base_q_idx')}")
        opts.append(f"depth={header.get('max_partition_depth')}"
                    + ("" if header.get("depth_overridden") else "(default)"))
        opts.append(f"deblock={int(bool(header.get('enable_deblock')))}")
        opts.append(f"cfl={int(bool(header.get('enable_cfl')))}")
        opts.append(f"full_intra={int(not header.get('luma_dc_only', True))}")
        opts.append(f"single={int(bool(header.get('force_single_block')))}")
        opts.append(f"dc={int(bool(header.get('force_dc')))}")
        env = header.get("env") or {}
        env_text = " ".join(f"{k.replace('AV2ENC_M1_', '')}={v}"
                            for k, v in sorted(env.items())) or "-"
        print(f"dump:    {width}x{height}  " + " ".join(opts)
              + f"  tile={header.get('tile_bytes')}B frame={header.get('frame_bytes')}B")
        print(f"env:     {env_text}")
        print(f"ivf:     {ivf}")
        print(f"decoder: {decoder}")
        for s in stats:
            print_plane_report(s, dump.planes[s.name], dec_planes[s.name],
                               args.limit_sb)
        if total_diffs == 0:
            print(f"RESULT:  {status} (0 differing samples)")
        else:
            print(f"RESULT:  {status} {total_diffs} differing samples "
                  f"(Y {stats[0].diffs}, U {stats[1].diffs}, V {stats[2].diffs}), "
                  f"max|diff|={max_diff}")
    return 0 if total_diffs == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
