#!/usr/bin/env python3
"""Compare two IVF files byte-for-byte at the **frame payload** level.

This is the hard gate for the M1 GPU path. PSNR is deliberately not used: a
stream can stay above 50 dB while a block has desynced, and a CPU-fallback GPU
run would trivially "pass" a quality check. The GPU path must reproduce the
CPU reference frame bytes exactly.

What is compared
----------------
* IVF header geometry (fourcc, width, height, frame rate) and frame count.
* Every frame payload (the OBU temporal unit), byte for byte.
* MD5 of each payload and of the complete file.

The 12-byte IVF frame header (payload size + timestamp) is reported but not
part of the byte gate; only the payloads are the bitstream. Timestamps are
informational because they are not normative stream data.

Exit codes: 0 payloads identical, 1 any difference, 2 usage/setup error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path

IVF_HEADER_BYTES = 32
IVF_FRAME_HEADER_BYTES = 12


def die(message: str) -> "NoReturn":  # noqa: F821 - string annotation
    print(f"ivf_compare: error: {message}", file=sys.stderr)
    raise SystemExit(2)


@dataclass
class Ivf:
    path: Path
    file_bytes: int
    header_len: int
    fourcc: str
    width: int
    height: int
    rate_num: int
    rate_den: int
    declared_frames: int
    timestamps: list = field(default_factory=list)
    payloads: list = field(default_factory=list)
    trailing: int = 0

    def file_md5(self) -> str:
        digest = hashlib.md5()
        with self.path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()


def load_ivf(path: Path) -> Ivf:
    data = path.read_bytes()
    if len(data) < IVF_HEADER_BYTES or data[:4] != b"DKIF":
        die(f"{path}: not an IVF file (missing DKIF header)")
    header_len = struct.unpack_from("<H", data, 6)[0]
    fourcc = data[8:12].decode("latin-1")
    width, height = struct.unpack_from("<HH", data, 12)
    rate_num, rate_den, declared = struct.unpack_from("<III", data, 16)
    if header_len < IVF_HEADER_BYTES or header_len > len(data):
        die(f"{path}: bad IVF header length {header_len}")

    payloads = []
    timestamps = []
    offset = header_len
    while offset + IVF_FRAME_HEADER_BYTES <= len(data):
        size = struct.unpack_from("<I", data, offset)[0]
        timestamp = struct.unpack_from("<Q", data, offset + 4)[0]
        start = offset + IVF_FRAME_HEADER_BYTES
        end = start + size
        if size == 0 and end > len(data):
            break
        if end > len(data):
            die(f"{path}: truncated frame payload at offset {offset} "
                f"(want {size} bytes, have {len(data) - start})")
        payloads.append(data[start:end])
        timestamps.append(timestamp)
        offset = end

    return Ivf(
        path=path,
        file_bytes=len(data),
        header_len=header_len,
        fourcc=fourcc,
        width=width,
        height=height,
        rate_num=rate_num,
        rate_den=rate_den,
        declared_frames=declared,
        timestamps=timestamps,
        payloads=payloads,
        trailing=len(data) - offset,
    )


def md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def hexdump(data: bytes, start: int, length: int = 16) -> str:
    chunk = data[start:start + length]
    return " ".join(f"{byte:02x}" for byte in chunk)


def first_difference(a: bytes, b: bytes):
    """Return (offset, a_byte, b_byte, differing_count) for the payloads."""
    limit = min(len(a), len(b))
    first = None
    differing = 0
    for index in range(limit):
        if a[index] != b[index]:
            differing += 1
            if first is None:
                first = (index, a[index], b[index])
    differing += abs(len(a) - len(b))
    if first is None and len(a) != len(b):
        index = limit
        first = (index, a[index] if index < len(a) else -1,
                 b[index] if index < len(b) else -1)
    return first, differing


def compare(a: Ivf, b: Ivf, limit_report: int = 8) -> dict:
    problems = []
    if a.fourcc != b.fourcc:
        problems.append(f"fourcc {a.fourcc!r} vs {b.fourcc!r}")
    if (a.width, a.height) != (b.width, b.height):
        problems.append(f"size {a.width}x{a.height} vs {b.width}x{b.height}")
    if (a.rate_num, a.rate_den) != (b.rate_num, b.rate_den):
        problems.append(f"frame rate {a.rate_num}/{a.rate_den} vs "
                        f"{b.rate_num}/{b.rate_den}")
    if len(a.payloads) != len(b.payloads):
        problems.append(f"frame count {len(a.payloads)} vs {len(b.payloads)}")

    frames = []
    differing_frames = 0
    differing_bytes = 0
    for index in range(min(len(a.payloads), len(b.payloads))):
        left = a.payloads[index]
        right = b.payloads[index]
        same = left == right
        entry = {
            "frame": index,
            "bytes": len(left),
            "bytes_other": len(right),
            "md5": md5(left),
            "md5_other": md5(right),
            "identical": same,
        }
        if not same:
            differing_frames += 1
            first, count = first_difference(left, right)
            differing_bytes += count
            entry["first_diff_offset"] = first[0]
            entry["first_diff_cpu"] = first[1]
            entry["first_diff_gpu"] = first[2]
            entry["differing_bytes"] = count
            entry["context_cpu"] = hexdump(left, max(0, first[0] - 8), 16)
            entry["context_gpu"] = hexdump(right, max(0, first[0] - 8), 16)
        frames.append(entry)

    total_payload = sum(len(p) for p in a.payloads)
    return {
        "ok": not problems and differing_frames == 0,
        "problems": problems,
        "frames": frames,
        "differing_frames": differing_frames,
        "differing_bytes": differing_bytes,
        "payload_bytes": total_payload,
        "file_md5_cpu": a.file_md5(),
        "file_md5_gpu": b.file_md5(),
        "_limit_report": limit_report,
    }


def print_report(a: Ivf, b: Ivf, result: dict,
                 label_a: str = "cpu", label_b: str = "gpu") -> None:
    print(f"{label_a} ivf:{' ' * max(1, 6 - len(label_a))}{a.path}  "
          f"({a.width}x{a.height} {a.fourcc}, "
          f"{len(a.payloads)} frame(s), {a.file_bytes} B)")
    print(f"{label_b} ivf:{' ' * max(1, 6 - len(label_b))}{b.path}  "
          f"({b.width}x{b.height} {b.fourcc}, "
          f"{len(b.payloads)} frame(s), {b.file_bytes} B)")
    if a.declared_frames != len(a.payloads) or b.declared_frames != len(b.payloads):
        print(f"note:     IVF header declares {a.declared_frames}/"
              f"{b.declared_frames} frames, found {len(a.payloads)}/"
              f"{len(b.payloads)}")

    for problem in result["problems"]:
        print(f"  MISMATCH {problem}")

    limited = result["_limit_report"]
    shown = 0
    for entry in result["frames"]:
        if not entry["identical"] or shown < limited:
            status = "IDENTICAL" if entry["identical"] else "DIFFERS"
            print(f"  frame {entry['frame']}: payload {entry['bytes']} B  "
                  f"md5 {label_a}={entry['md5']} {label_b}={entry['md5_other']}  "
                  f"{status}")
            if not entry["identical"]:
                print(f"    first byte offset {entry['first_diff_offset']} "
                      f"(0x{entry['first_diff_offset']:x}): "
                      f"{label_a}={entry['first_diff_cpu']} "
                      f"{label_b}={entry['first_diff_gpu']}, "
                      f"{entry['differing_bytes']} byte(s) differ")
                print(f"    {label_a} ctx: {entry['context_cpu']}")
                print(f"    {label_b} ctx: {entry['context_gpu']}")
            shown += 1
        if shown >= limited and not entry["identical"]:
            remaining = result["differing_frames"] - 1
            if remaining > 0:
                print(f"  ... {remaining} more differing frame(s)")
            break

    print(f"  payload bytes: {result['payload_bytes']}")
    print(f"  file md5:      {label_a}={result['file_md5_cpu']} "
          f"{label_b}={result['file_md5_gpu']}")
    if result["ok"]:
        print(f"RESULT: IVF FRAME BYTES IDENTICAL "
              f"({len(a.payloads)} frame(s), {result['payload_bytes']} bytes)")
    else:
        print(f"RESULT: IVF FRAME BYTES DIFFER "
              f"({result['differing_frames']} frame(s), "
              f"{result['differing_bytes']} payload byte(s))")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Byte-exact IVF frame-payload comparison (no PSNR).")
    parser.add_argument("cpu", type=Path, help="reference IVF (CPU M1)")
    parser.add_argument("gpu", type=Path, help="IVF under test (GPU M1)")
    parser.add_argument("--json", action="store_true",
                        help="print a machine-readable JSON result")
    parser.add_argument("--limit-report", type=int, default=8,
                        help="max frame lines to print in detail (default 8)")
    parser.add_argument("--label-a", default="cpu",
                        help="label for the first file (default cpu)")
    parser.add_argument("--label-b", default="gpu",
                        help="label for the second file (default gpu)")
    args = parser.parse_args()

    for path in (args.cpu, args.gpu):
        if not path.is_file():
            die(f"no such file: {path}")

    a = load_ivf(args.cpu)
    b = load_ivf(args.gpu)
    result = compare(a, b, args.limit_report)

    if args.json:
        payload = {key: value for key, value in result.items()
                   if not key.startswith("_")}
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print_report(a, b, result, args.label_a, args.label_b)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
