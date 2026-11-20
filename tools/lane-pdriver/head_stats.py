#!/usr/bin/env python3
"""Head/tail byte and skip-share statistics for the P-driver lane.

Reads the encoder's own `frame bits:` line (payload only, headers excluded) and,
when present, the AV2ENC_DECISION_DUMP leaf stream.  Prints per-segment bytes,
bytes per picture and the inter-leaf skip share, which is what the head
under-spend claim in docs/rd-pdriver.md is measured on.

  tools/lane-pdriver/head_stats.py --segment head one=/tmp/a.log two=/tmp/b.log
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

SEGMENTS = {
    "key": (0, 0),
    "head": (1, 32),
    "mid": (33, 43),
    "tail": (44, 59),
    "all": (0, 59),
}

BSIZE_NAMES = {
    0: "4x4", 1: "4x8", 2: "8x4", 3: "8x8", 4: "8x16", 5: "16x8", 6: "16x16",
    7: "16x32", 8: "32x16", 9: "32x32", 10: "32x64", 11: "64x32", 12: "64x64",
    19: "4x16", 20: "16x4", 21: "8x32", 22: "32x8", 23: "16x64", 24: "64x16",
}


def frame_bits(path: Path) -> list[int]:
    for line in path.read_text(errors="replace").splitlines():
        if "frame bits:" in line:
            return [int(x) for x in line.split("frame bits:")[1].split()]
    raise SystemExit(f"{path}: no 'frame bits:' line")


def load_decisions(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--segment", default="head", choices=sorted(SEGMENTS))
    ap.add_argument("--pair", action="append", default=[], metavar="LABEL=LOG")
    ap.add_argument("--decisions", action="append", default=[],
                    metavar="LABEL=PATH")
    args = ap.parse_args()

    start, end = SEGMENTS[args.segment]
    span = end - start + 1
    for spec in args.pair:
        label, raw = spec.split("=", 1)
        bits = frame_bits(Path(raw))
        seg = sum(bits[start:end + 1])
        print(f"{label:10s} {args.segment:5s} payload {seg:7d} B  "
              f"{seg / span:7.1f} B/picture  total {sum(bits):7d} B")

    for spec in args.decisions:
        label, raw = spec.split("=", 1)
        rows = load_decisions(Path(raw))
        seg = [r for r in rows if start <= r["order_hint"] <= end]
        inter = [r for r in seg if r["is_inter"]]
        skip = sum(1 for r in inter if r["skip"])
        zero_eob = sum(1 for r in inter
                       if not r["skip"] and r["eob_y"] == 0
                       and r["eob_u"] == 0 and r["eob_v"] == 0)
        coded = sum(1 for r in inter if not r["skip"] and r["eob_y"] > 0)
        print(f"{label:10s} {args.segment:5s} leaves {len(seg):5d} "
              f"inter {len(inter):5d} skip {skip:4d} "
              f"({100.0 * skip / max(len(inter), 1):5.1f}%) "
              f"residual-arm {len(inter) - skip:4d} "
              f"eob_y>0 {coded:4d} all-zero {zero_eob:4d}")
        print(f"{'':10s} {args.segment:5s} inter sizes "
              f"{dict(Counter(BSIZE_NAMES.get(r['bsize'], r['bsize']) for r in inter).most_common(6))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
