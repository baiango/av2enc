#!/usr/bin/env python3
"""Compare two AV2ENC diagnostic dumps: tile symbol bits and leaf decisions.

The encoder writes these only when the env vars are set, and neither changes
the bitstream:

    AV2ENC_BIT_CENSUS=/tmp/base.jsonl
    AV2ENC_DECISION_DUMP=/tmp/base.dec
    build-release/av2enc clip.y4m -o /tmp/base.ivf --video ... --qp Q

`AV2ENC_BIT_CENSUS` appends one JSON object per tile: the information bits the
§8.2 coder charged, split by symbol family (`classes`) and by the residual
sub-families (`fine`).  `AV2ENC_DECISION_DUMP` appends one JSON object per
coded leaf with its mode/MV/TX/EOB decision.

Usage:
    # bit movement between two runs at the SAME QP
    python3 tools/bit_census.py --base base.jsonl --cand cand.jsonl
    python3 tools/bit_census.py --base base.jsonl --cand cand.jsonl --top 8

    # decision flips between two runs at the SAME QP
    python3 tools/bit_census.py --decisions base.dec cand.dec

    # both
    python3 tools/bit_census.py --base base.jsonl --cand cand.jsonl \
        --decisions base.dec cand.dec

The comparison is only meaningful at a fixed QP (or for two arms whose QP
ladder point is the same); the tool prints the payload totals so a QP mismatch
is visible.  It never reads or writes the bitstream.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def census_totals(rows: list[dict]) -> dict:
    classes: dict[str, list[float]] = defaultdict(lambda: [0, 0.0])
    fine: dict[str, list[float]] = defaultdict(lambda: [0, 0.0])
    per_frame: dict[int, float] = defaultdict(float)
    payload = 0
    info = 0.0
    for r in rows:
        payload += int(r.get("payload_bytes", 0))
        bits = float(r.get("info_bits", 0.0))
        info += bits
        per_frame[int(r.get("order_hint", -1))] += bits
        for name, (n, b) in (r.get("classes") or {}).items():
            classes[name][0] += n
            classes[name][1] += b
        for name, (n, b) in (r.get("fine") or {}).items():
            fine[name][0] += n
            fine[name][1] += b
    return {
        "classes": classes,
        "fine": fine,
        "per_frame": per_frame,
        "payload_bytes": payload,
        "info_bits": info,
        "tiles": len(rows),
    }


def fmt_delta(a: float, b: float) -> str:
    d = b - a
    pct = (100.0 * d / a) if a else 0.0
    return f"{a:10.1f} {b:10.1f} {d:+10.1f} {pct:+7.2f}%"


def print_table(title: str, a: dict, b: dict) -> None:
    names = sorted(set(a) | set(b), key=lambda k: -(max(a.get(k, [0, 0.0])[1],
                                                      b.get(k, [0, 0.0])[1])))
    print(f"\n{title}")
    print(f"  {'family':14s} {'base bits':>10s} {'cand bits':>10s} "
          f"{'delta':>10s}  {'delta%':>7s}")
    for name in names:
        an, ab = a.get(name, [0, 0.0])
        bn, bb = b.get(name, [0, 0.0])
        if ab == 0 and bb == 0:
            continue
        print(f"  {name:14s} {fmt_delta(ab, bb)}   "
              f"n {int(an)}->{int(bn)}")


def compare_census(base_rows: list[dict], cand_rows: list[dict], top: int) -> None:
    base = census_totals(base_rows)
    cand = census_totals(cand_rows)
    print("bit census (information bits, one row per tile)")
    print(f"  tiles          base {base['tiles']}  cand {cand['tiles']}")
    print(f"  payload bytes  base {base['payload_bytes']}  "
          f"cand {cand['payload_bytes']}  "
          f"delta {cand['payload_bytes'] - base['payload_bytes']:+d}")
    print(f"  info bits      base {base['info_bits']:.0f}  "
          f"cand {cand['info_bits']:.0f}  "
          f"delta {cand['info_bits'] - base['info_bits']:+.0f} "
          f"({(cand['info_bits'] - base['info_bits']) / 8.0:+.1f} B)")
    print_table("classes", base["classes"], cand["classes"])
    print_table("residual detail", base["fine"], cand["fine"])

    frames = sorted(set(base["per_frame"]) | set(cand["per_frame"]))
    deltas = [(f, base["per_frame"].get(f, 0.0), cand["per_frame"].get(f, 0.0))
              for f in frames]
    deltas = [d for d in deltas if abs(d[2] - d[1]) > 1e-6]
    deltas.sort(key=lambda d: -abs(d[2] - d[1]))
    if deltas:
        print(f"\nper-frame bit deltas (top {min(top, len(deltas))})")
        print(f"  {'order_hint':>10s} {'base bits':>10s} {'cand bits':>10s} "
              f"{'delta B':>10s}")
        for f, a, b in deltas[:top]:
            print(f"  {f:10d} {a:10.0f} {b:10.0f} {(b - a) / 8.0:+10.1f}")
    else:
        print("\nper-frame: no change")


DECISION_FIELDS = (
    "bsize", "partition", "is_inter", "skip", "inter_mode", "ref", "compound",
    "mv_y", "mv_x", "mvd_y", "mvd_x", "y_mode", "uv_mode", "tx_size",
    "tx_type", "tx_partition", "eob_y", "eob_u", "eob_v", "segment",
    "qindex",
)


def decision_key(row: dict) -> tuple:
    return (int(row.get("order_hint", -1)),
            int(row.get("tile_mi_row", 0)),
            int(row.get("tile_mi_col", 0)),
            int(row.get("mi_row", -1)),
            int(row.get("mi_col", -1)))


def compare_decisions(base_rows: list[dict], cand_rows: list[dict],
                      top: int) -> None:
    base = {decision_key(r): r for r in base_rows}
    cand = {decision_key(r): r for r in cand_rows}
    missing = sorted(set(base) - set(cand))
    added = sorted(set(cand) - set(base))
    common = set(base) & set(cand)
    print("\nleaf decisions (same QP)")
    print(f"  leaves         base {len(base)}  cand {len(cand)}  "
          f"matched {len(common)}  removed {len(missing)}  added {len(added)}")

    flips: dict[str, int] = defaultdict(int)
    by_frame: dict[int, int] = defaultdict(int)
    examples: dict[str, list[tuple]] = defaultdict(list)
    for key in common:
        a = base[key]
        b = cand[key]
        frame = key[0]
        changed = False
        for field in DECISION_FIELDS:
            if a.get(field) != b.get(field):
                flips[field] += 1
                changed = True
                if len(examples[field]) < 3:
                    examples[field].append(
                        (key, a.get(field), b.get(field)))
        if changed:
            by_frame[frame] += 1

    if not flips:
        print("  decision fields: identical")
    else:
        print(f"  {'field':14s} {'flips':>8s}   examples (key, base -> cand)")
        for field, n in sorted(flips.items(), key=lambda kv: -kv[1]):
            ex = examples[field][0]
            print(f"  {field:14s} {n:8d}   {ex[0]} {ex[1]} -> {ex[2]}")
        frames = sorted(by_frame.items(), key=lambda kv: -kv[1])[:top]
        print(f"\n  leaves changed per frame (top {len(frames)})")
        for frame, n in frames:
            print(f"    order_hint {frame:4d}: {n}")
    if missing or added:
        print(f"\n  first removed: {missing[:3]}")
        print(f"  first added:   {added[:3]}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, help="baseline AV2ENC_BIT_CENSUS file")
    ap.add_argument("--cand", type=Path, help="candidate AV2ENC_BIT_CENSUS file")
    ap.add_argument("--decisions", nargs=2, type=Path, metavar=("BASE", "CAND"),
                    help="two AV2ENC_DECISION_DUMP files")
    ap.add_argument("--top", type=int, default=10,
                    help="rows in the per-frame tables (default 10)")
    args = ap.parse_args()

    if not args.base and not args.decisions:
        ap.error("give --base/--cand and/or --decisions")

    if args.base:
        if not args.cand:
            ap.error("--base requires --cand")
        compare_census(load_jsonl(args.base), load_jsonl(args.cand), args.top)
    if args.decisions:
        compare_decisions(load_jsonl(args.decisions[0]),
                          load_jsonl(args.decisions[1]), args.top)
    return 0


if __name__ == "__main__":
    sys.exit(main())
