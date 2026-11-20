#!/usr/bin/env python3
"""Tail coding-block / TX census for docs/rd-tail-floor.md.

Reads AV2ENC_DECISION_DUMP (one JSON line per coded leaf, pictures are
`order_hint`) and reports, for pictures 33-59:

  * coding-block size histogram (16x16, 16x8/8x16, 8x8, 4x4, ...)
  * inter leaf count, skip share, residual-leaf and coded-luma shares
  * luma TX size histogram of the residual leaves (8x8 / 8x4 / 4x4 ...)

    tools/lane-tail-floor/floor_stats.py --decisions cand.dec --json cand.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

BSIZE = {
    0: "4x4", 1: "4x8", 2: "8x4", 3: "8x8", 4: "8x16", 5: "16x8",
    6: "16x16", 7: "16x32", 8: "32x16", 9: "32x32", 10: "32x64",
    11: "64x32", 12: "64x64", 13: "64x128", 14: "128x64", 15: "128x128",
    19: "4x16", 20: "16x4", 21: "8x32", 22: "32x8", 23: "16x64",
    24: "64x16", 25: "4x32", 26: "32x4", 27: "8x64", 28: "64x8",
}
TXSIZE = {
    0: "4x4", 1: "8x8", 2: "16x16", 3: "32x32", 4: "64x64", 5: "4x8",
    6: "8x4", 7: "8x16", 8: "16x8", 9: "16x32", 10: "32x16", 11: "32x64",
    12: "64x32", 13: "4x16", 14: "16x4", 15: "8x32", 16: "32x8",
    17: "16x64", 18: "64x16", 19: "4x32", 20: "32x4", 21: "8x64",
    22: "64x8", 23: "4x64", 24: "64x4",
}


def load(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", required=True)
    ap.add_argument("--first", type=int, default=33)
    ap.add_argument("--last", type=int, default=59)
    ap.add_argument("--json")
    args = ap.parse_args()

    rows = [
        d for d in load(args.decisions)
        if args.first <= d.get("order_hint", -1) <= args.last
        and d.get("partition") == 0  # leaves only
    ]
    inter = [d for d in rows if d.get("is_inter")]
    skip = [d for d in inter if d.get("skip")]
    resid = [d for d in inter if not d.get("skip") and d.get("eob_y", 0) > 0]
    bsize = Counter(BSIZE.get(d.get("bsize"), str(d.get("bsize"))) for d in rows)
    bsize_inter = Counter(
        BSIZE.get(d.get("bsize"), str(d.get("bsize"))) for d in inter)
    tx = Counter(TXSIZE.get(d.get("tx_size"), str(d.get("tx_size"))) for d in resid)
    tx_all = Counter(
        TXSIZE.get(d.get("tx_size"), str(d.get("tx_size")))
        for d in inter if not d.get("skip"))
    out = {
        "frames": args.last - args.first + 1,
        "leaves": len(rows),
        "inter_leaves": len(inter),
        "skip_leaves": len(skip),
        "skip_share": len(skip) / len(inter) if inter else None,
        "coded_luma_leaves": len(resid),
        "coded_luma_share": len(resid) / len(inter) if inter else None,
        "bsize": dict(sorted(bsize.items())),
        "bsize_inter": dict(sorted(bsize_inter.items())),
        "tx_size_coded_luma": dict(sorted(tx.items())),
        "tx_size_residual_inter": dict(sorted(tx_all.items())),
        "eob_luma_total": sum(d.get("eob_y", 0) for d in inter),
        "mv_nonzero": sum(
            1 for d in inter if d.get("mv_y", 0) or d.get("mv_x", 0)),
    }
    print(json.dumps(out, indent=2))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(out, handle, indent=2)
            handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
