#!/usr/bin/env python3
"""Tail (pictures 33-59) byte / residual / skip / transform statistics.

Reads the encoder's existing diagnostics:
  AV2ENC_BIT_CENSUS=<path>     one JSON line per tile (payload_bytes, classes)
  AV2ENC_DECISION_DUMP=<path>  one JSON line per coded leaf (skip, eob, tx)

  tools/lane-tail/tail_stats.py --census base.census.jsonl --decisions base.dec \
      --json base.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", required=True)
    ap.add_argument("--decisions", required=True)
    ap.add_argument("--json")
    args = ap.parse_args()

    census = load_jsonl(args.census)
    decisions = load_jsonl(args.decisions)

    total_bytes = sum(row.get("payload_bytes", 0) for row in census)
    tail_bytes = sum(
        row.get("payload_bytes", 0)
        for row in census
        if 33 <= row.get("order_hint", -1) <= 59
    )
    total_bits = sum(row.get("info_bits", 0.0) for row in census)
    tail_bits = sum(
        row.get("info_bits", 0.0)
        for row in census
        if 33 <= row.get("order_hint", -1) <= 59
    )
    total_res = sum(
        row.get("classes", {}).get("residual", [0, 0.0])[1] for row in census
    )
    tail_res = sum(
        row.get("classes", {}).get("residual", [0, 0.0])[1]
        for row in census
        if 33 <= row.get("order_hint", -1) <= 59
    )
    tail_partition = sum(
        row.get("classes", {}).get("partition", [0, 0.0])[1]
        for row in census
        if 33 <= row.get("order_hint", -1) <= 59
    )
    tail_mode = sum(
        row.get("classes", {}).get("inter_mode", [0, 0.0])[1]
        for row in census
        if 33 <= row.get("order_hint", -1) <= 59
    )

    tail_leaves = [d for d in decisions if 33 <= d.get("order_hint", -1) <= 59]
    inter = [d for d in tail_leaves if d.get("is_inter")]
    skip = [d for d in inter if d.get("skip")]
    resid = [d for d in inter if not d.get("skip")]
    coded_y = [d for d in inter if d.get("eob_y", 0) > 0]
    tx_sizes = Counter()
    tx_parts = Counter()
    for d in resid:
        size = d.get("tx_size", 0)
        tx_sizes[size] += 1
        tx_parts[d.get("tx_partition", 0)] += 1

    out = {
        "frames": len(census),
        "payload_bytes_total": total_bytes,
        "payload_bytes_tail": tail_bytes,
        "tail_byte_share": tail_bytes / total_bytes if total_bytes else None,
        "info_bits_total": total_bits,
        "info_bits_tail": tail_bits,
        "residual_bits_total": total_res,
        "residual_bits_tail": tail_res,
        "residual_share_of_tail_bits": tail_res / tail_bits if tail_bits else None,
        "partition_bits_tail": tail_partition,
        "inter_mode_bits_tail": tail_mode,
        "tail_leaves": len(tail_leaves),
        "tail_inter_leaves": len(inter),
        "tail_skip_leaves": len(skip),
        "tail_skip_share": len(skip) / len(inter) if inter else None,
        "tail_residual_leaves": len(resid),
        "tail_residual_leaf_share": len(resid) / len(inter) if inter else None,
        "tail_coded_luma_leaves": len(coded_y),
        "tail_coded_luma_share": len(coded_y) / len(inter) if inter else None,
        "tail_tx_sizes": dict(sorted(tx_sizes.items())),
        "tail_tx_partitions": dict(sorted(tx_parts.items())),
    }
    print(json.dumps(out, indent=2))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(out, handle, indent=2)
            handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
