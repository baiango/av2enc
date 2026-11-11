#!/usr/bin/env python3
"""Compare a plan TSV dump against the CPU M1 trace.

Usage:
    compare_trace.py plan.tsv trace.log [--verbose]          # block order
    compare_trace.py --nodes tree.tsv trace.log [--verbose]  # preorder nodes

The trace is produced by the CPU reference:

    AV2ENC_M1_TRACE=1 ./build-m1/av2enc frame.y4m -o /tmp/out.ivf \
        --m1 --frames 1 2> trace.log

Block mode compares the leaf coding order: (mi_row, mi_col, block size).
Node mode compares the full partition preorder: (mi_row, mi_col, block size,
partition, forced).  The current CPU trace prints the tree twice (decision pass
and syntax-writer pass); node mode checks both passes against the same plan.

Exit status: 0 = identical order, 1 = mismatch or parse error.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

BLOCK_RE = re.compile(r"\bm1 block r=(\d+) c=(\d+) size=(\S+)")
NODE_RE = re.compile(
    r"\bm1 node r=(\d+) c=(\d+) size=(\S+) part=(\S+) forced=(\d+)"
)


def read_plan(
    path: Path, columns: tuple[str, ...], numeric: tuple[str, ...]
) -> list[tuple]:
    rows: list[tuple] = []
    header: list[str] | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if header is None:
                header = fields
                missing = set(columns).difference(header)
                if missing:
                    raise ValueError(f"plan header missing {sorted(missing)}")
                continue
            row = dict(zip(header, fields))
            try:
                values = []
                for column in columns:
                    values.append(
                        int(row[column]) if column in numeric else row[column]
                    )
                rows.append(tuple(values))
            except (KeyError, ValueError) as exc:
                raise ValueError(f"bad plan row: {line!r}") from exc
    if header is None:
        raise ValueError("plan file has no column header")
    return rows


def read_trace_blocks(path: Path) -> list[tuple]:
    rows: list[tuple] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = BLOCK_RE.search(line)
            if match:
                rows.append(
                    (int(match.group(1)), int(match.group(2)), match.group(3))
                )
    return rows


def read_trace_nodes(path: Path) -> list[tuple]:
    rows: list[tuple] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = NODE_RE.search(line)
            if match:
                rows.append(
                    (
                        int(match.group(1)),
                        int(match.group(2)),
                        match.group(3),
                        match.group(4),
                        int(match.group(5)),
                    )
                )
    return rows


def split_writer_passes(rows: list[tuple]) -> list[list[tuple]]:
    """The node trace repeats the same preorder for decision and writer."""
    if len(rows) % 2 == 0:
        half = len(rows) // 2
        if rows[:half] == rows[half:]:
            return [rows[:half], rows[half:]]
    return [rows]


def compare(
    label: str, plan_rows: list[tuple], trace_passes: list[list[tuple]]
) -> bool:
    ok = True
    for pass_index, trace_rows in enumerate(trace_passes):
        suffix = (
            f" (pass {pass_index + 1}/{len(trace_passes)})"
            if len(trace_passes) > 1
            else ""
        )
        total = max(len(plan_rows), len(trace_rows))
        for index in range(total):
            plan_row = plan_rows[index] if index < len(plan_rows) else None
            trace_row = trace_rows[index] if index < len(trace_rows) else None
            if plan_row != trace_row:
                print(
                    f"compare_trace: MISMATCH at {label} {index}{suffix}: "
                    f"plan={plan_row} trace={trace_row}",
                    file=sys.stderr,
                )
                print(
                    f"compare_trace: counts plan={len(plan_rows)} "
                    f"trace={len(trace_rows)}{suffix}",
                    file=sys.stderr,
                )
                ok = False
                break
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path, help="plan_dump --tsv/--tree output")
    parser.add_argument("trace", type=Path, help="AV2ENC_M1_TRACE=1 stderr log")
    parser.add_argument(
        "--nodes",
        action="store_true",
        help="compare the preorder tree (plan_dump --tree) instead of leaves",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="print the first/last rows"
    )
    args = parser.parse_args()

    try:
        if args.nodes:
            plan_rows = read_plan(
                args.plan,
                ("mi_row", "mi_col", "size", "partition", "forced"),
                ("mi_row", "mi_col", "forced"),
            )
            trace_rows = read_trace_nodes(args.trace)
            label = "node"
        else:
            plan_rows = read_plan(
                args.plan, ("mi_row", "mi_col", "size"),
                ("mi_row", "mi_col"),
            )
            trace_rows = read_trace_blocks(args.trace)
            label = "block"
    except (OSError, ValueError) as exc:
        print(f"compare_trace: {exc}", file=sys.stderr)
        return 1

    if not trace_rows:
        print(
            f"compare_trace: no 'm1 {label}' lines found in the trace",
            file=sys.stderr,
        )
        return 1

    trace_passes = split_writer_passes(trace_rows) if args.nodes else [trace_rows]
    if not compare(label, plan_rows, trace_passes):
        return 1

    passes = f" in {len(trace_passes)} passes" if args.nodes else ""
    print(
        f"compare_trace: OK {len(plan_rows)} {label}s, order matches{passes}",
        file=sys.stdout,
    )
    if args.verbose:
        print(f"  first={plan_rows[0]} last={plan_rows[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
