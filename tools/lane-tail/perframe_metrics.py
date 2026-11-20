#!/usr/bin/env python3
"""Per-frame Butteraugli / SSIMULACRA2 and the f33-43 / f44-59 segment means.

First-party lane analysis (stdlib only apart from the existing tools/ helpers).
`tools/extra_metrics.py` reports stream means; the tail report needs the two
segment rows the oracle's source-replacement ceilings are pooled over.

  tools/lane-tail/perframe_metrics.py --ref ref.y4m --test test.dec.y4m \
      --frames 60 --json out.json
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import extra_metrics  # noqa: E402

SEGMENTS = (("f0-32", 0, 32), ("f33-43", 33, 43), ("f44-59", 44, 59))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, type=Path)
    ap.add_argument("--test", required=True, type=Path)
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    for tool in ("ffmpeg", "butteraugli_main", "ssimulacra2"):
        if not extra_metrics.have(tool):
            print(f"perframe_metrics: missing {tool}", file=sys.stderr)
            return 77

    # BA/SS2 come from the same PNG extraction the oracle uses; the per-frame
    # PSNR-Y rows come from tools/perframe_psnr.py (FFmpeg).
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "ref").mkdir()
        (tmp / "test").mkdir()
        n = extra_metrics.extract(args.ref, tmp / "ref", args.frames or None)
        got = extra_metrics.extract(args.test, tmp / "test", args.frames or None)
        if got != n:
            print(
                f"perframe_metrics: ref {n} frames, test {got} frames",
                file=sys.stderr,
            )
            return 2
        ba = []
        ss = []
        for i in range(1, n + 1):
            b, s = extra_metrics.score_pair(
                tmp / "ref" / f"f{i:04d}.png", tmp / "test" / f"f{i:04d}.png"
            )
            ba.append(b)
            ss.append(s)

    rows = []
    for name, start, end in SEGMENTS:
        if end >= n:
            continue
        seg_ba = ba[start : end + 1]
        seg_ss = ss[start : end + 1]
        rows.append(
            {
                "segment": name,
                "frames": len(seg_ba),
                "butteraugli": sum(seg_ba) / len(seg_ba),
                "ssimulacra2": sum(seg_ss) / len(seg_ss),
            }
        )

    out = {
        "frames": n,
        "butteraugli_mean": sum(ba) / len(ba),
        "ssimulacra2_mean": sum(ss) / len(ss),
        "segments": rows,
        "per_frame": [
            {"frame": i, "butteraugli": ba[i], "ssimulacra2": ss[i]}
            for i in range(n)
        ],
    }
    print(f"frames={n} BA={out['butteraugli_mean']:.4f} "
          f"SS2={out['ssimulacra2_mean']:.3f}")
    for row in rows:
        print(f"{row['segment']}: BA={row['butteraugli']:.4f} "
              f"SS2={row['ssimulacra2']:.3f}")
    if args.json:
        args.json.write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
