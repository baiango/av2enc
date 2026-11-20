#!/usr/bin/env python3
"""Per-segment Butteraugli / SSIMULACRA2 / PSNR-Y for the P-driver lane.

Splits the 60-frame clip into the oracle's three pooled segments plus the key
picture and the head, and prints one row per (arm, segment) so a matched-rate
pair can be compared segment by segment.  Reuses tools/extra_metrics.py's PNG
extraction so the numbers are the oracle's own.

  tools/lane-pdriver/perframe_metrics.py --ref ref.y4m \
      --arm base@116=/tmp/base.dec.y4m --arm head@115=/tmp/head.dec.y4m \
      --json /tmp/segments.json
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
import perframe_psnr  # noqa: E402

# (name, first frame, last frame) inclusive, 0-based.
SEGMENTS = (
    ("key", 0, 0),
    ("head f1-32", 1, 32),
    ("f33-43", 33, 43),
    ("f44-59", 44, 59),
    ("all", 0, 59),
)


def score(ref: Path, test: Path, frames: int):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "ref").mkdir()
        (tmp / "test").mkdir()
        n = extra_metrics.extract(ref, tmp / "ref", frames or None)
        got = extra_metrics.extract(test, tmp / "test", frames or None)
        if got != n:
            raise SystemExit(f"perframe_metrics: ref {n} frames, test {got}")
        ba = []
        ss = []
        for i in range(1, n + 1):
            b, s = extra_metrics.score_pair(
                tmp / "ref" / f"f{i:04d}.png", tmp / "test" / f"f{i:04d}.png"
            )
            ba.append(b)
            ss.append(s)
    return n, ba, ss


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, type=Path)
    ap.add_argument("--arm", action="append", required=True,
                    metavar="LABEL=PATH", help="decoded Y4M for one arm")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    for tool in ("ffmpeg", "butteraugli_main", "ssimulacra2"):
        if not extra_metrics.have(tool):
            print(f"perframe_metrics: missing {tool}", file=sys.stderr)
            return 77

    out = {}
    for spec in args.arm:
        label, raw = spec.split("=", 1)
        path = Path(raw)
        n, ba, ss = score(args.ref, path, args.frames)
        _, _, ref_frames = perframe_psnr.read_y4m(args.ref, n)
        _, _, test_frames = perframe_psnr.read_y4m(path, n)
        psnr = [perframe_psnr.psnr_y(ref_frames[i], test_frames[i])
                for i in range(n)]
        rows = []
        for name, start, end in SEGMENTS:
            if end >= n:
                continue
            rows.append({
                "segment": name,
                "frames": end - start + 1,
                "butteraugli": sum(ba[start:end + 1]) / (end - start + 1),
                "ssimulacra2": sum(ss[start:end + 1]) / (end - start + 1),
                "psnr_y": sum(psnr[start:end + 1]) / (end - start + 1),
            })
        out[label] = {"frames": n, "segments": rows,
                      "butteraugli_mean": sum(ba) / len(ba),
                      "ssimulacra2_mean": sum(ss) / len(ss),
                      "psnr_y_mean": sum(psnr) / len(psnr)}

    for label, data in out.items():
        print(f"{label}: BA={data['butteraugli_mean']:.4f} "
              f"SS2={data['ssimulacra2_mean']:.3f} "
              f"PSNR-Y={data['psnr_y_mean']:.4f}")
        for row in data["segments"]:
            print(f"  {row['segment']:11s} BA={row['butteraugli']:.4f} "
                  f"SS2={row['ssimulacra2']:.3f} "
                  f"PSNR-Y={row['psnr_y']:.4f}")
    if args.json:
        args.json.write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
