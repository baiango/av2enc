#!/usr/bin/env python3
"""Butteraugli and SSIMULACRA2 for a decoded Y4M pair, per frame, averaged.

CVVDP is the project's primary metric (tools/quality.py). These two are
secondary perceptual measures that the mature-encoder literature quotes, so the
README table carries all three on the same streams.

  tools/extra_metrics.py --ref ref.y4m --test test.y4m [--frames N] [--json OUT]

Requires `butteraugli_main` (libjxl) and `ssimulacra2` on PATH. Both are
full-reference image metrics, so frames are extracted to PNG and scored
individually; the reported number is the arithmetic mean across frames.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def have(tool: str) -> bool:
    return shutil.which(tool) is not None


def extract(y4m: Path, out_dir: Path, limit: int | None) -> int:
    cmd = ["ffmpeg", "-v", "error", "-i", str(y4m)]
    if limit:
        cmd += ["-frames:v", str(limit)]
    cmd += [str(out_dir / "f%04d.png")]
    subprocess.run(cmd, check=True)
    return len(list(out_dir.glob("f*.png")))


def score_pair(ref_png: Path, test_png: Path) -> tuple[float, float]:
    ba = subprocess.run(["butteraugli_main", str(ref_png), str(test_png)],
                        capture_output=True, text=True)
    m = re.search(r"([-0-9.]+)", (ba.stdout or "") + (ba.stderr or ""))
    butter = float(m.group(1)) if m else float("nan")
    ss = subprocess.run(["ssimulacra2", str(ref_png), str(test_png)],
                        capture_output=True, text=True)
    m = re.search(r"([-0-9.]+)\s*$", (ss.stdout or "").strip())
    ssimu = float(m.group(1)) if m else float("nan")
    return butter, ssimu


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, type=Path)
    ap.add_argument("--test", required=True, type=Path)
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    missing = [t for t in ("ffmpeg", "butteraugli_main", "ssimulacra2")
               if not have(t)]
    if missing:
        print("extra_metrics: missing " + ", ".join(missing), file=sys.stderr)
        return 77

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "ref").mkdir()
        (tmp / "test").mkdir()
        n = extract(args.ref, tmp / "ref", args.frames or None)
        extract(args.test, tmp / "test", args.frames or None)
        butters, ssimus = [], []
        for i in range(1, n + 1):
            b, s = score_pair(tmp / "ref" / f"f{i:04d}.png",
                              tmp / "test" / f"f{i:04d}.png")
            butters.append(b)
            ssimus.append(s)

    out = {
        "frames": n,
        "butteraugli_mean": sum(butters) / len(butters) if butters else None,
        "ssimulacra2_mean": sum(ssimus) / len(ssimus) if ssimus else None,
    }
    print(f"frames={n} butteraugli={out['butteraugli_mean']:.4f} "
          f"ssimulacra2={out['ssimulacra2_mean']:.3f}")
    if args.json:
        args.json.write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
