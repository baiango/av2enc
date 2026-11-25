#!/usr/bin/env python3
"""Screen many av2enc arms at a few fixed QPs on one picture, ranked by K.

`tools/hk_eval.py` is the ratified measurement: it lands each arm at a byte
target and interpolates.  This tool exists only to decide *which* arms deserve
that treatment.  It encodes each arm at the same QPs, decodes with the pinned
dav2d, scores every point and prints bytes / K, so an arm that cannot reach the
baseline at similar bytes is dropped in one pass rather than a full walk.

The arms come from a JSON file, because each one needs its own env:

    [
      {"label": "uv23",       "env": {}, "flags": ["--video-gdf",
                                                  "--video-uv-qp-offset", "-23"]},
      {"label": "uv23_dlam256", "env": {"AV2ENC_RD_DRIVER_LAMBDA": "256"},
                                  "flags": ["--video-gdf",
                                            "--video-uv-qp-offset", "-23"]}
    ]

    tools/hk_screen.py --in x.y4m --spec arms.json --qp 186 188 --out DIR

Every arm is additionally run with the args-level `--env` (default none).
Results are appended to `DIR/screen.jsonl`, so a killed screen resumes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
AV2ENC = REPO / "build-pub/av2enc"
DAV2D = REPO / "tools/bin/dav2d-macos"


def run(cmd, env=None):
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True,
                          check=False, env=env)


def kpi(jod: float, ba: float, ss2: float) -> float:
    return 100.0 * (0.50 * max(0.0, min(1.0, jod / 10.0))
                    + 0.25 / (1.0 + max(ba, 0.0) / 3.0)
                    + 0.25 * max(0.0, min(1.0, (ss2 + 50.0) / 150.0)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True, type=Path)
    ap.add_argument("--spec", required=True, type=Path)
    ap.add_argument("--qp", type=int, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--env", action="append", default=[],
                    help="K=V applied to every arm")
    ap.add_argument("--frames", type=int, default=1)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    log = args.out / "screen.jsonl"
    done = set()
    if log.exists():
        for line in log.read_text().splitlines():
            try:
                r = json.loads(line)
            
                done.add((r["label"], r.get("tag"), r["qp"]))
            except (ValueError, KeyError):
                continue

    arms = json.loads(args.spec.read_text())
    base_env = dict(os.environ)
    for kv in args.env:
        k, _, v = kv.partition("=")
        base_env[k] = v

    for arm in arms:
        label = arm["label"]
        env = dict(base_env)
        env.update(arm.get("env", {}))
        # The cache directory is per (label, env, flags): reusing a label across
        # screens with a different env silently scored another arm's stream.
        key = json.dumps([label, arm.get("env", {}), arm["flags"]],
                         sort_keys=True)
        tag = hashlib.sha1(key.encode()).hexdigest()[:8]
        arm_dir = args.out / f"{label}-{tag}"
        arm_dir.mkdir(parents=True, exist_ok=True)
        for qp in args.qp:
            if (label, tag, qp) in done:
                continue
            stem = arm_dir / f"{label}_q{qp}"
            ivf = stem.with_suffix(".ivf")
            if not (ivf.exists() and ivf.stat().st_size > 0):
                cmd = [AV2ENC, args.src, "-o", ivf, "--video",
                       "--frames", args.frames, "--qp", qp,
                       "--cpu-threads", args.threads, *arm["flags"]]
                p = run(cmd, env=env)
                if p.returncode != 0 or not ivf.exists():
                    msg = (p.stderr or p.stdout).strip().splitlines()
                    first = next((l for l in msg if l.startswith("av2enc:")),
                                 "no message")
                    print(f"  {label} qp{qp}: REJECTED {first[:110]}",
                          flush=True)
                    continue
            y4m = stem.with_name(stem.name + "_dec.y4m")
            if not y4m.exists():
                p = run([DAV2D, "-q", "--strict", "1", "-i", ivf, "-o", y4m,
                         "--threads", "1"])
                if p.returncode != 0 or not y4m.exists():
                    print(f"  {label} qp{qp}: DECODE FAILED", flush=True)
                    continue
            qj = stem.with_name(stem.name + "_quality.json")
            ej = stem.with_name(stem.name + "_extra.json")
            if not (qj.exists() and ej.exists()):
                run([sys.executable, REPO / "tools/quality.py", "--ref",
                     args.src, "--test", y4m, "--json", qj])
                run([sys.executable, REPO / "tools/extra_metrics.py", "--ref",
                     args.src, "--test", y4m, "--json", ej])
            q, e = json.loads(qj.read_text()), json.loads(ej.read_text())
            row = {"label": label, "tag": tag, "qp": qp,
                   "flags": arm["flags"], "env": arm.get("env", {}),
                   "bytes": ivf.stat().st_size,
                   "jod": q["cvvdp_jod"], "ba": e["butteraugli_mean"],
                   "ss2": e["ssimulacra2_mean"],
                   "psnr_y": q["psnr"]["psnr_y"]}
            row["K"] = kpi(row["jod"], row["ba"], row["ss2"])
            with log.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"  {label:24s} qp{qp} {row['bytes']:7d} B  "
                  f"JOD {row['jod']:6.3f} BA {row['ba']:6.2f} "
                  f"SS2 {row['ss2']:7.2f}  K {row['K']:6.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
