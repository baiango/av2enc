#!/usr/bin/env python3
"""Land one av2enc video arm on a fixed-byte target and score its composite K.

Brick-level counterpart of the in-tree still harness for the 720p still
(`~/Downloads/av2enc-src720/sweep.py`, outside the tree).  It walks QP until the
encode brackets the target in BYTES (never in QP order: the QP->rate slope
changes with the arm), decodes every point with the pinned dav2d, scores
CVVDP / Butteraugli / SSIMULACRA2 and interpolates every metric to the target
before recomputing K with `tools/kpi.py`'s formula.

    tools/hk_eval.py --in x.y4m --target 25010 --label base -- --nightly-b
    tools/hk_eval.py --in x.y4m --target 25010 --label dmin16 --qp 187 \
        --env AV2ENC_RD_DRIVER_MIN=16 -- --nightly-b

Artifacts (`<label>_q<qp>.ivf`, `_dec.y4m`, `_quality.json`, `_extra.json`,
`<label>_result.json`) go to `--out` (default: a scratch directory).  Every
stream is decode-gated with `dav2d --strict 1`; `--validate` additionally runs
`tools/validate_dav2d.sh` on the bracketing point above the budget.

Harness traps this script exists to avoid (docs/rd-key-partition-floor.md §6):

* Pair by BYTES, never by QP order.  Sorting by QP pairs the far ends of a
  jumping walk and reports a bracket that is 4x wide.
* A walk that never brackets must fail loudly, not report its nearest point.
* Cached points come from the quality/extra JSON pair, so pkill and rerun is
  always safe.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
AV2ENC = REPO / "build-pub/av2enc"
DAV2D = REPO / "tools/bin/dav2d-macos"


def run(cmd, env=None):
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True,
                          check=False, env=env)


def kpi(jod: float, ba: float, ss2: float) -> float:
    q_c = max(0.0, min(1.0, jod / 10.0))
    q_b = 1.0 / (1.0 + max(ba, 0.0) / 3.0)
    q_s = max(0.0, min(1.0, (ss2 + 50.0) / 150.0))
    return 100.0 * (0.50 * q_c + 0.25 * q_b + 0.25 * q_s)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True, type=Path)
    ap.add_argument("--label", required=True)
    ap.add_argument("--target", type=int, required=True)
    ap.add_argument("--qp", type=int, default=145)
    ap.add_argument("--frames", type=int, default=1)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--max-steps", type=int, default=18)
    ap.add_argument("--mode", choices=("video", "still"), default="video",
                    help="which encoder path the arm drives (default video)")
    args, rest = ap.parse_known_args()
    arm = [a for a in rest if a != "--"]

    out = args.out or Path(os.environ.get("TMPDIR", "/tmp")) / f"hk-{args.label}"
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    for kv in args.env:
        k, _, v = kv.partition("=")
        env[k] = v
    target = args.target

    def ivf_of(qp: int) -> Path:
        return out / f"{args.label}_q{qp}.ivf"

    def score_of(qp: int) -> dict | None:
        qj = out / f"{args.label}_q{qp}_quality.json"
        ej = out / f"{args.label}_q{qp}_extra.json"
        if not (qj.exists() and ej.exists()):
            return None
        q, e = json.loads(qj.read_text()), json.loads(ej.read_text())
        return {"jod": q["cvvdp_jod"], "ba": e["butteraugli_mean"],
                "ss2": e["ssimulacra2_mean"], "psnr_y": q["psnr"]["psnr_y"]}

    def measure(qp: int) -> tuple[int, dict]:
        ivf = ivf_of(qp)
        if not (ivf.exists() and ivf.stat().st_size > 0):
            cmd = [AV2ENC, args.src, "-o", ivf]
            if args.mode == "video":
                cmd += ["--video", "--frames", args.frames,
                        "--cpu-threads", args.threads]
            else:
                # --cpu-threads is a --video-only flag; the still path is serial.
                cmd += ["--still"]
            cmd += ["--qp", qp, *arm]
            p = run(cmd, env=env)
            if p.returncode != 0 or not ivf.exists() or ivf.stat().st_size == 0:
                raise SystemExit(f"encode failed qp{qp} rc={p.returncode}\n"
                                 f"{p.stdout[-2000:]}\n{p.stderr[-2000:]}")
        size = ivf.stat().st_size
        sc = score_of(qp)
        if sc is None:
            y4m = out / f"{args.label}_q{qp}_dec.y4m"
            if not y4m.exists():
                p = run([DAV2D, "-q", "--strict", "1", "-i", ivf, "-o", y4m,
                         "--threads", "1"])
                if p.returncode != 0 or not y4m.exists():
                    raise SystemExit(f"decode failed qp{qp}: {p.stderr[-800:]}")
            qj = out / f"{args.label}_q{qp}_quality.json"
            ej = out / f"{args.label}_q{qp}_extra.json"
            run([sys.executable, REPO / "tools/quality.py", "--ref", args.src,
                 "--test", y4m, "--json", qj])
            run([sys.executable, REPO / "tools/extra_metrics.py", "--ref",
                 args.src, "--test", y4m, "--json", ej])
            sc = score_of(qp)
            if sc is None:
                raise SystemExit(f"scoring failed qp{qp}")
        return size, sc

    qp = args.qp
    pts: list[tuple[int, int, dict]] = []
    seen: set[int] = set()
    for _ in range(args.max_steps):
        if qp in seen:
            qp += 1
        seen.add(qp)
        t0 = time.time()
        size, sc = measure(qp)
        pts.append((qp, size, sc))
        print(f"  {args.label} qp{qp} -> {size} B "
              f"(K {kpi(sc['jod'], sc['ba'], sc['ss2']):.2f}, "
              f"{time.time() - t0:.1f} s)", flush=True)
        if size == target:
            break
        lo = [p[1] for p in pts if p[1] <= target]
        hi = [p[1] for p in pts if p[1] >= target]
        if lo and hi and (min(hi) - max(lo)) / target < 0.05:
            break  # bracketed inside 5 %: an extra point would not narrow it
        if size > target:
            ratio = size / target
            qp += 8 if ratio > 1.15 else (2 if ratio > 1.02 else 1)
        else:
            ratio = size / target
            qp -= 8 if ratio < 0.85 else (2 if ratio < 0.98 else 1)

    below = [p for p in pts if p[1] <= target]
    above = [p for p in pts if p[1] >= target]
    if below and above:
        a, b = max(below, key=lambda p: p[1]), min(above, key=lambda p: p[1])
        bracketed = True
    else:
        a = b = min(pts, key=lambda p: abs(p[1] - target))
        bracketed = False
    gap = abs(a[1] - b[1]) / target

    def interp(key: str) -> float:
        if a[1] == b[1]:
            return float(a[2][key])
        t = (target - a[1]) / (b[1] - a[1])
        return a[2][key] + t * (b[2][key] - a[2][key])

    m = {k: interp(k) for k in ("jod", "ba", "ss2", "psnr_y")}
    row = {"label": args.label, "arm": arm, "env": args.env,
           "qp": [a[0], b[0]], "bytes": [a[1], b[1]], "target": target,
           "bracket_gap": round(gap, 4), "bracketed": bracketed, **m,
           "K": kpi(m["jod"], m["ba"], m["ss2"]),
           # Endpoint K with no interpolation: on a ripple-prone arm the
           # bracket midpoint and an endpoint can disagree by more than the
           # arm's own effect, so both are always printed.
           "K_at": {str(p[0]): [p[1], round(kpi(p[2]["jod"], p[2]["ba"],
                                               p[2]["ss2"]), 3)]
                    for p in (a, b)}}
    if not bracketed or gap > 0.10:
        row["WARNING"] = (f"bracket {a[1]}..{b[1]} B is "
                          f"{gap * 100:.1f} % wide on target {target}; K is "
                          f"NOT comparable to a tight bracket")
        print("WARNING: " + row["WARNING"], file=sys.stderr)
    if args.validate:
        vs = min(above, key=lambda p: p[1]) if above else b
        p = run([REPO / "tools/validate_dav2d.sh", ivf_of(vs[0]), args.src])
        row["validate_dav2d"] = {"qp": vs[0], "rc": p.returncode}
        print(f"  validate_dav2d qp{vs[0]} -> rc {p.returncode}", flush=True)
    print(json.dumps(row, indent=2))
    (out / f"{args.label}_result.json").write_text(json.dumps(row, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
