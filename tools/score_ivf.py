#!/usr/bin/env python3
"""Score IVF files produced by tools/modal_rd.py on this Mac.

The Modal half only encodes: it returns the IVF bytes and a manifest. This
script decodes each stream with the pinned `dav2d-macos` and runs the same
quality pipeline every local measurement uses (CVVDP on MPS, Butteraugli and
SSIMULACRA2 from the local libjxl), so cloud-encoded arms are scored with the
exact metric implementation the pins were recorded against.

Usage
-----
    python3 tools/score_ivf.py --manifest /tmp/modal-encode/manifest.json
    python3 tools/score_ivf.py --manifest ... --parallel 3 --out results.json
    python3 tools/score_ivf.py --manifest ... --target-kbps 146.288   # 720p pin

Parallelism: metric jobs are deterministic, so running several at once only
changes wall time. Three is the measured sweet spot on the M4 (MPS contention
above that usually costs more than it saves).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CLIPS = {
    "240p": ROOT / "testdata/nasa_480x240_30fps_60f.y4m",
    "720p": ROOT / "testdata/space8k_1440x720_60f.y4m",
}


def _run(cmd: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, env=env)


def score_one(job: dict) -> dict:
    """Decode one IVF and return its metrics. Raises on any failure."""
    ivf = Path(job["ivf"])
    if not ivf.is_file():
        raise RuntimeError(f"missing {ivf}")
    digest = hashlib.md5(ivf.read_bytes()).hexdigest()
    want = job.get("md5", "")
    if want and not digest.startswith(want[:8]):
        raise RuntimeError(f"{ivf}: md5 {digest} does not match manifest {want}")

    clip = job.get("clip", "240p")
    ref = CLIPS.get(clip)
    if ref is None:
        raise RuntimeError(f"{ivf}: unknown clip {clip!r}")
    env = os.environ.copy()
    env["DAV2D"] = str(ROOT / "tools/bin/dav2d-macos")

    with tempfile.TemporaryDirectory(prefix="score-ivf-") as td:
        td = Path(td)
        y4m = td / "out.y4m"
        decoded = _run(
            [
                str(ROOT / "tools/bin/dav2d-macos"),
                "--strict",
                "1",
                "-i",
                str(ivf),
                "-o",
                str(y4m),
                "--muxer",
                "yuv4mpeg2",
            ]
        )
        log = (decoded.stdout or "") + (decoded.stderr or "")
        if decoded.returncode != 0 or "Error decoding" in log:
            raise RuntimeError(f"{ivf}: decode failed: {log[-400:]}")

        q = td / "q.json"
        e = td / "e.json"
        k = td / "k.json"
        for cmd in (
            [sys.executable, str(ROOT / "tools/quality.py"), "--ref", str(ref),
             "--ivf", str(ivf), "--json", str(q)],
            [sys.executable, str(ROOT / "tools/extra_metrics.py"), "--ref", str(ref),
             "--test", str(y4m), "--json", str(e)],
            [sys.executable, str(ROOT / "tools/kpi.py"), "--quality-json", str(q),
             "--extra-json", str(e), "--json", str(k)],
        ):
            proc = _run(cmd, env=env)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"{ivf}: {cmd[1]} failed: "
                    f"{((proc.stderr or '') + (proc.stdout or ''))[-400:]}"
                )
        kpi = json.loads(k.read_text())

    return {
        "name": job["name"],
        "qp": int(job["qp"]),
        "clip": clip,
        "bytes": ivf.stat().st_size,
        "kbps": ivf.stat().st_size * 8 / 2.0 / 1000.0,
        "md5": digest,
        "jod": kpi["cvvdp"],
        "ba": kpi["butteraugli"],
        "ss2": kpi["ssimulacra2"],
        "kpi": kpi["kpi"],
    }


def interpolate(rows: list[dict], target: float) -> dict | None:
    good = sorted((r for r in rows if "error" not in r), key=lambda r: r["kbps"])
    if not good:
        return None
    if len(good) == 1:
        r = good[0]
        return {"kpi": r["kpi"], "jod": r["jod"], "ba": r["ba"],
                "ss2": r["ss2"], "single": True}
    for lo, hi in zip(good, good[1:]):
        if lo["kbps"] <= target <= hi["kbps"]:
            span = hi["kbps"] - lo["kbps"]
            t = 0.0 if span == 0 else (target - lo["kbps"]) / span
            return {
                "kpi": lo["kpi"] + t * (hi["kpi"] - lo["kpi"]),
                "jod": lo["jod"] + t * (hi["jod"] - lo["jod"]),
                "ba": lo["ba"] + t * (hi["ba"] - lo["ba"]),
                "ss2": lo["ss2"] + t * (hi["ss2"] - lo["ss2"]),
            }
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--target-kbps", type=float, default=0.0,
                    help="KPI interpolation point; default 100 for 240p, 146.288 for 720p")
    ap.add_argument("--bracket-only", action="store_true",
                    help="score only the two points bracketing the target per config")
    ap.add_argument("--out", type=Path, help="write the raw rows here")
    args = ap.parse_args()

    jobs = json.loads(args.manifest.read_text())
    if not jobs:
        print("empty manifest", file=sys.stderr)
        return 1
    clips = {job.get("clip", "240p") for job in jobs}
    target = args.target_kbps or (146.288 if clips == {"720p"} else 100.0)

    if args.bracket_only:
        grouped: dict[str, list[dict]] = {}
        for job in jobs:
            grouped.setdefault(job["name"], []).append(job)
        selected = []
        for name, rows in grouped.items():
            rows = sorted(rows, key=lambda r: r["kbps"])
            pair = None
            for lo, hi in zip(rows, rows[1:]):
                if lo["kbps"] <= target <= hi["kbps"]:
                    pair = (lo, hi)
                    break
            if pair is None:
                pair = (rows[0], rows[-1])
                print(f"warning: {name} does not bracket {target:.3f} kbps; "
                      f"using {pair[0]['kbps']:.3f}/{pair[1]['kbps']:.3f}",
                      file=sys.stderr)
            selected.extend(pair)
        print(f"bracket-only: scoring {len(selected)} of {len(jobs)} points",
              flush=True)
        jobs = selected

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        futures = {pool.submit(score_one, job): job for job in jobs}
        for i, fut in enumerate(futures, 1):
            job = futures[fut]
            try:
                row = fut.result()
                rows.append(row)
                print(
                    f"[{i:2d}/{len(jobs)}] {row['name']:28s} qp={row['qp']:<4d} "
                    f"{row['kbps']:8.3f} kbps  JOD {row['jod']:.4f}  "
                    f"BA {row['ba']:.4f}  SS2 {row['ss2']:.3f}  K {row['kpi']:.2f}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[{i:2d}/{len(jobs)}] ERROR {job['name']} qp={job['qp']}: {exc}",
                      file=sys.stderr, flush=True)

    by_name: dict[str, list[dict]] = {}
    for row in rows:
        by_name.setdefault(row["name"], []).append(row)

    baseline = interpolate(by_name.get("baseline", []), target)
    base_k = baseline["kpi"] if baseline else None
    table = [
        f"Target {target:.3f} kbps. Deltas are against this run's baseline arm.",
        "",
        "| config | QP points (kbps / K) | K@target | ΔK |",
        "|---|---|---:|---:|",
    ]
    summaries = {}
    for name in sorted(by_name):
        points = sorted(by_name[name], key=lambda r: r["qp"])
        cells = " · ".join(f"{r['qp']}: {r['kbps']:.3f} / {r['kpi']:.2f}" for r in points)
        interp = interpolate(points, target)
        k = interp["kpi"] if interp else float("nan")
        dk = (k - base_k) if (interp and base_k is not None) else float("nan")
        summaries[name] = interp
        table.append(f"| {name} | {cells} | {k:.2f} | {dk:+.2f} |")
    print("\n".join(table))

    if args.out:
        args.out.write_text(json.dumps({
            "target_kbps": target,
            "rows": rows,
            "interpolated": summaries,
            "table": "\n".join(table),
        }, indent=2) + "\n")
        print(f"wrote {args.out}")
    return 0 if len(rows) == len(jobs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
