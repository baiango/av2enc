#!/usr/bin/env python3
"""Measure a tool's matched-rate oracle envelope, not a single K point.

The loop's gate is mechanical: this tool encodes every arm of one candidate
over its own QP bracket, scores each arm with ``tools/oracle.py`` (AVM decoder),
and reports

    envelope = max over arms of K at the target bitrate
    baseline = the arm named "base" (empty env, default-off gates)
    delta    = envelope - baseline

A candidate is progress only if ``delta`` clears the threshold.  A patch that
moves one arm while the envelope stays put is a realization gain and the gate
says so.

The spec is written by the iteration (``build/agent-loop/spec.json``):

    {
      "ref": "testdata/nasa_480x240_30fps_60f.y4m",
      "bin": "build-release/av2enc",
      "base_args": ["--video", "--frames", "60", "--cpu-threads", "8",
                    "--nightly"],
      "qp": 106,
      "target_kbps": 100.0,
      "arms": [
        {"name": "base", "env": {}},
        {"name": "cand", "env": {"AV2ENC_TOOL": "1"}}
      ]
    }

``base`` must be present with an empty env, so the baseline is measured on the
same build and a default-on patch cannot hide behind a separate tree.

QP bracketing is automatic: the tool encodes the start QP, steps toward the
target, and stops when one point is above and one below the target.  This is
what removes the two-point interpolation hand-waving: the bracket is measured,
and a candidate that cannot bracket is reported as ``no-bracket`` rather than
given a number.

Usage:
  tools/envelope.py --spec build/agent-loop/spec.json
  tools/envelope.py --spec spec.json --threshold 0.15 --json out.json

Exit: 0 gate pass, 1 gate fail / measurement error, 2 usage.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ORACLE = ROOT / "tools" / "oracle.py"
THRESHOLD = 0.15
MAX_ENCODES = 12


def rate_kbps(path: Path, frames: int) -> float:
    return path.stat().st_size * 8.0 / (frames / 30.0) / 1000.0


def run(cmd: list[str], cwd: Path, env: dict | None = None,
        timeout: float = 1800.0) -> subprocess.CompletedProcess:
    full = os.environ.copy()
    if env:
        full.update({k: str(v) for k, v in env.items()})
    return subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True,
                          env=full, timeout=timeout)


def next_qp(points: dict[int, float], target: float) -> int:
    """Secant step on log(rate) vs QP, clamped to a sane jump."""
    qps = sorted(points)
    qp = qps[-1]
    rate = points[qp]
    if len(qps) >= 2 and rate > 0 and target > 0:
        other = qps[-2] if qps[-2] != qp else qps[0]
        d_qp = qp - other
        if d_qp and points[other] > 0:
            slope = (math.log(rate) - math.log(points[other])) / d_qp
            if slope < 0:
                jump = (math.log(target) - math.log(rate)) / slope
                step = max(-32, min(32, round(jump)))
                candidate = qp + step
                if candidate != qp:
                    return max(0, min(255, candidate))
    step = max(2, min(24, round(abs(rate / target - 1.0) * 20.0)))
    return max(0, min(255, qp + (step if rate > target else -step)))


def bracket_arm(worktree: Path, spec: dict, arm: dict, outdir: Path) -> dict:
    bin_path = (worktree / spec.get("bin", "build-release/av2enc")).resolve()
    ref = (worktree / spec["ref"]).resolve()
    base_args = list(spec.get("base_args", []))
    frames = int(spec.get("frames", 60))
    target = float(spec.get("target_kbps", 100.0))
    env = dict(arm.get("env", {}))
    qp = int(spec.get("qp", 106))
    seen: dict[int, float] = {}

    def encode(at: int) -> None:
        out = outdir / f"{arm['name']}-qp{at}.ivf"
        cmd = [str(bin_path), str(ref), "-o", str(out), *base_args,
               "--qp", str(at)]
        proc = run(cmd, worktree, env)
        if proc.returncode != 0:
            raise RuntimeError(
                f"{arm['name']} qp{at}: encode failed: "
                f"{(proc.stderr or proc.stdout or '').strip()[-300:]}")
        seen[at] = rate_kbps(out, frames)

    encode(qp)
    while len(seen) < MAX_ENCODES:
        above = [q for q in seen if seen[q] >= target]
        below = [q for q in seen if seen[q] < target]
        if above and below:
            lo_qp, hi_qp = max(above), min(below)
            if hi_qp - lo_qp <= 2:
                break
            mid = (lo_qp + hi_qp) // 2
            if mid in seen or mid in (lo_qp, hi_qp):
                break
            encode(mid)
            continue
        nxt = next_qp(seen, target)
        if nxt in seen:
            break
        encode(nxt)
    above = [q for q in seen if seen[q] >= target]
    below = [q for q in seen if seen[q] < target]
    above_qp = max(above) if above else None
    below_qp = min(below) if below else None
    above = above_qp
    below = below_qp
    points = sorted(seen)
    return {
        "arm": arm["name"],
        "env": env,
        "target_kbps": target,
        "bracket": bool(above is not None and below is not None),
        "points": [{"qp": q, "kbps": round(seen[q], 3),
                    "ivf": str(outdir / f"{arm['name']}-qp{q}.ivf")}
                   for q in points],
        "above": above,
        "below": below,
    }


def measure_ceiling(worktree: Path, spec: dict, arm: dict, outdir: Path,
                    segment: str) -> float | None:
    """Source-replacement K for one segment of the arm's primary stream."""
    result = arm.get("_result")
    above, below = result["above"], result["below"]
    if above is None or below is None:
        return None
    point = below
    cmd = [sys.executable, str(ORACLE),
           "--ref", str((worktree / spec["ref"]).resolve()),
           "--target-kbps", str(result["target_kbps"]),
           "--json", str(outdir / f"{arm['name']}.segments.json"),
           "--ivf", f"{arm['name']}@{point}="
           + str(outdir / f"{arm['name']}-qp{point}.ivf")]
    proc = run(cmd, worktree)
    if proc.returncode != 0:
        return None
    report = json.loads((outdir / f"{arm['name']}.segments.json").read_text())
    for row in report["candidates"][0].get("segments", []):
        if row.get("replacement") == "source" and row.get("segment") == segment:
            return float(row["kpi"])
    return None


def score_arm(worktree: Path, spec: dict, arm: dict, outdir: Path) -> dict:
    result = bracket_arm(worktree, spec, arm, outdir)
    if not result["bracket"]:
        result["k"] = None
        result["method"] = "no-bracket"
        return result
    chosen = sorted({result["above"], result["below"]})
    cmd = [sys.executable, str(ORACLE),
           "--ref", str((worktree / spec["ref"]).resolve()),
           "--target-kbps", str(result["target_kbps"]),
           "--no-segments",
           "--json", str(outdir / f"{arm['name']}.oracle.json")]
    for qp in chosen:
        cmd += ["--ivf", f"{arm['name']}@{qp}="
                + str(outdir / f"{arm['name']}-qp{qp}.ivf")]
    proc = run(cmd, worktree)
    if proc.returncode != 0:
        raise RuntimeError(f"{arm['name']}: oracle failed: "
                           f"{(proc.stderr or proc.stdout or '')[-300:]}")
    report = json.loads((outdir / f"{arm['name']}.oracle.json").read_text())
    matched = report["candidates"][0]["matched"]
    result["k"] = float(matched["kpi"])
    result["method"] = matched.get("method", "linear")
    result["matched_kbps"] = float(matched["kbps"])
    return result


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--worktree", type=Path, default=Path.cwd())
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--ceiling-floor", type=float, default=None,
                    help="best known source-replacement ceiling for the target "
                         "segment; PASS also requires beating it")
    ap.add_argument("--ceiling-segment", default="f0-32",
                    help="segment whose source-replacement ceiling is gated")
    ap.add_argument("--base-env", default="{}",
                    help="JSON env shared by every arm; the established recipe "
                         "is the baseline instead of being an arm that wins")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--keep", action="store_true",
                    help="keep encoded streams (default: temporary directory)")
    args = ap.parse_args()

    worktree = args.worktree.resolve()
    spec = json.loads(args.spec.read_text())
    try:
        base_env = json.loads(args.base_env)
    except json.JSONDecodeError as exc:
        print(f"envelope: --base-env is not JSON: {exc}", file=sys.stderr)
        return 2
    if not isinstance(base_env, dict):
        print("envelope: --base-env must be a JSON object", file=sys.stderr)
        return 2
    # Every arm is the established recipe plus its own knob.  The established
    # recipe is the baseline, never an arm: otherwise a lane can "discover" the
    # already-committed stack and report it as a new envelope gain.
    arms = []
    for arm in spec.get("arms") or []:
        merged = dict(arm)
        merged["env"] = {**base_env, **dict(arm.get("env") or {})}
        merged["knobs"] = dict(arm.get("env") or {})
        arms.append(merged)
    names = [arm.get("name") for arm in arms]
    if "base" not in names:
        print("envelope: spec must contain an arm named 'base' with empty env",
              file=sys.stderr)
        return 2
    if arms[names.index("base")].get("knobs"):
        print("envelope: the base arm must have an empty env", file=sys.stderr)
        return 2

    outdir = (Path(tempfile.mkdtemp(prefix="av2enc-envelope-"))
              if not args.keep else (worktree / "build" / "agent-loop" /
                                     "envelope-streams"))
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        scored = [score_arm(worktree, spec, arm, outdir) for arm in arms]
    except Exception as exc:  # noqa: BLE001 - the loop must record, not crash
        print(f"envelope: measurement error: {exc}", file=sys.stderr)
        return 1

    base = next(row for row in scored if row["arm"] == "base")
    if base["k"] is None:
        print("envelope: the base arm did not bracket the target", file=sys.stderr)
        return 1
    measured = [row for row in scored if row["k"] is not None]
    best = max(measured, key=lambda row: row["k"])
    delta = best["k"] - base["k"]
    envelope_ceiling = None
    if args.ceiling_floor is not None:
        envelope_ceiling = measure_ceiling(
            worktree, spec, {"_result": best, "name": best["arm"]}, outdir,
            args.ceiling_segment)
    gate_ok = delta >= args.threshold
    ceiling_delta = None
    if args.ceiling_floor is not None:
        if envelope_ceiling is None:
            gate_ok = False
        else:
            ceiling_delta = envelope_ceiling - args.ceiling_floor
            gate_ok = gate_ok and ceiling_delta >= args.threshold
    verdict = "PASS" if gate_ok else "FAIL"

    report = {
        "schema": "av2enc-envelope/1",
        "target_kbps": spec.get("target_kbps", 100.0),
        "threshold": args.threshold,
        "base_env": base_env,
        "baseline_k": round(base["k"], 4),
        "base_k": round(base["k"], 4),
        "envelope_k": round(best["k"], 4),
        "delta": round(delta, 4),
        "best_arm": best["arm"],
        "ceiling_segment": args.ceiling_segment,
        "ceiling_floor": args.ceiling_floor,
        "envelope_ceiling": (round(envelope_ceiling, 4)
                             if envelope_ceiling is not None else None),
        "ceiling_delta": (round(ceiling_delta, 4)
                          if ceiling_delta is not None else None),
        "verdict": verdict,
        "arms": scored,
    }
    text = json.dumps(report, indent=2)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n")
    print(text)
    ceiling = ("" if envelope_ceiling is None else
               f"  ceiling[{args.ceiling_segment}] {envelope_ceiling:.3f}"
               f" vs {args.ceiling_floor:.3f}")
    print(f"envelope: base {base['k']:.3f}  best {best['k']:.3f} "
          f"({best['arm']})  delta {delta:+.3f}{ceiling}  {verdict}",
          flush=True)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
