#!/usr/bin/env python3
"""Measure the matched-rate oracle envelope with the encodes on Modal.

This is the drop-in replacement for ``tools/envelope.py``: same CLI, same
JSON report fields, same bracket/QP decision rules.  The difference is where
the bytes are made:

* the lane worktree (including uncommitted source) is mounted into a Modal
  image and built with the ``release`` preset, so the spec's
  ``build-release/av2enc`` maps to ``/src/build-release/av2enc``;
* every ``(arm, QP)`` encode is one 1-core / 200 MiB container
  (``--cpu-threads 1``), up to 24 at once;
* the IVF bytes come back over the RPC and ``tools/oracle.py`` scores them on
  this Mac exactly as ``tools/envelope.py`` does.

Scoring is **local only in this build**.  The remote metrics image is out of
scope: there is no pinned Linux ``avmdec`` in the tree, and a Linux decoder
would have to be a new tooling pin with its own parity check first.
``--score remote`` / ``--score auto`` exit 2 instead of silently measuring a
different metric stack.

The QP bracket is adaptive, so the scheduler mirrors
``envelope.bracket_arm``: each round every open arm proposes its next QP,
the round's encodes go out as one Modal batch, and the bitrates drive the
next proposal.  The rate is computed with ``envelope.rate_kbps`` and the
proposal with ``envelope.next_qp``, so the same byte counts produce the same
bracket as the local tool.  The measured streams are byte-identical to the
local 8-thread release build (validated on a 2-arm/5-point spec); only the
encode host and thread count change.

The spec's ``bin`` is ignored remotely: the image always builds release.  A
local run of the same spec still uses ``bin``.  ``--worktree`` is the mount
(``--repo`` is an alias checked against it).

Usage:
  tools/envelope_modal.py --spec build/agent-loop/spec.json \
      --worktree ../rd-loop-1 --json /tmp/modal-envelope.json

Env:
  AV2ENC_MODAL_MAX_CONTAINERS   encode pool size (default 24)

Exit: 0 gate pass, 1 gate fail / measurement error, 2 usage.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

try:  # the local contract helpers; absent when the module is imported remotely
    import envelope as contract  # noqa: E402  (next_qp, rate_kbps, oracle)
except ImportError:  # pragma: no cover - only inside an encode container
    contract = None  # type: ignore[assignment]

import modal  # noqa: E402

APP = modal.App("av2enc-envelope")
MAX_ENCODE_CONTAINERS = int(os.environ.get("AV2ENC_MODAL_MAX_CONTAINERS", "24"))
ENCODE_BIN = "/src/build-release/av2enc"
REF = "/data/ref.y4m"

# Same snapshot rules as tools/modal_rd.py: build products and media stay out
# of the image; the lane's src/ (committed or not) is what gets compiled.
IGNORE = [
    ".git",
    ".venv",
    ".venv-cvvdp",
    "**/build*",
    "**/__pycache__",
    "testdata",
    "bench",
    "spec",
    "artifacts",
    "out",
    "docs",
    "refactor",
    "tools/bin",
    "**/*.ivf",
    "**/*.y4m",
    "**/*.yuv",
    "**/*.o",
    "**/*.a",
    "**/*.so",
    "**/*.dylib",
]

BASE_PACKAGES = [
    "build-essential",
    "cmake",
    "ninja-build",
    "git",
    "nasm",
    "meson",
    "pkg-config",
]


def build_encode_image(worktree: Path, ref: Path) -> modal.Image:
    """Release build of the (possibly dirty) lane worktree."""
    return (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install(*BASE_PACKAGES)
        .add_local_dir(str(worktree), "/src", copy=True, ignore=IGNORE)
        .add_local_file(str(ref), REF, copy=True)
        .run_commands(
            "cd /src && cmake --preset release && "
            "cmake --build --preset release --target av2enc -j\"$(nproc)\""
        )
    )


def _one_thread(flags: list[str]) -> list[str]:
    """Drop any --cpu-threads in the spec and pin the worker to its one core."""
    out: list[str] = []
    i = 0
    while i < len(flags):
        if flags[i] == "--cpu-threads" and i + 1 < len(flags):
            i += 2
            continue
        if flags[i].startswith("--cpu-threads="):
            i += 1
            continue
        out.append(flags[i])
        i += 1
    return out + ["--cpu-threads", "1"]


def _encode_impl(job: dict) -> dict:
    import hashlib
    import subprocess
    import time

    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in (job.get("env") or {}).items()})
    out = Path("/tmp/out.ivf")
    cmd = [
        ENCODE_BIN, REF, "-o", str(out),
        *_one_thread(list(job["flags"])),
        "--qp", str(int(job["qp"])),
    ]
    started = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{job['arm']} qp{job['qp']}: encode failed: "
            f"{(proc.stderr or proc.stdout or '').strip()[-800:]}")
    data = out.read_bytes()
    return {
        "arm": job["arm"],
        "qp": int(job["qp"]),
        "bytes": len(data),
        "md5": hashlib.md5(data).hexdigest(),
        "encode_seconds": round(time.time() - started, 2),
        "started": started,
        "finished": time.time(),
        "ivf_b64": base64.b64encode(data).decode("ascii"),
    }


def _peak_concurrency(rows: list[dict]) -> int:
    events: list[tuple[float, int]] = []
    for row in rows:
        if "started" in row and "finished" in row:
            events.append((float(row["started"]), 1))
            events.append((float(row["finished"]), -1))
    current = peak = 0
    for _, delta in sorted(events):
        current += delta
        peak = max(peak, current)
    return peak


def _next_qp(seen: dict[int, float], spec: dict, target: float) -> int | None:
    """One step of envelope.bracket_arm's state machine (None = finished)."""
    if not seen:
        return int(spec.get("qp", 106))
    if len(seen) >= contract.MAX_ENCODES:
        return None
    above = [q for q in seen if seen[q] >= target]
    below = [q for q in seen if seen[q] < target]
    if above and below:
        lo_qp, hi_qp = max(above), min(below)
        if hi_qp - lo_qp <= 2:
            return None
        mid = (lo_qp + hi_qp) // 2
        if mid in seen or mid in (lo_qp, hi_qp):
            return None
        return mid
    nxt = contract.next_qp(seen, target)
    return None if nxt in seen else nxt


def _points_row(arm: dict, spec: dict, seen: dict[int, float],
                outdir: Path) -> dict:
    target = float(spec.get("target_kbps", 100.0))
    above = [q for q in seen if seen[q] >= target]
    below = [q for q in seen if seen[q] < target]
    return {
        "arm": arm["name"],
        "env": dict(arm["env"]),
        "target_kbps": target,
        "bracket": bool(above and below),
        "points": [
            {"qp": q, "kbps": round(seen[q], 3),
             "ivf": str(outdir / f"{arm['name']}-qp{q}.ivf")}
            for q in sorted(seen)
        ],
        "above": max(above) if above else None,
        "below": min(below) if below else None,
    }


def _score_local(worktree: Path, ref: Path, spec: dict, row: dict,
                 outdir: Path) -> dict:
    """Score the bracket with the same oracle call envelope.score_arm makes."""
    target = float(spec.get("target_kbps", 100.0))
    chosen = sorted({row["above"], row["below"]})
    cmd = [sys.executable, str(contract.ORACLE),
           "--ref", str(ref), "--target-kbps", str(target), "--no-segments",
           "--json", str(outdir / f"{row['arm']}.oracle.json")]
    for qp in chosen:
        cmd += ["--ivf", f"{row['arm']}@{qp}="
                + str(outdir / f"{row['arm']}-qp{qp}.ivf")]
    proc = contract.run(cmd, worktree)
    if proc.returncode != 0:
        raise RuntimeError(f"{row['arm']}: oracle failed: "
                           f"{(proc.stderr or proc.stdout or '')[-400:]}")
    return json.loads(
        (outdir / f"{row['arm']}.oracle.json").read_text()
    )["candidates"][0]["matched"]


def _dispatch(fn, jobs: list[dict], retries: int = 3) -> list[dict]:
    """Map with retries; one preempted container must not kill a measurement."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            return list(fn.map(jobs))
        except Exception as exc:  # noqa: BLE001 - retry transient Modal faults
            last = exc
            print(f"envelope_modal: batch failed ({attempt + 1}/{retries}): "
                  f"{exc}", file=sys.stderr, flush=True)
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Modal batch failed after {retries} attempts: {last}")


def main() -> int:
    if contract is None:
        raise SystemExit("envelope_modal: tools/envelope.py not importable")
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--worktree", type=Path, default=Path.cwd())
    ap.add_argument("--repo", type=Path, default=None,
                    help="alias for --worktree (the tree mounted into the image)")
    ap.add_argument("--threshold", type=float, default=contract.THRESHOLD)
    ap.add_argument("--ceiling-floor", type=float, default=None,
                    help="reference ceiling for the target segment "
                         "(diagnostic, reported only; the gated quantity is "
                         "the ceiling delta vs the base arm)")
    ap.add_argument("--ceiling-segment", default="f0-32",
                    help="segment whose source-replacement ceiling is gated")
    ap.add_argument("--best-known", type=float, default=None,
                    help="current best matched K at the target (85.099); PASS "
                         "requires envelope >= best-known + threshold")
    ap.add_argument("--min-arms", type=int, default=0,
                    help="minimum non-base arms (a knob sweep, not one point)")
    ap.add_argument("--base-env", default="{}",
                    help="JSON env shared by every arm; the established recipe "
                         "is the baseline instead of being an arm that wins")
    ap.add_argument("--score", choices=("local", "remote", "auto"),
                    default="local",
                    help="local (default, the only supported mode): encode on "
                         "Modal, score with tools/oracle.py on this Mac. "
                         "remote/auto are disabled in this build.")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--keep", action="store_true",
                    help="keep encoded streams (default: temporary directory)")
    args = ap.parse_args()

    if args.score != "local":
        print("envelope_modal: remote scoring is not supported in this build; "
              "use --score local (the default). The metrics image is out of "
              "scope: there is no pinned Linux avmdec, and the encode-only "
              "path returns the IVF files for tools/oracle.py.",
              file=sys.stderr)
        return 2

    worktree = (args.repo or args.worktree).expanduser().resolve()
    if args.repo and args.worktree != Path.cwd():
        explicit = args.worktree.expanduser().resolve()
        if explicit != args.repo.expanduser().resolve():
            print("envelope_modal: --repo and --worktree disagree",
                  file=sys.stderr)
            return 2
    spec = json.loads(args.spec.read_text())
    try:
        base_env = json.loads(args.base_env)
    except json.JSONDecodeError as exc:
        print(f"envelope_modal: --base-env is not JSON: {exc}", file=sys.stderr)
        return 2
    if not isinstance(base_env, dict):
        print("envelope_modal: --base-env must be a JSON object", file=sys.stderr)
        return 2
    arms = []
    for arm in spec.get("arms") or []:
        merged = dict(arm)
        merged["env"] = {**base_env, **dict(arm.get("env") or {})}
        merged["knobs"] = dict(arm.get("env") or {})
        arms.append(merged)
    names = [arm.get("name") for arm in arms]
    if "base" not in names:
        print("envelope_modal: spec must contain an arm named 'base' with "
              "empty env", file=sys.stderr)
        return 2
    if arms[names.index("base")].get("knobs"):
        print("envelope_modal: the base arm must have an empty env",
              file=sys.stderr)
        return 2

    ref = (worktree / spec["ref"]).resolve()
    if not ref.is_file():
        print(f"envelope_modal: missing ref: {ref}", file=sys.stderr)
        return 2
    base_args = list(spec.get("base_args", []))
    frames = int(spec.get("frames", 60))
    target = float(spec.get("target_kbps", 100.0))

    outdir = (Path(tempfile.mkdtemp(prefix="av2enc-envelope-modal-"))
              if not args.keep else (worktree / "build" / "agent-loop" /
                                     "envelope-streams"))
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"envelope_modal: worktree {worktree}")
    print(f"envelope_modal: ref {ref} ({frames} frames, target {target} kbps)")
    print(f"envelope_modal: {len(arms)} arms, encode containers<="
          f"{MAX_ENCODE_CONTAINERS}, score local", flush=True)

    encode_image = build_encode_image(worktree, ref)
    encode_fn = APP.function(
        image=encode_image, cpu=1, memory=200, timeout=1800,
        max_containers=MAX_ENCODE_CONTAINERS)(_encode_impl)

    seen: dict[str, dict[int, float]] = {arm["name"]: {} for arm in arms}
    done: set[str] = set()
    encode_rows: list[dict] = []
    score_wall = 0.0

    try:
        with APP.run():
            encode_started = time.time()
            while True:
                pending: list[dict] = []
                for arm in arms:
                    if arm["name"] in done:
                        continue
                    qp = _next_qp(seen[arm["name"]], spec, target)
                    if qp is None:
                        done.add(arm["name"])
                        continue
                    pending.append({
                        "arm": arm["name"],
                        "qp": qp,
                        "flags": base_args,
                        "env": arm["env"],
                    })
                if not pending:
                    break
                print("  round: " + ", ".join(
                    f"{job['arm']}@{job['qp']}" for job in pending), flush=True)
                rows = _dispatch(encode_fn, pending)
                for job, row in zip(pending, rows):
                    path = outdir / f"{job['arm']}-qp{job['qp']}.ivf"
                    path.write_bytes(base64.b64decode(row["ivf_b64"]))
                    # Rate is the contract's own formula, so the bracket sees
                    # the same number the local tool would.
                    seen[job["arm"]][job["qp"]] = contract.rate_kbps(
                        path, frames)
                    encode_rows.append(row)
            encode_wall = time.time() - encode_started

            rows = [_points_row(arm, spec, seen[arm["name"]], outdir)
                    for arm in arms]
            score_started = time.time()
            scored = []
            for row in rows:
                if not row["bracket"]:
                    row["k"] = None
                    row["method"] = "no-bracket"
                    scored.append(row)
                    continue
                matched = _score_local(worktree, ref, spec, row, outdir)
                row["k"] = float(matched["kpi"])
                row["method"] = matched.get("method", "linear")
                row["matched_kbps"] = float(matched["kbps"])
                scored.append(row)
            score_wall = time.time() - score_started
    except Exception as exc:  # noqa: BLE001 - the loop must record, not crash
        print(f"envelope_modal: measurement error: {exc}", file=sys.stderr)
        return 1

    base = next(row for row in scored if row["arm"] == "base")
    if base["k"] is None:
        print("envelope_modal: the base arm did not bracket the target",
              file=sys.stderr)
        return 1
    measured = [row for row in scored if row["k"] is not None]
    best = max(measured, key=lambda row: row["k"])
    delta = best["k"] - base["k"]
    envelope_ceiling = None
    base_ceiling = None
    ceiling_delta = None
    ceiling_arm = None
    if args.ceiling_floor is not None:
        # Ceilings are pooling bounds from the local AVM oracle; the matched K
        # above came from the same oracle, so this is the local contract path.
        ceilings = {}
        for row in scored:
            if row["k"] is None:
                continue
            point = row["below"] if row["below"] is not None else row["above"]
            value = contract.measure_ceiling(
                worktree, spec, {"name": row["arm"], "env": row["env"]},
                outdir, args.ceiling_segment, point)
            if value is not None:
                ceilings[row["arm"]] = value
                row["ceiling"] = round(value, 4)
        if ceilings:
            ceiling_arm = max(ceilings, key=ceilings.get)
            envelope_ceiling = ceilings[ceiling_arm]
            base_ceiling = ceilings.get("base")
            if base_ceiling is not None:
                ceiling_delta = envelope_ceiling - base_ceiling
    gate_ok = delta >= args.threshold
    candidate_arms = [row for row in scored if row["arm"] != "base"]
    if args.min_arms and len(candidate_arms) < args.min_arms:
        gate_ok = False
    if args.ceiling_floor is not None:
        gate_ok = (ceiling_delta is not None
                   and ceiling_delta >= args.threshold and gate_ok)
    target_k = None
    if args.best_known is not None:
        target_k = args.best_known + args.threshold
        gate_ok = gate_ok and best["k"] >= target_k
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
        "best_known": args.best_known,
        "target_k": target_k,
        "candidate_arms": len(candidate_arms),
        "gap_svt": round(88.8 - best["k"], 3),
        "gap_avm": round(90.2 - best["k"], 3),
        "ceiling_segment": args.ceiling_segment,
        "ceiling_floor": args.ceiling_floor,
        "base_ceiling": (round(base_ceiling, 4)
                         if base_ceiling is not None else None),
        "envelope_ceiling": (round(envelope_ceiling, 4)
                             if envelope_ceiling is not None else None),
        "ceiling_arm": ceiling_arm,
        "ceiling_delta": (round(ceiling_delta, 4)
                          if ceiling_delta is not None else None),
        "verdict": verdict,
        "arms": scored,
        "backend": {
            "encode": "modal",
            "score": "local",
            "mount": str(worktree),
            "bin": ENCODE_BIN,
            "max_containers": MAX_ENCODE_CONTAINERS,
            "peak_concurrency": _peak_concurrency(encode_rows),
            "encode_wall_seconds": round(encode_wall, 1),
            "score_wall_seconds": round(score_wall, 1),
            "encode_jobs": len(encode_rows),
            "encodes": [
                {key: row[key] for key in
                 ("arm", "qp", "bytes", "md5", "encode_seconds")}
                for row in encode_rows
            ],
        },
    }
    text = json.dumps(report, indent=2)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n")
    print(text)
    ceiling = ("" if envelope_ceiling is None else
               f"  ceiling[{args.ceiling_segment}] {envelope_ceiling:.3f}"
               f" base {base_ceiling:.3f} d {ceiling_delta:+.3f}"
               f" (published {args.ceiling_floor:.3f})")
    target_text = ("" if target_k is None else
                   f"  target {target_k:.3f} (best known "
                   f"{args.best_known:.3f})")
    print(f"envelope: base {base['k']:.3f}  best {best['k']:.3f} "
          f"({best['arm']}, {len(candidate_arms)} knob arms)  "
          f"delta {delta:+.3f}{target_text}{ceiling}  {verdict}", flush=True)
    print(f"envelope_modal: encode {report['backend']['encode_jobs']} jobs "
          f"in {encode_wall:.0f}s (peak "
          f"{report['backend']['peak_concurrency']}), score local in "
          f"{score_wall:.0f}s", flush=True)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
