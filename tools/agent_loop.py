#!/usr/bin/env python3
"""Run an OpenCode agent for N tool-improvement iterations.

The script owns the loop. Each invocation performs exactly one iteration, then
exits; the next iteration starts from the ledger. This is deliberately not an
agent that decides how many iterations to run.

Example:
    python3 tools/agent_loop.py --iterations 20

The default worktree is ../agent-loop from the current repository. It is
created from main if missing. The agent reads the queue and updates the ledger;
the script logs every run and never stops after a single iteration.

No third-party code; the agent prompt carries the project rules.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import selectors
import shutil
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_MODEL = "opencode-go/deepseek-v4.1-flash#max"
DEFAULT_AGENT = "general"
DEFAULT_PROMPT = """You are one iteration of an autonomous tool-improvement loop.
Work in {{worktree}}.

Read {{ledger}}, docs/tool-quality.md, docs/tool-improvement-queue-2.md,
docs/tool-improvement-queue-3.md, docs/realization-gaps.md,
docs/negative-results.md and docs/tool-improvement-loop.md first.

Perform EXACTLY ONE iteration. Do not start a second one.
1. Pick the next unblocked target from {{ledger}} if it has one; otherwise from
   docs/tool-quality.md and docs/tool-improvement-queue-2.md.
   Do NOT target key/intra (closed: +0.00 K), eff-perc/filter decision,
   refresh or rate knobs (closed: 21 attempts, 0 kept), or HF RDOQ/trellis
   (closed). Prioritize P-tail prediction/residual/coefficient coding and the
   kept skip/exact-rate composition.
2. Baseline, then ONE contained default-off implementation change. No new
   syntax, no preset/pin change, no third-party code or source copying.
3. Write build/agent-loop/spec.json for the script's envelope gate:
   {"ref": ..., "bin": "build-release/av2enc", "frames": 60,
    "base_args": ["--video","--frames","60","--cpu-threads","8","--fast-decode"],
    "qp": 106, "target_kbps": 100.0,
    "arms": [{"name":"base","env":{}},
             {"name":"cand","env":{"AV2ENC_TOOL":"1"}}]}
   `base` must be present with an empty env and every gate must be default-off.
   Put every knob value of the candidate in arms; the script computes the
   envelope (max K over arms) and the delta against the base arm itself.
4. Do NOT decide keep/revert from your own K numbers. Leave source changes
   uncommitted; the script runs tools/envelope.py after this call, records the
   result, keeps the patch on PASS and reverts src/ on FAIL.
5. Update {{ledger}} with target, patch, the measured bracket, and blockers.
   Do not claim an oracle gain the script did not measure.
6. Run the gate-off pins and tools/test.sh; they must pass.

Do not push or merge. 4:2:0 only. End with one line: ITERATION RESULT: ...
"""


def find_cli(explicit: str | None) -> str:
    if explicit:
        return explicit
    for name in ("OPENCODE_CLI", "OPENCODE_BIN"):
        value = os.environ.get(name)
        if value:
            return value
    for name in ("opencode", "opencode-cli"):
        found = shutil.which(name)
        if found:
            return found
    patterns = [
        str(Path.home() / "Library/Application Support/ai.opencode.desktop/cli/*/opencode-cli"),
        "/Applications/OpenCode.app/Contents/MacOS/*",
    ]
    for pattern in patterns:
        hits = sorted(glob.glob(pattern), reverse=True)
        for hit in hits:
            if os.path.isfile(hit) and os.access(hit, os.X_OK):
                return hit
    raise SystemExit("opencode CLI not found; set OPENCODE_CLI")


def run(cmd: list[str], cwd: Path, timeout: float | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True,
                          timeout=timeout)


def ensure_worktree(repo: Path, worktree: Path, branch: str) -> None:
    if worktree.exists():
        p = run(["git", "-C", str(worktree), "rev-parse", "--is-inside-work-tree"], repo)
        if p.returncode != 0:
            raise SystemExit(f"{worktree} exists but is not a git worktree")
        return
    worktree.parent.mkdir(parents=True, exist_ok=True)
    p = run(["git", "-C", str(repo), "worktree", "add", str(worktree), "-b", branch, "main"], repo)
    if p.returncode != 0:
        p = run(["git", "-C", str(repo), "worktree", "add", str(worktree), branch], repo)
    if p.returncode != 0:
        raise SystemExit((p.stdout or "") + (p.stderr or ""))


def link_fixtures(repo: Path, worktree: Path) -> None:
    for name in ("testdata", "tools/bin"):
        dst = worktree / name
        src = repo / name
        if dst.exists() or dst.is_symlink() or not src.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(src)


def run_agent(cmd: list[str], cwd: Path, log: Path,
              timeout: float | None) -> tuple[int, str]:
    """Run the agent and stream its JSON lines to `log` as they arrive.

    The old loop wrote the log only after the call returned, so a running
    iteration was invisible.  Streaming makes the status view real and keeps
    the same artifact (`<iteration>.json`) for the gate and post-mortems.
    """
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    out: list[str] = []
    with log.open("w") as handle:
        proc = subprocess.Popen(cmd, cwd=str(cwd), text=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, bufsize=1)
        assert proc.stdout is not None
        sel = selectors.DefaultSelector()
        sel.register(proc.stdout, selectors.EVENT_READ)
        try:
            while True:
                if timeout and time.monotonic() - started >= timeout:
                    proc.kill()
                    break
                events = sel.select(timeout=2.0)
                if not events:
                    if proc.poll() is not None:
                        break
                    continue
                line = proc.stdout.readline()
                if line == "":
                    break
                handle.write(line)
                handle.flush()
                out.append(line)
            proc.wait(timeout=10)
        finally:
            sel.close()
    return proc.returncode, "".join(out)


def ledger_hash(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def last_text(stdout: str) -> str:
    text = ""
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "text":
            value = event.get("part", {}).get("text")
            if value:
                text = value
    return text.strip()


def run_envelope(worktree: Path, args, iteration: int, log_dir: Path) -> str:
    spec = worktree / args.envelope_spec
    if not spec.is_file():
        if args.envelope_revert:
            run(["git", "-C", str(worktree), "checkout", "--", "src"], worktree)
            run(["git", "-C", str(worktree), "clean", "-fd", "src"], worktree)
        return "MISSING (no spec; src/ reverted)"
    out = log_dir / f"envelope-{iteration:03d}.json"
    cmd = [sys.executable, str(args.envelope_tool_resolved),
           "--spec", str(spec), "--worktree", str(worktree),
           "--threshold", str(args.envelope_threshold),
           "--base-env", args.envelope_base_env, "--json", str(out)]
    if args.envelope_ceiling_floor is not None:
        cmd += ["--ceiling-floor", str(args.envelope_ceiling_floor),
                "--ceiling-segment", args.envelope_ceiling_segment]
    if args.envelope_best_known is not None:
        cmd += ["--best-known", str(args.envelope_best_known)]
    if args.envelope_min_arms:
        cmd += ["--min-arms", str(args.envelope_min_arms)]
    proc = run(cmd, worktree)
    text = (proc.stdout or "") + (proc.stderr or "")
    verdict = "ERROR" if proc.returncode == 0 else "FAIL"
    report = None
    if out.is_file():
        try:
            report = json.loads(out.read_text())
            verdict = report.get("verdict", "ERROR")
        except json.JSONDecodeError:
            pass
    record = {
        "at": time.time(),
        "iteration": iteration,
        "verdict": verdict,
        "exit": proc.returncode,
        "report": report,
    }
    ledger = worktree / args.envelope_log
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    if verdict in ("FAIL", "MISSING") and args.envelope_revert:
        run(["git", "-C", str(worktree), "checkout", "--", "src"], worktree)
        run(["git", "-C", str(worktree), "clean", "-fd", "src"], worktree)
        return (f"FAIL {report.get('delta', 0.0):+.3f} K; reverted src/")
    if verdict == "PASS":
        return (f"PASS {report.get('delta', 0.0):+.3f} K "
                f"({report.get('best_arm')})")
    if verdict == "SKIP":
        return "SKIP (no spec)"
    return text.strip().splitlines()[-1] if text.strip() else "ERROR"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iterations", type=int, default=10,
                    help="agent calls; 0 = until stopped (default 10)")
    ap.add_argument("--repo", type=Path, default=Path.cwd(),
                    help="source repository (default: cwd)")
    ap.add_argument("--worktree", type=Path, default=None,
                    help="worktree for the loop (default: <repo>/../agent-loop)")
    ap.add_argument("--branch", default="exp/agent-loop",
                    help="branch to create when the worktree is missing")
    ap.add_argument("--ledger", default="docs/agent-loop-ledger.md",
                    help="ledger path relative to the worktree")
    ap.add_argument("--prompt-file", type=Path,
                    help="prompt template with {{iteration}}, {{worktree}}, {{ledger}}")
    ap.add_argument("--model", default=os.environ.get("OPENCODE_MODEL", DEFAULT_MODEL))
    ap.add_argument("--agent", default=os.environ.get("OPENCODE_AGENT", DEFAULT_AGENT))
    ap.add_argument("--cli", default=None, help="path to opencode CLI")
    ap.add_argument("--log-dir", type=Path, default=None,
                    help="log directory (default: <worktree>/build/agent-loop)")
    ap.add_argument("--max-hours", type=float, default=0.0,
                    help="wall-clock budget; 0 means no limit")
    ap.add_argument("--max-stale", type=int, default=0,
                    help="stop after N unchanged ledgers; 0 means no stale stop")
    ap.add_argument("--iteration-timeout", type=float, default=0.0,
                    help="seconds per agent call; 0 means no timeout")
    ap.add_argument("--sleep", type=float, default=2.0,
                    help="seconds between iterations")
    ap.add_argument("--stop-file", type=Path, default=None,
                    help="stop before the next iteration when this file exists")
    ap.add_argument("--envelope-gate", action="store_true",
                    help="run tools/envelope.py after each agent call")
    ap.add_argument("--envelope-tool", default=os.environ.get("AV2ENC_ENVELOPE", ""),
                    help="measurement tool (default: tools/envelope.py; set "
                         "tools/envelope_modal.py to run the ten-lane gate on "
                         "Modal, or use env AV2ENC_ENVELOPE)")
    ap.add_argument("--envelope-spec", default="build/agent-loop/spec.json",
                    help="candidate spec, relative to the worktree")
    ap.add_argument("--envelope-threshold", type=float, default=0.15,
                    help="required envelope delta (default 0.15 K)")
    ap.add_argument("--envelope-base-env", default="{}",
                    help="JSON env shared by every arm (the established recipe)")
    ap.add_argument("--envelope-best-known", type=float, default=None,
                    help="current best matched K; PASS must beat it + delta")
    ap.add_argument("--envelope-min-arms", type=int, default=0,
                    help="minimum non-base arms (a knob sweep)")
    ap.add_argument("--envelope-ceiling-floor", type=float, default=None,
                    help="diagnostic segment ceiling (not gated)")
    ap.add_argument("--envelope-ceiling-segment", default="f0-32",
                    help="segment whose ceiling is gated")
    ap.add_argument("--envelope-log", default="build/agent-loop/envelope.jsonl",
                    help="append-only gate ledger, relative to the worktree")
    ap.add_argument("--envelope-revert", action="store_true",
                    help="on FAIL revert src/ in the worktree")
    ap.add_argument("--fail-fast", action="store_true",
                    help="stop on the first agent failure")
    ap.add_argument("--dry-run", action="store_true",
                    help="print commands and prompts without invoking the agent")
    args = ap.parse_args()

    args.envelope_tool_resolved = (
        Path(args.envelope_tool).expanduser().resolve() if args.envelope_tool
        else Path(__file__).resolve().parent / "envelope.py")
    if not args.envelope_tool_resolved.is_file():
        raise SystemExit(f"envelope tool not found: {args.envelope_tool_resolved}")

    repo = args.repo.resolve()
    check = run(["git", "-C", str(repo), "rev-parse", "--is-inside-work-tree"], repo)
    if check.returncode != 0:
        raise SystemExit(f"not a git repository: {repo}")
    worktree = (args.worktree or (repo.parent / "agent-loop")).resolve()
    ensure_worktree(repo, worktree, args.branch)
    link_fixtures(repo, worktree)

    ledger = worktree / args.ledger
    if not ledger.exists():
        ledger.parent.mkdir(parents=True, exist_ok=True)
        ledger.write_text(
            "# Agent loop ledger\n\n"
            "| # | target | patch | probe | K @100 | verdict |\n"
            "|---:|---|---|---|---:|---|\n")

    if args.prompt_file:
        template = args.prompt_file.read_text()
    else:
        template = DEFAULT_PROMPT

    cli = find_cli(args.cli)
    log_dir = (args.log_dir or (worktree / "build" / "agent-loop")).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run and args.iterations <= 0:
        raise SystemExit("--dry-run needs --iterations > 0")

    started = time.monotonic()
    completed = failed = stale = 0
    i = 0
    while True:
        if args.iterations > 0 and i >= args.iterations:
            break
        i += 1
        if args.stop_file and args.stop_file.exists():
            print(f"stop file present: {args.stop_file}", flush=True)
            break
        if args.max_hours > 0 and (time.monotonic() - started) / 3600 >= args.max_hours:
            print("wall-clock budget reached", flush=True)
            break

        prompt = (template
                  .replace("{{iteration}}", str(i))
                  .replace("{{worktree}}", str(worktree))
                  .replace("{{ledger}}", str(ledger)))
        title = f"agent-loop {i:03d}"
        cmd = [cli, "run", "--auto", "--format", "json", "--archived",
               "--model", args.model, "--agent", args.agent,
               "--title", title, prompt]
        label = f"{i}/{args.iterations}" if args.iterations > 0 else f"{i}/inf"

        if args.dry_run:
            print(f"[{label}] {cli} run --auto --format json --archived "
                  f"--model {args.model} --agent {args.agent} --title {title!r} "
                  f"<prompt {len(prompt)} chars>", flush=True)
            continue

        if args.envelope_gate:
            (worktree / args.envelope_spec).unlink(missing_ok=True)
        before = ledger_hash(ledger)
        log = log_dir / f"{i:03d}.json"
        print(f"[{label}] {title} ({len(prompt)} char prompt)", flush=True)
        t0 = time.monotonic()
        try:
            rc, captured = run_agent(
                cmd, worktree, log, args.iteration_timeout or None)
        except Exception as exc:  # noqa: BLE001 - a broken call must not stop the loop
            elapsed = time.monotonic() - t0
            print(f"  agent call failed after {elapsed:.1f}s: {exc}; continuing",
                  flush=True)
            failed += 1
            stale += 1
            if args.fail_fast:
                break
            if args.max_stale and stale >= args.max_stale:
                print("stale ledger limit reached", flush=True)
                break
            if args.sleep:
                time.sleep(args.sleep)
            continue
        elapsed = time.monotonic() - t0
        after = ledger_hash(ledger)
        result = last_text(captured) or "(no text event)"
        changed = before != after
        print(f"  rc={rc} {elapsed:.1f}s ledger={'changed' if changed else 'UNCHANGED'}",
              flush=True)
        print(f"  {result[:1000]}", flush=True)
        if changed:
            stale = 0
        else:
            stale += 1
        if args.envelope_gate and rc == 0:
            try:
                gate = run_envelope(worktree, args, i, log_dir)
                print(f"  envelope: {gate}", flush=True)
            except Exception as exc:  # noqa: BLE001 - never kill the loop
                print(f"  envelope: measurement error: {exc}", flush=True)
        if rc != 0:
            failed += 1
            if args.fail_fast:
                break
        else:
            completed += 1
        if args.max_stale and stale >= args.max_stale:
            print("stale ledger limit reached", flush=True)
            break
        if args.sleep:
            time.sleep(args.sleep)

    print(f"summary: completed={completed} failed={failed} "
          f"logs={log_dir} ledger={ledger}", flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
