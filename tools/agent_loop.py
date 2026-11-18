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
3. Measure the target probe and natural 240p QP104/106/108 with the AVM oracle.
4. Keep only a sign-consistent >= +0.15 K gain or a clear probe-gap closure;
   otherwise commit the diagnostic and revert the code.
5. Update {{ledger}} with target, patch, probe/K, kept/negative, and blockers.
   Commit the ledger and any kept patch.
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
    ap.add_argument("--fail-fast", action="store_true",
                    help="stop on the first agent failure")
    ap.add_argument("--dry-run", action="store_true",
                    help="print commands and prompts without invoking the agent")
    args = ap.parse_args()

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

        before = ledger_hash(ledger)
        log = log_dir / f"{i:03d}.json"
        print(f"[{label}] {title} ({len(prompt)} char prompt)", flush=True)
        t0 = time.monotonic()
        try:
            proc = run(cmd, worktree, timeout=args.iteration_timeout or None)
        except subprocess.TimeoutExpired as exc:
            elapsed = time.monotonic() - t0
            log.write_text((exc.stdout or "") + (exc.stderr or ""))
            print(f"  timeout after {elapsed:.1f}s; continuing", flush=True)
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
        log.write_text((proc.stdout or "") + (proc.stderr or ""))
        after = ledger_hash(ledger)
        result = last_text(proc.stdout or "") or "(no text event)"
        changed = before != after
        print(f"  rc={proc.returncode} {elapsed:.1f}s ledger={'changed' if changed else 'UNCHANGED'}",
              flush=True)
        print(f"  {result[:1000]}", flush=True)
        if changed:
            stale = 0
        else:
            stale += 1
        if proc.returncode != 0:
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
