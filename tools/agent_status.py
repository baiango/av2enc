#!/usr/bin/env python3
"""One-screen status for the five tool-improvement lanes.

Shows, per lane: whether the runner is alive, which iteration is in flight,
how long it has been running, the last tool call and text from the agent's
streamed JSON log, the last mechanical gate result, and the size of the
uncommitted source change.

Usage:
  tools/agent_status.py
  tools/agent_status.py --prefix agent-loop-oracle --lanes 5
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = re.compile(r"Your exclusive target:\s*(.*)")
ROW = re.compile(r"^\|\s*\d+\s*\|")


def ps_table() -> list[tuple[int, str, str]]:
    proc = subprocess.run(["ps", "-axo", "pid=,etime=,command="],
                          text=True, capture_output=True)
    rows = []
    for line in proc.stdout.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) == 3 and parts[0].isdigit():
            rows.append((int(parts[0]), parts[1], parts[2]))
    return rows


def human(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 90:
        return f"{seconds}s"
    if seconds < 5400:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


def tail_json(path: Path, window: int = 2_000_000) -> dict:
    try:
        data = path.read_bytes()
    except OSError:
        return {}
    if len(data) > window:
        data = data[-window:]
        data = data.split(b"\n", 1)[-1]
    info = {"text": None, "tool": None, "hint": None,
            "start": None, "events": 0}
    for raw in data.splitlines():
        try:
            event = json.loads(raw)
        except Exception:  # noqa: BLE001
            continue
        info["events"] += 1
        kind = event.get("type")
        part = event.get("part") or {}
        if info["start"] is None and event.get("timestamp"):
            info["start"] = event["timestamp"] / 1000.0
        if kind == "text":
            text = (part.get("text") or "").strip()
            if text:
                info["text"] = text.splitlines()[-1][:180]
        elif kind == "tool_use":
            info["tool"] = part.get("tool")
            state = part.get("state") or {}
            data_in = state.get("input") or {}
            if isinstance(data_in, dict):
                for key in ("command", "description", "filePath", "path",
                            "pattern"):
                    if data_in.get(key):
                        hint = str(data_in[key]).replace("\n", " ")
                        info["hint"] = hint[:160]
                        break
    return info


def last_gate(path: Path) -> dict | None:
    try:
        lines = [line for line in path.read_text().splitlines() if line.strip()]
    except OSError:
        return None
    if not lines:
        return None
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=ROOT)
    ap.add_argument("--prefix", default="agent-loop-oracle")
    ap.add_argument("--lanes", type=int, default=5)
    args = ap.parse_args()

    repo = args.repo.resolve()
    procs = ps_table()
    stop = (repo.parent / "agent-loop" / "STOP").exists()
    print(f"status {time.strftime('%H:%M:%S')}  root={repo}  "
          f"stop-file={'PRESENT' if stop else 'no'}")
    live = 0
    for i in range(1, args.lanes + 1):
        worktree = (repo.parent / f"{args.prefix}-{i}").resolve()
        if not worktree.exists():
            print(f"lane {i}: no worktree {worktree}")
            continue
        def owns(cmd: str) -> bool:
            if "agent_loop.py" not in cmd:
                return False
            parts = cmd.split()
            for j, token in enumerate(parts):
                if token == "--worktree" and j + 1 < len(parts):
                    try:
                        return Path(parts[j + 1]).resolve() == worktree
                    except OSError:
                        return False
            return False

        runner = next((pid for pid, _e, cmd in procs if owns(cmd)), None)
        if runner:
            live += 1
        target = ""
        task = worktree / "docs" / "lane-tasks" / f"lane-{i}.md"
        try:
            for line in task.read_text().splitlines():
                if line.startswith("# "):
                    target = line[2:].strip()[:90]
                    break
        except OSError:
            prompt = repo / "build" / "agent-loop" / f"lane-{i}-prompt.txt"
            try:
                for line in prompt.read_text().splitlines():
                    match = TARGET.search(line)
                    if match:
                        target = match.group(1)[:90]
                        break
            except OSError:
                pass

        logs = sorted((worktree / "build" / "agent-loop").glob("[0-9][0-9][0-9].json"),
                      key=lambda p: p.stat().st_mtime, reverse=True) \
            if (worktree / "build" / "agent-loop").is_dir() else []
        log = logs[0] if logs else None
        info = tail_json(log) if log else {}
        elapsed = 0.0
        if log:
            started = info.get("start") or log.stat().st_birthtime
            elapsed = time.time() - started
        gate = last_gate(worktree / "build" / "agent-loop" / "envelope.jsonl")
        try:
            src_changes = len(subprocess.run(
                ["git", "-C", str(worktree), "status", "--short", "src"],
                text=True, capture_output=True).stdout.splitlines())
        except Exception:  # noqa: BLE001
            src_changes = -1
        ledger = worktree / "docs" / f"agent-loop-ledger-{i}.md"
        rows = 0
        try:
            rows = sum(1 for line in ledger.read_text().splitlines()
                       if ROW.match(line))
        except OSError:
            pass

        state = "RUN " if runner else "IDLE"
        name = log.name if log else "-"
        print(f"lane {i} {state} iter {name} {human(elapsed):>7}  "
              f"src {src_changes} files  ledger {rows} rows  :: {target}")
        if info.get("tool"):
            print(f"        now: {info['tool']}: {info.get('hint') or ''}")
        if info.get("text"):
            print(f"        last: {info['text']}")
        if gate:
            report = gate.get("report") or {}
            ceiling = ""
            if report.get("envelope_ceiling") is not None:
                ceiling = (f" ceiling[{report.get('ceiling_segment')}] "
                           f"{report['envelope_ceiling']:.3f} vs "
                           f"{report.get('ceiling_floor'):.3f}")
            age = ""
            if gate.get("at"):
                age = f" ({human(time.time() - gate['at'])} ago)"
            print(f"        gate{age}: it{gate.get('iteration')} "
                  f"{gate.get('verdict')} env {report.get('envelope_k')} "
                  f"base {report.get('baseline_k')} "
                  f"d {report.get('delta')}{ceiling}")
    print(f"runners alive: {live}/{args.lanes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
