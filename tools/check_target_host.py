#!/usr/bin/env python3
"""Verify that quality reports came from the target 8 GB RTX 3060.

Target host:
  - GPU memory enumerated as 8192 MiB
  - compute capability 8.6 (Ampere)

The 12 GB dev box (12288 MiB) is a compatibility proxy only. Its speed and QBS
numbers are not target validation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def check(path: Path) -> tuple[bool, str]:
    d = json.loads(path.read_text())
    host = d.get("host") or {}
    target = d.get("target_host") or {}
    mem = host.get("gpu_memory_mib")
    cc = str(host.get("gpu_compute_cap", "")).strip()
    is_target = bool(target.get("is_target", False))
    if is_target and mem == 8192 and cc == "8.6":
        return True, f"{mem} MiB, cc {cc}"
    return False, f"{mem} MiB, cc {cc or 'unknown'}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="+", type=Path)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    rows = []
    ok = True
    for p in args.reports:
        good, detail = check(p)
        ok = ok and good
        rows.append({"report": str(p), "target": good, "detail": detail})
        print(f"{'PASS' if good else 'FAIL'}  {p}  ({detail})")

    if args.json:
        args.json.write_text(json.dumps({"target": ok, "reports": rows}, indent=2) + "\n")

    if not ok:
        print(
            "Not target validation: need an RTX 3060 enumerating 8192 MiB "
            "(compute capability 8.6). The 12 GB dev box is a proxy.",
            file=sys.stderr,
        )
        return 1
    print("PASS: target 8 GB RTX 3060")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
