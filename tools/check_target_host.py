#!/usr/bin/env python3
"""Verify that quality reports came from a target host.

Target hosts:
  - RTX 3060 12 GB (development/feature baseline): 12288 MiB, cc 8.6
  - RTX 3080 Ti (production): 12288 MiB, cc 8.6
  - RTX 5090 (maximum speed): >= 30000 MiB, cc 12.0

8 GB cards, RTX 4070 Super and RTX 5070 are no longer targets.
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
    valid = (mem == 12288 and cc == "8.6") or (
        isinstance(mem, int) and mem >= 30000 and cc == "12.0"
    )
    if is_target and valid:
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
            "Not target validation: need an RTX 3060 12 GB or RTX 3080 Ti "
            "(12288 MiB, cc 8.6), or an RTX 5090 (>= 30000 MiB, cc 12.0).",
            file=sys.stderr,
        )
        return 1
    print("PASS: target host (RTX 3060 12 GB / 3080 Ti / 5090)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
