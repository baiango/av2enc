#!/usr/bin/env python3
"""Regenerate every file in spec/tables/ from the pinned spec artefacts.

    python3 tools/spec_extract/build_all.py

Steps:
  1. extract_constants.py  - spec section 3 constant table (values)  -> constants.csv/json
  2. extract_tables.py     - sections 9.2/9.3/9.6 + prose enums      -> the rest
  3. extract_sections.py   - HTML heading map                        -> spec_sections.json

The PDF is the source of truth for numeric tables. The HTML is used only for the
constant table and the section map (the PDF prints both too, but with layout
damage that makes them unreliable to parse).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def run(script: str, *args: str) -> None:
    cmd = [sys.executable, str(HERE / script), *args]
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> None:
    run("extract_constants.py")
    run("extract_tables.py")
    run("extract_sections.py")
    print("done")


if __name__ == "__main__":
    main()
