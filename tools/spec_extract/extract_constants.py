#!/usr/bin/env python3
"""Extract the constant table (spec section 3, "Symbols") from the AV2 HTML spec.

The constants live in one 338-row HTML table.  Values are integer expressions
over earlier constants, so they are evaluated in table order with a tiny
arithmetic evaluator (ints, names, + - * /, << >>, parentheses).

The HTML is fetched once and cached under spec/tables/_cache/ (git-ignored).
Only the official specification is read.

Usage:
    python3 tools/spec_extract/extract_constants.py [--html path]
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import operator
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "spec" / "tables"
CACHE = OUT / "_cache"
URL = "https://av2.aomedia.org/v1.0.0/index.html"

_BIN = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv,
    ast.Div: operator.floordiv,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
    ast.BitOr: operator.or_,
    ast.BitAnd: operator.and_,
    ast.Mod: operator.mod,
}


def evaluate(expr: str, env: dict[str, int]) -> int:
    tree = ast.parse(expr, mode="eval")

    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in env:
                raise KeyError(node.id)
            return env[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            return _BIN[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -walk(node.operand)
        raise ValueError(f"unsupported expression: {expr!r}")

    return walk(tree)


def fetch_html(path: Path | None) -> Path:
    if path is not None:
        return path
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = CACHE / "spec.html"
    if not cached.exists():
        print(f"fetching {URL} -> {cached}")
        urllib.request.urlretrieve(URL, cached)
    return cached


def parse_table(html_path: Path) -> list[tuple[str, str, str]]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "lxml")
    for table in soup.find_all("table"):
        text = table.get_text(" ", strip=True)
        if "MAXQ_8_BITS" in text and "Symbol name" in text:
            rows = []
            for tr in table.find_all("tr"):
                cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
                if len(cells) != 3 or cells[0] == "Symbol name":
                    continue
                rows.append((cells[0], cells[1], cells[2]))
            return rows
    raise SystemExit("constants table not found in HTML spec")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", type=Path, default=None)
    args = ap.parse_args()

    html = fetch_html(args.html)
    rows = parse_table(html)
    if len(rows) < 300:
        raise SystemExit(f"expected ~338 constants, found {len(rows)}")

    OUT.mkdir(parents=True, exist_ok=True)
    env: dict[str, int] = {}
    unresolved: list[str] = []
    out_rows = []
    for name, value, desc in rows:
        value = value.replace("\u2212", "-").replace(" ", "")
        try:
            number = evaluate(value, env)
            env[name] = number
            out_rows.append((name, str(number), value, desc))
        except (KeyError, ValueError, SyntaxError):
            unresolved.append(name)
            out_rows.append((name, "", value, desc))

    with (OUT / "constants.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "value", "expression", "description"])
        w.writerows(out_rows)

    (OUT / "constants.json").write_text(json.dumps(env, indent=1, sort_keys=True) + "\n")

    if unresolved:
        print(f"note: {len(unresolved)} constants not evaluated "
              f"(reserved/unknown names): {unresolved[:8]}")
    for key in ("MAXQ_8_BITS", "MAXQ_10_BITS", "MAXQ_OFFSET", "MAX_TILE_WIDTH",
                "MAX_TILE_AREA", "MAX_TILE_COLS", "MAX_TILE_ROWS"):
        if key in env:
            print(f"  {key} = {env[key]}")


if __name__ == "__main__":
    main()
