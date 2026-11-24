#!/usr/bin/env bash
# Glued-statement check: catch a newline lost between a block-opening brace and
# the statement that follows it.
#
# The defect looks like this (seen twice on this tree, in rdoq.cpp and in seven
# functions across the video/svc/cuda paths):
#
#     bool write_ivf(...) {  av2enc::IvfWriter writer;      // two statements
#     if (!write(...)) {        error = "failed: " + e;     // one line
#
# It compiles because the statements are independent, so no compiler, test or
# bitstream gate notices.  The only real symptom is an over-long line.
#
# Detection: a line whose last code '{' is followed by non-empty code that ends
# in ';' with no '}' on the same line.  The scan skips '//' comments and string
# and character literals, so JSON builders (`append("{\"type\":...")`) and spec
# quotes in comments do not match.  That separates a lost newline from the two
# legitimate shapes that look similar:
#   * one-line accessors and inline bodies   `{ return x; }`   (has '}')
#   * aggregate initialisers                 `{a, b, c,`       (no ';')
#
# Usage:
#   tools/refactor/check_glued_statements.sh [--bin PATH] [--dav2d PATH]
#
# Exit: 0 clean, 1 a glued statement, 2 usage error.
set -uo pipefail

while [ "$#" -gt 0 ]; do
    case "$1" in
        --bin|--dav2d) shift 2 ;;   # tools/run_gates.sh compatibility
        -h|--help) sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "check_glued_statements: unknown argument: $1" >&2; exit 2 ;;
    esac
done

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 2

command -v python3 >/dev/null 2>&1 || {
    echo "check_glued_statements: python3 required" >&2; exit 2; }

python3 - <<'PY'
import pathlib, subprocess, sys

files = subprocess.run(
    ["bash", "-c", "find src -name '*.cpp' -o -name '*.h'"],
    capture_output=True, text=True).stdout.split()

# Generated tables and golden fixtures are not hand-written; they legitimately
# carry very wide rows and are exempt.
EXEMPT = ("_tables.h", "_table.h", "generated", "goldens", "provenance")


def code_only(line):
    """Blank out // comments and string/char literals, keeping columns."""
    out, i, n = [], 0, len(line)
    quote = None
    while i < n:
        c = line[i]
        if quote:
            if c == "\\" and i + 1 < n:
                out.append("  "); i += 2; continue
            if c == quote:
                quote = None
            out.append(" "); i += 1; continue
        if c in "\"'":
            quote = c; out.append(" "); i += 1; continue
        if c == "/" and i + 1 < n and line[i + 1] == "/":
            out.append(" " * (n - i)); break
        out.append(c); i += 1
    return "".join(out)


bad = []
for f in files:
    if any(x in f for x in EXEMPT):
        continue
    for i, line in enumerate(pathlib.Path(f).read_text().splitlines(), 1):
        code = code_only(line)
        bi = code.rfind("{")
        if bi < 0:
            continue
        body = code[bi + 1:].strip()
        if not body or body.startswith("}"):
            continue
        if body.endswith(";") and "}" not in body:
            bad.append((f, i, line.strip()))

if bad:
    print(f"check_glued_statements: {len(bad)} lost newline(s)\n", file=sys.stderr)
    for f, i, s in bad:
        print(f"  {f}:{i}", file=sys.stderr)
        print(f"      {s[:120]}", file=sys.stderr)
    print("\nSplit the statement onto its own line.", file=sys.stderr)
    sys.exit(1)

print("check_glued_statements: OK (no glued statements)")
PY
