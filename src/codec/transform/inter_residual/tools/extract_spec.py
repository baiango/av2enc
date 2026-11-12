#!/usr/bin/env python3
"""Dump one AV2 spec section from the cached HTML as plain text.

Usage: extract_spec.py <section-id> [...]
The cache lives at spec/tables/_cache/spec.html (see spec/tables/README.md).
This helper is documentation tooling for the M2 inter-residual component; it is
not part of the build.
"""
import html
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
CACHE = os.path.join(ROOT, "spec", "tables", "_cache", "spec.html")


def text_of(fragment):
    fragment = re.sub(r"<br\s*/?>", "\n", fragment)
    fragment = re.sub(r"</(p|div|li|tr|pre|table|h[1-6])>", "\n", fragment)
    fragment = re.sub(r"</t[dh]>", " | ", fragment)
    fragment = re.sub(r"<[^>]+>", "", fragment)
    fragment = html.unescape(fragment)
    fragment = re.sub(r"[ \t]+", " ", fragment)
    fragment = re.sub(r"\n\s*\n+", "\n", fragment)
    return fragment.strip()


def main(argv):
    with open(CACHE, encoding="utf-8", errors="replace") as handle:
        doc = handle.read()
    heads = [
        (m.start(), m.group(1))
        for m in re.finditer(r'<h[1-6][^>]*id="([^"]+)"', doc)
    ]
    for wanted in argv:
        hit = next(((pos, sid) for pos, sid in heads if sid == wanted), None)
        if hit is None:
            print(f"!! section {wanted} not found")
            continue
        start = hit[0]
        nxt = next((pos for pos, _ in heads if pos > start), len(doc))
        print(f"===== {wanted} =====")
        print(text_of(doc[start:nxt]))
        print()


if __name__ == "__main__":
    main(sys.argv[1:])
