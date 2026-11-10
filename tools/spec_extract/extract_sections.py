#!/usr/bin/env python3
"""Emit the spec's section map (id -> number/title) as JSON.

Uses the same cached HTML as extract_constants.py. This is only a navigation aid
for the docs; nothing normative is derived from it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from extract_constants import CACHE, URL, fetch_html

OUT = Path(__file__).resolve().parents[2] / "spec" / "tables"


def main() -> None:
    from bs4 import BeautifulSoup

    html_path = fetch_html(None)
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "lxml")
    sections = []
    for h in soup.find_all(["h2", "h3", "h4", "h5"]):
        sid = h.get("id")
        if not sid:
            continue
        title = h.get_text(" ", strip=True)
        sections.append({"id": sid, "level": int(h.name[1]), "title": title})
    if len(sections) < 500:
        raise SystemExit(f"section map looks wrong: {len(sections)} headings")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "spec_sections.json").write_text(json.dumps(sections, indent=0) + "\n")
    print(f"wrote {len(sections)} headings to {OUT / 'spec_sections.json'}")


if __name__ == "__main__":
    sys.exit(main())
