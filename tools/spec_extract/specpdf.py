"""Helpers to pull normative tables out of the AV2 v1.0.0 specification PDF.

The spec HTML (av2.aomedia.org/v1.0.0) omits the numeric tables: section 9.3
("Default CDF tables") and 9.6 ("1d transform tables") only say "is defined in
the <name>.h header file".  The PDF renders the same sections with the actual
numbers, so the PDF is the source for every generated table in spec/tables/.

Only the official spec PDF is read here. No encoder source code.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

SPEC_PDF = Path(__file__).resolve().parents[2] / "spec" / "AV2_Spec_v1.0.0.pdf"
SPEC_SHA256 = "e9916f091e4e83446aad6b4601641c5b292e569c144c4163b26a4497573b533f"

_PAGE_MARK = re.compile(r"\n<<<PAGE \d+>>>\n")
_PAGE_HEAD = re.compile(r"\nAV2 Specification Page \d+ of \d+\n")


def pdf_text(pdf_path: Path = SPEC_PDF) -> str:
    """Return the PDF text with page furniture stripped.

    Page headers/footers are removed so that arrays that straddle a page break
    parse as one contiguous initializer (several do).
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("pypdf is required: python3 -m pip install pypdf") from exc

    if not pdf_path.exists():
        raise SystemExit(f"missing spec PDF: {pdf_path}")

    digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    if digest != SPEC_SHA256:
        print(
            f"warning: {pdf_path.name} sha256 {digest} != pinned {SPEC_SHA256}",
            file=sys.stderr,
        )

    reader = PdfReader(str(pdf_path))
    chunks = []
    for page in reader.pages:
        chunks.append(f"\n<<<PAGE {page.page_number}>>>\n")
        chunks.append(page.extract_text() or "")
    txt = "".join(chunks)
    txt = _PAGE_MARK.sub("\n", txt)
    txt = _PAGE_HEAD.sub("\n", txt)
    return txt


class CArray:
    """A parsed C-style array definition."""

    def __init__(self, name: str, dims_decl: list[str], data, source_line: int):
        self.name = name
        self.dims_decl = dims_decl  # raw text between the brackets
        self.data = data            # nested lists / ints / identifiers
        self.source_line = source_line

    @property
    def shape(self) -> list[int]:
        return infer_shape(self.data)

    @property
    def values(self) -> list:
        return flatten(self.data)

    def __repr__(self) -> str:  # pragma: no cover
        return f"CArray({self.name}, shape={self.shape})"


def infer_shape(obj) -> list[int]:
    shape = []
    while isinstance(obj, list):
        shape.append(len(obj))
        if not obj:
            break
        obj = obj[0]
    return shape


def flatten(obj) -> list:
    if isinstance(obj, list):
        out = []
        for item in obj:
            out.extend(flatten(item))
        return out
    return [obj]


_TOKEN = re.compile(r"\s*(?:(?P<num>-?\d+)|(?P<ident>[A-Za-z_][A-Za-z0-9_]*)|(?P<brace>[{}])|(?P<comma>,))")


def _sanitize_body(body: str) -> tuple[str, int]:
    """Drop prose lines that the PDF's two-column layout interleaves into arrays.

    Array initializer lines always end with ',', '{' or '}'.  Running text does
    not (it ends with '.' or a word), so this removes e.g. the four "For a
    transform size t ..." lines that are interleaved into Tx_Size_Sqr_Up.
    """
    kept = []
    dropped = 0
    body = re.sub(r"//[^\n]*", "", body)
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
    for line in body.split("\n"):
        stripped = line.strip()
        if not stripped:
            kept.append("")
            continue
        if re.fullmatch(r"[\s\dA-Za-z_,\-+{}]+", stripped) and not re.search(r"\b[a-z]{2,}\b", stripped):
            kept.append(line)
            continue
        dropped += 1
    return "\n".join(kept), dropped


def _tokenize(body: str):
    body = re.sub(r"//[^\n]*", "", body)
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
    pos = 0
    n = len(body)
    while pos < n:
        m = _TOKEN.match(body, pos)
        if not m:
            # Anything else (stray text from a page break) -> stop cleanly.
            raise ValueError(f"unexpected input at {pos}: {body[pos:pos+40]!r}")
        pos = m.end()
        if m.group("num") is not None:
            yield ("num", int(m.group("num")))
        elif m.group("ident") is not None:
            yield ("ident", m.group("ident"))
        elif m.group("brace") is not None:
            yield ("brace", m.group("brace"))
        elif m.group("comma") is not None:
            yield ("comma", ",")


def _parse_value(tokens, idx):
    """Parse one value (scalar or brace-delimited list) starting at tokens[idx]."""
    kind, val = tokens[idx]
    if kind == "brace" and val == "{":
        idx += 1
        items = []
        while True:
            kind2, val2 = tokens[idx]
            if kind2 == "brace" and val2 == "}":
                return items, idx + 1
            if kind2 == "comma":
                idx += 1
                continue
            item, idx = _parse_value(tokens, idx)
            items.append(item)
    if kind in ("num", "ident"):
        return val, idx + 1
    raise ValueError(f"unexpected token {kind}:{val}")


def find_array(text: str, name: str) -> CArray | None:
    """Locate `name[...]... = { ... }` and parse the initializer."""
    pattern = re.compile(
        r"^[ \t]*" + re.escape(name) + r"\s*((?:\[[^\]\n]*\]\s*)+)=\s*\{",
        re.M,
    )
    m = pattern.search(text)
    if not m:
        return None
    dims_decl = re.findall(r"\[([^\]]*)\]", m.group(1))
    start = text.index("{", m.start())
    depth = 0
    end = None
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end is None:
        raise ValueError(f"unbalanced braces for {name}")
    body = text[start : end + 1]
    body, dropped = _sanitize_body(body)
    if dropped:
        print(f"note: {name}: dropped {dropped} interleaved prose line(s)", file=sys.stderr)
    tokens = list(_tokenize(body))
    value, idx = _parse_value(tokens, 0)
    if idx != len(tokens):
        # Trailing junk inside the array body means our brace scan ran into the
        # next definition; treat as a parse failure rather than silent garbage.
        raise ValueError(f"{name}: {len(tokens) - idx} trailing tokens")
    line = text.count("\n", 0, m.start()) + 1
    return CArray(name, dims_decl, value, line)


def find_arrays(text: str, prefix: str) -> list[CArray]:
    names = []
    for m in re.finditer(r"^[ \t]*(" + re.escape(prefix) + r"[A-Za-z0-9_]*)\s*(?:\[[^\]\n]*\]\s*)+=\s*\{", text, re.M):
        names.append(m.group(1))
    out = []
    for name in dict.fromkeys(names):
        arr = find_array(text, name)
        if arr is not None:
            out.append(arr)
    return out


def section(text: str, start_marker: str, end_marker: str) -> str:
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    return text[start:end]


def dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, separators=(",", ":"), sort_keys=False) + "\n")


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(f"spec_extract: {msg}")
