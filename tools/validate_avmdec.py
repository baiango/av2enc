#!/usr/bin/env python3
"""Validate an AV2 IVF stream with AVM's reference decoder (`avmdec`).

The pinned `dav2d` f9dafbc has a §7.21.6 output-drain defect for a non-key
refresh that overwrites an implicit picture (a chained single-key multi-GOP
stream emits only the final store contents).  AVM's `avmdec` runs the drain
correctly, so chained structures are validated here until the `dav2d` pin
moves.  Flat key+P streams must still pass `tools/validate_dav2d.sh`.

Usage:
  tools/validate_avmdec.py out.ivf --ref source.y4m [--recon enc.y4m]
                          [--json out.json] [--keep]

Exit codes: 0 ok, 1 decode/recon failure, 2 usage.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from quality import decode_avm, find_avmdec  # noqa: E402


def payload(path: Path) -> bytes:
    data = path.read_bytes()
    start = data.find(b"FRAME")
    return data[start:] if start >= 0 else b""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ivf", type=Path)
    ap.add_argument("--ref", type=Path, required=True,
                    help="source Y4M (geometry for the normalized output)")
    ap.add_argument("--recon", type=Path,
                    help="encoder --video-recon Y4M to compare byte for byte")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()

    if not args.ivf.is_file():
        print(f"validate_avmdec: missing {args.ivf}", file=sys.stderr)
        return 2
    avmdec = find_avmdec()
    tmp = Path(tempfile.mkdtemp(prefix="avmdec-val-"))
    out = (tmp if args.keep else tmp) / "decoded.y4m"
    try:
        decode_avm(avmdec, args.ivf, args.ref, out, args.threads)
    except SystemExit as exc:
        print(f"validate_avmdec: avmdec failed: {exc}", file=sys.stderr)
        return 1

    frames = payload(out).count(b"FRAME")
    ok = frames > 0
    recon_ok = None
    if args.recon is not None:
        recon_ok = payload(args.recon) == payload(out)
        ok = ok and recon_ok
    report = {
        "decoder": avmdec,
        "ivf": str(args.ivf),
        "ivf_bytes": args.ivf.stat().st_size,
        "frames": frames,
        "recon_equal": recon_ok,
        "ok": ok,
    }
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    print(f"avmdec: {frames} frame(s), recon="
          f"{'ok' if recon_ok else ('n/a' if recon_ok is None else 'MISMATCH')}, "
          f"{'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
