#!/usr/bin/env python3
"""Pinned dav2d gate for the `dc_sign_horz_vert` minimal streams.

`gate_dc_sign_horz_vert` writes four 32x16 IVF files:

  hv_pos.ivf   H_DCT 32x16 luma, coefficient at (row 1, col 0), sign = 0,
               dc_sign_horz_vert coded with the normative context 0
  hv_zero.ivf  eob = 1 (no horz-vert coefficient, DCT_DCT control)
  hv_ctx1.ivf  same as hv_pos but the horz-vert sign is coded with ctx 1
  hv_neg.ivf   same as hv_pos but sign = 1

Gate:
  * hv_pos and hv_zero must decode with `dav2d --strict 1 --threads 1`,
    and both IVF and decoded Y4M MD5s must match the pins;
  * hv_ctx1 must be rejected (the decoder derives ctx 0, so any other
    context desynchronises);
  * hv_neg must decode with `--strict 0` to a pinned, different image, and
    is *documented* to be rejected by this dav2d build under `--strict 1`
    (a decoder-side check on negative 1D-transform coefficients; see
    README.md).  The gate pins that behaviour too, so a decoder upgrade that
    changes it is noticed.

Usage:
  python3 gate_dav2d.py --dir DIR [--decoder PATH] [--update]

Exit codes: 0 pass, 1 gate failure, 2 setup error.
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[4]

# Pinned 2026-11-12 against tools/bin/dav2d-macos (f9dafbc).
IVF_MD5 = {
    "hv_pos": "ada97e3f12b8582c65e5c2ac0826b0ec",
    "hv_zero": "a398d9cfb44d6ed8ceed044121e6a65b",
    "hv_ctx1": "75d363fcdc1ce67f53eae2445675340b",
    "hv_neg": "9275c76e44a3492229750c3737c53d6d",
}
DECODED_MD5 = {
    "hv_pos": "7a1349d9de949b949a05682fc606af86",
    "hv_zero": "54d3959bbd3c80c05af8f900e417a069",
    "hv_neg": "ebfbfaa4b8ecc8eb96547ca22a23c986",
}


def md5_file(path: pathlib.Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_decoder(explicit: str) -> pathlib.Path:
    if explicit:
        return pathlib.Path(explicit)
    for candidate in (REPO / "tools/bin/dav2d", REPO / "tools/bin/dav2d-macos"):
        if candidate.is_file() and candidate.stat().st_mode & 0o111:
            return candidate
    raise SystemExit("gate: no dav2d binary (set DAV2D or --decoder)")


def decode(decoder: pathlib.Path, ivf: pathlib.Path, y4m: pathlib.Path,
           strict: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(decoder), "-q", "-i", str(ivf), "-o", str(y4m),
         "--threads", "1", "--strict", str(strict)],
        capture_output=True, text=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--decoder", default="")
    parser.add_argument("--update", action="store_true",
                        help="print discovered pins and exit 0")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.dir)
    if not out_dir.is_dir():
        print(f"gate: no such directory: {out_dir}", file=sys.stderr)
        return 2
    decoder = find_decoder(args.decoder)
    if not (decoder.is_file() and decoder.stat().st_mode & 0o111):
        print(f"gate: decoder not executable: {decoder}", file=sys.stderr)
        return 2

    failures = 0
    ivf_md5: dict[str, str] = {}
    decoded: dict[str, str] = {}

    for name in ("hv_pos", "hv_zero", "hv_ctx1", "hv_neg"):
        ivf = out_dir / f"{name}.ivf"
        if not ivf.is_file():
            print(f"FAIL {name}: missing {ivf}", file=sys.stderr)
            failures += 1
            continue
        ivf_md5[name] = md5_file(ivf)

    for name in ("hv_pos", "hv_zero"):
        ivf = out_dir / f"{name}.ivf"
        if name not in ivf_md5:
            continue
        y4m = out_dir / f"{name}.y4m"
        result = decode(decoder, ivf, y4m, 1)
        if result.returncode != 0:
            print(f"FAIL {name}: dav2d --strict 1 rejected the stream",
                  file=sys.stderr)
            print(result.stdout + result.stderr, file=sys.stderr)
            failures += 1
            continue
        decoded[name] = md5_file(y4m)
        if not args.update:
            if IVF_MD5[name] and ivf_md5[name] != IVF_MD5[name]:
                print(f"FAIL {name}: IVF md5 {ivf_md5[name]}, expected "
                      f"{IVF_MD5[name]}", file=sys.stderr)
                failures += 1
            if DECODED_MD5[name] and decoded[name] != DECODED_MD5[name]:
                print(f"FAIL {name}: decoded md5 {decoded[name]}, expected "
                      f"{DECODED_MD5[name]}", file=sys.stderr)
                failures += 1
        print(f"{name}: ivf={ivf_md5[name]} decoded={decoded[name]}")

    # Wrong-context control must be rejected.
    if "hv_ctx1" in ivf_md5:
        ivf = out_dir / "hv_ctx1.ivf"
        result = decode(decoder, ivf, out_dir / "hv_ctx1.y4m", 1)
        if result.returncode == 0 and not args.update:
            print("FAIL hv_ctx1: dav2d accepted a non-normative "
                  "dc_sign_horz_vert context", file=sys.stderr)
            failures += 1
        else:
            print(f"hv_ctx1: rejected as expected (ivf={ivf_md5['hv_ctx1']})")
        if not args.update and IVF_MD5["hv_ctx1"] and \
                ivf_md5["hv_ctx1"] != IVF_MD5["hv_ctx1"]:
            print(f"FAIL hv_ctx1: IVF md5 {ivf_md5['hv_ctx1']}, expected "
                  f"{IVF_MD5['hv_ctx1']}", file=sys.stderr)
            failures += 1

    # Negative 1D coefficient: pinned under --strict 0, documented strict-1
    # rejection.
    if "hv_neg" in ivf_md5:
        ivf = out_dir / "hv_neg.ivf"
        y4m = out_dir / "hv_neg.y4m"
        result = decode(decoder, ivf, y4m, 0)
        if result.returncode != 0:
            print("FAIL hv_neg: dav2d --strict 0 rejected the stream",
                  file=sys.stderr)
            print(result.stdout + result.stderr, file=sys.stderr)
            failures += 1
        else:
            decoded["hv_neg"] = md5_file(y4m)
            if not args.update and DECODED_MD5["hv_neg"] and \
                    decoded["hv_neg"] != DECODED_MD5["hv_neg"]:
                print(f"FAIL hv_neg: decoded md5 {decoded['hv_neg']}, "
                      f"expected {DECODED_MD5['hv_neg']}", file=sys.stderr)
                failures += 1
            print(f"hv_neg: ivf={ivf_md5['hv_neg']} "
                  f"decoded={decoded['hv_neg']} (strict 0)")
        strict_result = decode(decoder, ivf, out_dir / "hv_neg.strict.y4m", 1)
        if strict_result.returncode == 0 and not args.update:
            print("FAIL hv_neg: expected this dav2d build to reject negative "
                  "1D coefficients under --strict 1; update README gates",
                  file=sys.stderr)
            failures += 1
        elif not args.update:
            print("hv_neg: rejected under --strict 1 (pinned decoder check)")
        if not args.update and IVF_MD5["hv_neg"] and \
                ivf_md5["hv_neg"] != IVF_MD5["hv_neg"]:
            print(f"FAIL hv_neg: IVF md5 {ivf_md5['hv_neg']}, expected "
                  f"{IVF_MD5['hv_neg']}", file=sys.stderr)
            failures += 1

    if args.update:
        print("IVF_MD5 = {")
        for name, value in ivf_md5.items():
            print(f'    "{name}": "{value}",')
        print("}")
        print("DECODED_MD5 = {")
        for name, value in decoded.items():
            print(f'    "{name}": "{value}",')
        print("}")
        return 0

    # The sign must change the reconstructed image.
    if "hv_pos" in decoded and "hv_neg" in decoded:
        if decoded["hv_pos"] == decoded["hv_neg"]:
            print("FAIL: hv_pos and hv_neg decoded identically; the "
                  "dc_sign_horz_vert sign was not consumed", file=sys.stderr)
            failures += 1

    if failures:
        print(f"gate: {failures} failure(s)", file=sys.stderr)
        return 1
    print("gate: PASS (strict pins + context control + sign control)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
