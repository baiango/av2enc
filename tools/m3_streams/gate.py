#!/usr/bin/env python3
"""M3 per-tool stream gate.

For each wireable tool, build the smallest valid 64x64 still stream that uses
ONLY that tool plus the M1 baseline, then:

  1. check the pinned IVF byte count and MD5;
  2. decode with the pinned ``dav2d --strict 1``;
  3. compare every reconstructed sample against the encoder dump
     (``tools/still_verify/still_dump.cpp``);
  4. prove the tool is active: the stream and the decoded pixels must differ
     from the M1 baseline.

Tools with no integrated syntax are checked against the source tree instead of
being faked. If a hardcoded-off syntax path disappears (or a CLI flag appears),
the gate reports ``STALE-BLOCK`` and fails: the tool now needs a real stream.

Usage:
    tools/m3_streams/gate.py
    tools/m3_streams/gate.py --bin build-m1/av2enc --dav2d tools/bin/dav2d-macos
    tools/m3_streams/gate.py --no-build --out-dir /tmp/m3_streams

Exit codes: 0 all pass, 1 a stream or blocker check failed, 2 infrastructure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MANIFEST = HERE / "m3_cases.json"
COMPARE = HERE / "compare_recon.py"
DUMP_SRC = ROOT / "tools" / "still_verify" / "still_dump.cpp"
IVF_OVERHEAD = 32 + 12  # IVF file header + one IVF frame header


class GateError(Exception):
    pass


def run(cmd: list[str], *, capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=ROOT,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        text=True,
    )


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_default_bin() -> Path | None:
    for rel in ("build-m1/av2enc", "build/av2enc", "build-cuda/av2enc"):
        candidate = ROOT / rel
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def find_default_lib() -> Path | None:
    for rel in ("build-m1/libav2enc_core.a", "build/libav2enc_core.a"):
        candidate = ROOT / rel
        if candidate.is_file():
            return candidate
    return None


def build_dump(out: Path, lib: Path | None, no_build: bool) -> Path:
    if no_build and out.is_file():
        return out
    if lib is None:
        lib = find_default_lib()
    if lib is None or not lib.is_file():
        raise GateError(
            "no libav2enc_core.a found; pass --lib or build the CPU reference "
            "into build-m1/"
        )
    cxx = os.environ.get("CXX", "c++")
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        cxx,
        "-O2",
        "-std=c++20",
        "-I",
        str(ROOT / "src"),
        str(DUMP_SRC),
        "-o",
        str(out),
        str(lib),
    ]
    proc = run(cmd)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise GateError(f"failed to build still_dump with {cxx}")
    return out


def parse_dump_header(path: Path) -> dict:
    with path.open("rb") as f:
        if f.readline() != b"StillDUMP1\n":
            raise GateError(f"{path}: bad still_dump magic")
        line = f.readline()
    try:
        return json.loads(line)
    except json.JSONDecodeError as exc:
        raise GateError(f"{path}: bad still_dump JSON header: {exc}") from exc


def parse_compare_tsv(text: str) -> dict:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise GateError("compare_recon.py produced no TSV output")
    parts = lines[-1].split("\t")
    if len(parts) != 6:
        raise GateError(f"unexpected compare_recon.py output: {lines[-1]!r}")
    return {
        "recon_status": parts[0],
        "diff_y": int(parts[1]),
        "diff_u": int(parts[2]),
        "diff_v": int(parts[3]),
        "max_diff": int(parts[4]),
        "decoded_yuv_md5": parts[5],
    }


def md_cell(text: str) -> str:
    return text.replace("|", "\\|")


def check_expected_options(header: dict, expected: dict) -> list[str]:
    errors = []
    for key, value in expected.items():
        got = header.get(key)
        if got != value:
            errors.append(f"dump {key}={got!r}, expected {value!r}")
    return errors


def run_case(
    case: dict,
    *,
    fixture: Path,
    fixture_meta: dict,
    bin_path: Path,
    dump_bin: Path,
    dav2d: str | None,
    out_dir: Path,
    baseline: dict | None,
) -> dict:
    case_id = case["id"]
    ivf = out_dir / f"{case_id}.ivf"
    dec = out_dir / f"{case_id}.decoded.y4m"
    dump = out_dir / f"{case_id}.dump"
    log = out_dir / f"{case_id}.encode.log"

    errors: list[str] = []
    row = {
        "id": case_id,
        "tool": case["tool"],
        "status": "FAIL",
        "encode_args": case["encode_args"],
        "dump_args": case["dump_args"],
        "notes": [],
    }

    encode_cmd = [
        str(bin_path),
        str(fixture),
        "-o",
        str(ivf),
        "--frames",
        "1",
        *case["encode_args"],
    ]
    proc = run(encode_cmd)
    log.write_text(
        f"$ {' '.join(encode_cmd)}\n\n[stdout]\n{proc.stdout}\n[stderr]\n{proc.stderr}\n"
    )
    if proc.returncode != 0 or not ivf.is_file():
        errors.append(f"encode failed ({proc.returncode}); see {log}")
        row["errors"] = errors
        return row

    size = ivf.stat().st_size
    md5 = md5_file(ivf)
    row["bytes"] = size
    row["md5"] = md5

    if size != case["expected_bytes"]:
        errors.append(f"IVF size {size}, pinned {case['expected_bytes']}")
    if md5 != case["expected_md5"]:
        errors.append(f"IVF MD5 {md5}, pinned {case['expected_md5']}")

    dump_cmd = [
        str(dump_bin),
        "-i",
        str(fixture),
        "-o",
        str(dump),
        *case["dump_args"],
    ]
    proc = run(dump_cmd)
    if proc.returncode != 0 or not dump.is_file():
        errors.append("still_dump failed")
        row["errors"] = errors
        return row

    header = parse_dump_header(dump)
    row["tile_bytes"] = header.get("tile_bytes")
    row["frame_bytes"] = header.get("frame_bytes")
    for key in ("width", "height"):
        if header.get(key) != fixture_meta[key]:
            errors.append(
                f"dump {key}={header.get(key)!r}, expected {fixture_meta[key]!r}"
            )
    if header.get("base_q_idx") != fixture_meta["qp"]:
        errors.append(
            f"dump base_q_idx={header.get('base_q_idx')!r}, "
            f"expected {fixture_meta['qp']}"
        )
    errors.extend(check_expected_options(header, case["expected_options"]))

    if header.get("tile_bytes") != case["expected_tile_bytes"]:
        errors.append(
            f"tile_bytes {header.get('tile_bytes')}, pinned "
            f"{case['expected_tile_bytes']}"
        )
    if header.get("frame_bytes") != case["expected_frame_bytes"]:
        errors.append(
            f"frame_bytes {header.get('frame_bytes')}, pinned "
            f"{case['expected_frame_bytes']}"
        )
    expected_ivf = int(header.get("frame_bytes", -1)) + IVF_OVERHEAD
    if size != expected_ivf:
        errors.append(
            f"IVF size {size} != dump frame_bytes + {IVF_OVERHEAD} "
            f"({expected_ivf}); CLI and dump encodes differ"
        )

    compare_cmd = [
        sys.executable,
        str(COMPARE),
        "--ivf",
        str(ivf),
        "--dump",
        str(dump),
        "--keep-dec",
        str(dec),
        "--tsv",
    ]
    if dav2d:
        compare_cmd += ["--decoder", dav2d]
    proc = run(compare_cmd)
    if proc.returncode == 2 or not proc.stdout.strip():
        errors.append(f"compare_recon infrastructure failure:\n{proc.stderr}")
        row["errors"] = errors
        return row
    result = parse_compare_tsv(proc.stdout)
    row.update(result)
    if proc.returncode != 0 or result["recon_status"] != "BIT-EXACT":
        errors.append(
            f"recon mismatch: Y {result['diff_y']}, U {result['diff_u']}, "
            f"V {result['diff_v']}, max|diff| {result['max_diff']}"
        )
    if result["decoded_yuv_md5"] != case["expected_decoded_yuv_md5"]:
        errors.append(
            f"decoded YUV MD5 {result['decoded_yuv_md5']}, pinned "
            f"{case['expected_decoded_yuv_md5']}"
        )

    if baseline is not None:
        if md5 == baseline.get("md5"):
            errors.append("IVF is identical to baseline: tool is not coded")
        if result["decoded_yuv_md5"] == baseline.get("decoded_yuv_md5"):
            errors.append(
                "decoded pixels are identical to baseline: tool has no effect"
            )
        else:
            row["notes"].append("decoded pixels differ from baseline")

    if not errors:
        row["status"] = "PASS"
    row["errors"] = errors
    return row


def check_blocked(entry: dict) -> dict:
    errors = []
    checks = []
    for ev in entry["evidence"]:
        path = ROOT / ev["file"]
        if not path.is_file():
            errors.append(f"evidence file missing: {ev['file']}")
            checks.append({"file": ev["file"], "ok": False, "detail": "missing"})
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if ev.get("strip_comments"):
            text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
            text = re.sub(r"//[^\n]*", "", text)
        found = re.search(ev["pattern"], text, re.MULTILINE) is not None
        ok = found if ev["expect"] == "present" else not found
        detail = (
            f"pattern {'found' if found else 'not found'} (expected "
            f"{ev['expect']})"
        )
        checks.append(
            {
                "file": ev["file"],
                "pattern": ev["pattern"],
                "expect": ev["expect"],
                "ok": ok,
                "detail": detail,
            }
        )
        if not ok:
            errors.append(f"{ev['file']}: {detail}: {ev['note']}")
    return {
        "id": entry["id"],
        "tool": entry["tool"],
        "status": "BLOCKED" if not errors else "STALE-BLOCK",
        "missing_syntax": entry["missing_syntax"],
        "reason": entry["reason"],
        "checks": checks,
        "errors": errors,
    }


def write_outputs(
    out_dir: Path,
    fixture: str,
    verified: list[dict],
    blocked: list[dict],
    manifest: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "fixture": fixture,
                "decoder": manifest["decoder"],
                "verified": verified,
                "blocked": blocked,
            },
            f,
            indent=2,
        )
        f.write("\n")

    tsv = [
        "id\ttool\tstatus\tbytes\tmd5\tdecoded_yuv_md5\ttile_bytes\tframe_bytes\trecon_diffs"
    ]
    for row in verified:
        tsv.append(
            "\t".join(
                [
                    row["id"],
                    row["tool"],
                    row["status"],
                    str(row.get("bytes", "")),
                    row.get("md5", ""),
                    row.get("decoded_yuv_md5", ""),
                    str(row.get("tile_bytes", "")),
                    str(row.get("frame_bytes", "")),
                    str(
                        row.get("diff_y", 0)
                        + row.get("diff_u", 0)
                        + row.get("diff_v", 0)
                    ),
                ]
            )
        )
    (out_dir / "results.tsv").write_text("\n".join(tsv) + "\n")

    lines = [
        "# M3 per-tool stream gate results",
        "",
        f"Fixture: `{fixture}` (64x64, 8-bit 4:2:0, one frame, QP 90)",
        f"Decoder: `{manifest['decoder']['binary']}` at `{manifest['decoder']['version']}`",
        "",
        "## Verified streams",
        "",
        "| Tool | Status | Flags added to `--still` | IVF bytes | IVF MD5 | Decoded YUV MD5 | Encoder recon |",
        "|---|---|---|---:|---|---|---|",
    ]
    for row in verified:
        flags = " ".join(row["encode_args"][1:]) or "(baseline)"
        recon = (
            "BIT-EXACT"
            if row["status"] == "PASS"
            else (
                f"Y {row.get('diff_y', '?')} / U {row.get('diff_u', '?')} / "
                f"V {row.get('diff_v', '?')}"
            )
        )
        lines.append(
            "| {tool} | {status} | `{flags}` | {bytes} | `{md5}` | `{dec}` | {recon} |".format(
                tool=md_cell(row["tool"]),
                status=row["status"],
                flags=md_cell(flags),
                bytes=row.get("bytes", "?"),
                md5=row.get("md5", "?"),
                dec=row.get("decoded_yuv_md5", "?"),
                recon=recon,
            )
        )
    lines += [
        "",
        "## Blocked tools (no integrated syntax)",
        "",
        "| Tool | Status | Missing syntax |",
        "|---|---|---|",
    ]
    for row in blocked:
        lines.append(
            "| {} | {} | {} |".format(
                md_cell(row["tool"]),
                row["status"],
                md_cell("; ".join(row["missing_syntax"])),
            )
        )
    lines.append("")
    (out_dir / "matrix.md").write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--bin", type=Path, default=None, help="av2enc binary")
    parser.add_argument("--dav2d", default=None, help="dav2d binary")
    parser.add_argument("--dump-bin", type=Path, default=None)
    parser.add_argument("--lib", type=Path, default=None, help="libav2enc_core.a")
    parser.add_argument("--out-dir", type=Path, default=HERE / "out")
    parser.add_argument("--no-build", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    fixture = ROOT / manifest["fixture"]

    bin_path = args.bin or find_default_bin()
    if bin_path is None or not Path(bin_path).is_file():
        print(
            "gate: no av2enc binary; pass --bin (for example build-m1/av2enc)",
            file=sys.stderr,
        )
        return 2
    bin_path = Path(bin_path)
    if not os.access(bin_path, os.X_OK):
        print(f"gate: {bin_path} is not executable", file=sys.stderr)
        return 2

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_bin = args.dump_bin or (HERE / "build" / "still_dump")
    try:
        dump_bin = build_dump(Path(dump_bin), args.lib, args.no_build)
    except GateError as exc:
        print(f"gate: {exc}", file=sys.stderr)
        return 2

    cases = manifest["cases"]
    baseline_id = manifest["baseline_id"]
    if not cases or cases[0]["id"] != baseline_id:
        print("gate: manifest must list the baseline case first", file=sys.stderr)
        return 2

    verified: list[dict] = []
    baseline_row = None
    failures = 0
    for case in cases:
        baseline = None if case["id"] == baseline_id else baseline_row
        try:
            row = run_case(
                case,
                fixture=fixture,
                fixture_meta=manifest["fixture_format"],
                bin_path=bin_path,
                dump_bin=dump_bin,
                dav2d=args.dav2d,
                out_dir=out_dir,
                baseline=baseline,
            )
        except GateError as exc:
            print(f"gate: {exc}", file=sys.stderr)
            return 2
        if case["id"] == baseline_id:
            baseline_row = row
        verified.append(row)
        if row["status"] != "PASS":
            failures += 1

    try:
        blocked = [check_blocked(entry) for entry in manifest["blocked"]]
    except re.error as exc:
        print(f"gate: bad blocker evidence regex: {exc}", file=sys.stderr)
        return 2
    stale = [row for row in blocked if row["status"] != "BLOCKED"]

    write_outputs(out_dir, manifest["fixture"], verified, blocked, manifest)

    print("M3 per-tool stream gate")
    print(f"  fixture : {manifest['fixture']}")
    print(f"  encoder : {bin_path}")
    print(f"  dump    : {dump_bin}")
    print(f"  decoder : {args.dav2d or 'auto-detected'}")
    print()
    for row in verified:
        detail = (
            f"{row.get('bytes', '?')} B, md5 {row.get('md5', '?')}, "
            f"decoded {row.get('decoded_yuv_md5', '?')}"
        )
        print(f"  {row['status']:4s}  {row['id']:10s} {detail}")
        for err in row.get("errors", []):
            print(f"        {err}")
    for row in blocked:
        print(f"  {row['status']:11s} {row['id']:10s} {row['tool']}")
        for err in row.get("errors", []):
            print(f"        {err}")
    print()
    print(f"  verified: {sum(1 for r in verified if r['status'] == 'PASS')}/{len(verified)}")
    print(f"  blocked : {sum(1 for r in blocked if r['status'] == 'BLOCKED')}/{len(blocked)}")
    print(f"  outputs : {out_dir}")

    return 0 if failures == 0 and not stale else 1


if __name__ == "__main__":
    sys.exit(main())
