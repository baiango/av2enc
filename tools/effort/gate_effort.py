#!/usr/bin/env python3
"""Budget/rejection/byte-identity gate for the effort ranking.

Run through tools/effort/gate_effort.sh.  The gate never recalibrates and never
writes the table; it checks the committed table and the compiled-in selector.

Checks:
  1. gen_table.py --check (include matches the JSON).
  2. Every feature has measured fields; the ranking is score-ascending.
  3. Budget invariant for --cpu-used 0..8 and an --effort sweep: the selected
     prefix and used score match the table, and used <= budget.
  4. Explicit flags are kept and not charged; conflicts are skipped.
  5. Default still/basic streams stay byte-identical to refactor/baseline.json.
  6. A still prefix decodes with dav2d --strict 1 and is measured.
  7. A representative --video stream decodes with --strict 1; --effort --video
     is rejected because no video rows are calibrated yet.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run(cmd: list[str], env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, text=True, capture_output=True, env=env)


def md5_file(path: Path) -> str:
    import hashlib
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_bin(path: str | None) -> Path:
    if path:
        p = Path(path).expanduser().resolve()
        if not p.is_file() or not os.access(p, os.X_OK):
            raise SystemExit(f"gate_effort: error: --bin is not executable: {p}")
        return p
    for candidate in (ROOT / "build-int" / "av2enc", ROOT / "build" / "av2enc"):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    raise SystemExit("gate_effort: error: no av2enc binary; pass --bin PATH")


def find_decoder(path: str | None) -> Path | None:
    if path:
        p = Path(path).expanduser().resolve()
        return p if p.is_file() and os.access(p, os.X_OK) else None
    if os.environ.get("DAV2D"):
        p = Path(os.environ["DAV2D"])
        if p.is_file() and os.access(p, os.X_OK):
            return p.resolve()
    for candidate in (ROOT / "tools" / "bin" / "dav2d",
                      ROOT / "tools" / "bin" / "dav2d-macos"):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    found = shutil.which("dav2d")
    return Path(found).resolve() if found else None


class Gate:
    def __init__(self, verbose: bool) -> None:
        self.verbose = verbose
        self.failures = 0
        self.checks = 0
        self.skips = 0

    def ok(self, message: str) -> None:
        self.checks += 1
        print(f"PASS  {message}")

    def skip(self, message: str) -> None:
        self.skips += 1
        print(f"SKIP  {message}")

    def fail(self, message: str, detail: str = "") -> None:
        self.checks += 1
        self.failures += 1
        print(f"FAIL  {message}")
        if detail:
            for line in detail.strip().splitlines()[-10:]:
                print(f"      {line}")

    def check(self, condition: bool, message: str, detail: str = "") -> bool:
        if condition:
            self.ok(message)
        else:
            self.fail(message, detail)
        return condition


BUDGET_RE = re.compile(
    r"av2enc: effort: budget ([\d.]+) \(([^)]*)\); selected (\d+) feature\(s\); "
    r"used ([\d.]+) of ([\d.]+)")
FEATURE_RE = re.compile(
    r"^  #(\d+) (--[A-Za-z0-9-]+) \((marginal |score )([\d.]+);", re.M)
EXPLICIT_RE = re.compile(r"^  explicit \(caller, not charged\):(.*)$", re.M)


def plan(binary: Path, fixture: Path, tmp: Path, extra: list[str]) -> tuple[int, str]:
    cmd = [str(binary), str(fixture), "-o", str(tmp / "plan.ivf"),
           "--effort-plan"] + extra
    p = run(cmd)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def parse_plan(text: str) -> tuple[float, int, float, list[str]] | None:
    m = BUDGET_RE.search(text)
    if not m:
        return None
    flags = [fm.group(2) for fm in FEATURE_RE.finditer(text)]
    return float(m.group(1)), int(m.group(3)), float(m.group(4)), flags


def effective(row: dict) -> tuple[float, bool]:
    """Return (cost, marginal_used) exactly like the runtime selector."""
    if row.get("marginal_eligible") is False:
        return -1.0, False
    marginal = row.get("marginal_score")
    if isinstance(marginal, (int, float)) and marginal >= 0:
        return float(marginal), True
    return float(row["score"]), False


def expected_prefix(table: dict, budget: float,
                    explicit: set[str] | None = None) -> tuple[list[str], float]:
    explicit = explicit or set()
    rows = [r for r in table["features"] if r.get("mode") == "still" and r.get("eligible")]
    rows.sort(key=lambda r: (float(r["score"]), str(r["flag"])))
    used = 0.0
    flags: list[str] = []
    for row in rows:
        if row["flag"] in explicit:
            continue
        cost, _ = effective(row)
        if cost < 0:
            continue
        if used + cost > budget + 1e-9:
            break
        used += cost
        flags.append(row["flag"])
    return flags, used


def cpu_used_budget(k: int) -> float:
    value = 10.0 * (1000.0 ** ((8 - k) / 8.0))
    return float(max(10, min(10000, int(math.floor(value + 0.5)))))


def decode_strict(decoder: Path, ivf: Path) -> tuple[int, str]:
    p = run([str(decoder), "-q", "-i", str(ivf), "--muxer", "md5", "-o", "-",
             "--threads", "1", "--strict", "1"])
    lines = [line.strip() for line in (p.stdout or "").splitlines() if line.strip()]
    return p.returncode, lines[-1] if lines else ""


def measure(binary: Path, decoder: Path, ref: Path, ivf: Path, tmp: Path) -> dict:
    qjson = tmp / (ivf.stem + ".json")
    env = dict(os.environ)
    env["DAV2D"] = str(decoder)
    env.setdefault("DAV2D_THREADS", "1")
    p = run([sys.executable, str(ROOT / "tools" / "quality.py"),
             "--ref", str(ref), "--ivf", str(ivf), "--json", str(qjson)], env=env)
    if p.returncode == 0 and qjson.is_file():
        data = json.loads(qjson.read_text())
        return {
            "metric": "cvvdp",
            "quality": float(data["cvvdp_jod"]),
            "bitrate_kbps": float(data["bitrate_kbps"]),
        }
    # Documented fallback: PSNR, clearly labelled.
    dec = tmp / (ivf.stem + ".decoded.y4m")
    d = run([str(decoder), "-q", "-i", str(ivf), "-o", str(dec), "--threads", "1"])
    if d.returncode != 0:
        raise RuntimeError(f"dav2d failed for {ivf}")
    f = run(["ffmpeg", "-hide_banner", "-v", "info", "-i", str(dec),
             "-i", str(ref), "-lavfi", "[0:v][1:v]psnr", "-f", "null", "-"])
    text = (f.stdout or "") + "\n" + (f.stderr or "")
    m = re.search(r"average:([0-9.]+)", text)
    if not m:
        raise RuntimeError(f"could not parse PSNR for {ivf}")
    with ref.open("rb") as fh:
        header = fh.readline().decode("ascii", "replace")
    fm = re.search(r"F(\d+):(\d+)", header)
    num, den = (int(fm.group(1)), int(fm.group(2))) if fm else (1, 1)
    with dec.open("rb") as fh:
        frames = sum(1 for line in fh if line.startswith(b"FRAME"))
    duration = frames * den / num if num else 1.0
    return {
        "metric": "psnr",
        "quality": float(m.group(1)),
        "bitrate_kbps": ivf.stat().st_size * 8 / duration / 1000 if duration else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin")
    ap.add_argument("--dav2d")
    ap.add_argument("--table", default=str(ROOT / "tools" / "effort" / "effort_table.json"))
    ap.add_argument("--quick", action="store_true", help="skip the 720p video measurement")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    gate = Gate(args.verbose)
    binary = find_bin(args.bin)
    decoder = find_decoder(args.dav2d)
    if decoder is None:
        print("gate_effort: SKIP (exit 77): no dav2d decoder "
              "(set DAV2D or install tools/bin/dav2d)")
        return 77

    fixture = ROOT / "tests" / "data" / "basic_640x360_1f.y4m"
    if not fixture.is_file():
        print(f"gate_effort: SKIP (exit 77): missing fixture {fixture}")
        return 77

    table_path = Path(args.table).expanduser().resolve()
    if not table_path.is_file():
        print(f"gate_effort: error: table not found: {table_path}")
        return 1
    table = json.loads(table_path.read_text())

    tmp = Path(tempfile.mkdtemp(prefix="av2enc-effort-gate-"))
    try:
        print("== effort gate ==")
        print(f"  binary:  {binary}")
        print(f"  decoder: {decoder}")

        # 1. generated include matches the JSON
        p = run([sys.executable, str(ROOT / "tools" / "effort" / "gen_table.py"),
                 "--table", str(table_path),
                 "--out", str(ROOT / "src" / "pipeline" / "planner" / "effort_table.inc"),
                 "--check"])
        gate.check(p.returncode == 0, "gen_table --check",
                   (p.stdout or "") + (p.stderr or ""))

        # 2. table completeness and ranking
        required = ["flag", "mode", "score", "d_cvvdp_pct", "d_time_pct",
                    "d_rate_pct", "bitrate_anchor_kbps", "resolution_class",
                    "repeats", "confidence", "source", "eligible"]
        missing = []
        for row in table.get("features", []):
            for key in required:
                if key not in row:
                    missing.append(f"{row.get('flag', '?')}.{key}")
            if not row.get("eligible") and not row.get("exclude_reason"):
                missing.append(f"{row['flag']}.exclude_reason")
        gate.check(not missing, "table has measured fields for every feature",
                   "\n".join(missing))
        eligible = [r for r in table.get("features", [])
                    if r.get("mode") == "still" and r.get("eligible")]
        gate.check(bool(eligible), "at least one eligible feature is published")
        if not eligible:
            raise SystemExit(1)
        ranked = sorted(eligible, key=lambda r: (float(r["score"]), str(r["flag"])))
        expected_ranking = [r["flag"] for r in ranked]
        gate.check(table.get("ranking") == expected_ranking,
                   "ranking is score-ascending", str(table.get("ranking")))
        gate.check(table.get("prefix_validation") is not None,
                   "top-K prefix validation row is recorded")

        # 2b. --effort-plan needs neither an input nor an output file.
        for label, cmd in (
            ("without input", [str(binary), "--effort", "100", "--effort-plan"]),
            ("without output", [str(binary), str(fixture), "--effort", "100",
                                "--effort-plan"]),
            ("without either", [str(binary), "--cpu-used", "8",
                                "--effort-plan"]),
        ):
            p = run(cmd)
            text = (p.stdout or "") + (p.stderr or "")
            gate.check(p.returncode == 0 and "av2enc: effort: budget" in text,
                       f"--effort-plan works {label} (rc={p.returncode})", text)

        # 3. budget invariant
        costs = [effective(r)[0] for r in ranked]
        cutoffs = []
        running = 0.0
        for cost in costs:
            running += max(cost, 0.0)
            cutoffs.extend({int(math.floor(running)), int(math.ceil(running))})
        budgets = {10, 20, 24, 50, 56, 100, 133, 316, 750, 1778, 4217, 10000}
        budgets.update(b for b in cutoffs if 10 <= b <= 10000)
        for budget in sorted(budgets):
            rc, text = plan(binary, fixture, tmp, ["--effort", str(budget)])
            parsed = parse_plan(text)
            exp_flags, exp_used = expected_prefix(table, float(budget))
            if rc != 0 or parsed is None:
                gate.fail(f"--effort {budget} did not print a selection", text)
                continue
            got_budget, got_count, used, flags = parsed
            gate.check(abs(got_budget - budget) < 0.5 and used <= budget + 0.05,
                       f"--effort {budget}: used {used:.1f} <= {budget}")
            gate.check(got_count == len(exp_flags) and flags == exp_flags,
                       f"--effort {budget}: prefix {flags} "
                       f"(expected {exp_flags})", text)
            gate.check(abs(used - exp_used) <= 0.15,
                       f"--effort {budget}: used {used:.1f} "
                       f"(expected {exp_used:.1f})", text)

        for k in range(9):
            expected = cpu_used_budget(k)
            rc, text = plan(binary, fixture, tmp, ["--cpu-used", str(k)])
            parsed = parse_plan(text)
            exp_flags, exp_used = expected_prefix(table, expected)
            if rc != 0 or parsed is None:
                gate.fail(f"--cpu-used {k} did not print a selection", text)
                continue
            got_budget, got_count, used, flags = parsed
            gate.check(abs(got_budget - expected) < 0.5,
                       f"--cpu-used {k}: budget {got_budget:.0f} "
                       f"(expected {expected:.0f})", text)
            gate.check(used <= expected + 0.05 and got_count == len(exp_flags) and
                       flags == exp_flags,
                       f"--cpu-used {k}: prefix {flags}, used {used:.1f}",
                       text)

        # 4. explicit flags and conflicts
        top = ranked[0]["flag"]
        top_cost = effective(ranked[0])[0]
        rc, text = plan(binary, fixture, tmp, ["--effort", "10000", top])
        parsed = parse_plan(text)
        exp_flags, exp_used = expected_prefix(table, 10000.0, {top})
        gate.check(rc == 0 and parsed is not None and
                   parsed[3] == exp_flags and abs(parsed[2] - exp_used) <= 0.15,
                   f"explicit {top}: kept, not charged (used {exp_used:.1f})", text)
        gate.check(f"explicit (caller, not charged):" in text,
                   "explicit feature is reported", text)
        rc, text = plan(binary, fixture, tmp, ["--effort", "10000", "--no-cfl"])
        parsed = parse_plan(text)
        gate.check(rc == 0 and parsed is not None and
                   "--still-mhccp" not in parsed[3],
                   "--no-cfl: --still-mhccp is skipped", text)
        gate.check("conflicts with an explicit or higher-ranked flag" in text,
                   "--no-cfl: conflict is reported", text)
        rc, text = plan(binary, fixture, tmp, ["--effort", str(int(top_cost)), top])
        parsed = parse_plan(text)
        gate.check(rc == 0 and parsed is not None and parsed[3] == [],
                   f"explicit {top} at its own cost: nothing else is charged",
                   text)

        for extra, label in [
            (["--still-gpu"], "GPU"),
            (["--video"], "video"),
            (["--lossless"], "lossless"),
            (["--tiles", "2"], "tiles"),
            (["--bit-depth", "10"], "10-bit"),
            (["--chroma", "444"], "4:4:4"),
        ]:
            p = run([str(binary), str(fixture), "-o", str(tmp / "reject.ivf"),
                     "--effort", "10"] + extra)
            gate.check(p.returncode == 2,
                       f"--effort with {label} is rejected (rc={p.returncode})",
                       (p.stderr or ""))
            p = run([str(binary), str(fixture), "-o", str(tmp / "reject.ivf"),
                     "--effort-plan", "--cpu-used", "0"] + extra)
            gate.check(p.returncode == 2,
                       f"--cpu-used with {label} is rejected (rc={p.returncode})",
                       (p.stderr or ""))

        # 5. default byte identity against the pinned baseline
        baseline_path = ROOT / "refactor" / "baseline.json"
        if not baseline_path.is_file():
            gate.skip("refactor/baseline.json is missing; byte identity not checked")
        else:
            baseline = json.loads(baseline_path.read_text())
            streams = baseline.get("streams", {})
            checked = 0
            for name in sorted(streams):
                rec = streams[name]
                fixture_rec = rec.get("fixture", "")
                if fixture_rec == "generated" or not fixture_rec:
                    continue
                src = ROOT / fixture_rec
                if not src.is_file():
                    continue
                out = tmp / f"base-{name}.ivf"
                cmd = [str(binary), str(src), "-o", str(out)] + list(rec["args"])
                p = run(cmd)
                if p.returncode != 0:
                    gate.fail(f"default stream {name} failed to encode", p.stderr)
                    continue
                enc = md5_file(out)
                rc_dec, dec = decode_strict(decoder, out)
                gate.check(enc == rec.get("encoder_md5") and
                           dec == rec.get("decoded_md5") and rc_dec == 0,
                           f"default stream {name} byte-identical",
                           f"encoder {enc} vs {rec.get('encoder_md5')}\n"
                           f"decoded {dec} vs {rec.get('decoded_md5')}")
                gate.check("effort:" not in (p.stderr or ""),
                           f"default stream {name} did not activate the ranking")
                checked += 1
            if checked == 0:
                gate.skip("no pinned default fixtures are present")

        # 6. still end-to-end: baseline vs the full ranked prefix
        still_ref = ROOT / "testdata" / "shine_640x360_1f.y4m"
        if not still_ref.is_file():
            still_ref = fixture
        qp = "170"
        base_ivf = tmp / "still-base.ivf"
        best_ivf = tmp / "still-prefix.ivf"
        times = {"base": [], "prefix": []}
        for label, extra, out in [
            ("base", [], base_ivf),
            ("prefix", ["--effort", "10000"], best_ivf),
        ]:
            for _ in range(3):
                t0 = time.perf_counter()
                p = run([str(binary), str(still_ref), "-o", str(out),
                         "--frames", "1", "--qp", qp, "--still"] + extra)
                times[label].append(time.perf_counter() - t0)
                if p.returncode != 0:
                    gate.fail(f"still {label} encode failed", p.stderr)
                    break
            rc_dec, _ = decode_strict(decoder, out)
            gate.check(rc_dec == 0, f"still {label} decodes with --strict 1")
        if base_ivf.is_file() and best_ivf.is_file():
            gate.check(md5_file(base_ivf) != md5_file(best_ivf),
                       "effort prefix changes the stream (opt-in by design)")
            try:
                mb = measure(binary, decoder, still_ref, base_ivf, tmp)
                ml = measure(binary, decoder, still_ref, best_ivf, tmp)
                gate.check(True, f"still ranked prefix measured ({mb['metric']})")
                print(f"      content: {still_ref.name} (fixed QP {qp}, "
                      f"not the iso-rate anchor)")
                print(f"      base:   {mb['bitrate_kbps']:.0f} kbps, "
                      f"quality {mb['quality']:.4f}, "
                      f"encode {min(times['base']):.3f}s")
                print(f"      prefix: {ml['bitrate_kbps']:.0f} kbps, "
                      f"quality {ml['quality']:.4f}, "
                      f"encode {min(times['prefix']):.3f}s")
            except Exception as exc:  # noqa: BLE001 - gate reports the detail
                gate.fail("still measurement failed", str(exc))

        # 7. video end-to-end (default path) and the rejected effort+video combo
        video_src = ROOT / "testdata" / "shine_1280x720_60f.y4m"
        if args.quick:
            gate.skip("video measurement skipped (--quick)")
        elif not video_src.is_file():
            gate.skip(f"video fixture missing: {video_src}")
        else:
            video_ivf = tmp / "video.ivf"
            p = run([str(binary), str(video_src), "-o", str(video_ivf),
                     "--video", "--frames", "4", "--p-frames", "3", "--qp", qp])
            gate.check(p.returncode == 0, "video default encode", p.stderr)
            if video_ivf.is_file():
                rc_dec, _ = decode_strict(decoder, video_ivf)
                gate.check(rc_dec == 0, "video stream decodes with --strict 1")
                try:
                    mv = measure(binary, decoder, video_src, video_ivf, tmp)
                    gate.check(True, f"video measured ({mv['metric']})")
                    print(f"      video: {mv['bitrate_kbps']:.0f} kbps, "
                          f"quality {mv['quality']:.4f}")
                except Exception as exc:  # noqa: BLE001
                    gate.fail("video measurement failed", str(exc))
        p = run([str(binary), str(fixture), "-o", str(tmp / "ve.ivf"),
                 "--effort", "10", "--video"])
        gate.check(p.returncode == 2,
                   "--effort with --video is rejected (no video rows)",
                   p.stderr)

        print()
        print(f"== summary: {gate.checks - gate.failures}/{gate.checks} passed, "
              f"{gate.skips} skipped ==")
        if gate.failures:
            print(f"gate_effort: FAILED ({gate.failures} check(s))")
            return 1
        print("gate_effort: OK")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
