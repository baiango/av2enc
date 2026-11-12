#!/usr/bin/env python3
"""Calibrate the av2enc effort table: one measured row per optional feature.

The runtime ranks features by score ascending and takes a prefix while the
cumulative score fits the budget.  This tool measures each feature solo against
the same mode with every optional tool off, on a QP grid so quality can be
interpolated at the 1000 kbps anchor.  It also records the 300/1500/2500 kbps
anchor points when the sweep brackets them.

An optional marginal pass (--marginal K, default 5) re-measures the cumulative
prefix of the top K eligible features.  That gives a measured marginal cost for
a feature *given the higher-ranked features*, which catches the worst
non-additivity with O(K) encodes instead of an exhaustive search.  The runtime
uses the marginal cost when present and falls back to the solo score otherwise.

The tool refuses to write a row with a missing measurement.  A feature that
cannot be bracketed at 1000 kbps is a setup error, not a score of zero.

Usage:
  tools/effort/effort_calibrate.py --bin build-int/av2enc
  tools/effort/effort_calibrate.py --bin build-int/av2enc --marginal 5
  tools/effort/effort_calibrate.py --bin build-int/av2enc --dry-run

Defaults:
  features tools/effort/features.json
  fixture  testdata/shine_640x360_1f.y4m   (fallback: tests/data/basic_640x360_1f.y4m)
  anchors  300,1000,1500,2500 kbps; score anchor 1000
  qps      110,130,150,170,190,210,230
  repeats  7 (child CPU time; enabled p75 / baseline p25 bound)
  out      tools/effort/effort_table.json
  inc      src/pipeline/planner/effort_table.inc

The table is host-specific.  CPU/GPU model, decoder identity and metric version
are recorded; re-run this tool on the target host before trusting the scores
there.  See docs/effort.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
DEFAULT_FIXTURES = [
    ROOT / "testdata" / "shine_640x360_1f.y4m",
    ROOT / "tests" / "data" / "basic_640x360_1f.y4m",
]


def die(message: str) -> "None":
    raise SystemExit(f"effort_calibrate: error: {message}")


def run(cmd: list[str], env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, text=True, capture_output=True, env=env)


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_bin(path: str | None) -> Path:
    if path:
        p = Path(path).expanduser().resolve()
        if not p.is_file() or not os.access(p, os.X_OK):
            die(f"--bin is not executable: {p}")
        return p
    for candidate in (ROOT / "build-int" / "av2enc", ROOT / "build" / "av2enc"):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    die("no av2enc binary; pass --bin PATH")


def find_fixture(path: str | None) -> Path:
    if path:
        p = Path(path).expanduser().resolve()
        if not p.is_file():
            die(f"fixture not found: {p}")
        return p
    for candidate in DEFAULT_FIXTURES:
        if candidate.is_file():
            return candidate.resolve()
    die("no 640x360 fixture found (looked for testdata/shine_640x360_1f.y4m "
        "and tests/data/basic_640x360_1f.y4m); pass --fixture")


def y4m_info(path: Path) -> dict:
    with path.open("rb") as f:
        header = f.readline().decode("ascii", "replace").rstrip("\n")
        import re
        wm = re.search(r"W(\d+)\s+H(\d+)", header)
        fm = re.search(r"F(\d+):(\d+)", header)
        cm = re.search(r"C([0-9A-Za-z]+)", header)
        if not wm:
            die(f"bad Y4M header: {header}")
        width, height = int(wm.group(1)), int(wm.group(2))
        fps_num, fps_den = (int(fm.group(1)), int(fm.group(2))) if fm else (0, 0)
        chroma = cm.group(1) if cm else "420"
        frames = 0
        size = width * height
        if chroma.startswith("420"):
            size += 2 * ((width + 1) // 2) * ((height + 1) // 2)
        elif chroma.startswith("422"):
            size += 2 * ((width + 1) // 2) * height
        elif chroma.startswith("444"):
            size += 2 * width * height
        else:
            die(f"unsupported chroma in {path}: {chroma}")
        while True:
            line = f.readline()
            if not line:
                break
            if line.startswith(b"FRAME"):
                if len(f.read(size)) != size:
                    die(f"truncated frame in {path}")
                frames += 1
            else:
                die(f"expected FRAME in {path}, got {line[:16]!r}")
    return {
        "path": str(path),
        "md5": md5_file(path),
        "width": width,
        "height": height,
        "fps_num": fps_num,
        "fps_den": fps_den,
        "chroma": chroma,
        "frames": frames,
        "resolution_class": resolution_class(width, height),
    }


def resolution_class(width: int, height: int) -> str:
    if height <= 360:
        return "360p"
    if height <= 720:
        return "720p"
    if height <= 1080:
        return "1080p"
    if height <= 1440:
        return "1440p"
    return "2160p"


def host_info() -> dict:
    info = {
        "platform": sys.platform,
        "machine": os.uname().machine if hasattr(os, "uname") else "",
    }
    if sys.platform == "darwin":
        p = run(["sysctl", "-n", "machdep.cpu.brand_string"])
        info["cpu_model"] = p.stdout.strip() if p.returncode == 0 else None
        p = run(["sysctl", "-n", "hw.physicalcpu"])
        info["cpu_physical_cores"] = int(p.stdout.strip()) if p.returncode == 0 else None
        p = run(["sysctl", "-n", "hw.ncpu"])
        info["cpu_logical_cores"] = int(p.stdout.strip()) if p.returncode == 0 else None
        p = run(["sysctl", "-n", "hw.memsize"])
        info["ram_gib"] = round(int(p.stdout.strip()) / 1024**3, 1) if p.returncode == 0 else None
        gpu = None
        p = run(["system_profiler", "SPDisplaysDataType", "-detailLevel", "mini"])
        if p.returncode == 0:
            for line in (p.stdout or "").splitlines():
                if "Chipset Model:" in line:
                    gpu = line.split(":", 1)[1].strip()
                    break
        info["gpu"] = gpu or "Apple integrated (no CUDA)"
        info["cuda"] = False
    else:
        model = None
        try:
            for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
                if line.startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
        except OSError:
            pass
        info["cpu_model"] = model
        info["cpu_logical_cores"] = os.cpu_count()
        p = run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap",
                 "--format=csv,noheader"])
        info["gpu"] = [x.strip() for x in p.stdout.splitlines() if x.strip()] if p.returncode == 0 else None
    return info


def find_decoder() -> str | None:
    if os.environ.get("DAV2D"):
        return os.environ["DAV2D"]
    for candidate in (ROOT / "tools" / "bin" / "dav2d",
                      ROOT / "tools" / "bin" / "dav2d-macos"):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    found = shutil.which("dav2d")
    return found


def metric_env() -> dict:
    env = dict(os.environ)
    env.setdefault("DAV2D_THREADS", "1")
    return env


def measure_quality(ivf: Path, ref: Path, tmp: Path, allow_psnr: bool) -> dict:
    """Run tools/quality.py; return metric dict.  PSNR fallback is explicit."""
    qjson = tmp / (ivf.stem + ".quality.json")
    cmd = [sys.executable, str(TOOLS / "quality.py"),
           "--ref", str(ref), "--ivf", str(ivf), "--json", str(qjson)]
    p = run(cmd, env=metric_env())
    if p.returncode == 0 and qjson.is_file():
        data = json.loads(qjson.read_text())
        return {
            "metric": "cvvdp",
            "quality": float(data["cvvdp_jod"]),
            "version": str(data.get("cvvdp_version", "unknown")),
            "display": str(data.get("cvvdp_display", "standard_fhd")),
            "device": str(data.get("cvvdp_device", "unknown")),
            "bitrate_kbps": float(data["bitrate_kbps"]),
            "psnr_avg": data.get("psnr", {}).get("psnr_avg"),
        }
    if not allow_psnr:
        sys.stderr.write((p.stdout or "")[-3000:])
        sys.stderr.write((p.stderr or "")[-3000:])
        die(f"CVVDP metrics failed for {ivf}; re-run with --allow-psnr-fallback "
            "to record a documented PSNR row instead")
    # Documented fallback: decode with dav2d and score PSNR with ffmpeg.
    decoder = find_decoder()
    if not decoder:
        die("CVVDP failed and no dav2d decoder is available for the PSNR fallback")
    dec = tmp / (ivf.stem + ".decoded.y4m")
    d = run([decoder, "-q", "-i", str(ivf), "-o", str(dec), "--threads", "1"])
    if d.returncode != 0 or not dec.is_file():
        die(f"dav2d failed during the PSNR fallback for {ivf}")
    f = run(["ffmpeg", "-hide_banner", "-v", "info", "-i", str(dec), "-i", str(ref),
             "-lavfi", "[0:v][1:v]psnr", "-f", "null", "-"])
    text = (f.stdout or "") + "\n" + (f.stderr or "")
    import re
    m = re.search(r"average:([0-9.]+)", text)
    if not m:
        die(f"could not parse PSNR for {ivf}")
    size = ivf.stat().st_size
    info = y4m_info(ref)
    with dec.open("rb") as fh:
        fh.readline()
        frames = sum(1 for line in fh if line.startswith(b"FRAME"))
    duration = (frames * info["fps_den"] / info["fps_num"]
                if info["fps_num"] else 1.0)
    return {
        "metric": "psnr_fallback",
        "quality": float(m.group(1)),
        "version": "ffmpeg psnr",
        "display": "n/a",
        "device": "cpu",
        "bitrate_kbps": size * 8 / duration / 1000 if duration > 0 else 0.0,
        "psnr_avg": float(m.group(1)),
    }


def encode_once(binary: Path, fixture: Path, out: Path, args: list[str]) -> tuple[float, float]:
    """Return (wall_seconds, cpu_seconds) for one encoder invocation."""
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.perf_counter()
    p = run([str(binary), str(fixture), "-o", str(out)] + args, env=metric_env())
    wall = time.perf_counter() - t0
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = ((after.ru_utime - before.ru_utime) +
           (after.ru_stime - before.ru_stime))
    if p.returncode != 0:
        sys.stderr.write((p.stdout or "")[-2000:])
        sys.stderr.write((p.stderr or "")[-2000:])
        die(f"encode failed: {' '.join(args)} (rc={p.returncode})")
    if not out.is_file() or out.stat().st_size == 0:
        die(f"encode produced no output: {' '.join(args)}")
    return wall, cpu


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("empty percentile")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def measure_config(binary: Path, fixture: Path, args: list[str], qps: list[int],
                   repeats: int, tmp: Path, label: str, allow_psnr: bool) -> dict:
    """Encode one configuration on the QP grid and measure it.

    Encode cost is child CPU time (user+sys).  The conservative bound is the
    75th percentile of the enabled configuration over the 25th percentile of
    the baseline, so d_time_pct is biased upward rather than optimistic.
    """
    points = []
    warm = tmp / f"{label}-warm.ivf"
    encode_once(binary, fixture, warm, args + ["--qp", str(qps[0])])
    for qp in qps:
        out = tmp / f"{label}-q{qp}.ivf"
        wall_times = []
        cpu_times = []
        md5s = []
        for _ in range(repeats):
            wall, cpu = encode_once(binary, fixture, out, args + ["--qp", str(qp)])
            wall_times.append(wall)
            cpu_times.append(cpu)
            md5s.append(md5_file(out))
        if len(set(md5s)) != 1:
            die(f"{label} QP {qp} is not deterministic across {repeats} runs "
                f"({', '.join(sorted(set(md5s)))})")
        m = measure_quality(out, fixture, tmp, allow_psnr)
        spread = 0.0
        if cpu_times and statistics.median(cpu_times) > 0:
            spread = ((max(cpu_times) - min(cpu_times)) /
                      statistics.median(cpu_times) * 100.0)
        points.append({
            "qp": qp,
            "bitrate_kbps": m["bitrate_kbps"],
            "quality": m["quality"],
            "time_seconds": max(cpu_times),
            "time_lower_seconds": min(cpu_times),
            "time_p25_seconds": percentile(cpu_times, 0.25),
            "time_p75_seconds": percentile(cpu_times, 0.75),
            "time_median_seconds": statistics.median(cpu_times),
            "wall_seconds": max(wall_times),
            "time_spread_pct": spread,
            "md5": md5s[0],
        })
    points.sort(key=lambda p: p["bitrate_kbps"])
    return {
        "points": points,
        "metric": m["metric"],
        "metric_version": m["version"],
        "metric_display": m["display"],
        "metric_device": m["device"],
    }


def interp_at_rate(points: list[dict], target: float) -> float | None:
    if len(points) < 2:
        return None
    if target < points[0]["bitrate_kbps"] or target > points[-1]["bitrate_kbps"]:
        return None
    for a, b in zip(points, points[1:]):
        if a["bitrate_kbps"] <= target <= b["bitrate_kbps"]:
            if b["bitrate_kbps"] == a["bitrate_kbps"]:
                return (a["quality"] + b["quality"]) / 2
            f = ((math.log(target) - math.log(a["bitrate_kbps"])) /
                 (math.log(b["bitrate_kbps"]) - math.log(a["bitrate_kbps"])))
            return a["quality"] + f * (b["quality"] - a["quality"])
    return None


def time_at_rate(points: list[dict], target: float, key: str = "time_seconds") -> float | None:
    if len(points) < 2:
        return None
    if target < points[0]["bitrate_kbps"] or target > points[-1]["bitrate_kbps"]:
        return None
    for a, b in zip(points, points[1:]):
        if a["bitrate_kbps"] <= target <= b["bitrate_kbps"]:
            if b["bitrate_kbps"] == a["bitrate_kbps"]:
                if key in ("time_lower_seconds", "time_p25_seconds"):
                    return min(a[key], b[key])
                return max(a[key], b[key])
            f = ((math.log(target) - math.log(a["bitrate_kbps"])) /
                 (math.log(b["bitrate_kbps"]) - math.log(a["bitrate_kbps"])))
            return a[key] + f * (b[key] - a[key])
    return None


def rate_at_quality(points: list[dict], target: float) -> float | None:
    """Bitrate at a target quality (linear between adjacent QP points)."""
    pts = sorted(points, key=lambda p: p["quality"])
    if len(pts) < 2 or target < pts[0]["quality"] or target > pts[-1]["quality"]:
        return None
    for a, b in zip(pts, pts[1:]):
        if a["quality"] <= target <= b["quality"]:
            if b["quality"] == a["quality"]:
                return max(a["bitrate_kbps"], b["bitrate_kbps"])
            f = (target - a["quality"]) / (b["quality"] - a["quality"])
            return math.exp(math.log(a["bitrate_kbps"]) +
                            f * (math.log(b["bitrate_kbps"]) -
                                 math.log(a["bitrate_kbps"])))
    return None


def anchor_qp(points: list[dict], anchor: float) -> int | None:
    if len(points) < 2:
        return None
    if anchor < points[0]["bitrate_kbps"] or anchor > points[-1]["bitrate_kbps"]:
        return None
    nearest = min(points,
                  key=lambda p: abs(math.log(p["bitrate_kbps"] / anchor)))
    return int(nearest["qp"])


def paired_time(binary: Path, fixture: Path, base_args: list[str],
                prev_flags: list[str], prev_qp: int,
                cand_flags: list[str], cand_qp: int,
                repeats: int, tmp: Path, label: str) -> dict:
    """Alternating A/B CPU-time measurement at each config's anchor QP.

    The separate QP sweeps can drift with CPU frequency; pairing the baseline
    and the candidate run by run cancels that drift.  The p75 ratio is the
    conservative bound used for the score.
    """
    base_out = tmp / f"{label}-pair-prev.ivf"
    cand_out = tmp / f"{label}-pair-cand.ivf"
    encode_once(binary, fixture, base_out,
                base_args + prev_flags + ["--qp", str(prev_qp)])
    encode_once(binary, fixture, cand_out,
                base_args + cand_flags + ["--qp", str(cand_qp)])
    ratios = []
    for index in range(repeats):
        if index % 2 == 0:
            _, t_prev = encode_once(binary, fixture, base_out,
                                    base_args + prev_flags + ["--qp", str(prev_qp)])
            _, t_cand = encode_once(binary, fixture, cand_out,
                                    base_args + cand_flags + ["--qp", str(cand_qp)])
        else:
            _, t_cand = encode_once(binary, fixture, cand_out,
                                    base_args + cand_flags + ["--qp", str(cand_qp)])
            _, t_prev = encode_once(binary, fixture, base_out,
                                    base_args + prev_flags + ["--qp", str(prev_qp)])
        if t_prev > 0:
            ratios.append(t_cand / t_prev)
    if not ratios:
        die(f"paired timing produced no samples for {label}")
    return {
        "d_time_pct": (percentile(ratios, 0.75) - 1.0) * 100.0,
        "median_pct": (statistics.median(ratios) - 1.0) * 100.0,
        "p75_ratio": percentile(ratios, 0.75),
        "ratios": ratios,
        "prev_qp": prev_qp,
        "cand_qp": cand_qp,
    }


def row_metrics(base: dict, on: dict, anchor: float) -> dict | None:
    q_base = interp_at_rate(base["points"], anchor)
    q_on = interp_at_rate(on["points"], anchor)
    # Conservative bounds: baseline p25, enabled p75.  Quality is deterministic.
    t_base = time_at_rate(base["points"], anchor, "time_p25_seconds")
    t_on = time_at_rate(on["points"], anchor, "time_p75_seconds")
    if None in (q_base, q_on, t_base, t_on) or t_base <= 0:
        return None
    rate_on_at_qbase = rate_at_quality(on["points"], q_base)
    if rate_on_at_qbase is None:
        return None
    d_jod = (q_on - q_base) / q_base * 100.0
    d_time = (t_on - t_base) / t_base * 100.0
    d_rate = (rate_on_at_qbase / anchor - 1.0) * 100.0
    return {
        "bitrate_anchor_kbps": anchor,
        "quality_base": q_base,
        "quality_on": q_on,
        "time_base_seconds": t_base,
        "time_on_seconds": t_on,
        "d_cvvdp_pct": d_jod,
        "d_time_pct": d_time,
        "d_rate_pct": d_rate,
        "quality_gain_pct": -d_rate,
    }


def score_row(m: dict | None) -> tuple[float | None, str | None]:
    """Return (raw score, exclusion reason).  Higher = slower/less efficient."""
    if m is None:
        return None, "missing measurement at the 1000 kbps anchor"
    gain = m["quality_gain_pct"]
    d_time = max(0.0, m["d_time_pct"])
    if gain <= 0.0:
        if m["d_cvvdp_pct"] <= 0.0 and m["d_time_pct"] > 0.0:
            return None, "CVVDP does not improve at equal bitrate and the feature costs time"
        return None, "no equal-CVVDP rate gain"
    if m["d_cvvdp_pct"] <= 0.0:
        return None, "CVVDP does not improve at the fixed bitrate anchor"
    if d_time <= 0.0:
        return 0.0, None
    return 1000.0 * d_time / max(0.25, gain), None


def confidence_for(spread_pct: float) -> str:
    if spread_pct <= 10.0:
        return "high"
    if spread_pct <= 25.0:
        return "medium"
    return "low"


def conflicts_with(flag: str, conflicts: dict[str, list[str]],
                   selected: list[str], explicit: list[str]) -> bool:
    for other in conflicts.get(flag, []):
        if other in selected or other in explicit:
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", help="av2enc binary (default: build-int/av2enc)")
    ap.add_argument("--features", default=str(ROOT / "tools" / "effort" / "features.json"))
    ap.add_argument("--fixture", help="Y4M fixture (default: first existing 640x360)")
    ap.add_argument("--qps", default="110,130,150,170,190,210,230")
    ap.add_argument("--anchors", default="300,1000,1500,2500")
    ap.add_argument("--score-anchor", type=float, default=1000.0)
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--frames", type=int, default=None,
                    help="frames to encode (default: still=1, video=4)")
    ap.add_argument("--marginal", type=int, default=5,
                    help="measure the cumulative prefix of the top K features (0 = off)")
    ap.add_argument("--out", default=str(ROOT / "tools" / "effort" / "effort_table.json"))
    ap.add_argument("--inc", default=str(ROOT / "src" / "pipeline" / "planner" /
                                        "effort_table.inc"))
    ap.add_argument("--allow-psnr-fallback", action="store_true",
                    help="record PSNR rows if CVVDP is unavailable (marked in the table)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the measurement plan and exit without encoding")
    ap.add_argument("--keep", action="store_true", help="keep temporary outputs")
    args = ap.parse_args()

    if args.repeats < 1:
        die("--repeats must be >= 1")
    qps = [int(x) for x in args.qps.split(",") if x.strip()]
    anchors = [float(x) for x in args.anchors.split(",") if x.strip()]
    if args.score_anchor not in anchors:
        die("--score-anchor must be one of --anchors")
    if not qps or len(set(qps)) != len(qps):
        die("--qps must be a non-empty list of distinct integers")
    if args.marginal < 0:
        die("--marginal must be >= 0")

    features_path = Path(args.features).expanduser().resolve()
    config = json.loads(features_path.read_text())
    mode = config.get("mode", "still")
    features = config.get("features", [])
    if not features:
        die("features.json has no features")
    if mode not in ("still", "video"):
        die("mode must be still or video")
    frames = args.frames if args.frames is not None else (1 if mode == "still" else 4)

    binary = find_bin(args.bin)
    fixture = find_fixture(args.fixture)
    finfo = y4m_info(fixture)
    if not finfo["fps_num"]:
        die("fixture has no frame rate; bitrate anchors cannot be normalised")

    base_args = (["--still"] if mode == "still" else ["--video"]) + ["--frames", str(frames)]
    configs: list[tuple[str, list[str]]] = [("baseline", [])]
    for feature in features:
        configs.append((feature["flag"].lstrip("-"), [feature["flag"]]))

    if args.dry_run:
        print("effort_calibrate: dry run")
        print(f"  binary:  {binary}")
        print(f"  fixture: {fixture} ({finfo['width']}x{finfo['height']}, "
              f"{finfo['resolution_class']}, {finfo['frames']} frame(s))")
        print(f"  mode:    {mode}, {frames} frame(s) per encode")
        print(f"  qps:     {qps}")
        print(f"  anchors: {anchors} (score anchor {args.score_anchor})")
        print(f"  repeats: {args.repeats}; marginal prefix: {args.marginal}")
        print(f"  rows:    {len(configs)} configurations x {len(qps)} QPs")
        for name, flags in configs:
            print(f"    {name:28s} {' '.join(flags) if flags else '(none)'}")
        return 0

    tmp = Path(tempfile.mkdtemp(prefix="av2enc-effort-"))
    try:
        measured: dict[str, dict] = {}
        for name, flags in configs:
            print(f"effort_calibrate: measuring {name} ({' '.join(flags) or 'baseline'})",
                  file=sys.stderr)
            measured[name] = measure_config(
                binary, fixture, base_args + flags, qps, args.repeats, tmp, name,
                args.allow_psnr_fallback)

        base = measured["baseline"]
        first = next(iter(measured.values()))
        metric_name = first["metric"]
        if metric_name == "cvvdp" and first["metric_version"] in ("unknown", ""):
            die("CVVDP version could not be identified; refusing to write a row")
        if metric_name == "psnr_fallback":
            print("effort_calibrate: WARNING: CVVDP unavailable; rows are marked "
                  "psnr_fallback", file=sys.stderr)

        if args.score_anchor not in [a for a in anchors
                                     if row_metrics(base, base, a) is not None]:
            die(f"baseline does not bracket the score anchor {args.score_anchor} kbps "
                f"(grid {qps}); widen --qps")

        source_base = (f"tools/effort/effort_calibrate.py; metric={metric_name} "
                       f"{first['metric_version']} ({first['metric_display']}); "
                       f"fixture md5 {finfo['md5']}; {args.repeats} paired "
                       f"repeats (p75 A/B CPU-time ratio); "
                       f"host {host_info().get('cpu_model') or 'unknown'}")

        base_qp = anchor_qp(base["points"], args.score_anchor)
        if base_qp is None:
            die("baseline has no QP near the score anchor")
        paired: dict[str, dict] = {}
        for feature in features:
            flag = feature["flag"]
            name = flag.lstrip("-")
            feat_qp = anchor_qp(measured[name]["points"], args.score_anchor)
            if feat_qp is None:
                die(f"{flag} does not bracket the score anchor "
                    f"{args.score_anchor} kbps")
            print(f"effort_calibrate: paired timing {name} "
                  f"(baseline QP {base_qp}, feature QP {feat_qp})",
                  file=sys.stderr)
            paired[name] = paired_time(
                binary, fixture, base_args, [], base_qp, [flag], feat_qp,
                args.repeats, tmp, name)

        rows = []
        for feature in features:
            flag = feature["flag"]
            name = flag.lstrip("-")
            m = row_metrics(base, measured[name], args.score_anchor)
            if m is not None:
                m["d_time_pct"] = paired[name]["d_time_pct"]
            raw, reason = score_row(m)
            ratio_spread = 0.0
            ratios = paired[name]["ratios"]
            if ratios and min(ratios) > 0:
                ratio_spread = ((max(ratios) - min(ratios)) /
                                statistics.median(ratios) * 100.0)
            row = {
                "flag": flag,
                "mode": mode,
                "score_raw": raw,
                "score": 0.0,
                "eligible": raw is not None and reason is None,
                "exclude_reason": reason,
                "d_cvvdp_pct": m["d_cvvdp_pct"] if m else None,
                "d_time_pct": m["d_time_pct"] if m else None,
                "d_rate_pct": m["d_rate_pct"] if m else None,
                "quality_gain_pct": m["quality_gain_pct"] if m else None,
                "bitrate_anchor_kbps": args.score_anchor,
                "resolution_class": finfo["resolution_class"],
                "repeats": args.repeats,
                "confidence": confidence_for(ratio_spread),
                "time_spread_pct": ratio_spread,
                "paired_time_p75_pct": paired[name]["d_time_pct"],
                "paired_time_median_pct": paired[name]["median_pct"],
                "anchor_qp": paired[name]["cand_qp"],
                "source": source_base,
                "conflicts": list(feature.get("conflicts", [])),
                "requires_cfl": bool(feature.get("requires_cfl", False)),
                "anchors": {},
            }
            for anchor in anchors:
                am = row_metrics(base, measured[name], anchor)
                row["anchors"][str(int(anchor))] = None if am is None else {
                    "d_cvvdp_pct": am["d_cvvdp_pct"],
                    "d_time_pct": am["d_time_pct"],
                    "d_rate_pct": am["d_rate_pct"],
                }
            rows.append(row)

        eligible = [r for r in rows if r["eligible"] and r["score_raw"] is not None]
        if not eligible:
            die("no eligible feature; nothing can be ranked (check the fixture "
                "and the anchor)")
        if min(r["score_raw"] for r in eligible) <= 0.0:
            die("timing produced a non-positive cost for the most efficient "
                "feature; increase --repeats or use an idle host")
        # The literal formula produces 1e5..1e6 on this corpus because CVVDP
        # gains are <0.3%; scale so the best measured feature costs exactly the
        # minimum budget (10) and the published 10..10000 range spans the table.
        # Ordering and ratios are unchanged.
        scale = 10.0 / min(r["score_raw"] for r in eligible)
        for row in rows:
            if row["score_raw"] is not None:
                row["score"] = round(row["score_raw"] * scale, 1)

        ranked = sorted(eligible, key=lambda r: (r["score"], r["flag"]))
        conflicts = {f["flag"]: list(f.get("conflicts", [])) for f in features}
        prefix_rows = []
        if args.marginal > 0 and ranked:
            selected: list[str] = []
            previous_name = "baseline"
            previous_flags: list[str] = []
            for rank, row in enumerate(ranked, start=1):
                if rank > args.marginal:
                    break
                flag = row["flag"]
                if conflicts_with(flag, conflicts, selected, []):
                    continue
                candidate = selected + [flag]
                if rank == 1:
                    # The marginal cost of the first feature is its solo cost.
                    # Reusing the paired solo measurement avoids timing noise
                    # changing the minimum budget.
                    row["marginal_score_raw"] = row["score_raw"]
                    row["marginal_score"] = row["score"]
                    row["marginal_eligible"] = row["eligible"]
                    row["marginal_exclude_reason"] = row["exclude_reason"]
                    row["marginal_d_time_pct"] = row["d_time_pct"]
                    row["marginal_d_rate_pct"] = row["d_rate_pct"]
                    row["marginal_d_cvvdp_pct"] = row["d_cvvdp_pct"]
                    row["marginal_paired_time_p75_pct"] = row["d_time_pct"]
                    prefix_rows.append({
                        "rank": rank,
                        "flags": candidate,
                        "score": row["marginal_score"],
                        "total_score": row["score"],
                        "marginal_d_time_pct": row["d_time_pct"],
                        "marginal_d_rate_pct": row["d_rate_pct"],
                        "total_d_time_pct": row["d_time_pct"],
                        "total_d_rate_pct": row["d_rate_pct"],
                        "measured": True,
                    })
                    selected = candidate
                    previous_name = flag.lstrip("-")
                    previous_flags = candidate
                    continue
                name = "prefix-" + str(rank)
                flags = candidate
                print(f"effort_calibrate: measuring marginal prefix {rank}: "
                      f"{' '.join(flags)}", file=sys.stderr)
                measured[name] = measure_config(
                    binary, fixture, base_args + flags, qps, args.repeats, tmp,
                    name, args.allow_psnr_fallback)
                mm = row_metrics(measured[previous_name], measured[name],
                                 args.score_anchor)
                total = row_metrics(base, measured[name], args.score_anchor)
                prev_qp = anchor_qp(measured[previous_name]["points"],
                                    args.score_anchor)
                cand_qp = anchor_qp(measured[name]["points"], args.score_anchor)
                if prev_qp is not None and cand_qp is not None:
                    print(f"effort_calibrate: paired marginal {rank} "
                          f"(prefix QP {prev_qp}, candidate QP {cand_qp})",
                          file=sys.stderr)
                    pm = paired_time(binary, fixture, base_args,
                                     previous_flags, prev_qp, flags, cand_qp,
                                     args.repeats, tmp, name)
                    if mm is not None:
                        mm["d_time_pct"] = pm["d_time_pct"]
                    row["marginal_paired_time_p75_pct"] = pm["d_time_pct"]
                    print(f"effort_calibrate: paired total {rank} "
                          f"(baseline QP {base_qp})", file=sys.stderr)
                    pm_total = paired_time(binary, fixture, base_args, [],
                                          base_qp, flags, cand_qp,
                                          args.repeats, tmp, name + "-total")
                    if total is not None:
                        total["d_time_pct"] = pm_total["d_time_pct"]
                mraw, mreason = score_row(mm)
                praw, _ = score_row(total)
                row["marginal_score_raw"] = mraw
                row["marginal_score"] = (round(mraw * scale, 1)
                                         if mraw is not None else None)
                row["marginal_eligible"] = mraw is not None and mreason is None
                row["marginal_exclude_reason"] = mreason
                if mm is not None:
                    row["marginal_d_time_pct"] = mm["d_time_pct"]
                    row["marginal_d_rate_pct"] = mm["d_rate_pct"]
                    row["marginal_d_cvvdp_pct"] = mm["d_cvvdp_pct"]
                prefix_rows.append({
                    "rank": rank,
                    "flags": candidate,
                    "score": row["marginal_score"],
                    "total_score": round(praw * scale, 1) if praw is not None else None,
                    "marginal_d_time_pct": mm["d_time_pct"] if mm else None,
                    "marginal_d_rate_pct": mm["d_rate_pct"] if mm else None,
                    "total_d_time_pct": total["d_time_pct"] if total else None,
                    "total_d_rate_pct": total["d_rate_pct"] if total else None,
                    "measured": mm is not None,
                })
                selected = candidate
                previous_name = name
                previous_flags = candidate

        # The final selected prefix is the last measured cumulative set; report
        # it once so the reviewer can compare the additive sum with a real
        # encode of the whole prefix.
        prefix_validation = None
        if prefix_rows:
            last = prefix_rows[-1]
            def effective(row: dict) -> float:
                marginal = row.get("marginal_score")
                return (float(marginal)
                        if isinstance(marginal, (int, float)) and marginal >= 0
                        else float(row["score"]))
            prefix_validation = {
                "flags": last["flags"],
                "additive_score": round(sum(
                    effective(next(r for r in rows if r["flag"] == f))
                    for f in last["flags"]), 1),
                "score": last["total_score"],
                "measured_d_time_pct": last["total_d_time_pct"],
                "measured_d_rate_pct": last["total_d_rate_pct"],
                "source": source_base,
            }

        table = {
            "schema": 2,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "tool": "tools/effort/effort_calibrate.py",
            "mode": mode,
            "score_anchor_kbps": args.score_anchor,
            "anchors_kbps": [int(a) for a in anchors],
            "qp_grid": qps,
            "repeats": args.repeats,
            "frames_per_encode": frames,
            "marginal_k": args.marginal,
            "score_scale": scale,
            "score_scale_note": (
                "score = round(score_raw * score_scale, 1). score_raw is the "
                "literal 1000*d_time_pct/max(0.25, quality_gain_pct) formula "
                "with quality_gain_pct = -d_rate_pct (equal-CVVDP bitrate gain). "
                "The literal d_cvvdp_pct denominator gives 1e5..1e6 on this "
                "corpus, outside every legal budget, so the scale maps the most "
                "efficient measured feature to the minimum budget (10) without "
                "changing the order or the ratios."),
            "metric": {
                "name": metric_name,
                "version": first["metric_version"],
                "display": first["metric_display"],
                "device": first["metric_device"],
                "psnr_fallback": metric_name == "psnr_fallback",
            },
            "host": host_info(),
            "decoder": {"path": find_decoder(), "name": Path(find_decoder()).name
                        if find_decoder() else None},
            "encoder": {"path": str(binary), "sha256": hashlib.sha256(
                binary.read_bytes()).hexdigest(),
                "git_commit": run(["git", "-C", str(ROOT), "rev-parse", "HEAD"]).stdout.strip()},
            "fixture": finfo,
            "features_file": str(features_path.relative_to(ROOT))
                             if features_path.is_relative_to(ROOT) else str(features_path),
            "ranking": [r["flag"] for r in ranked],
            "features": rows,
            "prefix_validation": prefix_validation,
            "uncalibrated": config.get("uncalibrated", []),
        }

        out = Path(args.out).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(table, indent=2) + "\n")
        print(f"effort_calibrate: wrote {out}", file=sys.stderr)

        inc = Path(args.inc).expanduser().resolve()
        rc = run([sys.executable, str(ROOT / "tools" / "effort" / "gen_table.py"),
                  "--table", str(out), "--out", str(inc)]).returncode
        if rc != 0:
            die("gen_table.py failed; the table was written but the runtime "
                "table was not regenerated")
        print(f"effort_calibrate: wrote {inc}", file=sys.stderr)
        print(json.dumps({
            "ranking": [
                {"rank": i + 1, "flag": r["flag"], "score": r["score"],
                 "marginal": r.get("marginal_score"),
                 "eligible": r["eligible"], "d_time_pct": r.get("d_time_pct"),
                 "d_rate_pct": r.get("d_rate_pct"),
                 "confidence": r["confidence"]}
                for i, r in enumerate(ranked)
            ],
            "excluded": [
                {"flag": r["flag"], "reason": r["exclude_reason"]}
                for r in rows if not r["eligible"]
            ],
        }, indent=2))
        return 0
    finally:
        if not args.keep:
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            print(f"effort_calibrate: kept {tmp}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
