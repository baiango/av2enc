#!/usr/bin/env python3
"""Wall/GPU benchmark for the M1 GPU path (``--m1-gpu``).

Reports, per resolution (64x64 / 640x360 / 1080p by default):

* wall time per frame (median and min over timed runs) and equivalent fps;
* GPU kernel time, and kernel time as a fraction of wall;
* host launch count (CUDA launch API calls) and kernel instance count;
* VRAM high-water mark;
* the evidence used to conclude the run really used the GPU.

Metric sources, best first
--------------------------
1. ``AV2ENC_GPU_STATS`` (also exported as ``AV2ENC_M1_GPU_STATS``): a JSON file
   the encoder/GPU orchestrator writes at exit. Recognised keys are documented
   in ``tools/m1_gpu/README.md``. ``kernel_ns`` measured with CUDA events is
   authoritative (``"cuda_events": true``).
2. ``nsys profile --trace=cuda`` + ``nsys stats`` (one extra untimed run per
   resolution). The container has Nsight Systems 2025.1.3; ``ncu`` counters are
   blocked and are not used.
3. ``nvidia-smi`` polling for a device-memory high-water mark. This includes
   the arena reservation and other tenants, so it is reported as a sampled
   upper bound, never as "VRAM used".

CPU rows are opt-in (``--cpu-baseline``) and are labelled a correctness oracle,
not a performance target (AGENTS.md).

Exit codes: 0 at least one GPU row measured, 1 failure, 2 usage, 3 every GPU
row skipped (``--m1-gpu`` absent, stub, or unattested -> possible CPU
fallback). ``ALLOW_SKIP=1``/``--allow-skip`` maps 3 to 0 for CI.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

TARGET_NAME = "NVIDIA GeForce RTX 3060"
TARGET_MEM_MIB = 8192
TARGET_CC = "8.6"

LAUNCH_APIS = {
    "cudaLaunchKernel",
    "cudaLaunchKernelExC",
    "cudaLaunchCooperativeKernel",
    "cudaLaunchCooperativeKernelMultiDevice",
    "cudaGraphLaunch",
    "cuLaunchKernel",
    "cuLaunchKernelEx",
}

FALLBACK_RE = re.compile(
    r"(frame bytes came from the CPU|cpu[ _-]+(m1[ _-]+)?reference|"
    r"stage hooks are not linked|frame_bytes_from_cpu)",
    re.IGNORECASE,
)

STUB_RE = re.compile(
    r"(not\s+implemented|unsupported|stub|not\s+built|not\s+available|"
    r"no\s+cuda\s+device|cuda_error_no_device|disabled|blocked|todo|skip|"
    r"requires\s+a\s+cuda\s+build|cpu[-\s]only)",
    re.IGNORECASE,
)


def die(message: str) -> "NoReturn":  # noqa: F821 - string annotation
    print(f"bench: error: {message}", file=sys.stderr)
    raise SystemExit(2)


def run(cmd, env=None, timeout=None):
    return subprocess.run(cmd, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True, timeout=timeout)


def root_dir() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass
class GpuInfo:
    name: str
    memory_mib: int
    compute_cap: str
    driver: str = ""


def gpu_info() -> "GpuInfo | None":
    if not shutil.which("nvidia-smi"):
        return None
    try:
        query = run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap,"
                     "driver_version", "--format=csv,noheader"], timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if query.returncode != 0 or not query.stdout.strip():
        return None
    parts = [part.strip() for part in query.stdout.splitlines()[0].split(",")]
    if len(parts) < 3:
        return None
    match = re.search(r"(\d+)", parts[1])
    memory = int(match.group(1)) if match else 0
    return GpuInfo(name=parts[0], memory_mib=memory, compute_cap=parts[2],
                   driver=parts[3] if len(parts) > 3 else "")


def encoder_supports_gpu(encoder: Path) -> bool:
    try:
        proc = run([str(encoder), "--help"], timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return "--m1-gpu" in (proc.stdout + proc.stderr)


def make_input(path: Path, width: int, height: int, force: bool = False) -> None:
    if path.is_file() and path.stat().st_size > 0 and not force:
        return
    if not shutil.which("ffmpeg"):
        die(f"ffmpeg not found; cannot generate {width}x{height} input")
    path.parent.mkdir(parents=True, exist_ok=True)
    src_w, src_h = width, height
    prefilter = []
    tmp = None
    if width > 640:
        # Match tools/m1_verify/run_matrix.sh: build 640x360, Lanczos upscale.
        tmp = path.with_name(f".src-{path.stem}.y4m")
        src_w, src_h = 640, 360
    cmd = [
        "ffmpeg", "-hide_banner", "-v", "error", "-y",
        "-f", "lavfi", "-i",
        f"testsrc2=size={src_w}x{src_h}:rate=1",
        "-frames:v", "1", "-pix_fmt", "yuv420p",
    ]
    if tmp is not None:
        cmd += ["-f", "yuv4mpegpipe", str(tmp)]
        proc = run(cmd, timeout=300)
        if proc.returncode != 0:
            die(f"ffmpeg failed generating {tmp}: {proc.stderr.strip()}")
        cmd = [
            "ffmpeg", "-hide_banner", "-v", "error", "-y",
            "-i", str(tmp),
            "-vf", f"scale={width}:{height}:flags=lanczos",
            "-frames:v", "1", "-pix_fmt", "yuv420p",
            "-f", "yuv4mpegpipe", str(path),
        ]
    else:
        cmd += ["-f", "yuv4mpegpipe", str(path)]
    proc = run(cmd, timeout=300)
    if tmp is not None:
        tmp.unlink(missing_ok=True)
    if proc.returncode != 0 or not path.is_file():
        die(f"ffmpeg failed generating {width}x{height} input: "
            f"{proc.stderr.strip()}")


class VramSampler(threading.Thread):
    """Poll nvidia-smi device memory while a subprocess runs."""

    def __init__(self, interval: float = 0.05):
        super().__init__(daemon=True)
        self.interval = interval
        self.baseline = 0
        self.peak = 0
        self._halt = threading.Event()
        self.ok = shutil.which("nvidia-smi") is not None

    def _read(self) -> int:
        try:
            proc = run(["nvidia-smi", "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits"], timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            return 0
        if proc.returncode != 0:
            return 0
        values = []
        for line in proc.stdout.splitlines():
            line = line.strip()
            if line.isdigit():
                values.append(int(line))
        return max(values) if values else 0

    def run(self):
        if not self.ok:
            return
        self.baseline = self._read()
        self.peak = self.baseline
        while not self._halt.is_set():
            self.peak = max(self.peak, self._read())
            self._halt.wait(self.interval)

    def stop(self) -> None:
        self._halt.set()

    def delta_mib(self) -> int:
        return max(0, self.peak - self.baseline)


@dataclass
class Measurement:
    kernel_ns: int = 0
    kernels: int = 0
    launches: int = 0
    vram_peak_mib: int = 0
    vram_source: str = ""
    backend: str = ""
    cuda_events: bool = False
    cpu_reference: bool = False
    source: str = ""

    def attested(self) -> bool:
        if self.cpu_reference:
            return False
        return (self.backend.lower() in ("gpu", "m1-gpu", "cuda") or
                self.kernels > 0 or self.launches > 0)


def parse_stats_file(path: Path) -> "Measurement | None":
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    measurement = Measurement(source="stats")

    def as_int(*names) -> int:
        for name in names:
            value = data.get(name)
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                return int(value)
            if isinstance(value, str) and value.isdigit():
                return int(value)
        return 0

    measurement.cpu_reference = data.get("frame_bytes_from_cpu") is True
    measurement.backend = str(data.get("backend", data.get("path", "")))
    measurement.kernel_ns = as_int("kernel_ns", "gpu_kernel_ns", "kernel_time_ns")
    measurement.kernels = as_int("kernels", "kernel_count", "kernel_instances")
    measurement.launches = as_int("launches", "launch_count", "host_launches")
    measurement.cuda_events = bool(data.get("cuda_events", False))
    peak = as_int("vram_peak_bytes", "arena_peak_bytes", "device_peak_bytes")
    if peak:
        measurement.vram_peak_mib = int(peak / (1024 * 1024))
        measurement.vram_source = "stats"
    if measurement.kernel_ns and not measurement.cuda_events:
        measurement.source = "stats(kernel_ns)"
    elif measurement.kernel_ns:
        measurement.source = "stats(cuda events)"
    return measurement


def _nsys_rows(report: str, rep_file: Path) -> list:
    try:
        proc = run(["nsys", "stats", "--force-export=true", "--report", report,
                    "--format", "csv", str(rep_file)], timeout=300)
    except subprocess.TimeoutExpired:
        return []
    if proc.returncode != 0:
        return []
    lines = proc.stdout.splitlines()
    header = None
    for index, line in enumerate(lines):
        if line.startswith("Time (%)"):
            header = index
            break
    if header is None:
        return []
    reader = csv.DictReader(lines[header:])
    return [row for row in reader if row.get("Name")]


def nsys_measure(cmd, env, workdir: Path) -> "Measurement | None":
    if not shutil.which("nsys"):
        return None
    prefix = workdir / "nsys-profile"
    for suffix in (".nsys-rep", ".sqlite"):
        (workdir / f"{prefix.name}{suffix}").unlink(missing_ok=True)
    try:
        profile = run([
            "nsys", "profile", "--trace=cuda", "--sample=none",
            "--cpuctxsw=none", "--force-overwrite", "true",
            "-o", str(prefix),
        ] + [str(part) for part in cmd], env=env, timeout=1800)
    except subprocess.TimeoutExpired:
        return None
    rep_file = workdir / "nsys-profile.nsys-rep"
    if profile.returncode != 0 or not rep_file.is_file():
        return None
    measurement = Measurement(source="nsys")

    kernel_rows = _nsys_rows("cuda_gpu_kern_sum", rep_file)
    for row in kernel_rows:
        try:
            measurement.kernel_ns += int(row.get("Total Time (ns)", "0"))
            measurement.kernels += int(row.get("Instances", "0"))
        except ValueError:
            continue
    api_rows = _nsys_rows("cuda_api_sum", rep_file)
    for row in api_rows:
        if row.get("Name") in LAUNCH_APIS:
            try:
                measurement.launches += int(row.get("Num Calls", "0"))
            except ValueError:
                continue
    if measurement.kernels:
        measurement.backend = "gpu"
    return measurement


@dataclass
class Row:
    size: str
    path: str
    wall_ms: float = 0.0
    min_ms: float = 0.0
    fps: float = 0.0
    kernel_ms: float = 0.0
    launches: int = 0
    kernels: int = 0
    vram_mib: int = 0
    vram_source: str = ""
    attest: str = "n/a"
    status: str = "ok"
    note: str = ""


def _timed_run(cmd, env, timeout):
    try:
        return run(cmd, env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            cmd, 124, "", f"timed out after {timeout}s")


def time_runs(cmd, env, iterations: int, warmup: int, timeout):
    proc = None
    for _ in range(warmup):
        proc = _timed_run(cmd, env, timeout)
        if proc.returncode != 0:
            return None, proc
    walls = []
    for _ in range(iterations):
        start = time.perf_counter()
        proc = _timed_run(cmd, env, timeout)
        walls.append((time.perf_counter() - start) * 1000.0)
        if proc.returncode != 0:
            return None, proc
    return walls, proc


def tag_output(text: str) -> str:
    return text.strip().splitlines()[-1] if text.strip() else ""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="M1 GPU wall/kernel/VRAM benchmark with proxy labelling.")
    parser.add_argument("--encoder", default=None,
                        help="av2enc binary (default build-cuda/av2enc)")
    parser.add_argument("--workdir", default=None,
                        help="artifact directory (default a temp dir)")
    parser.add_argument("--sizes", default="64x64,640x360,1920x1080",
                        help="comma-separated WxH list")
    parser.add_argument("--iterations", type=int, default=3,
                        help="timed GPU runs per size (default 3; keep low)")
    parser.add_argument("--warmup", type=int, default=1,
                        help="warmup runs per size (default 1)")
    parser.add_argument("--qp", type=int, default=90)
    parser.add_argument("--cpu-baseline", action="store_true",
                        help="also time the CPU --m1 oracle (reference only)")
    parser.add_argument("--nsys", choices=("auto", "on", "off"), default="auto",
                        help="Nsight Systems kernel/launch metrics (default auto)")
    parser.add_argument("--vram-sample", choices=("auto", "on", "off"),
                        default="auto",
                        help="nvidia-smi memory polling fallback (default auto)")
    parser.add_argument("--gpu-args", default="",
                        help="extra arguments for --m1-gpu runs")
    parser.add_argument("--cpu-args", default="",
                        help="extra arguments for --m1 runs")
    parser.add_argument("--json", default=None, help="write JSON results here")
    parser.add_argument("--timeout", type=float, default=1800.0,
                        help="per-encode timeout in seconds")
    parser.add_argument("--keep", action="store_true",
                        help="keep the work directory")
    parser.add_argument("--allow-skip", action="store_true",
                        help="exit 0 when every GPU row is skipped")
    args = parser.parse_args()

    root = root_dir()
    encoder = Path(args.encoder) if args.encoder else root / "build-cuda" / "av2enc"
    if not encoder.is_file():
        die(f"encoder not found: {encoder} (build build-cuda/av2enc)")

    if args.workdir:
        workdir = Path(args.workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        workdir = Path(tempfile.mkdtemp(prefix="m1_gpu_bench_"))
        cleanup = not args.keep

    info = gpu_info()
    print("== m1 GPU benchmark ==")
    if info is None:
        print("hardware: unknown (no usable nvidia-smi)")
        print("TARGET:   NVIDIA GeForce RTX 3060, 8192 MiB, cc 8.6")
        print("WARNING:  not a GPU host or nvidia-smi unavailable; "
              "GPU numbers will be skipped.")
    else:
        print(f"hardware: {info.name}, {info.memory_mib} MiB, "
              f"cc {info.compute_cap}, driver {info.driver or '?'}")
        print(f"target:   {TARGET_NAME}, {TARGET_MEM_MIB} MiB, cc {TARGET_CC}")
        target = (info.name == TARGET_NAME and
                  info.memory_mib == TARGET_MEM_MIB and
                  info.compute_cap.startswith(TARGET_CC))
        if not target:
            print("WARNING:  PROXY HARDWARE — this is not the 8 GB RTX 3060 "
                  "target; treat throughput as indicative only "
                  "(docs/negative-results.md).")
        if info.memory_mib != TARGET_MEM_MIB:
            print(f"          enumerated memory {info.memory_mib} MiB != "
                  f"{TARGET_MEM_MIB} MiB; do not quote this as 8 GB "
                  f"validation.")
    print(f"encoder:  {encoder}")
    supported = encoder_supports_gpu(encoder)
    if not supported:
        print("gate:     encoder help has no --m1-gpu; GPU rows will be SKIPPED")
    print(f"workdir:  {workdir}")
    print()

    sizes = []
    for token in args.sizes.split(","):
        token = token.strip()
        if not token:
            continue
        match = re.fullmatch(r"(\d+)x(\d+)", token)
        if not match:
            die(f"bad size: {token!r} (want WxH)")
        sizes.append((int(match.group(1)), int(match.group(2))))

    gpu_extra = shlex.split(args.gpu_args) if args.gpu_args else []
    cpu_extra = shlex.split(args.cpu_args) if args.cpu_args else []
    nsys_available = shutil.which("nsys") is not None

    rows = []
    gpu_measured = 0
    failures = 0

    for width, height in sizes:
        size_label = f"{width}x{height}"
        input_path = workdir / f"in-{size_label}.y4m"
        make_input(input_path, width, height)

        # ---- CPU oracle (opt-in, reference only) ---------------------------
        if args.cpu_baseline:
            out = workdir / f"cpu-{size_label}.ivf"
            cmd = [str(encoder), str(input_path), "-o", str(out),
                   "--m1", "--qp", str(args.qp)] + cpu_extra
            walls, proc = time_runs(cmd, None, 1, 0, args.timeout)
            row = Row(size=size_label, path="cpu")
            if walls is None:
                row.status = "FAIL"
                row.note = tag_output(proc.stderr or proc.stdout)
                failures += 1
            else:
                row.wall_ms = walls[0]
                row.min_ms = walls[0]
                row.fps = 1000.0 / row.wall_ms if row.wall_ms else 0.0
                row.attest = "oracle"
                row.note = "CPU reference (not a performance target)"
            rows.append(row)
            print(f"{size_label:>10s}  cpu  wall {row.wall_ms:9.2f} ms  "
                  f"{row.status}")

        # ---- GPU path ------------------------------------------------------
        row = Row(size=size_label, path="gpu")
        if not supported:
            row.status = "SKIP"
            row.note = "encoder has no --m1-gpu"
            rows.append(row)
            print(f"{size_label:>10s}  gpu  SKIP (no --m1-gpu)")
            continue

        out = workdir / f"gpu-{size_label}.ivf"
        stats_path = workdir / f"gpu-{size_label}-stats.json"
        stats_path.unlink(missing_ok=True)
        env = dict(os.environ)
        env["AV2ENC_GPU_STATS"] = str(stats_path)
        env["AV2ENC_M1_GPU_STATS"] = str(stats_path)
        cmd = [str(encoder), str(input_path), "-o", str(out),
               "--m1-gpu", "--qp", str(args.qp)] + gpu_extra

        # Warmup + timed wall runs.
        sampler = None
        start_sampling = args.vram_sample == "on" or (
            args.vram_sample == "auto" and shutil.which("nvidia-smi"))
        if start_sampling:
            sampler = VramSampler()
            sampler.start()
        try:
            walls, proc = time_runs(cmd, env, max(1, args.iterations),
                                    max(0, args.warmup), args.timeout)
        finally:
            if sampler is not None:
                sampler.stop()
                sampler.join(timeout=5)
        if walls is None:
            text = (proc.stderr or "") + (proc.stdout or "")
            if STUB_RE.search(text):
                row.status = "SKIP"
                row.note = f"GPU path blocked/stub: {tag_output(text)}"
            else:
                row.status = "FAIL"
                row.note = f"exit {proc.returncode}: {tag_output(text)}"
                failures += 1
            rows.append(row)
            print(f"{size_label:>10s}  gpu  {row.status} ({row.note})")
            continue

        text = ((proc.stderr or "") + (proc.stdout or "")) if proc else ""
        if FALLBACK_RE.search(text):
            row.status = "SKIP"
            row.note = ("CPU-reference staging: frame bytes did not come "
                        "from GPU stages")
            rows.append(row)
            print(f"{size_label:>10s}  gpu  SKIP ({row.note})")
            continue

        row.wall_ms = statistics.median(walls)
        row.min_ms = min(walls)
        row.fps = 1000.0 / row.wall_ms if row.wall_ms else 0.0
        if not out.is_file() or out.stat().st_size == 0:
            row.status = "SKIP"
            row.note = "GPU run wrote no IVF (stub/blocked)"
            rows.append(row)
            print(f"{size_label:>10s}  gpu  SKIP (no IVF written)")
            continue

        measurement = parse_stats_file(stats_path) or Measurement()
        if measurement.source:
            row.note = measurement.source

        need_nsys = nsys_available and args.nsys != "off" and (
            args.nsys == "on" or not measurement.attested() or
            (measurement.kernel_ns == 0 and measurement.kernels == 0))
        if need_nsys:
            measured = nsys_measure(cmd, env, workdir)
            if measured is not None:
                if measured.kernel_ns and not measurement.kernel_ns:
                    measurement.kernel_ns = measured.kernel_ns
                if measured.kernels and not measurement.kernels:
                    measurement.kernels = measured.kernels
                if measured.launches and not measurement.launches:
                    measurement.launches = measured.launches
                if measured.attested():
                    measurement.backend = measurement.backend or "gpu"
                measurement.source = measurement.source or measured.source
                row.note = measurement.source or "nsys"

        if not measurement.vram_peak_mib and sampler is not None:
            measurement.vram_peak_mib = sampler.delta_mib()
            measurement.vram_source = "nvidia-smi sampled delta"

        row.vram_mib = measurement.vram_peak_mib
        row.vram_source = measurement.vram_source
        row.kernel_ms = measurement.kernel_ns / 1e6
        row.launches = measurement.launches
        row.kernels = measurement.kernels

        if measurement.attested():
            row.attest = "gpu"
            gpu_measured += 1
        else:
            row.status = "SKIP"
            row.attest = "none"
            row.note = ("unattested GPU execution (possible CPU fallback); "
                        "no stats/nsys evidence, wall time not a GPU number")
        rows.append(row)
        if row.status == "ok":
            print(f"{size_label:>10s}  gpu  wall {row.wall_ms:9.2f} ms  "
                  f"kernel {row.kernel_ms:8.3f} ms  "
                  f"{row.launches} launches  {row.fps:7.2f} fps")
        else:
            print(f"{size_label:>10s}  gpu  {row.status} ({row.note})")
        print()

    # ---- table -------------------------------------------------------------
    print("results")
    header = (f"{'size':>10s}  {'path':4s}  {'wall_ms':>10s}  {'min_ms':>9s}  "
              f"{'fps':>8s}  {'kernel_ms':>9s}  {'kern%':>6s}  "
              f"{'launch':>7s}  {'kerns':>6s}  {'vram_MiB':>9s}  "
              f"{'attest':6s}  status")
    print(header)
    print("-" * len(header))
    for row in rows:
        kern_pct = (100.0 * row.kernel_ms / row.wall_ms
                    if row.kernel_ms and row.wall_ms else 0.0)
        note = row.note
        if len(note) > 72:
            note = note[:69] + "..."
        print(f"{row.size:>10s}  {row.path:4s}  {row.wall_ms:10.2f}  "
              f"{row.min_ms:9.2f}  {row.fps:8.2f}  {row.kernel_ms:9.3f}  "
              f"{kern_pct:6.1f}  {row.launches:7d}  {row.kernels:6d}  "
              f"{row.vram_mib:9d}  {row.attest:6s}  {row.status}"
              + (f"  {note}" if note and row.status != "ok" else ""))
    print()
    print("notes")
    print("  wall_ms  median of timed runs; min_ms is the best run")
    print("  kernel_ms  GPU kernel time from CUDA events/stats, else nsys sum")
    print("  kern%    kernel time / wall (not wall-clock concurrency aware)")
    print("  launch   CUDA launch API calls; kerns = kernel instances")
    print("  vram_MiB high-water mark; source is stats when available, else a")
    print("           sampled nvidia-smi delta (includes the arena reservation")
    print("           and other tenants — an upper bound, not 'VRAM used')")
    print("  fps      1 / wall_ms for a one-frame still; not a video claim")

    if args.json:
        payload = {
            "hardware": None if info is None else info.__dict__,
            "target": {"name": TARGET_NAME, "memory_mib": TARGET_MEM_MIB,
                       "compute_cap": TARGET_CC},
            "gpu_flag_present": supported,
            "rows": [row.__dict__ for row in rows],
        }
        Path(args.json).write_text(json.dumps(payload, indent=2))
        print(f"json:     {args.json}")

    if failures:
        verdict = "FAIL"
        code = 1
    elif gpu_measured == 0:
        verdict = "SKIP"
        code = 3
    else:
        verdict = "OK"
        code = 0
    print(f"RESULT: {verdict} ({gpu_measured} GPU row(s) measured)")
    if cleanup:
        shutil.rmtree(workdir, ignore_errors=True)
    else:
        print(f"kept:    {workdir}")
    if code == 3 and (args.allow_skip or os.environ.get("ALLOW_SKIP") == "1"):
        return 0
    return code


if __name__ == "__main__":
    sys.exit(main())
