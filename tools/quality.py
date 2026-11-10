#!/usr/bin/env python3
"""Quality report for an AV2 encode.

Computes:
  - bitrate (when given an IVF)
  - PSNR (ffmpeg)
  - SSIM (ffmpeg)
  - ColorVideoVDP / CVVDP (JOD, PyTorch, MPS on Apple Silicon)

The decoder is dav2d. CVVDP is run with a fixed display model so numbers are
comparable across encoder revisions.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DAV2D = ROOT / "tools" / "bin" / "dav2d"
DEFAULT_DAV2D_MACOS = ROOT / "tools" / "bin" / "dav2d-macos"


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, text=True, capture_output=True, **kw)


def _sysctl(name: str) -> str | None:
    p = run(["sysctl", "-n", name])
    return p.stdout.strip() if p.returncode == 0 else None


def _linux_cpu() -> dict:
    model = None
    cores_per_socket = None
    sockets: set[str] = set()
    try:
        text = Path("/proc/cpuinfo").read_text(errors="replace")
    except OSError:
        return {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = (x.strip() for x in line.split(":", 1))
        if key == "model name" and not model:
            model = value
        elif key == "cpu cores" and cores_per_socket is None:
            try:
                cores_per_socket = int(value)
            except ValueError:
                pass
        elif key == "physical id":
            sockets.add(value)
    physical = None
    if cores_per_socket:
        physical = cores_per_socket * max(1, len(sockets))
    return {"cpu_model": model, "cpu_physical_cores": physical}


def _mem_gb() -> float | None:
    if sys.platform == "darwin":
        v = _sysctl("hw.memsize")
        return round(int(v) / 1e9, 1) if v else None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return round(int(line.split()[1]) / 1024**2, 1)
    except OSError:
        pass
    return None


def _lscpu_field(prefix: str) -> str | None:
    if not shutil.which("lscpu"):
        return None
    p = run(["lscpu"])
    for line in p.stdout.splitlines():
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip()
    return None


def host_info() -> dict:
    info = {
        "hostname": platform.node(),
        "os": platform.platform(),
        "machine": platform.machine(),
        "cpu_logical_cores": os.cpu_count(),
        "ram_gb": _mem_gb(),
    }
    if sys.platform == "darwin":
        info["cpu_model"] = _sysctl("machdep.cpu.brand_string") or platform.processor()
        phys = _sysctl("hw.physicalcpu")
        info["cpu_physical_cores"] = int(phys) if phys else None
    else:
        info.update(_linux_cpu())
        info["cpu_max_mhz"] = _lscpu_field("CPU max MHz:")
        info["cpu_scaling_pct"] = _lscpu_field("CPU(s) scaling MHz:")

    gpus = []
    if shutil.which("nvidia-smi"):
        p = run([
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version,compute_cap",
            "--format=csv,noheader",
        ])
        if p.returncode == 0:
            gpus = [x.strip() for x in p.stdout.splitlines() if x.strip()]
    info["gpu"] = gpus or None
    if gpus:
        parts = [x.strip() for x in gpus[0].split(",")]
        if len(parts) >= 4:
            info["gpu_name"] = parts[0]
            mem = parts[1].split()
            try:
                info["gpu_memory_mib"] = int(mem[0])
            except (ValueError, IndexError):
                pass
            info["gpu_driver"] = parts[2]
            info["gpu_compute_cap"] = parts[3]
    return info


def find_dav2d() -> str:
    if os.environ.get("DAV2D"):
        return os.environ["DAV2D"]
    for p in (DEFAULT_DAV2D, DEFAULT_DAV2D_MACOS):
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)
    found = shutil.which("dav2d")
    if found:
        return found
    raise SystemExit("dav2d not found; set DAV2D=/path/to/dav2d")


def y4m_header(path: Path) -> str:
    with path.open("rb") as f:
        return f.readline().decode("ascii", "replace").rstrip("\n")


def y4m_geometry(header: str) -> tuple[int, int, str, int, int]:
    m = re.search(r"W(\d+)\s+H(\d+)", header)
    if not m:
        raise SystemExit(f"bad Y4M header: {header}")
    w, h = int(m.group(1)), int(m.group(2))
    fm = re.search(r"F(\d+):(\d+)", header)
    if not fm:
        raise SystemExit(f"bad Y4M frame rate: {header}")
    fps_num, fps_den = int(fm.group(1)), int(fm.group(2))
    cm = re.search(r"C([0-9A-Za-z]+)", header)
    chroma = cm.group(1) if cm else "420"
    return w, h, chroma, fps_num, fps_den


def frame_bytes(header: str) -> int:
    w, h, chroma, _, _ = y4m_geometry(header)
    if chroma.startswith("420"):
        cw, ch = (w + 1) // 2, (h + 1) // 2
        return w * h + 2 * cw * ch
    if chroma.startswith("422"):
        cw = (w + 1) // 2
        return w * h + 2 * cw * h
    if chroma.startswith("444"):
        return 3 * w * h
    raise SystemExit(f"unsupported chroma in {header}")


def read_frames(path: Path) -> tuple[str, list[bytes]]:
    header = y4m_header(path)
    size = frame_bytes(header)
    frames: list[bytes] = []
    with path.open("rb") as f:
        f.readline()
        while True:
            line = f.readline()
            if not line:
                break
            if not line.startswith(b"FRAME"):
                raise SystemExit(f"{path}: expected FRAME, got {line[:20]!r}")
            buf = f.read(size)
            if len(buf) != size:
                raise SystemExit(f"{path}: truncated frame {len(frames)}")
            frames.append(buf)
    return header, frames


def write_y4m(path: Path, header: str, frames: list[bytes]) -> None:
    with path.open("wb") as f:
        f.write((header + "\n").encode("ascii"))
        for fr in frames:
            f.write(b"FRAME\n")
            f.write(fr)


def normalize_pair(ref: Path, test: Path, outdir: Path) -> tuple[Path, Path, int, str]:
    ref_header, ref_frames = read_frames(ref)
    test_header, test_frames = read_frames(test)

    rw, rh, _, rfps_n, rfps_d = y4m_geometry(ref_header)
    tw, th, _, _, _ = y4m_geometry(test_header)
    if (rw, rh) != (tw, th):
        raise SystemExit(f"dimension mismatch: ref {rw}x{rh}, test {tw}x{th}")

    n = min(len(ref_frames), len(test_frames))
    if n == 0:
        raise SystemExit("no frames")
    if len(ref_frames) != len(test_frames):
        print(
            f"warning: ref has {len(ref_frames)} frames, test has {len(test_frames)}; "
            f"using first {n}",
            file=sys.stderr,
        )

    ref_header = re.sub(r"F\d+:\d+", f"F{rfps_n}:{rfps_d}", ref_header)
    test_header = re.sub(r"F\d+:\d+", f"F{rfps_n}:{rfps_d}", test_header)

    ref_out = outdir / "ref.y4m"
    test_out = outdir / "test.y4m"
    write_y4m(ref_out, ref_header, ref_frames[:n])
    write_y4m(test_out, test_header, test_frames[:n])
    return ref_out, test_out, n, f"{rfps_n}/{rfps_d}"


def parse_psnr(text: str) -> dict[str, float]:
    m = re.search(
        r"PSNR y:([0-9.]+) u:([0-9.]+) v:([0-9.]+) average:([0-9.]+)",
        text,
    )
    if not m:
        raise SystemExit("could not parse PSNR")
    return {
        "psnr_y": float(m.group(1)),
        "psnr_u": float(m.group(2)),
        "psnr_v": float(m.group(3)),
        "psnr_avg": float(m.group(4)),
    }


def parse_ssim(text: str) -> dict[str, float]:
    m = re.search(r"SSIM Y:([0-9.]+).*?All:([0-9.]+)", text)
    if not m:
        raise SystemExit("could not parse SSIM")
    return {"ssim_y": float(m.group(1)), "ssim_all": float(m.group(2))}


def cvvdp_command() -> list[str]:
    if os.environ.get("CVVDP_CMD"):
        return os.environ["CVVDP_CMD"].split()
    if shutil.which("cvvdp"):
        return ["cvvdp"]
    try:
        import pycvvdp  # noqa: F401
    except Exception:
        raise SystemExit("cvvdp not found; pip install cvvdp or set CVVDP_CMD")
    return [sys.executable, "-m", "pycvvdp.run_cvvdp"]


def default_device() -> str:
    if os.environ.get("CVVDP_DEVICE"):
        return os.environ["CVVDP_DEVICE"]
    if sys.platform == "darwin":
        return "mps"
    return "cpu"


def run_cvvdp(ref: Path, test: Path) -> tuple[float, str]:
    cmd = cvvdp_command() + [
        "-t", str(test),
        "-r", str(ref),
        "-d", os.environ.get("CVVDP_DISPLAY", "standard_fhd"),
        "--device", default_device(),
    ]
    p = run(cmd)
    out = (p.stdout or "") + "\n" + (p.stderr or "")
    m = re.search(r"cvvdp=([-0-9.]+)\s*\[JOD\]", out)
    if not m:
        sys.stderr.write(out[-4000:])
        raise SystemExit("could not parse CVVDP output")
    ver = re.search(r"ColorVideoVDP v([0-9.]+)", out)
    return float(m.group(1)), (ver.group(1) if ver else "unknown")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, type=Path, help="reference Y4M")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--test", type=Path, help="distorted Y4M")
    src.add_argument("--ivf", type=Path, help="AV2 IVF; decoded with dav2d")
    ap.add_argument("--json", type=Path, help="write JSON report")
    ap.add_argument("--keep", action="store_true", help="keep normalized Y4M files")
    ap.add_argument("--encoder", help="encoder name for score_speed.py")
    ap.add_argument("--qp", help="encoder QP/setting label")
    ap.add_argument("--encode-seconds", type=float,
                    help="wall-clock encode time; adds fps/realtime_x")
    args = ap.parse_args()

    if not args.ref.is_file():
        raise SystemExit(f"missing ref: {args.ref}")

    tmp = Path(tempfile.mkdtemp(prefix="av2qual-"))
    try:
        if args.ivf:
            if not args.ivf.is_file():
                raise SystemExit(f"missing ivf: {args.ivf}")
            test_y4m = tmp / "decoded.y4m"
            dav2d = find_dav2d()
            p = run([dav2d, "-q", "-i", str(args.ivf), "-o", str(test_y4m),
                     "--threads", os.environ.get("DAV2D_THREADS", "0")])
            if p.returncode != 0 or not test_y4m.is_file():
                sys.stderr.write(p.stdout or "")
                sys.stderr.write(p.stderr or "")
                raise SystemExit("dav2d failed")
        else:
            test_y4m = args.test
            if not test_y4m.is_file():
                raise SystemExit(f"missing test: {test_y4m}")

        ref_n, test_n, n, fps = normalize_pair(args.ref, test_y4m, tmp)

        psnr_out = run([
            "ffmpeg", "-hide_banner", "-v", "info", "-i", str(test_n),
            "-i", str(ref_n), "-lavfi", "[0:v][1:v]psnr", "-f", "null", "-",
        ])
        psnr = parse_psnr((psnr_out.stdout or "") + "\n" + (psnr_out.stderr or ""))

        ssim_out = run([
            "ffmpeg", "-hide_banner", "-v", "info", "-i", str(test_n),
            "-i", str(ref_n), "-lavfi", "[0:v][1:v]ssim", "-f", "null", "-",
        ])
        ssim = parse_ssim((ssim_out.stdout or "") + "\n" + (ssim_out.stderr or ""))

        jod, ver = run_cvvdp(ref_n, test_n)

        host = host_info()
        target_mem = 8192
        is_target = (
            host.get("gpu_memory_mib") == target_mem
            and str(host.get("gpu_compute_cap", "")).strip() == "8.6"
        )

        report = {
            "ref": str(args.ref),
            "test": str(args.ivf or args.test),
            "frames": n,
            "fps": fps,
            "psnr": psnr,
            "ssim": ssim,
            "cvvdp_jod": jod,
            "cvvdp_version": ver,
            "cvvdp_display": os.environ.get("CVVDP_DISPLAY", "standard_fhd"),
            "cvvdp_device": default_device(),
            "host": host,
            "target_host": {
                "gpu_memory_mib": target_mem,
                "compute_cap": "8.6",
                "is_target": is_target,
            },
        }
        if args.ivf:
            size = args.ivf.stat().st_size
            num, den = (int(x) for x in fps.split("/"))
            duration = n * den / num
            report["ivf_bytes"] = size
            report["bitrate_kbps"] = size * 8 / duration / 1000

        if args.encoder:
            report["encoder"] = args.encoder
        if args.qp:
            report["qp"] = args.qp
        if args.encode_seconds and args.encode_seconds > 0:
            num, den = (int(x) for x in fps.split("/"))
            video_seconds = n * den / num
            report["encode_seconds"] = args.encode_seconds
            report["encode_fps"] = n / args.encode_seconds
            report["realtime_x"] = video_seconds / args.encode_seconds

        print(json.dumps(report, indent=2))
        if args.json:
            args.json.write_text(json.dumps(report, indent=2) + "\n")
        if args.keep:
            print(f"normalized files: {tmp}")
        return 0
    finally:
        if not args.keep:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
