#!/usr/bin/env python3
"""Quality report for an AV2 encode.

Computes:
  - bitrate (when given an IVF)
  - PSNR (ffmpeg)
  - SSIM (ffmpeg)
  - ColorVideoVDP / CVVDP (JOD, PyTorch, MPS on Apple Silicon)

The decoder is AVM `avmdec`. CVVDP is run with a fixed display model so numbers
are comparable across encoder revisions.
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
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AVM = ROOT / "tools" / "bin" / "avmdec"
DEFAULT_AVM_MACOS = ROOT / "tools" / "bin" / "avmdec-macos"


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


def find_avmdec() -> str:
    for name in ("AVMDEC", "AVM_DEC"):
        path = os.environ.get(name)
        if path:
            return path
    for p in (DEFAULT_AVM, DEFAULT_AVM_MACOS):
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)
    found = shutil.which("avmdec")
    if found:
        return found
    raise SystemExit("avmdec not found; set AVMDEC=/path/to/avmdec")


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


def chroma_family(chroma: str) -> str:
    if chroma.startswith("420"):
        return "420"
    if chroma.startswith("422"):
        return "422"
    if chroma.startswith("444"):
        return "444"
    return chroma


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


def decode_avm(avmdec: str, ivf: Path, ref: Path, out_y4m: Path,
               threads: str | None = None) -> int:
    """Decode an IVF with AVM to 8-bit I420 and wrap it as Y4M."""
    header = y4m_header(ref)
    if re.search(r"C[0-9A-Za-z]*p(10|12|16)", header):
        raise SystemExit("quality.py: AVM path currently supports 8-bit Y4M only")
    if not re.search(r"C420", header) and re.search(r"C(422|444)", header):
        raise SystemExit("quality.py: AVM path currently supports 4:2:0 Y4M only")
    raw = out_y4m.with_suffix(".i420")
    cmd = [avmdec, "--i420", "--summary", "-o", str(raw)]
    if threads:
        cmd += ["-t", str(threads)]
    cmd.append(str(ivf))
    p = run(cmd)
    if p.returncode != 0 or not raw.is_file():
        sys.stderr.write(p.stdout or "")
        sys.stderr.write(p.stderr or "")
        raise SystemExit("avmdec failed")
    data = raw.read_bytes()
    size = frame_bytes(header)
    if size <= 0 or len(data) == 0 or len(data) % size != 0:
        raise SystemExit(
            f"avmdec output {len(data)} B is not a whole number of "
            f"{header.split()[1] if len(header.split()) > 1 else '?'} frames")
    frames = [data[i:i + size] for i in range(0, len(data), size)]
    write_y4m(out_y4m, header, frames)
    return len(frames)


def normalize_pair(ref: Path, test: Path, outdir: Path) -> tuple[Path, Path, int, str]:
    ref_header, ref_frames = read_frames(ref)
    test_header, test_frames = read_frames(test)

    rw, rh, rchroma, rfps_n, rfps_d = y4m_geometry(ref_header)
    tw, th, tchroma, _, _ = y4m_geometry(test_header)
    if (rw, rh) != (tw, th):
        raise SystemExit(f"dimension mismatch: ref {rw}x{rh}, test {tw}x{th}")
    if chroma_family(rchroma) != chroma_family(tchroma):
        raise SystemExit(
            f"chroma mismatch: ref {chroma_family(rchroma)} vs test "
            f"{chroma_family(tchroma)}; 4:2:0 and 4:4:4 must be scored "
            f"separately")

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
    src.add_argument("--ivf", type=Path, help="AV2 IVF; decoded with AVM")
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
        decode_seconds = None
        decoder_name = None
        if args.ivf:
            if not args.ivf.is_file():
                raise SystemExit(f"missing ivf: {args.ivf}")
            test_y4m = tmp / "decoded.y4m"
            avmdec = find_avmdec()
            decoder_name = Path(avmdec).name
            threads = (os.environ.get("AVMDEC_THREADS") or
                       os.environ.get("AVM_DEC_THREADS"))
            t0 = time.perf_counter()
            decode_avm(avmdec, args.ivf, args.ref, test_y4m, threads)
            decode_seconds = time.perf_counter() - t0
        else:
            test_y4m = args.test
            if not test_y4m.is_file():
                raise SystemExit(f"missing test: {test_y4m}")

        ref_n, test_n, n, fps = normalize_pair(args.ref, test_y4m, tmp)
        ref_chroma = y4m_geometry(y4m_header(args.ref))[2]
        test_chroma = y4m_geometry(y4m_header(test_y4m))[2]

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
        target_mem = host.get("gpu_memory_mib")
        target_cc = str(host.get("gpu_compute_cap", "")).strip()
        is_target = (
            target_mem == 12288 and target_cc == "8.6"
        ) or (
            isinstance(target_mem, int)
            and target_mem >= 30000
            and target_cc == "12.0"
        )

        report = {
            "ref": str(args.ref),
            "test": str(args.ivf or args.test),
            "frames": n,
            "fps": fps,
            "chroma": chroma_family(ref_chroma),
            "ref_chroma": ref_chroma,
            "test_chroma": test_chroma,
            "psnr": psnr,
            "ssim": ssim,
            "cvvdp_jod": jod,
            "cvvdp_version": ver,
            "cvvdp_display": os.environ.get("CVVDP_DISPLAY", "standard_fhd"),
            "cvvdp_device": default_device(),
            "host": host,
            "target_host": {
                "gpu_memory_mib": target_mem,
                "compute_cap": target_cc,
                "is_target": is_target,
            },
        }
        if args.ivf:
            size = args.ivf.stat().st_size
            num, den = (int(x) for x in fps.split("/"))
            duration = n * den / num
            report["ivf_bytes"] = size
            report["bitrate_kbps"] = size * 8 / duration / 1000

        if decode_seconds is not None:
            num, den = (int(x) for x in fps.split("/"))
            video_seconds = n * den / num
            report["decoder"] = decoder_name
            report["decode_threads"] = (os.environ.get("AVMDEC_THREADS") or
                                        os.environ.get("AVM_DEC_THREADS") or "0")
            report["decode_seconds"] = decode_seconds
            report["decode_fps"] = n / decode_seconds if decode_seconds > 0 else None
            report["decode_realtime_x"] = (
                video_seconds / decode_seconds if decode_seconds > 0 else None
            )

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
