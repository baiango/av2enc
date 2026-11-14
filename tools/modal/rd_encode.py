#!/usr/bin/env python3
"""Encode RD candidates on Modal, decode and score them on this Mac.

The CPU encoder is built in the Modal image with portable LTO (no
host-specific -march=native).  The requested QPs are encoded in one container,
the IVF bytes come back over the RPC result, and the local driver then runs
``tools/quality.py``, which uses the local ``dav2d-macos`` decoder and the
local CVVDP install.  No decoder is needed in the Modal image.

Examples
--------
Baseline at the 100 kbps point::

    python3 tools/modal/rd_encode.py --qps 189

Candidate preset (whatever the RD work chooses)::

    python3 tools/modal/rd_encode.py --qps 188,189,190 \
        --extra "--video-rd-preset quality"

A different clip (the 10 s 300-frame NASA file)::

    AV2ENC_MODAL_CLIP=/tmp/nasa_480x240_30fps_300f.y4m \
    python3 tools/modal/rd_encode.py --qps 189 --frames 300

The source snapshot is taken from the current working tree (build/testdata
directories are excluded), so an uncommitted RD change is encoded.  Set
``AV2ENC_MODAL_ROOT`` to a worktree to encode that worktree instead.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import modal

# The module is imported again inside the remote container, where __file__ is
# just the mounted app file.  Keep every local path/mount operation behind
# modal.is_local(); the remote copy only needs the function definition.
if modal.is_local():
    APP_DIR = Path(__file__).resolve().parent
    DEFAULT_ROOT = APP_DIR.parents[1]
    DEFAULT_CLIP = DEFAULT_ROOT / "testdata" / "nasa_480x240_30fps_60f.y4m"
    ROOT = Path(os.environ.get("AV2ENC_MODAL_ROOT", DEFAULT_ROOT)).resolve()
    CLIP = Path(os.environ.get("AV2ENC_MODAL_CLIP", DEFAULT_CLIP)).resolve()
    CPU_FAST = os.environ.get("AV2ENC_MODAL_CPU_FAST", "0") != "0"
    THREADS = int(os.environ.get("AV2ENC_MODAL_THREADS", "8"))
else:  # pragma: no cover - only executed inside the Modal runtime
    APP_DIR = DEFAULT_ROOT = ROOT = CLIP = Path("/")
    CPU_FAST = False
    THREADS = 8

# Keep the source snapshot small and stable.  The encoder build only needs the
# CMake files, src, tests and tools/still_verify; media and build products are
# mounted/added separately.
SKIP_DIRS = {
    ".git", ".venv", ".venv-cvvdp", "build", "bench", "testdata",
    "__pycache__", "docs", "artifacts", "out",
}


def _snapshot(root: Path) -> Path:
    """Copy the source tree without build outputs or media."""
    dst = Path(tempfile.mkdtemp(prefix="av2enc-modal-src-"))

    def ignore(directory: str, names: list[str]) -> set[str]:
        ignored: set[str] = set()
        for name in names:
            if name in SKIP_DIRS or name.startswith("build"):
                ignored.add(name)
            elif name.endswith((".ivf", ".y4m", ".o", ".a", ".so", ".dylib",
                                ".dSYM")):
                ignored.add(name)
            elif directory.endswith("tools") and name in ("bin", "modal"):
                # tools/modal is the driver itself, not encoder source.
                ignored.add(name)
        return ignored

    shutil.copytree(root, dst, ignore=ignore, dirs_exist_ok=True)
    return dst


if modal.is_local():
    SNAPSHOT = _snapshot(ROOT)
    _CODEGEN = "-DAV2ENC_CPU_FAST=ON" if CPU_FAST else "-DAV2ENC_LTO=ON"
    image = (
        modal.Image.debian_slim(python_version="3.13")
        .apt_install("build-essential", "cmake", "ninja-build")
        .add_local_dir(str(SNAPSHOT), "/repo", copy=True)
        .run_commands(
            "cd /repo && "
            "cmake -S . -B build -G Ninja "
            "-DCMAKE_BUILD_TYPE=Release "
            "-DAV2ENC_ENABLE_CUDA=OFF "
            "-DAV2ENC_ALLOW_CPU_ONLY=ON "
            f"{_CODEGEN} && "
            "cmake --build build --target av2enc -j"
        )
        .add_local_file(str(CLIP), "/data/clip.y4m")
    )
else:  # pragma: no cover - the image is sent with the function definition
    SNAPSHOT = Path("/")
    image = modal.Image.debian_slim(python_version="3.13")

app = modal.App("av2enc-rd")


@app.function(image=image, cpu=8, memory=4096, timeout=1800)
def encode(qps: list[int], frames: int, threads: int, extra: list[str]) -> list[dict]:
    out: list[dict] = []
    for qp in qps:
        path = f"/tmp/rd_q{qp}.ivf"
        cmd = [
            "/repo/build/av2enc", "/data/clip.y4m",
            "-o", path,
            "--video",
            "--frames", str(frames),
            "--qp", str(qp),
            "--cpu-threads", str(threads),
            *extra,
        ]
        start = time.monotonic()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.monotonic() - start
        if proc.returncode != 0:
            raise RuntimeError(
                f"encoder failed for QP {qp}: {' '.join(cmd)}\n"
                f"{proc.stderr[-4000:]}"
            )
        data = Path(path).read_bytes()
        out.append({
            "qp": qp,
            "ivf": data,
            "bytes": len(data),
            "encode_seconds": elapsed,
            "cmd": cmd,
            "stderr_tail": proc.stderr.strip().splitlines()[-2:],
        })
    return out


def _score(ref: Path, ivf: Path, qp: int, encode_seconds: float) -> dict:
    quality = Path(__file__).resolve().parents[1] / "quality.py"
    proc = subprocess.run(
        [sys.executable, str(quality), "--ref", str(ref), "--ivf", str(ivf),
         "--qp", str(qp), "--encode-seconds", str(encode_seconds)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-4000:])
    return json.loads(proc.stdout)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qps", default="189",
                    help="comma-separated QP list (default 189)")
    ap.add_argument("--frames", type=int,
                    default=int(os.environ.get("AV2ENC_MODAL_FRAMES", "60")),
                    help="frames to encode (default 60 or AV2ENC_MODAL_FRAMES)")
    ap.add_argument("--threads", type=int, default=THREADS,
                    help=f"CPU workers (default {THREADS})")
    ap.add_argument("--extra", default="",
                    help="extra encoder flags, shell-quoted")
    ap.add_argument("--out-dir", type=Path, default=Path("/tmp/av2enc-modal-rd"),
                    help="local directory for IVF + quality JSON")
    ap.add_argument("--no-score", action="store_true",
                    help="only encode; do not decode/score locally")
    args = ap.parse_args()

    qps = [int(x) for x in args.qps.split(",")]
    extra = shlex.split(args.extra)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = "candidate" if extra else "baseline"
    print(f"source: {ROOT}")
    print(f"clip:   {CLIP} ({args.frames} frames)")
    print(f"flags:  {' '.join(extra) or '(default)'}")
    print(f"qps:    {qps}")

    t0 = time.monotonic()
    with app.run():
        results = encode.remote(qps, args.frames, args.threads, extra)
    print(f"modal encode wall: {time.monotonic() - t0:.1f} s")

    rows = []
    for result in results:
        qp = result["qp"]
        ivf = args.out_dir / f"{tag}_q{qp}.ivf"
        ivf.write_bytes(result["ivf"])
        row = {
            "tag": tag, "qp": qp, "bytes": result["bytes"],
            "encode_seconds": result["encode_seconds"],
            "ivf": str(ivf),
        }
        if not args.no_score:
            report = _score(CLIP, ivf, qp, result["encode_seconds"])
            jpath = args.out_dir / f"{tag}_q{qp}.json"
            jpath.write_text(json.dumps(report, indent=2) + "\n")
            row.update({
                "kbps": report["bitrate_kbps"],
                "cvvdp_jod": report["cvvdp_jod"],
                "psnr_y": report["psnr"]["psnr_y"],
                "ssim_all": report["ssim"]["ssim_all"],
                "decoder": report.get("decoder"),
                "decode_seconds": report.get("decode_seconds"),
            })
        rows.append(row)

    summary = {
        "source": str(ROOT),
        "clip": str(CLIP),
        "frames": args.frames,
        "threads": args.threads,
        "extra": extra,
        "rows": rows,
    }
    (args.out_dir / f"{tag}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")

    if args.no_score:
        for row in rows:
            print(f"QP {row['qp']}: {row['bytes']} bytes, "
                  f"{row['encode_seconds']:.2f} s -> {row['ivf']}")
    else:
        print("\n| tag | QP | bytes | kbps | CVVDP | PSNR-Y | encode s |")
        print("|---|---:|---:|---:|---:|---:|---:|")
        for row in rows:
            print(f"| {tag} | {row['qp']} | {row['bytes']} | "
                  f"{row['kbps']:.1f} | {row['cvvdp_jod']:.3f} | "
                  f"{row['psnr_y']:.2f} | {row['encode_seconds']:.2f} |")
    print(f"\noutputs: {args.out_dir}")
    print(f"summary: {args.out_dir / f'{tag}_summary.json'}")


if __name__ == "__main__":
    main()
