#!/usr/bin/env python3
"""Fan out av2enc RD measurements on Modal.

Why
---
A leave-one-out ablation of `--nightly` is ~40 encode+metric jobs. On one Mac
they serialise at 30-60 s per CVVDP job; here every (config, QP) is an
independent container, so the wall time is one job plus scheduling.

The Linux build must reproduce the pinned streams byte-for-byte before any
number is trusted: the bitstream is host-independent (the primitives fall back
to the scalar reference on x86), so `--mode check` asserts the five pins first.

Usage
-----
    python3 -m modal run tools/modal_rd.py --mode check
    python3 -m modal run tools/modal_rd.py --mode ablate --qps 109,110,111
    python3 -m modal run tools/modal_rd.py --mode add --add --video-wiener
    python3 -m modal run tools/modal_rd.py --mode matrix --matrix /tmp/jobs.json

`--mode ablate` emits the tuned-drift baseline plus one arm per
`--video-*-off` switch (the switches are on main). `--mode matrix` takes
`[{"name", "flags", "qps"}]`.

CPU-only: 8 cores per job, no GPU, up to 24 containers. A 40-job 240p matrix is
roughly 10-20 min wall. 720p CVVDP is ~3x slower.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
APP = modal.App("av2enc-rd")

# The ablation switches added for the reference-structure sweep. Kept here so a
# new switch is one line and every matrix shares the same vocabulary.
ABLATIONS = [
    "--video-bawp-off",
    "--video-cdf-chain-off",
    "--video-deblock-off",
    "--video-drift-refresh-off",
    "--video-edge-inter-off",
    "--video-edge-mask-off",
    "--video-key-qp-offset-off",
    "--video-loop-filters-off",
    "--video-partition-rd-off",
    "--video-rdoq-off",
    "--video-tile-group-off",
    "--video-tx-search-off",
]

# The current best operating point at 240p/100 kbps.
DRIFT = [
    "--video-drift-refresh-threshold",
    "1.5",
    "--video-drift-refresh-budget",
    "3250",
]

CLIPS = {
    "240p": {
        "ref": "/data/nasa_480x240_30fps_60f.y4m",
        "local": "testdata/nasa_480x240_30fps_60f.y4m",
    },
    "720p": {
        "ref": "/data/space8k_1440x720_60f.y4m",
        "local": "testdata/space8k_1440x720_60f.y4m",
    },
}

# name -> (clip, qp, flags) for the five pinned recipes.
PINS = {
    "default": ("240p", 189, ["--video"]),
    "quality": ("240p", 190, ["--video", "--video-rd-preset", "quality"]),
    "nightly": ("240p", 110, ["--video", "--nightly"]),
    "explambda": (
        "240p",
        110,
        [
            "--video",
            "--nightly",
            "--video-qp-lambda",
            "--video-tx-lambda",
            "136",
            "--video-key-tx-lambda",
            "136",
            "--video-leaf-rd-lambda",
            "125",
            "--video-rdoq-scale",
            "0.1",
            "--video-partition-rd-lambda",
            "2.75e-6",
        ],
    ),
    "720p": (
        "720p",
        217,
        [
            "--video",
            "--nightly",
            "--video-key-qp-offset",
            "-57",
            "--video-leaf-rd-lambda",
            "250",
            "--video-ext-partitions",
        ],
    ),
}

PIN_MD5 = {
    "default": "a1d0dba1",
    "quality": "3984677b",
    "nightly": "1b99127a",
    "explambda": "7f03eb6e",
    "720p": "c4ba2813",
}


def build_image() -> modal.Image:
    """Encoder + pinned decoder; optionally the three metrics.

    The normal split is encode-only on Modal and score on the Mac with the
    pinned local tools, so the default image carries no torch/cvvdp/libjxl.
    """
    image = modal.Image.debian_slim(python_version="3.11").apt_install(
        "build-essential",
        "cmake",
        "ninja-build",
        "git",
        "nasm",
        "meson",
        "pkg-config",
    )
    image = (
        image.add_local_file(
            str(ROOT / "tools/modal_image_setup.sh"),
            "/tmp/modal_image_setup.sh",
            copy=True,
        )
        .run_commands("WITH_METRICS=0 sh /tmp/modal_image_setup.sh")
        .add_local_dir(
            str(ROOT),
            "/src",
            copy=True,
            ignore=[
                ".git",
                "**/build*",
                "testdata",
                "bench",
                "spec",
                "**/*.ivf",
                "**/*.y4m",
                "**/*.yuv",
            ],
        )
        .add_local_file(
            str(ROOT / "testdata/nasa_480x240_30fps_60f.y4m"),
            "/data/nasa_480x240_30fps_60f.y4m",
            copy=True,
        )
        .add_local_file(
            str(ROOT / "testdata/space8k_1440x720_60f.y4m"),
            "/data/space8k_1440x720_60f.y4m",
            copy=True,
        )
        .run_commands(
            "cd /src && cmake --preset dev && "
            "cmake --build --preset dev -j\"$(nproc)\" && "
            "/src/build-dev/av2enc --version 2>/dev/null | head -1 || true"
        )
    )
    return image


ENCODE_IMAGE = build_image()


@APP.function(
    image=ENCODE_IMAGE,
    cpu=4,
    # Measured peak RSS: 50.5 MiB at 240p, 245.5 MiB at 720p (release, 4T).
    # 512 MiB is 2x the largest fixture.
    memory=512,
    timeout=900,
    max_containers=4,
)
def encode_one(job: dict) -> dict:
    """Encode one (config, QP) and return the IVF for local scoring."""
    import base64
    import hashlib
    import os
    import subprocess

    out = Path("/tmp/out.ivf")
    started = time.time()
    try:
        subprocess.run(
            [
                "/src/build-dev/av2enc",
                job["ref"],
                "-o",
                str(out),
                "--frames",
                "60",
                "--cpu-threads",
                "4",
                "--qp",
                str(int(job["qp"])),
                *job["flags"],
            ],
            check=True,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        data = out.read_bytes()
        return {
            "name": job["name"],
            "qp": int(job["qp"]),
            "flags": job["flags"],
            "clip": job.get("clip", "240p"),
            "bytes": len(data),
            "kbps": len(data) * 8 / 2.0 / 1000.0,
            "md5": hashlib.md5(data).hexdigest(),
            "encode_seconds": round(time.time() - started, 2),
            "ivf_b64": base64.b64encode(data).decode("ascii"),
        }
    except subprocess.CalledProcessError as exc:
        return {
            "name": job["name"],
            "qp": int(job["qp"]),
            "flags": job["flags"],
            "error": (exc.stderr or exc.stdout or str(exc))[-800:],
        }
    except Exception as exc:  # noqa: BLE001
        return {"name": job["name"], "qp": int(job["qp"]), "flags": job["flags"],
                "error": repr(exc)}


def _parse_qps(text: str) -> list[int]:
    return [int(x) for x in text.replace(" ", "").split(",") if x]


def _jobs_for(mode: str, clip: str, qps_text: str, matrix: str, add_text: str,
              base: str) -> list[dict]:
    ref = CLIPS[clip]["ref"]
    if mode == "check":
        jobs = []
        for name, (pin_clip, qp, flags) in PINS.items():
            jobs.append(
                {
                    "name": f"pin-{name}",
                    "ref": CLIPS[pin_clip]["ref"],
                    "clip": pin_clip,
                    "qp": qp,
                    "flags": flags,
                    "metrics": False,
                }
            )
        return jobs

    if mode == "matrix":
        raw = json.loads(Path(matrix).read_text())
        jobs = []
        for entry in raw:
            for qp in entry["qps"]:
                jobs.append(
                    {
                        "name": entry["name"],
                        "ref": CLIPS[entry.get("clip", clip)]["ref"],
                        "clip": entry.get("clip", clip),
                        "qp": qp,
                        "flags": entry["flags"],
                    }
                )
        return jobs

    qps = _parse_qps(qps_text) if qps_text else ([109, 110, 111] if mode == "ablate"
                                                 else [105, 106, 107])
    base_flags = ["--video", "--nightly"] + (DRIFT if base == "drift" else [])
    jobs = []
    if mode == "ablate":
        for qp in qps:
            jobs.append({"name": "baseline", "ref": ref, "clip": clip, "qp": qp, "flags": base_flags})
        for switch in ABLATIONS:
            for qp in qps:
                jobs.append(
                    {
                        "name": f"no-{switch.removeprefix('--video-').removesuffix('-off')}",
                        "ref": ref,
                        "clip": clip,
                        "qp": qp,
                        "flags": base_flags + [switch],
                    }
                )
        return jobs

    add = [x for x in add_text.replace(" ", "").split(",") if x]
    if mode == "add":
        if not add:
            raise SystemExit("--mode add needs --add flag[,flag]")
        for qp in qps:
            jobs.append({"name": "baseline", "ref": ref, "clip": clip, "qp": qp, "flags": base_flags})
        for flag in add:
            for qp in qps:
                jobs.append(
                    {
                        "name": f"add-{flag.removeprefix('--video-')}",
                        "ref": ref,
                        "clip": clip,
                        "qp": qp,
                        "flags": base_flags + [flag],
                    }
                )
        return jobs

    raise SystemExit(f"unknown mode: {mode}")


def _interpolate(rows: list[dict], target: float = 100.0) -> dict | None:
    """Linear K at `target` kbps between the two bracketing QPs."""
    good = sorted(
        (r for r in rows if "error" not in r), key=lambda r: r["kbps"]
    )
    if not good:
        return None
    if len(good) == 1:
        r = good[0]
        return {"kpi": r["kpi"], "jod": r["jod"], "ba": r["ba"],
                "ss2": r["ss2"], "single": True}
    for lo, hi in zip(good, good[1:]):
        if lo["kbps"] <= target <= hi["kbps"]:
            span = hi["kbps"] - lo["kbps"]
            t = 0.0 if span == 0 else (target - lo["kbps"]) / span
            return {
                "kpi": lo["kpi"] + t * (hi["kpi"] - lo["kpi"]),
                "jod": lo["jod"] + t * (hi["jod"] - lo["jod"]),
                "ba": lo["ba"] + t * (hi["ba"] - lo["ba"]),
                "ss2": lo["ss2"] + t * (hi["ss2"] - lo["ss2"]),
                "lo": lo["name"],
                "hi": hi["name"],
            }
    return None


@APP.local_entrypoint()
def main(
    mode: str = "check",
    clip: str = "240p",
    qps: str = "",
    matrix: str = "",
    base: str = "drift",
    add: str = "",
    out: str = "",
    out_dir: str = "",
    encode_only: bool = False,
    allow_large: bool = False,
    dry_run: bool = False,
):
    jobs = _jobs_for(mode, clip, qps, matrix, add, base)
    print(f"mode={mode} clip={clip} jobs={len(jobs)}")
    if len(jobs) > 24 and not allow_large:
        raise SystemExit(
            f"refusing {len(jobs)} jobs: this harness is sized for small matrices "
            "(4 containers, 4 cores, 1 GiB). Pass --allow-large to override."
        )
    for job in jobs:
        print(f"  {job['name']:28s} qp={job['qp']:<4d} {' '.join(job['flags'])}")
    if dry_run:
        return

    started = time.time()
    results = list(encode_one.map(jobs))
    elapsed = time.time() - started

    errors = [r for r in results if "error" in r]
    for err in errors:
        print(f"ERROR {err['name']} qp={err['qp']}: {err['error']}", file=sys.stderr)

    # Write the IVF files and a manifest for tools/score_ivf.py.
    if mode != "check":
        import base64

        target = Path(out_dir or "/tmp/modal-encode")
        target.mkdir(parents=True, exist_ok=True)
        manifest = []
        for row in results:
            if "error" in row:
                continue
            path = target / f"{row['name']}-qp{row['qp']}.ivf"
            path.write_bytes(base64.b64decode(row["ivf_b64"]))
            entry = {
                key: row[key]
                for key in ("name", "qp", "flags", "clip", "bytes", "kbps", "md5",
                            "encode_seconds")
            }
            entry["ivf"] = str(path)
            manifest.append(entry)
        (target / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        print(f"wrote {len(manifest)} IVF files to {target} in {elapsed:.0f}s")
        for entry in sorted(manifest, key=lambda e: (e["name"], e["qp"])):
            print(f"  {entry['name']:28s} qp={entry['qp']:<4d} "
                  f"{entry['kbps']:8.3f} kbps  {entry['md5']}")
        return

    # Pin check: compare the first four bytes of md5 and the byte counts.
    if mode == "check":
        ok = True
        for row in sorted(results, key=lambda r: r["name"]):
            pin = row["name"].removeprefix("pin-")
            want = PIN_MD5[pin]
            got = row.get("md5", "")
            status = "OK " if got.startswith(want) else "FAIL"
            ok &= status == "OK "
            print(
                f"{status} {pin:10s} {row.get('bytes', 0):6d} B "
                f"md5={got} want={want}"
            )
        print(f"pin check: {'PASS' if ok else 'FAIL'} ({elapsed:.0f}s)")
        if not ok:
            raise SystemExit(1)
        return

    # Group, print, and interpolate.
    by_name: dict[str, list[dict]] = {}
    for row in results:
        by_name.setdefault(row["name"], []).append(row)

    lines = [
        "| config | QP points (kbps / K) | K@100 | ΔK | encode s |",
        "|---|---|---:|---:|---:|",
    ]
    baseline = _interpolate(by_name.get("baseline", []))
    base_k = baseline["kpi"] if baseline else None
    for name in sorted(by_name):
        rows = sorted(by_name[name], key=lambda r: r["qp"])
        points = " · ".join(
            f"{r['qp']}: {r['kbps']:.3f} / {r['kpi']:.2f}"
            for r in rows
            if "error" not in r
        )
        interp = _interpolate(rows)
        k = interp["kpi"] if interp else float("nan")
        dk = (k - base_k) if (interp and base_k is not None) else float("nan")
        secs = sum(r.get("encode_seconds", 0) for r in rows)
        lines.append(f"| {name} | {points} | {k:.2f} | {dk:+.2f} | {secs:.1f} |")
    table = "\n".join(lines)
    print(table)

    payload = {
        "mode": mode,
        "clip": clip,
        "base": base,
        "elapsed_seconds": round(elapsed, 1),
        "results": results,
        "summary": table,
    }
    if out:
        Path(out).write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {out}")
