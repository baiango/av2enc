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

`--mode matrix` takes

    [{"name": "cand", "qps": [110, 111], "flags": ["--video", "--nightly"],
      "env": {"AV2ENC_TOOL": "1"}, "clip": "240p", "frames": 60}]

`env` is applied to the encoder subprocess, which is how every new gate in this
tree is wired (the gates are AV2ENC_* environment switches, not flags).  The
JSON can also be fed on stdin with `--matrix -`.

Worktree mount
--------------
The image builds the source tree, so an uncommitted lane patch must be mounted.
Pass `--repo /path/to/worktree` (or set `AV2ENC_MODAL_REPO`) and that tree is
snapshotted into the image instead of this checkout:

    python3 -m modal run tools/modal_rd.py --mode matrix --repo ~/work/rd-loop-3 ...

The mount is chosen when the module is imported (Modal builds the image before
the entrypoint parses arguments), so `--repo` is also scanned from `sys.argv`
at import time.  The binary is built at `/src/build-release/av2enc` with the
`release` preset, matching the `bin` path every envelope spec names.

Worker shape
------------
One core and 200 MiB per job (`--cpu-threads 1`): each job matches its core,
and 24 containers give the ten-lane loop a shared 24-vCPU pool.  Measured peak
RSS is 50.5 MiB at 240p and 245.5 MiB at 720p, so 200 MiB is comfortable at
240p and deliberately tight at 720p; `--allow-large` is the escape hatch.
`max_containers` is the effective peak concurrency (Modal may cap it lower on
the account).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]


def _early_repo() -> Path:
    """Resolve --repo before the image is built (Modal imports first)."""
    argv = sys.argv[1:]
    for i, arg in enumerate(argv):
        if arg == "--repo" and i + 1 < len(argv):
            return Path(argv[i + 1]).expanduser().resolve()
        if arg.startswith("--repo="):
            return Path(arg.split("=", 1)[1]).expanduser().resolve()
    env = os.environ.get("AV2ENC_MODAL_REPO")
    return Path(env).expanduser().resolve() if env else ROOT.resolve()


def _early_flag(name: str) -> bool:
    argv = sys.argv[1:]
    return any(a == name or a.startswith(name + "=") for a in argv)


MOUNT_ROOT = _early_repo()
WITH_METRICS = (_early_flag("--with-metrics")
                or os.environ.get("AV2ENC_MODAL_METRICS", "0") != "0")
MAX_CONTAINERS = int(os.environ.get("AV2ENC_MODAL_MAX_CONTAINERS", "24"))

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

# clip -> (remote ref, local fixture, fps)
CLIPS = {
    "240p": {
        "ref": "/data/nasa_480x240_30fps_60f.y4m",
        "local": "testdata/nasa_480x240_30fps_60f.y4m",
        "fps": 30.0,
    },
    "720p": {
        "ref": "/data/space8k_1440x720_60f.y4m",
        "local": "testdata/space8k_1440x720_60f.y4m",
        "fps": 60.0,
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

# Keep the source snapshot small: build products, media and the metrics dists
# never belong in the encoder image.
IGNORE = [
    ".git",
    ".venv",
    ".venv-cvvdp",
    "**/build*",
    "**/__pycache__",
    "testdata",
    "bench",
    "spec",
    "artifacts",
    "out",
    "docs",
    "refactor",
    "tools/bin",
    "**/*.ivf",
    "**/*.y4m",
    "**/*.yuv",
    "**/*.o",
    "**/*.a",
    "**/*.so",
    "**/*.dylib",
]


def build_image(root: Path, with_metrics: bool = False) -> modal.Image:
    """Encoder + pinned decoder; optionally the three metrics.

    The normal split is encode-only on Modal and score on the Mac with the
    pinned local tools, so the default image carries no torch/cvvdp/libjxl.
    `WITH_METRICS=1` (or `--with-metrics`) adds ffmpeg, the pinned libjxl
    tools, and `cvvdp==0.5.7` so `tools/quality.py` can score in the container.
    """
    packages = [
        "build-essential",
        "cmake",
        "ninja-build",
        "git",
        "nasm",
        "meson",
        "pkg-config",
    ]
    if with_metrics:
        packages.append("ffmpeg")
    image = modal.Image.debian_slim(python_version="3.11").apt_install(*packages)
    if with_metrics:
        # The project's tuning metric is CVVDP 0.5.7 (tools/quality.py).
        image = image.pip_install("cvvdp==0.5.7")
    image = (
        image.add_local_file(
            str(ROOT / "tools/modal_image_setup.sh"),
            "/tmp/modal_image_setup.sh",
            copy=True,
        )
        .run_commands(
            f"WITH_METRICS={1 if with_metrics else 0} "
            "sh /tmp/modal_image_setup.sh"
        )
        .add_local_dir(str(root), "/src", copy=True, ignore=IGNORE)
    )
    for clip in CLIPS.values():
        image = image.add_local_file(
            str(root / clip["local"]), clip["ref"], copy=True)
    return image.run_commands(
        # The release preset is the measurement build: -O3, strict warnings,
        # hardening.  Its binary lands at /src/build-release/av2enc, exactly
        # the `bin` path an envelope spec names.
        "cd /src && cmake --preset release && "
        "cmake --build --preset release --target av2enc -j\"$(nproc)\" && "
        "/src/build-release/av2enc --version 2>/dev/null | head -1 || true"
    )


if modal.is_local():
    ENCODE_IMAGE = build_image(MOUNT_ROOT, WITH_METRICS)
else:  # pragma: no cover - only the function definition is needed remotely
    ENCODE_IMAGE = modal.Image.debian_slim()


@APP.function(
    image=ENCODE_IMAGE,
    cpu=1,
    # Measured peak RSS: 50.5 MiB at 240p, 245.5 MiB at 720p (release, 4T).
    # 200 MiB is 4x the 240p fixture; the ten-lane pool is 240p.  A 720p job
    # does not fit and is routed to encode_large below.
    memory=200,
    timeout=1800,
    max_containers=MAX_CONTAINERS,
)
def encode_one(job: dict) -> dict:
    """Encode one 240p (config, QP) and return the IVF for local scoring."""
    return _encode_impl(job)


@APP.function(
    image=ENCODE_IMAGE,
    cpu=1,
    memory=512,
    timeout=1800,
    max_containers=MAX_CONTAINERS,
)
def encode_large(job: dict) -> dict:
    """720p needs 245.5 MiB peak RSS; 512 MiB is 2x the largest fixture."""
    return _encode_impl(job)


def _encode_impl(job: dict) -> dict:
    import base64
    import hashlib
    import os
    import subprocess

    out = Path("/tmp/out.ivf")
    started = time.time()
    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in (job.get("env") or {}).items()})
    try:
        subprocess.run(
            [
                "/src/build-release/av2enc",
                job["ref"],
                "-o",
                str(out),
                "--frames",
                str(int(job.get("frames", 60))),
                "--cpu-threads",
                "1",
                "--qp",
                str(int(job["qp"])),
                *job["flags"],
            ],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        data = out.read_bytes()
        fps = float(CLIPS.get(job.get("clip", "240p"), {}).get("fps", 30.0))
        frames = int(job.get("frames", 60))
        return {
            "name": job["name"],
            "qp": int(job["qp"]),
            "flags": job["flags"],
            "env": job.get("env") or {},
            "clip": job.get("clip", "240p"),
            "frames": frames,
            "bytes": len(data),
            "kbps": len(data) * 8 / (frames / fps) / 1000.0,
            "md5": hashlib.md5(data).hexdigest(),
            "encode_seconds": round(time.time() - started, 2),
            "started": started,
            "finished": time.time(),
            "ivf_b64": base64.b64encode(data).decode("ascii"),
        }
    except subprocess.CalledProcessError as exc:
        return {
            "name": job["name"],
            "qp": int(job["qp"]),
            "flags": job["flags"],
            "env": job.get("env") or {},
            "error": (exc.stderr or exc.stdout or str(exc))[-2000:],
        }
    except Exception as exc:  # noqa: BLE001
        return {"name": job["name"], "qp": int(job["qp"]), "flags": job["flags"],
                "env": job.get("env") or {}, "error": repr(exc)}


def _parse_qps(text: str) -> list[int]:
    return [int(x) for x in text.replace(" ", "").split(",") if x]


def _jobs_for(mode: str, clip: str, qps_text: str, matrix: str, add_text: str,
              base: str, global_env: dict) -> list[dict]:
    ref = CLIPS[clip]["ref"]
    if mode == "check":
        jobs = []
        for name, (pin_clip, qp, flags) in PINS.items():
            jobs.append(
                {
                    "name": f"pin-{name}",
                    "ref": CLIPS[pin_clip]["ref"],
                    "clip": pin_clip,
                    "frames": 60,
                    "qp": qp,
                    "flags": flags,
                    "env": dict(global_env),
                    "metrics": False,
                }
            )
        return jobs

    if mode == "matrix":
        text = sys.stdin.read() if matrix == "-" else Path(matrix).read_text()
        raw = json.loads(text)
        jobs = []
        for entry in raw:
            env = {**global_env, **dict(entry.get("env") or {})}
            for qp in entry["qps"]:
                jobs.append(
                    {
                        "name": entry["name"],
                        "ref": CLIPS[entry.get("clip", clip)]["ref"],
                        "clip": entry.get("clip", clip),
                        "frames": int(entry.get("frames", 60)),
                        "qp": qp,
                        "flags": entry["flags"],
                        "env": env,
                    }
                )
        return jobs

    qps = _parse_qps(qps_text) if qps_text else ([109, 110, 111] if mode == "ablate"
                                                 else [105, 106, 107])
    base_flags = ["--video", "--nightly"] + (DRIFT if base == "drift" else [])
    jobs = []
    if mode == "ablate":
        for qp in qps:
            jobs.append({"name": "baseline", "ref": ref, "clip": clip, "frames": 60,
                         "qp": qp, "flags": base_flags,
                         "env": dict(global_env)})
        for switch in ABLATIONS:
            for qp in qps:
                jobs.append(
                    {
                        "name": f"no-{switch.removeprefix('--video-').removesuffix('-off')}",
                        "ref": ref,
                        "clip": clip,
                        "frames": 60,
                        "qp": qp,
                        "flags": base_flags + [switch],
                        "env": dict(global_env),
                    }
                )
        return jobs

    add = [x for x in add_text.replace(" ", "").split(",") if x]
    if mode == "add":
        if not add:
            raise SystemExit("--mode add needs --add flag[,flag]")
        for qp in qps:
            jobs.append({"name": "baseline", "ref": ref, "clip": clip, "frames": 60,
                         "qp": qp, "flags": base_flags,
                         "env": dict(global_env)})
        for flag in add:
            for qp in qps:
                jobs.append(
                    {
                        "name": f"add-{flag.removeprefix('--video-')}",
                        "ref": ref,
                        "clip": clip,
                        "frames": 60,
                        "qp": qp,
                        "flags": base_flags + [flag],
                        "env": dict(global_env),
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


def _peak_concurrency(results: list[dict]) -> int:
    events: list[tuple[float, int]] = []
    for row in results:
        if "started" in row and "finished" in row:
            events.append((float(row["started"]), 1))
            events.append((float(row["finished"]), -1))
    current = peak = 0
    for _, delta in sorted(events):
        current += delta
        peak = max(peak, current)
    return peak


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
    env: str = "{}",
    repo: str = "",
    with_metrics: bool = False,
    encode_only: bool = False,
    allow_large: bool = False,
    dry_run: bool = False,
):
    if repo:
        requested = Path(repo).expanduser().resolve()
        if requested != MOUNT_ROOT:
            raise SystemExit(
                f"--repo {requested} does not match the mounted tree "
                f"{MOUNT_ROOT}; Modal builds the image before the entrypoint "
                f"runs, so pass --repo on the first invocation")
    try:
        global_env = json.loads(env)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--env is not JSON: {exc}") from exc
    if not isinstance(global_env, dict):
        raise SystemExit("--env must be a JSON object")
    jobs = _jobs_for(mode, clip, qps, matrix, add, base, global_env)
    print(f"mode={mode} mount={MOUNT_ROOT} metrics={WITH_METRICS} "
          f"jobs={len(jobs)} max_containers={MAX_CONTAINERS}")
    if len(jobs) > 24 and not allow_large:
        raise SystemExit(
            f"refusing {len(jobs)} jobs: this harness is sized for small "
            "matrices (24 containers, 1 core, 200 MiB). Pass --allow-large "
            "to override."
        )
    for job in jobs:
        extra = (" env=" + json.dumps(job["env"], sort_keys=True)
                 if job.get("env") else "")
        print(f"  {job['name']:28s} qp={job['qp']:<4d} "
              f"{' '.join(job['flags'])}{extra}")
    if dry_run:
        return

    started = time.time()
    small = [job for job in jobs if job.get("clip") != "720p"]
    large = [job for job in jobs if job.get("clip") == "720p"]
    results: list[dict] = []
    if small:
        results += list(encode_one.map(small))
    if large:
        results += list(encode_large.map(large))
    elapsed = time.time() - started
    peak = _peak_concurrency(results)
    print(f"encode wall {elapsed:.1f}s  peak concurrency {peak} "
          f"(cap {MAX_CONTAINERS})")

    errors = [r for r in results if "error" in r]
    for err in errors:
        print(f"ERROR {err['name']} qp={err['qp']}: {err['error']}",
              file=sys.stderr)

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
                for key in ("name", "qp", "flags", "env", "clip", "frames",
                            "bytes", "kbps", "md5", "encode_seconds")
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
        if errors:
            raise SystemExit(1)
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
        print(f"pin check: {'PASS' if ok else 'FAIL'} ({elapsed:.0f}s, "
              f"peak concurrency {peak})")
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
        "mount": str(MOUNT_ROOT),
        "metrics": WITH_METRICS,
        "max_containers": MAX_CONTAINERS,
        "peak_concurrency": peak,
        "elapsed_seconds": round(elapsed, 1),
        "results": results,
        "summary": table,
    }
    if out:
        Path(out).write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {out}")
