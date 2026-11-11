#!/usr/bin/env python3
"""M3 quality / BD-rate harness for the av2enc still-picture path.

What this does
--------------
1. Extracts exactly one frame (frame 0 by default) from the benchmark clip at
   640x360 and at native 1920x1080, and hashes the canonical Y4M fixture.
2. Probes the encoder's QP support.  The current M1 writer pins
   ``base_q_idx = 90`` until the coefficient-CDF qctx work lands; the probe
   records the rejection instead of pretending a curve exists.
3. Encodes the M1 baseline and every M3 tool that is actually wired into the
   CLI, one tool at a time, at every supported QP.
4. Runs the ``dav2d --strict 1`` decoder gate *and* the repository
   ``tools/validate_dav2d.sh`` gate on every stream before scoring it.
5. Scores every verified stream with ``tools/quality.py`` (PSNR, SSIM, CVVDP).
6. Enforces same-source / same-frame / same-host / same-decoder / same-CVVDP
   comparisons and refuses to call a mixed run a valid comparison.
7. Writes per-point JSON, a markdown report, an R-D plot, and a BD-rate report.

The harness never edits encoder code, ``main.cpp`` or ``CMakeLists.txt`` and
never claims a result for a tool without a decoded stream.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bd_rate import RDPoint, bd_psnr, bd_rate  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = ROOT / "tools"
QUALITY_PY = TOOLS_DIR / "quality.py"
VALIDATE_SH = TOOLS_DIR / "validate_dav2d.sh"
DEFAULT_ENCODER = ROOT / "build-m1" / "av2enc"
DEFAULT_SOURCE_360 = ROOT / "bench" / "avm_360p" / "shine_360p_2s.y4m"
DEFAULT_SOURCE_1080 = ROOT / "bench" / "avm_1080p" / "shine_1080p_2s.y4m"
DEFAULT_OUT = ROOT / "tools" / "m3_quality" / "results"
SCHEMA = "av2enc.m3_quality.v1"

# Verdict thresholds.  These are deliberately conservative and local to this
# report; the confidence is low because every point is one frame and (today)
# one QP.  A trade of rate against quality is never called a win without a
# multi-QP curve.
RATE_FLAT_PCT = 0.5
Q_UP_CVVDP_JOD = 0.02
Q_UP_PSNR_DB = 0.05
Q_DOWN_PSNR_DB = 0.05
Q_DOWN_CVVDP_JOD = 0.02
Q_DOWN_SSIM = 0.001
Q_UP_SSIM_TOL = 0.0005
MAX_WIN_RATE_COST_PCT = 1.0
HEURISTIC_DB_PER_PCT_RATE = 0.05  # diagnostic only, never a verdict

# BD-rate verdict thresholds and noise floors.  A BD-rate within +/-1% is
# called neutral.  A metric whose per-QP deltas never exceed its noise floor is
# not allowed to flip the verdict: near CVVDP saturation a 0.002 JOD delta is
# not a 2% rate difference, it is measurement noise.
BD_NEUTRAL_PCT = 1.0
CVVDP_NOISE_FLOOR_JOD = 0.02
PSNR_NOISE_FLOOR_DB = 0.02
CORE_BD_QPS = (90, 110, 135, 160, 185, 210)


@dataclass(frozen=True)
class ResolutionSpec:
    label: str
    width: int
    height: int
    default_source: Path
    provenance: str
    proxy_for: str | None = None


@dataclass(frozen=True)
class ToolSpec:
    id: str
    display: str
    kind: str  # baseline | control | m3
    flags: tuple[str, ...]
    candidate_flags: tuple[str, ...] = ()
    scope: str = ""
    not_wired_reason: str = ""

    @property
    def required_flag(self) -> str | None:
        if self.kind in ("baseline", "control"):
            return self.candidate_flags[0] if self.candidate_flags else "--m1"
        return self.candidate_flags[0] if len(self.candidate_flags) == 1 else None


RESOLUTIONS = {
    "640x360": ResolutionSpec(
        "640x360",
        640,
        360,
        DEFAULT_SOURCE_360,
        "downscaled_from_4k_source_at_10s",
        proxy_for="1920x1080",
    ),
    "1920x1080": ResolutionSpec(
        "1920x1080",
        1920,
        1080,
        DEFAULT_SOURCE_1080,
        "native_1920x1080_from_4k_source_at_10s",
        proxy_for=None,
    ),
}

TOOL_SPECS = (
    ToolSpec(
        "m1_baseline",
        "M1 baseline",
        "baseline",
        ("--m1",),
        scope="M1 still picture: partition tree, full intra search off (DC/CfL), "
        "variable transforms, deblock; no M3 filter/RDO tool",
    ),
    ToolSpec(
        "m1_full_intra",
        "M1 full intra (control)",
        "control",
        ("--m1", "--m1-full-intra"),
        ("--m1-full-intra",),
        scope="Control for RDO: full directional/smooth intra search without the "
        "rate term. Not an M3 tool; it is the anchor that isolates RDO.",
    ),
    ToolSpec(
        "cdef",
        "CDEF",
        "m3",
        ("--m1", "--m1-cdef", "--m1-cdef-strengths", "4"),
        ("--m1-cdef",),
        scope="Constrained directional enhancement filter, post-deblock, 4 strength "
        "sets; filtered planes replace the reconstruction.",
    ),
    ToolSpec(
        "ccso",
        "CCSO",
        "m3",
        ("--m1", "--m1-ccso", "--m1-ccso-bands", "6"),
        ("--m1-ccso",),
        scope="Band-offset-only cross-component sample offset, post-CDEF, 64 bands. "
        "Edge-classified CCSO and per-unit ccso_blk=0 are not implemented.",
    ),
    ToolSpec(
        "rdoq",
        "RDOQ",
        "m3",
        ("--m1", "--m1-rdoq"),
        ("--m1-rdoq",),
        scope="Full-trellis coefficient RDOQ at SSE-scale lambda 0.125; syntax is "
        "unchanged, only quantised levels change.",
    ),
    ToolSpec(
        "tx_type",
        "TX type",
        "m3",
        ("--m1", "--m1-tx-type"),
        ("--m1-tx-type",),
        scope="Intra luma transform-type search at SSE-scale lambda 0.125; chroma "
        "stays DCT_DCT and identity transforms remain disabled.",
    ),
    ToolSpec(
        "rdo_intra",
        "RDO (intra)",
        "m3",
        ("--m1", "--m1-full-intra", "--m1-rdo-intra"),
        ("--m1-rdo-intra",),
        scope="Rate-aware luma intra-mode decision over the existing candidate "
        "search. This is intra RDO, not partition/inter Lagrangian RDO.",
    ),
    ToolSpec(
        "wiener",
        "Wiener NS",
        "m3",
        (),
        ("--m1-wiener", "--m1-wiener-ns", "--wiener"),
        scope="Loop restoration / non-separable Wiener. Component and syntax exist "
        "but there is no M1 bitstream path or encoder-side coefficient fit.",
        not_wired_reason=(
            "No M1/M3 bitstream path. docs/m1-features.md records Wiener NS as "
            "'not integrated'; the restoration component has no reference "
            "implementation of read_wienerns_filter()/search_frame_filters()."
        ),
    ),
    ToolSpec(
        "delta_q",
        "Delta-Q",
        "m3",
        (),
        ("--m1-delta-q", "--m1-delta-q-res", "--delta-q"),
        scope="Frame/tile/per-block quantizer delta. Planning code exists, but the "
        "M1 writer emits delta_q_present = 0 and the CLI does not expose it.",
        not_wired_reason=(
            "No M1 stream. src/encoder/m1/m1_headers.cpp writes "
            "delta_q_present = 0; src/encoder/pipeline/README.md states the M1 "
            "writer emits one tile with delta_q_present = 0."
        ),
    ),
    ToolSpec(
        "qm",
        "QM",
        "m3",
        (),
        ("--m1-qm", "--qm", "--quant-matrix"),
        scope="Quantization matrices (QM OBU + setup). No encoder-side QM selection "
        "or OBU path is wired into the still-picture CLI.",
        not_wired_reason=(
            "No M1 stream. src/encoder/m2/post/tiled_frame.cpp writes "
            "using_qmatrix = 0, and no QM OBU/setup path exists for the still path."
        ),
    ),
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def run(
    cmd: list[str | Path],
    *,
    cwd: Path = ROOT,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(c) for c in cmd],
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def tail(text: str, limit: int = 2500) -> str:
    text = text.strip()
    return text if len(text) <= limit else "..." + text[-limit:]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def parse_float(text: str | None) -> float | None:
    if text is None:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def load_quality_module():
    spec = importlib.util.spec_from_file_location("av2enc_quality", QUALITY_PY)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {QUALITY_PY}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Y4M handling
# ---------------------------------------------------------------------------


@dataclass
class Y4mInfo:
    header: str
    width: int
    height: int
    chroma: str
    fps_num: int
    fps_den: int
    frame_bytes: int
    frames: int


def y4m_geometry(header: str) -> tuple[int, int, str, int, int]:
    m = re.search(r"W(\d+)\s+H(\d+)", header)
    if not m:
        raise SystemExit(f"bad Y4M header: {header}")
    w, h = int(m.group(1)), int(m.group(2))
    fm = re.search(r"F(\d+):(\d+)", header)
    if not fm:
        raise SystemExit(f"bad Y4M frame rate: {header}")
    cm = re.search(r"C([0-9A-Za-z]+)", header)
    chroma = cm.group(1) if cm else "420"
    return w, h, chroma, int(fm.group(1)), int(fm.group(2))


def y4m_frame_bytes(header: str) -> int:
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


def inspect_y4m(path: Path) -> Y4mInfo:
    with path.open("rb") as f:
        header = f.readline().decode("ascii", "replace").rstrip("\n")
        size = y4m_frame_bytes(header)
        frames = 0
        while True:
            marker = f.readline()
            if not marker:
                break
            if not marker.startswith(b"FRAME"):
                raise SystemExit(f"{path}: expected FRAME, got {marker[:20]!r}")
            data = f.read(size)
            if len(data) != size:
                raise SystemExit(f"{path}: truncated frame {frames}")
            frames += 1
    w, h, chroma, num, den = y4m_geometry(header)
    return Y4mInfo(header, w, h, chroma, num, den, size, frames)


def extract_one_frame(src: Path, index: int, dst: Path) -> Y4mInfo:
    info = inspect_y4m(src)
    if index < 0 or index >= info.frames:
        raise SystemExit(f"{src}: frame index {index} outside 0..{info.frames - 1}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("rb") as f:
        header = f.readline()
        for current in range(index + 1):
            marker = f.readline()
            if not marker.startswith(b"FRAME"):
                raise SystemExit(f"{src}: expected FRAME at {current}")
            payload = f.read(info.frame_bytes)
            if len(payload) != info.frame_bytes:
                raise SystemExit(f"{src}: truncated frame {current}")
        with dst.open("wb") as out:
            out.write(header)
            out.write(b"FRAME\n")
            out.write(payload)
    return inspect_y4m(dst)


# ---------------------------------------------------------------------------
# Host / decoder / quality provenance
# ---------------------------------------------------------------------------


def classify_host(host: dict) -> dict:
    gpus = host.get("gpu") or []
    mem = host.get("gpu_memory_mib")
    cc = str(host.get("gpu_compute_cap", "")).strip()
    if gpus:
        if mem == 8192 and cc == "8.6":
            klass = "target-gpu"
            target = True
        else:
            klass = "proxy-gpu"
            target = False
    else:
        klass = "cpu-reference"
        target = False
    return {
        "class": klass,
        "target_validation": target,
        "gpu_memory_mib": mem,
        "gpu_compute_cap": cc or None,
        "gpu": gpus or None,
    }


def host_key(host: dict) -> tuple:
    return (
        host.get("hostname"),
        host.get("os"),
        host.get("machine"),
        host.get("cpu_model"),
        tuple(host.get("gpu") or []),
    )


def find_dav2d() -> Path:
    env = os.environ.get("DAV2D")
    if env:
        p = Path(env)
        if p.is_file() and os.access(p, os.X_OK):
            return p
        raise SystemExit(f"DAV2D={env} is not executable")
    for candidate in (TOOLS_DIR / "bin" / "dav2d", TOOLS_DIR / "bin" / "dav2d-macos"):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    found = shutil.which("dav2d")
    if found:
        return Path(found)
    raise SystemExit("dav2d not found; set DAV2D=/path/to/dav2d")


def decoder_version(dav2d: Path) -> str:
    p = run([dav2d, "--version"])
    text = ((p.stdout or "") + "\n" + (p.stderr or "")).strip()
    lines = [line for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else "unknown"


def git_provenance() -> dict:
    def git(*args: str) -> str | None:
        p = run(["git", *args], cwd=ROOT)
        return p.stdout.strip() if p.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    status = git("status", "--short")
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status),
        "dirty_entries": [] if not status else status.splitlines(),
    }


# ---------------------------------------------------------------------------
# Tool availability
# ---------------------------------------------------------------------------


def flag_in_help(help_text: str, flag: str) -> bool:
    return (
        re.search(
            r"(?<![A-Za-z0-9_-])" + re.escape(flag) + r"(?![A-Za-z0-9_-])",
            help_text,
        )
        is not None
    )


def detect_tools(help_text: str, selected: set[str] | None) -> list[dict]:
    rows = []
    for spec in TOOL_SPECS:
        if selected and spec.id not in selected:
            continue
        if spec.kind in ("baseline", "control"):
            wired = flag_in_help(help_text, "--m1")
            if spec.kind == "control" and spec.candidate_flags:
                wired = wired and flag_in_help(help_text, spec.candidate_flags[0])
            used_flag = spec.candidate_flags[0] if spec.candidate_flags else "--m1"
            reason = "" if wired else "this build does not advertise the required flag"
        elif not spec.candidate_flags:
            wired = True
            used_flag = None
            reason = ""
        else:
            matches = [f for f in spec.candidate_flags if flag_in_help(help_text, f)]
            wired = bool(matches)
            used_flag = matches[0] if matches else None
            reason = "" if wired else spec.not_wired_reason
        rows.append(
            {
                "id": spec.id,
                "display": spec.display,
                "kind": spec.kind,
                "wired": wired,
                "flag": used_flag,
                "flags": list(spec.flags) if wired else [],
                "scope": spec.scope,
                "not_wired_reason": reason,
                "required_flag": spec.required_flag,
            }
        )
    return rows


def tool_by_id(tool_id: str) -> ToolSpec:
    for spec in TOOL_SPECS:
        if spec.id == tool_id:
            return spec
    raise KeyError(tool_id)


# ---------------------------------------------------------------------------
# QP probe
# ---------------------------------------------------------------------------


def probe_qps(
    encoder: Path,
    source: Path,
    qps: list[int],
    outdir: Path,
    dav2d: Path,
    timeout: float,
) -> list[dict]:
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for qp in qps:
        ivf = outdir / f"probe_qp{qp}.ivf"
        decoded = outdir / f"probe_qp{qp}.y4m"
        for p in (ivf, decoded):
            p.unlink(missing_ok=True)
        t0 = time.perf_counter()
        p = run(
            [encoder, source, "-o", ivf, "--frames", "1", "--qp", str(qp), "--m1"],
            timeout=timeout,
        )
        elapsed = time.perf_counter() - t0
        row = {
            "qp": qp,
            "returncode": p.returncode,
            "seconds": elapsed,
            "stdout": tail(p.stdout or "", 1200),
            "stderr": tail(p.stderr or "", 1200),
            "encode_ok": p.returncode == 0 and ivf.is_file(),
            "decoded": False,
            "bytes": ivf.stat().st_size if ivf.is_file() else None,
        }
        if row["encode_ok"]:
            d = run(
                [
                    dav2d,
                    "-q",
                    "-i",
                    ivf,
                    "-o",
                    decoded,
                    "--threads",
                    "1",
                    "--strict",
                    "1",
                ],
                timeout=timeout,
            )
            row["decoded"] = d.returncode == 0 and decoded.is_file()
            row["decode_stderr"] = tail((d.stdout or "") + (d.stderr or ""), 1200)
        rows.append(row)
        ivf.unlink(missing_ok=True)
        decoded.unlink(missing_ok=True)
    return rows


# ---------------------------------------------------------------------------
# Encoding / gate / quality
# ---------------------------------------------------------------------------


def encode_point(
    encoder: Path,
    encoder_sha256: str,
    tool: dict,
    source: Path,
    resolution: str,
    qp: int,
    out_root: Path,
    dav2d: Path,
    quality_seconds: bool,
    timeout: float,
    reuse: bool,
    frame_index: int,
) -> dict:
    tool_id = tool["id"]
    stem = f"{tool_id}_qp{qp}"
    ivf = out_root / "encodes" / resolution / f"{stem}.ivf"
    meta = out_root / "encodes" / resolution / f"{stem}.meta.json"
    qjson = out_root / "quality" / resolution / f"{stem}.json"
    logdir = out_root / "logs" / resolution
    logdir.mkdir(parents=True, exist_ok=True)
    ivf.parent.mkdir(parents=True, exist_ok=True)
    qjson.parent.mkdir(parents=True, exist_ok=True)

    base = {
        "tool": tool_id,
        "display": tool["display"],
        "kind": tool["kind"],
        "resolution": resolution,
        "qp": qp,
        "flags": tool["flags"],
        "ivf": str(ivf),
        "quality_report": str(qjson),
        "source": str(source),
        "source_sha256": sha256_file(source),
        "frame_index": frame_index,
        "encoder_sha256": encoder_sha256,
    }

    if reuse and ivf.is_file() and qjson.is_file() and meta.is_file():
        try:
            q = json.loads(qjson.read_text())
            stored = json.loads(meta.read_text())
            if (
                stored.get("encoder_sha256") == encoder_sha256
                and stored.get("source_sha256") == base["source_sha256"]
                and stored.get("qp") == qp
                and stored.get("tool") == tool_id
                and q.get("ivf_bytes") == ivf.stat().st_size
                and q.get("frames") == 1
                and q.get("ref") == str(source)
                and q.get("encoder") == tool_id
                and str(q.get("qp")) == str(qp)
            ):
                point = dict(base)
                point.update(
                    {
                        "status": "ok",
                        "reused": True,
                        "bytes": ivf.stat().st_size,
                        "ivf_md5": md5_file(ivf),
                        "ivf_sha256": sha256_file(ivf),
                        "quality": q,
                        "metrics": quality_metrics(q),
                        "decode_gate": {"reused": True, "strict": True},
                    }
                )
                return point
        except Exception:
            pass

    ivf.unlink(missing_ok=True)
    qjson.unlink(missing_ok=True)

    cmd = [
        encoder,
        source,
        "-o",
        ivf,
        "--frames",
        "1",
        "--qp",
        str(qp),
        *tool["flags"],
    ]
    t0 = time.perf_counter()
    p = run(cmd, timeout=timeout)
    encode_seconds = time.perf_counter() - t0
    (logdir / f"{stem}_encode.log").write_text(
        (p.stdout or "") + (p.stderr or "")
    )

    if p.returncode != 0 or not ivf.is_file():
        point = dict(base)
        point.update(
            {
                "status": "encode_failed",
                "error": tail((p.stdout or "") + (p.stderr or "")),
                "encode_seconds": encode_seconds,
            }
        )
        return point

    decoded = out_root / "work" / "decoded" / resolution / f"{stem}.y4m"
    decoded.parent.mkdir(parents=True, exist_ok=True)
    decoded.unlink(missing_ok=True)
    d = run(
        [
            dav2d,
            "-q",
            "-i",
            ivf,
            "-o",
            decoded,
            "--threads",
            "1",
            "--strict",
            "1",
        ],
        timeout=timeout,
    )
    decode_log = (d.stdout or "") + (d.stderr or "")
    (logdir / f"{stem}_decode_strict.log").write_text(decode_log)
    decoded_md5 = md5_file(decoded) if decoded.is_file() else None
    decoded_bytes = decoded.stat().st_size if decoded.is_file() else None
    if d.returncode != 0 or not decoded.is_file():
        point = dict(base)
        point.update(
            {
                "status": "decode_failed",
                "error": tail(decode_log),
                "encode_seconds": encode_seconds,
                "bytes": ivf.stat().st_size,
                "ivf_md5": md5_file(ivf),
            }
        )
        decoded.unlink(missing_ok=True)
        return point

    # Repository gate: tools/validate_dav2d.sh.  Run the literal wrapper too so
    # the report can say the gate the repo defines was executed.
    venv = os.environ.copy()
    venv["DAV2D"] = str(dav2d)
    venv["THREADS"] = "1"
    v = run(["bash", VALIDATE_SH, ivf], env=venv, timeout=timeout)
    validate_log = (v.stdout or "") + (v.stderr or "")
    (logdir / f"{stem}_validate.log").write_text(validate_log)
    validate_ok = v.returncode == 0 and "OK: dav2d decoded" in validate_log
    if not validate_ok:
        point = dict(base)
        point.update(
            {
                "status": "decode_failed",
                "error": "tools/validate_dav2d.sh failed: " + tail(validate_log),
                "encode_seconds": encode_seconds,
                "bytes": ivf.stat().st_size,
                "ivf_md5": md5_file(ivf),
                "decode_gate": {
                    "strict_ok": True,
                    "validate_script_ok": False,
                    "decoded_md5": decoded_md5,
                },
            }
        )
        decoded.unlink(missing_ok=True)
        return point

    # Score only after both decoder gates pass.
    qcmd = [
        sys.executable,
        QUALITY_PY,
        "--ref",
        source,
        "--ivf",
        ivf,
        "--json",
        qjson,
        "--encoder",
        tool_id,
        "--qp",
        str(qp),
    ]
    if quality_seconds:
        qcmd += ["--encode-seconds", f"{encode_seconds:.6f}"]
    qenv = os.environ.copy()
    qenv["DAV2D"] = str(dav2d)
    q = run(qcmd, env=qenv, timeout=max(timeout, 300.0))
    (logdir / f"{stem}_quality.log").write_text((q.stdout or "") + (q.stderr or ""))
    if q.returncode != 0 or not qjson.is_file():
        point = dict(base)
        point.update(
            {
                "status": "quality_failed",
                "error": tail((q.stdout or "") + (q.stderr or "")),
                "encode_seconds": encode_seconds,
                "bytes": ivf.stat().st_size,
                "ivf_md5": md5_file(ivf),
                "decode_gate": {
                    "strict_ok": True,
                    "validate_script_ok": True,
                    "decoded_md5": decoded_md5,
                    "decoded_bytes": decoded_bytes,
                },
            }
        )
        decoded.unlink(missing_ok=True)
        return point

    qdata = json.loads(qjson.read_text())
    if int(qdata.get("frames", 0)) != 1:
        point = dict(base)
        point.update(
            {
                "status": "quality_failed",
                "error": f"quality.py scored {qdata.get('frames')} frames, expected 1",
                "encode_seconds": encode_seconds,
            }
        )
        decoded.unlink(missing_ok=True)
        return point
    if int(qdata.get("ivf_bytes", -1)) != ivf.stat().st_size:
        point = dict(base)
        point.update(
            {
                "status": "quality_failed",
                "error": "quality.py IVF size does not match the stream on disk",
                "encode_seconds": encode_seconds,
            }
        )
        decoded.unlink(missing_ok=True)
        return point

    point = dict(base)
    point.update(
        {
            "status": "ok",
            "bytes": ivf.stat().st_size,
            "ivf_md5": md5_file(ivf),
            "ivf_sha256": sha256_file(ivf),
            "encode_seconds": encode_seconds,
            "quality": qdata,
            "metrics": quality_metrics(qdata),
            "decode_gate": {
                "strict_ok": True,
                "validate_script_ok": True,
                "decoded_md5": decoded_md5,
                "decoded_bytes": decoded_bytes,
                "decoder": qdata.get("decoder"),
                "decoder_version": qdata.get("decoder_version"),
            },
        }
    )
    decoded.unlink(missing_ok=True)
    write_json(
        meta,
        {
            "tool": tool_id,
            "qp": qp,
            "resolution": resolution,
            "source": str(source),
            "source_sha256": base["source_sha256"],
            "encoder_sha256": encoder_sha256,
            "ivf_sha256": point["ivf_sha256"],
            "quality_report": str(qjson),
        },
    )
    return point


def quality_metrics(q: dict) -> dict:
    psnr = q.get("psnr") or {}
    ssim = q.get("ssim") or {}
    return {
        "psnr_y": psnr.get("psnr_y"),
        "psnr_u": psnr.get("psnr_u"),
        "psnr_v": psnr.get("psnr_v"),
        "psnr_avg": psnr.get("psnr_avg"),
        "ssim_y": ssim.get("ssim_y"),
        "ssim_all": ssim.get("ssim_all"),
        "cvvdp_jod": q.get("cvvdp_jod"),
        "cvvdp_version": q.get("cvvdp_version"),
        "cvvdp_display": q.get("cvvdp_display"),
        "cvvdp_device": q.get("cvvdp_device"),
        "bitrate_kbps": q.get("bitrate_kbps"),
        "ivf_bytes": q.get("ivf_bytes"),
        "frames": q.get("frames"),
        "decoder": q.get("decoder"),
        "encode_seconds": q.get("encode_seconds"),
        "encode_fps": q.get("encode_fps"),
        "host": q.get("host") or {},
        "target_host": q.get("target_host") or {},
    }


# ---------------------------------------------------------------------------
# Deltas / verdicts
# ---------------------------------------------------------------------------


def delta(base: dict, test: dict) -> dict | None:
    b = base.get("metrics") or {}
    t = test.get("metrics") or {}
    keys = ("psnr_y", "psnr_avg", "ssim_all", "cvvdp_jod", "bitrate_kbps")
    if any(b.get(k) is None or t.get(k) is None for k in keys):
        return None
    d_rate = (t["bitrate_kbps"] / b["bitrate_kbps"] - 1.0) * 100.0
    d_psnr_y = t["psnr_y"] - b["psnr_y"]
    d_psnr_avg = t["psnr_avg"] - b["psnr_avg"]
    d_ssim = t["ssim_all"] - b["ssim_all"]
    d_cvvdp = t["cvvdp_jod"] - b["cvvdp_jod"]
    return {
        "d_rate_pct": d_rate,
        "d_bytes_pct": (t["ivf_bytes"] / b["ivf_bytes"] - 1.0) * 100.0,
        "d_psnr_y_db": d_psnr_y,
        "d_psnr_avg_db": d_psnr_avg,
        "d_ssim_all": d_ssim,
        "d_cvvdp_jod": d_cvvdp,
        "rate_ratio": t["bitrate_kbps"] / b["bitrate_kbps"],
        "rate_quality_score_db": d_psnr_y + HEURISTIC_DB_PER_PCT_RATE * (-d_rate),
    }


def classify(d: dict | None) -> dict:
    if d is None:
        return {
            "verdict": "unmeasured",
            "reasons": ["missing metrics"],
            "confidence": "none",
        }
    rate = d["d_rate_pct"]
    dp = d["d_psnr_y_db"]
    dc = d["d_cvvdp_jod"]
    ds = d["d_ssim_all"]
    q_up = (
        (dc >= Q_UP_CVVDP_JOD or dp >= Q_UP_PSNR_DB)
        and dp >= -Q_UP_CVVDP_JOD
        and dc >= -Q_UP_CVVDP_JOD
        and ds >= -Q_UP_SSIM_TOL
    )
    q_down = dc <= -Q_DOWN_CVVDP_JOD or dp <= -Q_DOWN_PSNR_DB or ds <= -Q_DOWN_SSIM
    q_flat = not q_up and not q_down
    rate_flat = abs(rate) < RATE_FLAT_PCT
    rate_down = rate <= -RATE_FLAT_PCT
    rate_up = rate >= RATE_FLAT_PCT

    reasons = [
        f"rate {rate:+.2f}%",
        f"PSNR-Y {dp:+.3f} dB",
        f"CVVDP {dc:+.3f} JOD",
        f"SSIM-All {ds:+.5f}",
    ]

    if rate_flat and q_flat:
        return {
            "verdict": "neutral",
            "reasons": reasons + ["no material rate or quality change"],
            "confidence": "low",
        }
    if rate_flat and q_up:
        return {
            "verdict": "net_win",
            "reasons": reasons + ["quality gain at a flat rate"],
            "confidence": "low",
        }
    if rate_flat and q_down:
        return {
            "verdict": "loss",
            "reasons": reasons + ["quality loss at a flat rate"],
            "confidence": "low",
        }
    if rate_down and not q_down:
        return {
            "verdict": "net_win",
            "reasons": reasons + ["rate saving without a material quality loss"],
            "confidence": "low",
        }
    if rate_down and q_down:
        return {
            "verdict": "tradeoff",
            "reasons": reasons
            + [
                "rate and quality both fall; iso-quality sign needs a multi-QP "
                "curve (BD-rate) to resolve"
            ],
            "confidence": "low",
        }
    if rate_up and q_up:
        if rate <= MAX_WIN_RATE_COST_PCT:
            return {
                "verdict": "net_win",
                "reasons": reasons
                + [f"quality gain for <= {MAX_WIN_RATE_COST_PCT:.1f}% rate"],
                "confidence": "low",
            }
        return {
            "verdict": "tradeoff",
            "reasons": reasons
            + ["quality gain costs more than the small-rate-win budget; needs a curve"],
            "confidence": "low",
        }
    if rate_up and not q_up:
        return {
            "verdict": "loss",
            "reasons": reasons + ["rate increase without a material quality gain"],
            "confidence": "low",
        }
    return {
        "verdict": "tradeoff",
        "reasons": reasons + ["unclassified rate/quality combination"],
        "confidence": "low",
    }


def aggregate_verdicts(verdicts: list[str]) -> str:
    if not verdicts:
        return "unmeasured"
    unique = set(verdicts)
    if len(unique) == 1:
        return verdicts[0]
    if unique <= {"net_win", "neutral"}:
        return "mixed"
    if unique <= {"loss", "neutral"}:
        return "mixed"
    if "tradeoff" in unique:
        return "mixed"
    return "mixed"


def bd_direction(pct: float | None, within_noise: bool) -> str:
    if pct is None:
        return "unmeasured"
    if within_noise or abs(pct) < BD_NEUTRAL_PCT:
        return "neutral"
    return "net_win" if pct < 0.0 else "loss"


def bd_resolution_verdict(block: dict) -> tuple[str, dict]:
    """Combine PSNR-Y and CVVDP BD-rate into one resolution verdict.

    PSNR is the primary stable metric. CVVDP is allowed to disagree only when
    its per-QP deltas exceed the JOD noise floor; near saturation it is treated
    as neutral rather than as a large rate difference.
    """
    p = block["psnr_y"]
    c = block["cvvdp"]
    if not p.get("available"):
        return "unmeasured", {
            "reason": p.get("reason", "PSNR-Y BD-rate unavailable"),
        }
    p_dir = bd_direction(p.get("bd_rate_pct"), bool(p.get("within_noise")))
    c_dir = bd_direction(c.get("bd_rate_pct"), bool(c.get("within_noise")))
    detail = {
        "psnr_bd_rate_pct": p.get("bd_rate_pct"),
        "psnr_bd_psnr_db": p.get("bd_psnr_db"),
        "psnr_within_noise": bool(p.get("within_noise")),
        "psnr_sensitivity_pct": p.get("sensitivity", {}).get("bd_rate_pct"),
        "cvvdp_bd_rate_pct": c.get("bd_rate_pct"),
        "cvvdp_within_noise": bool(c.get("within_noise")),
        "cvvdp_sensitivity_pct": c.get("sensitivity", {}).get("bd_rate_pct"),
        "psnr_direction": p_dir,
        "cvvdp_direction": c_dir,
    }
    if p_dir == "neutral" and c_dir == "neutral":
        return "neutral", detail
    if c_dir == "unmeasured":
        return p_dir, detail
    if p_dir == c_dir:
        return p_dir, detail
    if p_dir == "neutral":
        return c_dir, detail
    if c_dir == "neutral":
        return p_dir, detail
    return "metric_dependent", detail


# ---------------------------------------------------------------------------
# BD-rate
# ---------------------------------------------------------------------------


def build_bd_rate(
    points: list[dict],
    tools: list[dict],
    resolutions: list[str],
    method: str,
    primary_qps: list[int],
    requested_qps: list[int],
) -> dict:
    ok = [p for p in points if p["status"] == "ok"]
    supported = sorted({int(p["qp"]) for p in ok})
    if len(primary_qps) < 3:
        primary_qps = supported
    result = {
        "anchor": "m1_baseline",
        "method": method,
        "status": "ok",
        "note": (
            "BD-rate is computed on decoded output at every supported QP. "
            "The primary fit uses the core QP set (default 90..210) because "
            "QP60 is near CVVDP saturation and QP235 is a near-collapse point; "
            "the all-QP fit is reported as a sensitivity check. RDO (intra) is "
            "anchored to the full-intra control so its rate term is isolated "
            "from the full-intra mode search."
        ),
        "primary_qps": primary_qps,
        "sensitivity_qps": supported,
        "tools": {},
        "blocking_reason": None,
    }
    if len(supported) < 3:
        result["status"] = "not_computable"
        if len(requested_qps) < 3:
            result["blocking_reason"] = (
                f"only {len(requested_qps)} QP(s) were requested "
                f"({', '.join(str(q) for q in requested_qps)}); BD-rate needs "
                ">= 3. Pass a wider --qps set."
            )
        else:
            result["blocking_reason"] = (
                f"only {len(supported)} supported QP(s) "
                f"({', '.join(str(q) for q in supported)}); BD-rate needs >= 3. "
                "The encoder or decoder accepted no other QP; an older build "
                "pins base_q_idx = 90 until coefficient-CDF qctx lands."
            )

    for tool in tools:
        tid = tool["id"]
        if tid == "m1_baseline":
            continue
        anchor_id = "m1_full_intra" if tid == "rdo_intra" else "m1_baseline"
        result["tools"][tid] = {
            "display": tool["display"],
            "wired": tool["wired"],
            "anchor": anchor_id,
            "resolutions": {},
        }
        for res in resolutions:
            def pts(tool_id: str, metric: str, qps: list[int]) -> list[RDPoint]:
                return [
                    RDPoint(
                        p["metrics"][metric],
                        p["metrics"]["bitrate_kbps"],
                        f"qp{p['qp']}",
                    )
                    for p in ok
                    if p["resolution"] == res
                    and p["tool"] == tool_id
                    and int(p["qp"]) in qps
                ]

            base_pts = pts(anchor_id, "psnr_y", primary_qps)
            test_pts = pts(tid, "psnr_y", primary_qps)
            base_pts_all = pts(anchor_id, "psnr_y", supported)
            test_pts_all = pts(tid, "psnr_y", supported)
            base_pts_q = pts(anchor_id, "cvvdp_jod", primary_qps)
            test_pts_q = pts(tid, "cvvdp_jod", primary_qps)
            base_pts_q_all = pts(anchor_id, "cvvdp_jod", supported)
            test_pts_q_all = pts(tid, "cvvdp_jod", supported)

            psnr_primary = bd_rate(base_pts, test_pts, method=method)
            psnr_primary_psnr = bd_psnr(base_pts, test_pts, method=method)
            psnr_all = bd_rate(base_pts_all, test_pts_all, method=method)
            cvvdp_primary = bd_rate(base_pts_q, test_pts_q, method=method)
            cvvdp_all = bd_rate(base_pts_q_all, test_pts_q_all, method=method)

            cvvdp_deltas = [
                abs(p["delta_vs_baseline"]["d_cvvdp_jod"])
                for p in points
                if p["resolution"] == res
                and p["tool"] == tid
                and p["status"] == "ok"
                and p.get("delta_vs_baseline")
            ]
            psnr_deltas = [
                abs(p["delta_vs_baseline"]["d_psnr_y_db"])
                for p in points
                if p["resolution"] == res
                and p["tool"] == tid
                and p["status"] == "ok"
                and p.get("delta_vs_baseline")
            ]
            cvvdp_noise = bool(cvvdp_deltas) and max(cvvdp_deltas) < CVVDP_NOISE_FLOOR_JOD
            psnr_noise = bool(psnr_deltas) and max(psnr_deltas) < PSNR_NOISE_FLOOR_DB

            result["tools"][tid]["resolutions"][res] = {
                "anchor_points": len(base_pts),
                "test_points": len(test_pts),
                "psnr_y": {
                    **psnr_primary,
                    **psnr_primary_psnr,
                    "sensitivity": psnr_all,
                    "within_noise": psnr_noise,
                    "max_abs_delta_db": max(psnr_deltas) if psnr_deltas else None,
                },
                "cvvdp": {
                    **cvvdp_primary,
                    "sensitivity": cvvdp_all,
                    "within_noise": cvvdp_noise,
                    "max_abs_delta_jod": max(cvvdp_deltas) if cvvdp_deltas else None,
                },
            }
    return result


# ---------------------------------------------------------------------------
# Plots / markdown
# ---------------------------------------------------------------------------


def plot_rd(points: list[dict], resolution: str, outpath: Path) -> dict:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment-dependent
        return {"ok": False, "reason": f"matplotlib unavailable: {exc}"}

    ok = [p for p in points if p["status"] == "ok" and p["resolution"] == resolution]
    if not ok:
        return {"ok": False, "reason": "no verified points"}

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharex=True)
    groups: dict[str, list[dict]] = {}
    for p in ok:
        groups.setdefault(p["tool"], []).append(p)
    for tool_id, pts in groups.items():
        pts = sorted(pts, key=lambda p: p["metrics"]["bitrate_kbps"])
        xs = [p["metrics"]["bitrate_kbps"] for p in pts]
        ys_c = [p["metrics"]["cvvdp_jod"] for p in pts]
        ys_p = [p["metrics"]["psnr_y"] for p in pts]
        label = pts[0]["display"]
        marker = "*" if tool_id == "m1_baseline" else "o"
        size = 160 if tool_id == "m1_baseline" else 55
        for ax, ys in zip(axes, (ys_c, ys_p)):
            ax.plot(xs, ys, marker=marker, markersize=math.sqrt(size), linewidth=1.3, label=label)
            if len(pts) == 1:
                ax.annotate(
                    label,
                    (xs[0], ys[0]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=7,
                )
    axes[0].set_ylabel("CVVDP (JOD)")
    axes[0].set_title("CVVDP vs bitrate")
    axes[1].set_ylabel("PSNR-Y (dB)")
    axes[1].set_title("PSNR-Y vs bitrate")
    for ax in axes:
        ax.set_xlabel("bitrate (kbps, one-frame normalization)")
        ax.set_xscale("log")
        ax.grid(True, which="both", alpha=0.25)
    axes[1].legend(fontsize=7, loc="best")
    qps = sorted({int(p["qp"]) for p in ok})
    title = f"{resolution} — decoded output"
    if len(qps) == 1:
        title += f" (QP {qps[0]} only; no R-D curve yet)"
    fig.suptitle(title)
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=160)
    plt.close(fig)
    return {"ok": True, "path": str(outpath)}


def fmt(value, nd: int = 3, suffix: str = "") -> str:
    if value is None:
        return "—"
    if isinstance(value, str):
        return value
    return f"{value:.{nd}f}{suffix}"


def fmt_pct(value) -> str:
    if value is None:
        return "—"
    return f"{value:+.2f}%"


def fmt_db(value) -> str:
    if value is None:
        return "—"
    return f"{value:+.3f} dB"


def fmt_jod(value) -> str:
    if value is None:
        return "—"
    return f"{value:+.3f} JOD"


def fmt_ssim(value) -> str:
    if value is None:
        return "—"
    return f"{value:+.5f}"


VERDICT_LABELS = {
    "net_win": "net win",
    "metric_dependent": "metric-dependent",
    "tradeoff": "trade-off",
    "unmeasured": "unmeasured",
    "neutral": "neutral",
    "loss": "loss",
    "mixed": "mixed",
}


def pretty_verdict(value: str) -> str:
    return VERDICT_LABELS.get(value, value)


def write_markdown(
    path: Path,
    manifest: dict,
    summary: dict,
    points: list[dict],
    bd: dict,
    plot_info: dict[str, dict],
) -> None:
    lines: list[str] = []
    add = lines.append

    add("# M3 quality / BD-rate report")
    add("")
    add(f"Generated: `{manifest['generated_utc']}`")
    add("")
    verdict = summary["headline"]
    add(f"**Overall: {verdict}** — comparison_valid=`{summary['comparison_valid']}`, "
        f"target_validation=`{summary['target_validation']}`")
    add("")
    if summary["flags"]:
        add("**Flags:** " + "; ".join(f"`{f}`" for f in summary["flags"]))
        add("")

    add("## Provenance")
    add("")
    add(f"- Encoder: `{manifest['encoder']['path']}` "
        f"(sha256 `{manifest['encoder']['sha256'][:16]}…`)")
    if manifest["encoder"].get("path") != manifest["encoder"].get("source_path"):
        add(f"- Encoder snapshot source: `{manifest['encoder'].get('source_path')}` "
            f"(live binary unchanged during run: "
            f"`{manifest['encoder'].get('source_unchanged_during_run')}`)")
    add(f"- Source tree: `{manifest['git']['branch']}` @ `{manifest['git']['commit']}`"
        + (" (dirty)" if manifest["git"]["dirty"] else ""))
    add(f"- Decoder: `{manifest['decoder']['path']}` — {manifest['decoder']['version']}")
    add(f"- Metrics: `tools/quality.py`, CVVDP `{manifest['cvvdp']['version']}` "
        f"(`{manifest['cvvdp']['display']}`, `{manifest['cvvdp']['device']}`)")
    host = manifest["host"]
    add(f"- Host: `{host.get('hostname')}` — {host.get('os')}, "
        f"{host.get('cpu_model')}, {host.get('cpu_physical_cores') or host.get('cpu_logical_cores')} cores, "
        f"{host.get('ram_gb')} GB RAM")
    add(f"- Hardware class: `{manifest['hardware']['class']}` "
        f"(target validation: `{manifest['hardware']['target_validation']}`)")
    add("")
    add("### Sources (one canonical frame per resolution)")
    add("")
    add("| Resolution | Source | Frame | Y4M MD5 | Y4M SHA-256 | Provenance |")
    add("|---|---|---:|---|---|---|")
    for res, info in manifest["sources"].items():
        add(
            f"| {res} | `{info['source_path']}` | {info['frame_index']} | "
            f"`{info['md5']}` | `{info['sha256'][:16]}…` | {info['provenance']} |"
        )
    add("")
    if manifest.get("cross_resolution_check"):
        cr = manifest["cross_resolution_check"]
        add(f"Cross-resolution same-source check: "
            f"`{cr['status']}` — 1080p scaled to 640x360 vs 640x360 source "
            f"PSNR-Y `{fmt(cr.get('scaled_psnr_y'), 3, ' dB')}` "
            f"(pass ≥ {cr['pass_threshold_db']} dB).")
        add("")

    add("## QP support probe")
    add("")
    add("The encoder was asked for each QP; only rows that encoded **and decoded "
        "with `dav2d --strict 1`** are treated as supported.")
    add("")
    add("| Resolution | QP | Encode | Decode | Bytes | Diagnostic |")
    add("|---|---:|---|---|---:|---|")
    for res, rows in manifest["qp_probe"].items():
        for row in rows:
            diag = row.get("stderr") or row.get("decode_stderr") or ""
            diag = diag.replace("|", "\\|")
            if len(diag) > 110:
                diag = diag[:107] + "…"
            add(
                f"| {res} | {row['qp']} | {'ok' if row['encode_ok'] else 'rejected'} | "
                f"{'ok' if row['decoded'] else '—'} | {fmt(row.get('bytes'), 0)} | "
                f"{diag or '—'} |"
            )
    add("")
    add(f"Supported QPs: `{manifest['supported_qps']}`. "
        f"Requested: `{manifest['qps_requested']}`.")
    add("")

    encoded = [p for p in points if p["status"] != "not_wired"]
    verified = [p for p in encoded if p["status"] == "ok"]
    add("## Stream verification")
    add("")
    add(f"- Verified streams: **{len(verified)}/{len(encoded)}** encoded points.")
    add(f"- Decoder: `{manifest['decoder']['path']}` ({manifest['decoder']['version']}); "
        "every stream passed `dav2d --strict 1` and `tools/validate_dav2d.sh`.")
    add(f"- Unwired tool rows: {len([p for p in points if p['status'] == 'not_wired'])} "
        "(no stream, no metric).")
    add("- Per-point IVF MD5/SHA-256 and decode-gate results are in `summary.json`; "
        "raw logs are under `logs/`.")
    add("")

    add("## Tool availability")
    add("")
    add("| Tool | Kind | Wired | Flag | Scope |")
    add("|---|---|---|---|---|")
    for tool in summary["tools"]:
        flag = f"`{tool['flag']}`" if tool.get("flag") else "—"
        add(f"| {tool['display']} | {tool['kind']} | `{tool['wired']}` | {flag} | {tool['scope']} |")
    add("")
    not_wired = [t for t in summary["tools"] if not t["wired"]]
    if not_wired:
        add("**Not measured — no stream:**")
        for tool in not_wired:
            add(f"- **{tool['display']}**: {tool['not_wired_reason']}")
        add("")

    add("## Per-tool results")
    add("")
    add("Deltas are against the M1 baseline at the same QP/resolution, except "
        "RDO rows, which are anchored to the M1 full-intra control so the rate "
        "term is isolated. `Stream` means the IVF passed both `dav2d --strict 1` "
        "and `tools/validate_dav2d.sh`. A row without a verified stream has no "
        "metric and no verdict. `Point verdict` is a single-QP classification; "
        "the overall verdicts use BD-rate.")
    add("")
    for res in manifest["resolutions"]:
        add(f"### {res}")
        add("")
        add("| Tool | Scope | QP | Stream | IVF B | kbps | PSNR-Y | PSNR-avg | SSIM-All | CVVDP | Δrate | ΔPSNR-Y | ΔCVVDP | ΔSSIM | Point verdict |")
        add("|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for p in points:
            if p["resolution"] != res:
                continue
            m = p.get("metrics") or {}
            d = p.get("delta_vs_anchor")
            v = p.get("verdict") or {}
            stream = "verified" if p["status"] == "ok" else p["status"]
            if p["status"] != "ok":
                err = (p.get("error") or "").replace("\n", " ").replace("|", "\\|")
                if len(err) > 80:
                    err = err[:77] + "…"
                stream = f"{stream}: {err}" if err else stream
            add(
                f"| {p['display']} | {p['kind']} | {p['qp']} | {stream} | "
                f"{fmt(p.get('bytes'), 0)} | {fmt(m.get('bitrate_kbps'), 1)} | "
                f"{fmt(m.get('psnr_y'), 3)} | {fmt(m.get('psnr_avg'), 3)} | "
                f"{fmt(m.get('ssim_all'), 5)} | {fmt(m.get('cvvdp_jod'), 3)} | "
                f"{fmt_pct(d.get('d_rate_pct') if d else None)} | "
                f"{fmt_db(d.get('d_psnr_y_db') if d else None)} | "
                f"{fmt_jod(d.get('d_cvvdp_jod') if d else None)} | "
                f"{fmt_ssim(d.get('d_ssim_all') if d else None)} | "
                f"{v.get('verdict', '—')} |"
            )
        add("")
        plot = plot_info.get(res)
        if plot and plot.get("ok"):
            add(f"R-D plot: `{Path(plot['path']).name}`")
            add("")
        elif plot:
            add(f"R-D plot not written: {plot.get('reason')}")
            add("")

    add("## BD-rate vs M1 baseline")
    add("")
    add(f"Method: `{bd['method']}` interpolation, decoded output, anchor "
        f"`m1_baseline` (RDO rows use the full-intra control).")
    add("")
    if bd["status"] != "ok":
        add(f"**Not computable.** {bd['blocking_reason']}")
        add("")
        add("Pointwise rate deltas at the available QP are reported above. They "
            "are **not** BD-rate: a rate delta at one QP is confounded with the "
            "quality delta and cannot establish the iso-quality sign.")
    else:
        add("Negative BD-rate is better (less rate at equal quality). "
            "Positive BD-PSNR is better. `Core` is the primary fit; `All-QP` is "
            "a sensitivity check. `noise` means the metric's per-QP deltas stay "
            "below its noise floor, so it cannot drive the verdict.")
    add("")
    add("| Tool | Resolution | Metric | BD-rate (core) | BD-PSNR (core) | All-QP BD-rate | Noise |")
    add("|---|---|---|---:|---:|---:|---|")
    for tid, tool in bd["tools"].items():
        if not tool["wired"]:
            add(f"| {tool['display']} | — | — | — | — | — | not wired (no stream) |")
            continue
        for res, metrics in tool["resolutions"].items():
            for metric_key, metric_label in (("psnr_y", "PSNR-Y"), ("cvvdp", "CVVDP")):
                block = metrics[metric_key]
                if block.get("available"):
                    br = fmt_pct(block.get("bd_rate_pct"))
                    bp = fmt_db(block.get("bd_psnr_db")) if metric_key == "psnr_y" else "—"
                    sens = block.get("sensitivity") or {}
                    all_br = fmt_pct(sens.get("bd_rate_pct"))
                    noise = "within_noise" if block.get("within_noise") else "—"
                    add(f"| {tool['display']} | {res} | {metric_label} | {br} | {bp} | {all_br} | {noise} |")
                else:
                    reason = block.get("reason", "unavailable")
                    add(f"| {tool['display']} | {res} | {metric_label} | — | — | — | {reason} |")
    add("")

    add("## Verdicts")
    add("")
    add("| Tool | Scope | 640x360 | 1920x1080 | Overall | Confidence |")
    add("|---|---|---|---|---|---|")
    for tool in summary["tools"]:
        if tool["kind"] == "baseline":
            continue
        per = tool.get("point_verdicts") or {}
        add(
            f"| {tool['display']} | {tool['kind']} | {pretty_verdict(per.get('640x360', '—'))} | "
            f"{pretty_verdict(per.get('1920x1080', '—'))} | {pretty_verdict(tool['overall_verdict'])} | "
            f"{tool['confidence']} |"
        )
    add("")
    add("Verdict rules:")
    add("")
    add(f"- **BD-rate mode** (≥3 QPs): a metric is a win at ≤−{BD_NEUTRAL_PCT:.1f}% "
        f"BD-rate, a loss at ≥+{BD_NEUTRAL_PCT:.1f}%, neutral in between. The "
        "primary fit uses the core QP set; the all-QP fit is checked for sign "
        "flips.")
    add(f"- a metric whose per-QP deltas stay below its noise floor "
        f"(CVVDP <{CVVDP_NOISE_FLOOR_JOD:.2f} JOD, PSNR <{PSNR_NOISE_FLOOR_DB:.2f} dB) "
        "is treated as neutral and cannot flip the verdict.")
    add("- `metric_dependent` means PSNR-Y and CVVDP disagree beyond their noise "
        "floors. It is **not** a net win.")
    add("- `mixed` means the two resolutions disagree; read the per-resolution "
        "cells, not the headline.")
    add(f"- **single-QP fallback**: rate change is material at ±{RATE_FLAT_PCT:.1f}%; "
        f"quality gain at ≥{Q_UP_CVVDP_JOD:.2f} JOD or ≥{Q_UP_PSNR_DB:.2f} dB; "
        f"loss at ≤−{Q_DOWN_CVVDP_JOD:.2f} JOD, ≤−{Q_DOWN_PSNR_DB:.2f} dB, or "
        f"≤−{Q_DOWN_SSIM:.4f} SSIM-All. Rate down **and** quality down is "
        "`tradeoff`, not a win.")
    add("")

    add("## Bottom line")
    add("")
    for tool in summary["tools"]:
        if tool["kind"] != "m3":
            continue
        if not tool["wired"]:
            add(f"- **{tool['display']}**: unmeasured — {tool['not_wired_reason']}")
            continue
        per = tool.get("point_verdicts") or {}
        detail = tool.get("bd_detail") or {}
        bits = []
        for res in manifest["resolutions"]:
            v = per.get(res, "unmeasured")
            d = detail.get(res) or {}
            p = d.get("psnr_bd_rate_pct")
            c = d.get("cvvdp_bd_rate_pct")
            if p is not None:
                bits.append(
                    f"{pretty_verdict(v)} at {res} (PSNR-Y {fmt_pct(p)}, CVVDP {fmt_pct(c)}"
                    + (", CVVDP within noise" if d.get("cvvdp_within_noise") else "")
                    + ")"
                )
            else:
                bits.append(f"{pretty_verdict(v)} at {res}")
        add(f"- **{tool['display']}**: " + "; ".join(bits) + f". Overall: **{pretty_verdict(tool['overall_verdict'])}**.")
    add("")

    add("## Confidence caveats")
    add("")
    for caveat in summary["caveats"]:
        add(f"- {caveat}")
    add("")
    add("## Reproduction")
    add("")
    add("```sh")
    add("python3 tools/m3_quality/run_m3_quality.py \\")
    add("  --encoder build-m1/av2enc \\")
    add("  --out tools/m3_quality/results")
    add("```")
    add("")
    add(f"Raw points: `summary.json`, `manifest.json`, `bd_rate.json`; "
        f"per-point quality JSON under `quality/`; streams under `encodes/`; "
        f"logs under `logs/`.")
    add("")

    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--encoder", type=Path, default=DEFAULT_ENCODER)
    ap.add_argument("--source-360p", type=Path, default=DEFAULT_SOURCE_360)
    ap.add_argument("--source-1080p", type=Path, default=DEFAULT_SOURCE_1080)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--qps", default="60,90,110,135,160,185,210,235",
                    help="comma-separated QP probe list (default: 60,90,110,135,160,185,210,235)")
    ap.add_argument("--frame-index", type=int, default=0,
                    help="frame to extract from each source (default: 0)")
    ap.add_argument("--tools", help="comma-separated tool ids (default: all)")
    ap.add_argument("--no-controls", action="store_true",
                    help="skip the m1_full_intra control (also disables the RDO incremental view)")
    ap.add_argument("--skip-cross-res-check", action="store_true")
    ap.add_argument("--allow-mixed-hardware", action="store_true",
                    help="write the report but mark mixed-host comparisons invalid")
    ap.add_argument("--bd-method", choices=("pchip", "cubic"), default="pchip")
    ap.add_argument("--bd-qps",
                    help="comma-separated QP subset for the primary BD-rate fit "
                         "(default: the supported subset of 90,110,135,160,185,210; "
                         "all supported QPs are always reported as a sensitivity check)")
    ap.add_argument("--reuse", action="store_true",
                    help="reuse an existing IVF + quality JSON if the size matches")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--timeout", type=float, default=900.0)
    args = ap.parse_args()

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)

    encoder_source = args.encoder.resolve()
    if not encoder_source.is_file() or not os.access(encoder_source, os.X_OK):
        raise SystemExit(f"encoder not executable: {encoder_source}")
    encoder_sha = sha256_file(encoder_source)
    snapshot = out / "bin" / f"av2enc-{encoder_sha[:16]}"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if not snapshot.is_file() or sha256_file(snapshot) != encoder_sha:
        shutil.copy2(encoder_source, snapshot)
        os.chmod(snapshot, 0o755)
    # Run the snapshot, not the live build directory: another agent may rebuild
    # `build-m1/av2enc` while this sweep is running. A mixed-binary sweep is not
    # a valid comparison.
    encoder = snapshot
    dav2d = find_dav2d()
    quality_mod = load_quality_module()
    host = quality_mod.host_info()
    hardware = classify_host(host)
    dec_version = decoder_version(dav2d)
    git_info_start = git_provenance()

    qps = [int(x.strip()) for x in args.qps.split(",") if x.strip()]
    selected = None
    if args.tools:
        selected = {x.strip() for x in args.tools.split(",") if x.strip()}
    if args.no_controls:
        selected = (selected or {s.id for s in TOOL_SPECS}) - {"m1_full_intra"}

    # Encoder help / availability.
    help_p = run([encoder, "--help"])
    help_text = (help_p.stdout or "") + "\n" + (help_p.stderr or "")
    (out / "logs").mkdir(parents=True, exist_ok=True)
    (out / "logs" / "encoder_help.txt").write_text(help_text)
    tools = detect_tools(help_text, selected)
    if not any(t["id"] == "m1_baseline" for t in tools):
        raise SystemExit("the tool selection must include m1_baseline")

    encoder_info = {
        "path": str(encoder),
        "source_path": str(encoder_source),
        "sha256": encoder_sha,
        "md5": md5_file(encoder),
        "bytes": encoder.stat().st_size,
        "help_sha256": hashlib.sha256(help_text.encode()).hexdigest(),
        "original_sha256_at_start": encoder_sha,
        "source_unchanged_during_run": None,
    }
    git_info = git_info_start

    # Source extraction (same canonical frame for every tool at a resolution).
    sources: dict[str, dict] = {}
    source_paths: dict[str, Path] = {}
    source_specs = {"640x360": args.source_360p, "1920x1080": args.source_1080p}
    for res, spec in RESOLUTIONS.items():
        src = source_specs[res].resolve()
        if not src.is_file():
            raise SystemExit(f"missing source for {res}: {src}")
        extracted = out / "sources" / f"{res}_frame{args.frame_index}.y4m"
        info = extract_one_frame(src, args.frame_index, extracted)
        if (info.width, info.height) != (spec.width, spec.height):
            raise SystemExit(
                f"{src}: expected {spec.width}x{spec.height}, got {info.width}x{info.height}"
            )
        sources[res] = {
            "source_path": str(src),
            "source_sha256": sha256_file(src),
            "frame_index": args.frame_index,
            "frame_path": str(extracted),
            "width": info.width,
            "height": info.height,
            "chroma": info.chroma,
            "fps": f"{info.fps_num}/{info.fps_den}",
            "frame_bytes": info.frame_bytes,
            "md5": md5_file(extracted),
            "sha256": sha256_file(extracted),
            "provenance": spec.provenance,
            "proxy_for": spec.proxy_for,
        }
        source_paths[res] = extracted

    # Cross-resolution source check: scale 1080p to 640x360 and compare.
    cross = None
    if not args.skip_cross_res_check and "1920x1080" in source_paths:
        scaled = work / "cross_1080p_to_360p.y4m"
        p = run([
            "ffmpeg", "-hide_banner", "-v", "error", "-y",
            "-i", source_paths["1920x1080"],
            "-vf", "scale=640:360:flags=bicubic",
            "-pix_fmt", "yuv420p", "-frames:v", "1",
            "-f", "yuv4mpegpipe", scaled,
        ], timeout=args.timeout)
        psnr_y = None
        if p.returncode == 0 and scaled.is_file():
            q = run([
                "ffmpeg", "-hide_banner", "-v", "info",
                "-i", scaled, "-i", source_paths["640x360"],
                "-lavfi", "[0:v][1:v]psnr", "-f", "null", "-",
            ], timeout=args.timeout)
            m = re.search(
                r"PSNR y:([0-9.]+)",
                (q.stdout or "") + "\n" + (q.stderr or ""),
            )
            psnr_y = float(m.group(1)) if m else None
        if psnr_y is None:
            status = "unavailable"
        elif psnr_y >= 40.0:
            status = "pass"
        elif psnr_y >= 30.0:
            status = "warn"
        else:
            status = "fail"
        cross = {
            "status": status,
            "scaled_psnr_y": psnr_y,
            "pass_threshold_db": 40.0,
            "warn_threshold_db": 30.0,
            "note": "1080p frame scaled to 640x360 vs the 640x360 fixture; "
                    "high PSNR confirms the same source frame and scene.",
        }
        scaled.unlink(missing_ok=True)

    # QP probes at both resolutions.
    qp_probe: dict[str, list[dict]] = {}
    for res, src in source_paths.items():
        print(f"[probe] {res} QPs {qps}", file=sys.stderr)
        qp_probe[res] = probe_qps(
            encoder, src, qps, work / "probe" / res, dav2d, args.timeout
        )
    supported_by_res = {
        res: [row["qp"] for row in rows if row["encode_ok"] and row["decoded"]]
        for res, rows in qp_probe.items()
    }
    support_sets = [set(v) for v in supported_by_res.values()]
    supported = sorted(set.intersection(*support_sets)) if support_sets else []
    if not supported:
        raise SystemExit(
            "no QP encoded and decoded at both resolutions; see "
            f"{out / 'logs'} and the QP probe diagnostics"
        )
    if any(set(v) != set(supported) for v in supported_by_res.values()):
        print("warning: QP support differs by resolution; using the intersection",
              file=sys.stderr)

    # Encode the matrix at each supported QP.
    points: list[dict] = []
    measured_tools = [t for t in tools if t["wired"]]
    total = len(measured_tools) * len(supported) * len(RESOLUTIONS)
    index = 0
    for res in RESOLUTIONS:
        for qp in supported:
            for tool in measured_tools:
                index += 1
                print(
                    f"[{index}/{total}] {res} QP{qp} {tool['id']}",
                    file=sys.stderr,
                )
                point = encode_point(
                    encoder,
                    encoder_sha,
                    tool,
                    source_paths[res],
                    res,
                    qp,
                    out,
                    dav2d,
                    True,
                    args.timeout,
                    args.reuse,
                    args.frame_index,
                )
                points.append(point)
    # Add one row per unwired tool/resolution so the report covers the full M3
    # list without pretending there are per-QP measurements.
    for tool in tools:
        if tool["wired"]:
            continue
        for res in RESOLUTIONS:
            points.append({
                "tool": tool["id"],
                "display": tool["display"],
                "kind": tool["kind"],
                "resolution": res,
                "qp": "—",
                "flags": [],
                "status": "not_wired",
                "error": tool["not_wired_reason"],
                "metrics": {},
                "verdict": {
                    "verdict": "unmeasured",
                    "reasons": ["no verified stream"],
                    "confidence": "none",
                },
            })

    # Anchor every tool: RDO is compared with full-intra, others with M1.
    by_key = {
        (p["resolution"], p["tool"], p["qp"]): p
        for p in points
        if p["status"] == "ok"
    }
    for p in points:
        if p["status"] != "ok":
            continue
        anchor_id = "m1_full_intra" if p["tool"] == "rdo_intra" else "m1_baseline"
        base = by_key.get((p["resolution"], anchor_id, p["qp"]))
        if p["tool"] == "m1_baseline":
            p["anchor_tool"] = "m1_baseline"
            p["delta_vs_anchor"] = {
                "d_rate_pct": 0.0,
                "d_bytes_pct": 0.0,
                "d_psnr_y_db": 0.0,
                "d_psnr_avg_db": 0.0,
                "d_ssim_all": 0.0,
                "d_cvvdp_jod": 0.0,
                "rate_ratio": 1.0,
                "rate_quality_score_db": 0.0,
            }
            p["verdict"] = {
                "verdict": "baseline",
                "reasons": ["anchor"],
                "confidence": "n/a",
            }
            p["anchor_tool"] = "m1_baseline"
            continue
        p["anchor_tool"] = anchor_id
        d = delta(base, p) if base else None
        p["delta_vs_anchor"] = d
        p["verdict"] = classify(d)
        # Also record the baseline-relative delta for transparency.
        baseline = by_key.get((p["resolution"], "m1_baseline", p["qp"]))
        if baseline and p["tool"] != "m1_full_intra":
            p["delta_vs_baseline"] = delta(baseline, p)
        else:
            p["delta_vs_baseline"] = d if p["tool"] == "m1_full_intra" else None

    # Host / source / decoder / CVVDP consistency.
    ok_points = [p for p in points if p["status"] == "ok"]
    host_keys = {
        host_key(p["metrics"].get("host") or {})
        for p in ok_points
    }
    cvvdp_settings = {
        (
            p["metrics"].get("cvvdp_version"),
            p["metrics"].get("cvvdp_display"),
            p["metrics"].get("cvvdp_device"),
        )
        for p in ok_points
    }
    decoders = {p["metrics"].get("decoder") for p in ok_points}
    frames = {p["metrics"].get("frames") for p in ok_points}
    comparison_errors: list[str] = []
    if len(host_keys) > 1:
        comparison_errors.append(f"mixed hosts: {len(host_keys)} distinct host identities")
    if len(cvvdp_settings) > 1:
        comparison_errors.append(
            "mixed CVVDP settings: " + repr(sorted(cvvdp_settings, key=str))
        )
    if len(decoders) > 1:
        comparison_errors.append(f"mixed decoders: {sorted(decoders, key=str)}")
    for res in RESOLUTIONS:
        hashes = {p.get("source_sha256") for p in ok_points if p["resolution"] == res}
        if len(hashes) > 1:
            comparison_errors.append(
                f"mixed source frames at {res}: {len(hashes)} distinct Y4M hashes"
            )
    if frames != {1}:
        comparison_errors.append(f"non-single-frame quality reports: {sorted(frames, key=str)}")
    if cross and cross["status"] == "fail":
        comparison_errors.append(
            "cross-resolution source check failed; the 640x360 and 1920x1080 "
            "fixtures are not the same source frame"
        )
    comparison_valid = not comparison_errors
    if comparison_errors and args.allow_mixed_hardware:
        print(
            "warning: comparison is invalid but --allow-mixed-hardware was set; "
            "verdicts will be marked unmeasured",
            file=sys.stderr,
        )

    # BD-rate and per-tool aggregation.
    if args.bd_qps:
        primary_qps = [int(x.strip()) for x in args.bd_qps.split(",") if x.strip()]
        primary_qps = [q for q in primary_qps if q in supported]
    else:
        primary_qps = [q for q in CORE_BD_QPS if q in supported]
    if len(primary_qps) < 3:
        primary_qps = supported
    bd = build_bd_rate(
        points, tools, list(RESOLUTIONS), args.bd_method, primary_qps, qps
    )
    tool_summary = []
    for tool in tools:
        tid = tool["id"]
        per_res: dict[str, str] = {}
        per_res_basis: dict[str, str] = {}
        bd_detail: dict[str, dict] = {}
        reasons: list[str] = []
        bd_tool = bd.get("tools", {}).get(tid)
        for res in RESOLUTIONS:
            point_verdicts = [
                p["verdict"]["verdict"]
                for p in points
                if p["tool"] == tid
                and p["resolution"] == res
                and p["status"] == "ok"
            ]
            point_verdict = aggregate_verdicts(point_verdicts)
            bd_res = None
            if bd_tool:
                bd_res = bd_tool.get("resolutions", {}).get(res)
            bd_available = bool(
                bd_res
                and bd_res["psnr_y"].get("available")
                and len(supported) >= 3
            )
            if bd_available:
                bd_verdict, detail = bd_resolution_verdict(bd_res)
                per_res[res] = bd_verdict
                per_res_basis[res] = "bd_rate"
                bd_detail[res] = detail
                if detail.get("psnr_within_noise"):
                    reasons.append(f"{res}: PSNR within noise")
                if detail.get("cvvdp_within_noise"):
                    reasons.append(f"{res}: CVVDP within noise")
                if bd_verdict == "metric_dependent":
                    reasons.append(
                        f"{res}: PSNR {detail.get('psnr_direction')} vs "
                        f"CVVDP {detail.get('cvvdp_direction')}"
                    )
                sp = detail.get("psnr_sensitivity_pct")
                pp = detail.get("psnr_bd_rate_pct")
                if sp is not None and pp is not None and (sp < 0) != (pp < 0):
                    reasons.append(f"{res}: PSNR BD-rate sign flips on the all-QP fit")
            elif point_verdicts:
                per_res[res] = point_verdict
                per_res_basis[res] = "pointwise"
            else:
                per_res[res] = "unmeasured"
                per_res_basis[res] = "none"
        if tool["kind"] == "m3":
            if not tool["wired"]:
                overall = "unmeasured"
                conf = "none"
                reasons.append(tool["not_wired_reason"])
            elif not comparison_valid:
                overall = "unmeasured"
                conf = "none"
                reasons.append("comparison invalid: " + "; ".join(comparison_errors))
            else:
                measured = [v for v in per_res.values() if v != "unmeasured"]
                unmeasured = [res for res, v in per_res.items() if v == "unmeasured"]
                if "metric_dependent" in measured:
                    overall = "metric_dependent"
                elif "tradeoff" in measured:
                    overall = "mixed"
                elif len(set(measured)) == 1 and len(measured) == len(RESOLUTIONS):
                    overall = measured[0]
                elif measured and all(v in ("net_win", "neutral") for v in measured):
                    overall = "mixed"
                elif measured and all(v in ("loss", "neutral") for v in measured):
                    overall = "mixed"
                else:
                    overall = "mixed"
                if unmeasured:
                    overall = "mixed"
                    reasons.append(
                        "unmeasured resolution(s): " + ", ".join(unmeasured)
                    )
                uses_bd = any(v == "bd_rate" for v in per_res_basis.values())
                conf = "low" if uses_bd else ("very low" if measured else "none")
                if len(set(measured)) > 1:
                    reasons.append("resolutions disagree")
                if uses_bd:
                    reasons.append(
                        "BD-rate verdict uses the core QP fit; all-QP "
                        "sensitivity is in bd_rate.json"
                    )
                if not uses_bd and len(supported) < 3:
                    reasons.append("single-QP mode: no BD-rate curve")
        elif tool["kind"] == "control":
            overall = "control"
            conf = "n/a"
        else:
            overall = "baseline"
            conf = "n/a"
        tool_summary.append({
            **tool,
            "point_verdicts": per_res,
            "verdict_basis": per_res_basis,
            "bd_detail": bd_detail,
            "overall_verdict": overall,
            "confidence": conf,
            "reasons": reasons,
        })

    # Flags.
    flags: list[str] = []
    if hardware["class"] != "target-gpu":
        flags.append(f"non_target_hardware:{hardware['class']}")
    if not hardware["target_validation"]:
        flags.append("target_validation_false")
    if len(supported) == 1:
        flags.append(f"single_qp_only:QP{supported[0]}_until_qctx")
    if cross and cross["status"] != "pass":
        flags.append(f"cross_resolution_source_check:{cross['status']}")
    if sources["640x360"].get("proxy_for"):
        flags.append("resolution_proxy:640x360_downscaled_from_4k")
    if comparison_errors:
        flags.append("comparison_invalid")
    if any(not t["wired"] for t in tools if t["kind"] == "m3"):
        flags.append("unwired_m3_tools_present")

    if len(supported) < 3:
        if len(qps) >= 3:
            qp_caveat = (
                "QP-90-only: the encoder or decoder accepted only "
                f"{supported} from the requested {qps}. An older build pins "
                "base_q_idx = 90 until the coefficient-CDF qctx work lands. "
                "A standard BD-rate fit needs >=3 QPs."
            )
        else:
            qp_caveat = (
                f"Only QP(s) {supported} were requested/measured. A standard "
                "BD-rate fit needs >=3 QPs; pass a wider --qps set."
            )
    else:
        qp_caveat = (
            f"Multi-QP run with supported QPs {supported}. The primary BD-rate "
            f"fit uses {primary_qps}; the all-QP fit is a sensitivity check. "
            "QP60 is near CVVDP saturation and QP235 is a near-collapse point, "
            "so they are excluded from the primary fit but retained in the "
            "tables and plots."
        )
    caveats = [
        "One frame per operating point (the M1/M3 path is still-picture only). "
        "This is not a video-sequence result.",
        qp_caveat,
        "BD-rate here is a QP sweep on one frame per resolution, not a "
        "multi-frame corpus. It measures the rate-quality trade at fixed "
        "content; it is not a sequence-average result.",
        "CVVDP is bounded and nonlinear. If every per-QP CVVDP delta is below "
        f"{CVVDP_NOISE_FLOOR_JOD:.2f} JOD, the CVVDP BD-rate is marked "
        "`within_noise` and is not allowed to flip the verdict. "
        "`metric_dependent` means PSNR-Y and CVVDP disagree beyond their noise "
        "floors.",
        "Quality and bitrate are machine-independent, but encode speed and "
        "encode-fps are not target validation. "
        f"Hardware class is `{hardware['class']}`.",
        "The 640x360 input is a downscale of the same 4K source; it is a "
        "low-resolution benchmark, not a substitute for native 1080p.",
        "Content is one smooth TV-demo shot (LG QNED Shine). Absolute rates are "
        "easy-content numbers; do not generalize to hard footage.",
        "CVVDP is fixed to `standard_fhd` on the device shown in the report. "
        "JOD is not comparable across display models or viewing conditions.",
        "RDO here is rate-aware intra-mode decision (`--m1-rdo-intra`), not "
        "partition/inter Lagrangian RDO. Its anchor is the full-intra control, "
        "so the table's `delta_vs_anchor` isolates the rate term; the "
        "baseline-relative delta includes the full-intra search.",
        "CDEF/CCSO/RDOQ/TX-type are each enabled alone on top of the M1 "
        "baseline. No joint optimization or filter-order ablation is implied.",
        "A tool with no verified stream is reported as unmeasured; no metric or "
        "BD-rate is claimed for it.",
        "Verdict thresholds are local to this report and deliberately "
        "conservative. `tradeoff` means the iso-quality sign is unresolved.",
    ]
    if not comparison_valid:
        caveats.append("Comparison invalid: " + "; ".join(comparison_errors))

    # Plots.
    plot_info: dict[str, dict] = {}
    if not args.no_plot:
        for res in RESOLUTIONS:
            plot_info[res] = plot_rd(
                points, res, out / f"rd_curve_{res}.png"
            )

    # Re-check the live build after the sweep. The snapshot above is what every
    # encode used; this only records whether the source binary moved underneath
    # us and whether the source tree changed.
    encoder_source_sha_end = sha256_file(encoder_source)
    encoder_info["original_sha256_at_end"] = encoder_source_sha_end
    encoder_info["source_unchanged_during_run"] = (
        encoder_source_sha_end == encoder_info["original_sha256_at_start"]
    )
    git_info_end = git_provenance()
    if not encoder_info["source_unchanged_during_run"]:
        flags.append("encoder_binary_changed_during_run_snapshot_used")
        caveats.append(
            "The live build binary changed while the sweep was running. All "
            "encodes used the recorded snapshot, so the comparison is internally "
            "consistent; rerun to score the newer build."
        )
    if (
        git_info.get("commit") != git_info_end.get("commit")
        or git_info.get("dirty") != git_info_end.get("dirty")
    ):
        flags.append("source_tree_changed_during_run")
        caveats.append(
            "The source tree changed during the sweep. The report is tied to the "
            "recorded encoder snapshot and start/end commits."
        )

    manifest = {
        "schema": SCHEMA,
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "command": sys.argv,
        "root": str(ROOT),
        "git": git_info,
        "git_end": git_info_end,
        "source_changed_during_run": (
            git_info.get("commit") != git_info_end.get("commit")
            or git_info.get("dirty") != git_info_end.get("dirty")
        ),
        "encoder": encoder_info,
        "decoder": {"path": str(dav2d), "version": dec_version},
        "cvvdp": {
            "version": next(iter({p["metrics"].get("cvvdp_version") for p in ok_points}), None),
            "display": next(iter({p["metrics"].get("cvvdp_display") for p in ok_points}), None),
            "device": next(iter({p["metrics"].get("cvvdp_device") for p in ok_points}), None),
        },
        "host": host,
        "hardware": hardware,
        "qps_requested": qps,
        "qp_probe": qp_probe,
        "supported_qps": supported,
        "supported_qps_by_resolution": supported_by_res,
        "resolutions": list(RESOLUTIONS),
        "sources": sources,
        "cross_resolution_check": cross,
        "tools": tools,
        "comparison": {
            "valid": comparison_valid,
            "errors": comparison_errors,
            "allow_mixed_hardware": args.allow_mixed_hardware,
        },
    }
    write_json(out / "manifest.json", manifest)

    m3_tools = [t for t in tool_summary if t["kind"] == "m3"]
    verdict_counts: dict[str, int] = {}
    for tool in m3_tools:
        verdict_counts[tool["overall_verdict"]] = (
            verdict_counts.get(tool["overall_verdict"], 0) + 1
        )
    if not comparison_valid:
        headline = "comparison invalid — no net-win/loss claim"
    else:
        parts = [
            f"{verdict_counts[key]} {key}"
            for key in (
                "net_win",
                "neutral",
                "tradeoff",
                "loss",
                "metric_dependent",
                "mixed",
                "unmeasured",
            )
            if verdict_counts.get(key)
        ]
        headline = ", ".join(parts) if parts else "no measured M3 tool"

    summary = {
        "schema": SCHEMA,
        "generated_utc": manifest["generated_utc"],
        "comparison_valid": comparison_valid,
        "comparison_errors": comparison_errors,
        "target_validation": hardware["target_validation"],
        "hardware_class": hardware["class"],
        "supported_qps": supported,
        "flags": flags,
        "tools": tool_summary,
        "verdict_counts": verdict_counts,
        "headline": headline,
        "points": points,
        "bd_rate": bd,
        "caveats": caveats,
        "verdict_thresholds": {
            "rate_flat_pct": RATE_FLAT_PCT,
            "quality_up_cvvdp_jod": Q_UP_CVVDP_JOD,
            "quality_up_psnr_db": Q_UP_PSNR_DB,
            "quality_down_psnr_db": Q_DOWN_PSNR_DB,
            "quality_down_cvvdp_jod": Q_DOWN_CVVDP_JOD,
            "quality_down_ssim": Q_DOWN_SSIM,
            "max_win_rate_cost_pct": MAX_WIN_RATE_COST_PCT,
        },
    }
    write_json(out / "summary.json", summary)
    write_json(out / "bd_rate.json", bd)

    bd_md: list[str] = [
        "# BD-rate vs M1 baseline",
        "",
        f"Generated: `{manifest['generated_utc']}`  ",
        f"Method: `{bd['method']}`  ",
        f"Anchor: `m1_baseline` (RDO rows use the full-intra control)  ",
        f"Primary QPs: `{bd.get('primary_qps')}`  ",
        f"All supported QPs: `{supported}`  ",
        "",
    ]
    if bd["status"] != "ok":
        bd_md += [
            f"**Not computable:** {bd['blocking_reason']}",
            "",
            "| Tool | Resolution | Metric | BD-rate | Status |",
            "|---|---|---|---:|---|",
        ]
        for tid, tool in bd["tools"].items():
            if not tool["wired"]:
                bd_md.append(f"| {tool['display']} | — | — | — | not wired |")
                continue
            for res in RESOLUTIONS:
                block = tool["resolutions"][res]["psnr_y"]
                bd_md.append(
                    f"| {tool['display']} | {res} | PSNR-Y | — | "
                    f"{block.get('reason', 'unavailable')} |"
                )
        bd_md += [
            "",
            "Pointwise rate/quality deltas are in `summary.md` and "
            "`summary.json`. A one-QP rate delta is not a BD-rate.",
        ]
    else:
        bd_md += [
            "| Tool | Resolution | Metric | BD-rate (core) | BD-PSNR (core) | All-QP BD-rate | Noise |",
            "|---|---|---|---:|---:|---:|---|",
        ]
        for tid, tool in bd["tools"].items():
            if not tool["wired"]:
                continue
            for res in RESOLUTIONS:
                for key, label in (("psnr_y", "PSNR-Y"), ("cvvdp", "CVVDP")):
                    block = tool["resolutions"][res][key]
                    br = fmt_pct(block.get("bd_rate_pct"))
                    bp = fmt_db(block.get("bd_psnr_db")) if key == "psnr_y" else "—"
                    sens = block.get("sensitivity") or {}
                    all_br = fmt_pct(sens.get("bd_rate_pct"))
                    noise = "within_noise" if block.get("within_noise") else "—"
                    bd_md.append(
                        f"| {tool['display']} | {res} | {label} | {br} | {bp} | "
                        f"{all_br} | {noise} |"
                    )
    (out / "bd_rate.md").write_text("\n".join(bd_md) + "\n")

    write_markdown(out / "summary.md", manifest, summary, points, bd, plot_info)

    print("", file=sys.stderr)
    print(f"summary:  {out / 'summary.md'}", file=sys.stderr)
    print(f"json:     {out / 'summary.json'}", file=sys.stderr)
    print(f"bd-rate:  {out / 'bd_rate.md'}", file=sys.stderr)
    encoded_points = [p for p in points if p["status"] != "not_wired"]
    print(
        f"comparison_valid={comparison_valid} target_validation={hardware['target_validation']} "
        f"verified_points={len(ok_points)}/{len(encoded_points)} "
        f"(unwired_tools={len([t for t in tools if not t['wired']])})",
        file=sys.stderr,
    )

    if not comparison_valid:
        return 2
    if len(ok_points) != len(encoded_points):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
