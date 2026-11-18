#!/usr/bin/env python3
"""Per-tool evidence for the synthetic probe sweep.

The sweep (tools/tool_quality_run.py) answers "how big is the gap".  This script
answers "which implementation is responsible", using only first-party encoder
diagnostics that already exist in the tree:

  * IVF frame sizes: key-picture share vs P-chain bytes for every encoder.
  * Decoded luma PSNR per frame: key quality vs the P-chain delta.
  * AV2ENC_BLOCKDUMP: exact-MV rate, skip rate, non-zero-MV skips, residual
    leaves.  This separates "the ME search failed" from "the leaf RD chose
    skip / the residual coder could not repair the reference".
  * --video-trace: key TX-size histogram and BAWP application count.
  * Small ablations with existing flags: BAWP off on the fade, quarter-pel
    instead of eighth on me_eighth, leaf-RD off on me_half.

No encoder source is modified and no new syntax/tool is added.

Usage:
  tools/tool_probe_evidence.py --results build/tool-quality/results.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))
import tool_quality_run as tqr  # noqa: E402

NIGHTLY_FLAGS = [
    "--video-rd-preset", "quality", "--video-ccso-rd", "--video-edge-inter",
    "--video-partition-rd", "--video-key-qp-offset", "-27", "--video-rdoq",
    "--video-edge-mask", "--video-cdef", "--video-cdef-rd",
    "--video-qp-lambda", "--video-tile-flush-min", "--video-ref-signal-compact",
    "--video-tile-group-compact", "--video-ccso-header-rd",
    "--video-ccso-bands", "6", "--video-ccso-lambda", "22",
    "--video-ccso-refit", "2", "--video-ccso-unit-bits", "2",
    "--video-bawp", "--video-bawp-carrier", "--video-leaf-rd",
    "--video-subpel-eighth", "--video-key-intra-txpart",
    "--avg-cdf", "--adaptive-cdf", "--cdf-continuity",
    "--video-drift-refresh",
]


def run(cmd: list[str], env: dict | None = None, timeout: int = 7200):
    e = os.environ.copy()
    if env:
        e.update(env)
    return subprocess.run(cmd, cwd=ROOT, env=e, text=True,
                          capture_output=True, timeout=timeout)


def read_luma(path: Path) -> tuple[int, int, list]:
    with path.open("rb") as f:
        header = f.readline().decode("ascii", "replace")
        m = re.search(r"W(\d+)\s+H(\d+)", header)
        w, h = int(m.group(1)), int(m.group(2))
        frames = []
        while True:
            line = f.readline()
            if not line or not line.startswith(b"FRAME"):
                break
            y = f.read(w * h)
            f.read(w * h // 2)
            if len(y) != w * h:
                break
            frames.append(y)
    return w, h, frames


def psnr(a: bytes, b: bytes) -> float:
    import numpy as np
    x = np.frombuffer(a, dtype=np.uint8).astype(np.float64)
    y = np.frombuffer(b, dtype=np.uint8).astype(np.float64)
    mse = float(((x - y) ** 2).mean())
    return 99.0 if mse == 0 else 10.0 * math.log10(255.0 * 255.0 / mse)


def frame_psnr_summary(ref: Path, dec: Path) -> dict:
    w, h, r = read_luma(ref)
    w2, h2, d = read_luma(dec)
    n = min(len(r), len(d))
    vals = [psnr(r[i], d[i]) for i in range(n)]
    if not vals:
        return {}
    p = vals[1:]
    return {
        "frames": n,
        "key": round(vals[0], 3),
        "p_mean": round(sum(p) / len(p), 3) if p else None,
        "p_min": round(min(p), 3) if p else None,
        "p_max": round(max(p), 3) if p else None,
        "delta_p_key": round(sum(p) / len(p) - vals[0], 3) if p else None,
    }


def ivf_split(path: Path) -> dict:
    sizes = tqr.ivf_frame_sizes(path)
    if not sizes:
        return {}
    total = sum(sizes)
    return {
        "coded_frames": len(sizes),
        "key_bytes": sizes[0],
        "p_bytes_mean": round(sum(sizes[1:]) / max(1, len(sizes) - 1), 1),
        "total_bytes": total,
        "key_share": round(sizes[0] / total, 4) if total else None,
    }


def parse_blocks(path: Path, key_leaves: int, expected_mv: tuple | None) -> dict:
    lines = path.read_text(errors="replace").splitlines()
    p_lines = lines[key_leaves:] if key_leaves <= len(lines) else []
    inter = skip = nz_skip = residual = exact = 0
    mv_hist: dict[str, int] = {}
    for line in p_lines:
        f = line.split()
        if len(f) < 15:
            continue
        is_inter = int(f[4])
        mv_x, mv_y, sk = int(f[8]), int(f[9]), int(f[10])
        eob = int(f[11]) + int(f[12]) + int(f[13])
        if is_inter:
            inter += 1
            if sk:
                skip += 1
                if mv_x or mv_y:
                    nz_skip += 1
            else:
                residual += 1
            if expected_mv is not None and (mv_x, mv_y) == expected_mv:
                exact += 1
            key = f"{mv_x},{mv_y}"
            mv_hist[key] = mv_hist.get(key, 0) + 1
    return {
        "key_leaves": key_leaves,
        "p_leaves": len(p_lines),
        "inter": inter,
        "skip": skip,
        "skip_pct": round(100.0 * skip / inter, 1) if inter else None,
        "nz_skip": nz_skip,
        "nz_skip_pct": round(100.0 * nz_skip / max(1, skip), 1) if skip else None,
        "residual": residual,
        "exact_mv": exact,
        "exact_mv_pct": round(100.0 * exact / inter, 1) if inter else None,
        "mv_top": sorted(mv_hist.items(), key=lambda kv: -kv[1])[:4],
        "expected_mv": list(expected_mv) if expected_mv is not None else None,
    }


def tx_histogram(trace_text: str) -> dict:
    hist: dict[str, int] = {}
    for m in re.finditer(r"intra tx part r=\d+ c=\d+ (\d+x\d+)", trace_text):
        hist[m.group(1)] = hist.get(m.group(1), 0) + 1
    total = sum(hist.values())
    return {
        "total": total,
        "hist": dict(sorted(hist.items(), key=lambda kv: -kv[1])),
        "pct": {k: round(100.0 * v / total, 1) for k, v in
                sorted(hist.items(), key=lambda kv: -kv[1])} if total else {},
    }


def bawp_usage(trace_text: str) -> dict:
    total = applied = 0
    for m in re.finditer(r"cand bawp alpha=.* applied=(\d)", trace_text):
        total += 1
        applied += int(m.group(1))
    return {"candidates": total, "applied": applied,
            "applied_pct": round(100.0 * applied / total, 1) if total else None}


class Evidence:
    def __init__(self, args):
        self.args = args
        self.results = json.loads(args.results.read_text())["results"]
        self.probe_dir = args.probe_dir
        self.out = args.out_dir
        self.out.mkdir(parents=True, exist_ok=True)
        self.work = self.out / "evidence"
        self.work.mkdir(exist_ok=True)

    def score_stream(self, probe: str, tag: str, ivf: Path) -> dict:
        probe_path = self.probe_dir / f"{probe}.y4m"
        dec = self.work / f"{tag}.dec.y4m"
        n = tqr.decode_av2(str(self.args.avmdec), ivf, probe_path, dec)
        q = self.work / f"{tag}.q.json"
        e = self.work / f"{tag}.e.json"
        env = {"AVMDEC": str(self.args.avmdec)}
        for cmd in (
            [sys.executable, str(TOOLS / "quality.py"), "--ref", str(probe_path),
             "--test", str(dec), "--json", str(q)],
            [sys.executable, str(TOOLS / "extra_metrics.py"), "--ref",
             str(probe_path), "--test", str(dec), "--json", str(e)],
        ):
            p = run(cmd, env=env)
            if p.returncode:
                return {"error": (p.stdout + p.stderr)[-2000:]}
        p = run([sys.executable, str(TOOLS / "kpi.py"),
                 "--quality-json", str(q), "--extra-json", str(e),
                 "--json", str(self.work / f"{tag}.kpi.json")])
        if p.returncode:
            return {"error": (p.stdout + p.stderr)[-2000:]}
        quality = json.loads(q.read_text())
        extra = json.loads(e.read_text())
        kpi = json.loads((self.work / f"{tag}.kpi.json").read_text())
        return {
            "frames": n,
            "kbps": round(ivf.stat().st_size * 8.0 * 30.0 / n / 1000.0, 3),
            "bytes": ivf.stat().st_size,
            "cvvdp": quality["cvvdp_jod"],
            "psnr_y": quality["psnr"]["psnr_y"],
            "butteraugli": extra["butteraugli_mean"],
            "ssimulacra2": extra["ssimulacra2_mean"],
            "kpi": round(kpi["kpi"], 2),
        }

    def encode_nightly_variant(self, probe: str, tag: str, qp: int,
                               flags: list[str]) -> Path:
        probe_path = self.probe_dir / f"{probe}.y4m"
        out = self.work / f"{tag}.ivf"
        cmd = [str(self.args.av2enc), str(probe_path), "-o", str(out),
               "--video", "--frames", str(self.args.frames), "--qp", str(qp),
               "--cpu-threads", "8"] + flags
        p = run(cmd)
        if p.returncode or not out.is_file():
            raise RuntimeError(f"variant {tag} failed:\n{p.stdout[-2000:]}\n"
                               f"{p.stderr[-2000:]}")
        return out

    def av2_blocks(self, probe: str, enc_name: str, qp: int) -> dict:
        enc = tqr.ENCODERS[enc_name]
        probe_path = self.probe_dir / f"{probe}.y4m"
        dump = self.work / f"{probe}_{enc_name}_q{qp}.blocks"
        key_dump = self.work / f"{probe}_{enc_name}_q{qp}_key.blocks"
        ivf = self.work / f"{probe}_{enc_name}_q{qp}.ivf"
        env = dict(enc.env)
        env["AV2ENC_BLOCKDUMP"] = str(dump)
        cmd = enc.command(str(self.args.av2enc), self.args.svt, str(self.args.avm),
                          probe_path, ivf, qp, self.args.frames)
        p = run(cmd, env=env)
        if p.returncode or not dump.is_file():
            return {"error": f"block dump failed: {p.stderr[-1000:]}"}
        env1 = dict(env)
        env1["AV2ENC_BLOCKDUMP"] = str(key_dump)
        cmd1 = enc.command(str(self.args.av2enc), self.args.svt, str(self.args.avm),
                           probe_path, self.work / f"{probe}_{enc_name}_key.ivf",
                           qp, 1)
        p1 = run(cmd1, env=env1)
        if p1.returncode or not key_dump.is_file():
            return {"error": f"key block dump failed: {p1.stderr[-1000:]}"}
        key_leaves = len(key_dump.read_text(errors="replace").splitlines())
        meta_path = self.probe_dir / f"{probe}.json"
        expected = None
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text())
            units = meta.get("translation_units_eighth")
            if units:
                expected = (int(units[0]), int(units[1]))
        rec = parse_blocks(dump, key_leaves, expected)
        scored = (self.args.results.parent / "work" / probe / enc_name
                  / f"q{qp}.ivf")
        rec["matches_scored_ivf_bytes"] = (
            scored.is_file() and ivf.stat().st_size == scored.stat().st_size)
        return rec

    def trace(self, probe: str, enc_name: str, qp: int, frames: int,
              extra: list[str] | None = None) -> str:
        enc = tqr.ENCODERS[enc_name]
        probe_path = self.probe_dir / f"{probe}.y4m"
        out = self.work / f"{probe}_{enc_name}_trace.ivf"
        cmd = enc.command(str(self.args.av2enc), self.args.svt, str(self.args.avm),
                          probe_path, out, qp, frames) + ["--video-trace"]
        if extra:
            cmd += extra
        p = run(cmd, env=enc.env)
        return p.stdout + "\n" + p.stderr

    def run(self) -> dict:
        out = {"target_kbps": self.args.target_kbps, "probes": {}}
        for key, result in sorted(self.results.items()):
            probe, enc_name = key.split("/", 1)
            if "at_target" not in result:
                continue
            at = result["at_target"]
            nearest = min(result["points"],
                          key=lambda p: abs(p["kbps"] - self.args.target_kbps))
            qp = nearest["qp"]
            entry = {
                "at_target": {k: at[k] for k in
                              ("kbps", "cvvdp_jod", "butteraugli",
                               "ssimulacra2", "psnr_y", "kpi")},
                "qp": qp,
                "point_kbps": nearest["kbps"],
                "ivf": ivf_split(self.args.results.parent / "work" / probe
                                 / enc_name / f"q{qp}.ivf"),
            }
            dec = (self.args.results.parent / "work" / probe / enc_name
                   / f"q{qp}.dec.y4m")
            if dec.is_file():
                entry["per_frame"] = frame_psnr_summary(
                    self.probe_dir / f"{probe}.y4m", dec)
            if enc_name in ("nightly", "eff-perc"):
                entry["blocks"] = self.av2_blocks(probe, enc_name, qp)
            out["probes"].setdefault(probe, {})[enc_name] = entry

        # Tool-specific traces and ablations.
        for probe, enc_name in (("fade", "nightly"), ("me_eighth", "nightly"),
                                ("me_half", "nightly"),
                                ("detail_edges", "nightly")):
            res = self.results.get(f"{probe}/{enc_name}")
            if not res or "at_target" not in res:
                continue
            nearest = min(res["points"],
                          key=lambda p: abs(p["kbps"] - self.args.target_kbps))
            qp = nearest["qp"]
            e = out["probes"][probe][enc_name]
            if probe == "fade":
                text = self.trace(probe, enc_name, qp, 6)
                e["bawp_trace"] = bawp_usage(text)
                # Ablation: same QP, BAWP off.
                flags = [f for f in NIGHTLY_FLAGS
                         if f not in ("--video-bawp", "--video-bawp-carrier")]
                ivf = self.encode_nightly_variant(probe, f"{probe}_nobawp_q{qp}",
                                                  qp, flags)
                e["ablation_nobawp"] = self.score_stream(
                    probe, f"{probe}_nobawp_q{qp}", ivf)
            elif probe == "me_eighth":
                text = self.trace(probe, enc_name, qp, 6)
                e["tx_or_mv_trace_ok"] = bool(text)
                # Ablation: quarter-pel frame precision, no eighth stage.
                flags = [f for f in NIGHTLY_FLAGS
                         if f != "--video-subpel-eighth"]
                flags += ["--video-mv-precision", "quarter"]
                ivf = self.encode_nightly_variant(probe, f"{probe}_quarter_q{qp}",
                                                  qp, flags)
                e["ablation_quarter"] = self.score_stream(
                    probe, f"{probe}_quarter_q{qp}", ivf)
            elif probe == "me_half":
                # Ablation: leaf RD off, everything else nightly.
                flags = [f for f in NIGHTLY_FLAGS if f != "--video-leaf-rd"]
                ivf = self.encode_nightly_variant(probe, f"{probe}_noleafrd_q{qp}",
                                                  qp, flags)
                e["ablation_no_leaf_rd"] = self.score_stream(
                    probe, f"{probe}_noleafrd_q{qp}", ivf)
            elif probe == "detail_edges":
                text = self.trace(probe, enc_name, qp, 1)
                e["key_tx"] = tx_histogram(text)
        (self.out / "evidence.json").write_text(
            json.dumps(out, indent=2, sort_keys=True) + "\n")
        print(json.dumps({p: {e: {k: v for k, v in d.items()
                                  if k in ("at_target", "per_frame")}
                              for e, d in es.items()}
                          for p, es in out["probes"].items()}, indent=2))
        return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path,
                    default=ROOT / "build" / "tool-quality" / "results.json")
    ap.add_argument("--probe-dir", type=Path,
                    default=ROOT / "build" / "tool-probes")
    ap.add_argument("--out-dir", type=Path,
                    default=ROOT / "build" / "tool-quality")
    ap.add_argument("--av2enc", type=Path, default=tqr.DEFAULT_AV2ENC)
    ap.add_argument("--avm", type=Path, default=tqr.DEFAULT_AVM)
    ap.add_argument("--avmdec", type=Path, default=tqr.DEFAULT_AVMDEC)
    ap.add_argument("--svt", default=tqr.DEFAULT_SVT)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--target-kbps", type=float, default=100.0)
    args = ap.parse_args()
    Evidence(args).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
