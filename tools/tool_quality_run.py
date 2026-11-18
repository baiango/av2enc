#!/usr/bin/env python3
"""Encode the synthetic tool probes with four encoders and score the gap.

This is the measurement half of docs/tool-quality.md.  It generates (or reuses)
the clips from tools/tool_probes.py, finds a two-QP bracket around the target
rate for every encoder/probe, scores the bracket with the project's own
quality stack, and interpolates K at the target rate.

Encoders:
  nightly   av2enc --nightly
  eff-perc  av2enc --nightly + AV2ENC_EFF_PERC=1 + the documented drift recipe
  svt       SVT-AV1 4.3.0 preset 0, CRF (`--keyint -1 --scd 0`)
  avm       AVM e6f99d3 --end-usage=q --cpu-used=6

Scoring:
  * av2enc/AVM IVF -> AVM `avmdec` -> tools/quality.py (CVVDP + PSNR)
  * SVT IVF -> dav1d -> tools/quality.py --test
  * tools/extra_metrics.py for Butteraugli / SSIMULACRA2
  * tools/kpi.py for K

Everything is cached under --out-dir: rerunning resumes.  No encoder source is
modified and no third-party media is read.

Usage:
  tools/tool_quality_run.py --av2enc build-int/av2enc
  tools/tool_quality_run.py --only me_half fade --encoders nightly avm
  tools/tool_quality_run.py --list
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
PROBE_TOOL = TOOLS / "tool_probes.py"
DEFAULT_AV2ENC = ROOT / "build-int" / "av2enc"
DEFAULT_AVM = Path(os.environ.get("AVMENC") or shutil.which("avmenc") or
                   "/nonexistent/avmenc")
DEFAULT_AVMDEC = TOOLS / "bin" / "avmdec"
DEFAULT_SVT = "SvtAv1EncApp"
DEFAULT_DAV1D = "dav1d"
FPS = 30.0
PROBE_ORDER = [
    "me_half", "me_quarter", "me_eighth", "fade",
    "static_repeat", "near_static", "detail_edges", "grain",
]


def run(cmd: list[str], env: dict | None = None, timeout: int = 7200) -> subprocess.CompletedProcess:
    e = os.environ.copy()
    if env:
        e.update(env)
    return subprocess.run(cmd, cwd=ROOT, env=e, text=True,
                          capture_output=True, timeout=timeout)


def kbps_of(path: Path, frames: int, fps: float = FPS) -> float:
    return path.stat().st_size * 8.0 * fps / frames / 1000.0


def ivf_frame_sizes(path: Path) -> list[int]:
    data = path.read_bytes()
    if len(data) < 32 or data[:4] != b"DKIF":
        return []
    sizes = []
    off = 32
    while off + 12 <= len(data):
        (size,) = struct.unpack_from("<I", data, off)
        off += 12
        if size == 0 or off + size > len(data):
            break
        sizes.append(size)
        off += size
    return sizes


def decode_av2(avmdec: str, ivf: Path, ref: Path, out_y4m: Path) -> int:
    header = ref.open("rb").readline().decode("ascii", "replace").strip()
    import re
    m = re.search(r"W(\d+)\s+H(\d+)", header)
    if not m:
        raise RuntimeError(f"bad Y4M header: {header}")
    w, h = int(m.group(1)), int(m.group(2))
    if "C420" not in header:
        raise RuntimeError("probe scoring expects C420")
    raw = out_y4m.with_suffix(".i420")
    p = run([avmdec, "--i420", "--summary", "-o", str(raw), str(ivf)])
    if p.returncode != 0 or not raw.is_file():
        raise RuntimeError(f"avmdec failed for {ivf}:\n{p.stdout}\n{p.stderr}")
    data = raw.read_bytes()
    fsz = w * h * 3 // 2
    if len(data) == 0 or len(data) % fsz:
        raise RuntimeError(f"avmdec output {len(data)} B is not {fsz}-B frames")
    with out_y4m.open("wb") as fh:
        fh.write((header + "\n").encode("ascii"))
        for i in range(0, len(data), fsz):
            fh.write(b"FRAME\n")
            fh.write(data[i:i + fsz])
    return len(data) // fsz


def decode_svt(dav1d: str, ivf: Path, out_y4m: Path) -> int:
    p = run([dav1d, "-q", "-i", str(ivf), "-o", str(out_y4m)])
    if p.returncode != 0 or not out_y4m.is_file():
        raise RuntimeError(f"dav1d failed for {ivf}:\n{p.stdout}\n{p.stderr}")
    data = out_y4m.read_bytes()
    return data.count(b"FRAME\n")


@dataclass
class Encoder:
    name: str
    qp_lo: int
    qp_hi: int
    start: dict = field(default_factory=dict)
    env: dict = field(default_factory=dict)
    kind: str = "av2"

    def start_qp(self, probe: str) -> int:
        return self.start.get(probe, self.start.get("_", 195))

    def command(self, bin_av2enc: str, svt: str, avm: str,
                probe: Path, out: Path, qp: int, frames: int) -> list[str]:
        if self.kind == "svt":
            return [svt, "-i", str(probe), "-b", str(out), "--preset", "0",
                    "--crf", str(qp), "--fps-num", "30", "--fps-denom", "1",
                    "--keyint", "-1", "--scd", "0", "--progress", "0"]
        if self.kind == "avm":
            return [avm, str(probe), "-o", str(out), "--ivf",
                    "--end-usage=q", f"--qp={qp}", "--cpu-used=6",
                    "--threads=8", f"--limit={frames}", "--psnr=1"]
        cmd = [bin_av2enc, str(probe), "-o", str(out), "--video",
               "--frames", str(frames), "--qp", str(qp),
               "--cpu-threads", "8", "--nightly"]
        if self.name == "eff-perc":
            cmd += ["--video-drift-refresh-threshold", "1.5",
                    "--video-drift-refresh-budget", "3250"]
        return cmd


ENCODERS = {
    "nightly": Encoder("nightly", 1, 255,
                       start={"me_half": 140, "me_quarter": 140,
                              "me_eighth": 140, "fade": 170,
                              "static_repeat": 170, "near_static": 170,
                              "detail_edges": 180, "grain": 170, "_": 185}),
    "eff-perc": Encoder("eff-perc", 1, 255,
                        env={"AV2ENC_EFF_PERC": "1"},
                        start={"me_half": 145, "me_quarter": 145,
                               "me_eighth": 145, "fade": 170,
                               "static_repeat": 170, "near_static": 170,
                               "detail_edges": 180, "grain": 170, "_": 185}),
    "svt": Encoder("svt", 1, 70, kind="svt",
                   start={"me_half": 38, "me_quarter": 38, "me_eighth": 37,
                          "fade": 53, "static_repeat": 57, "near_static": 57,
                          "detail_edges": 60, "grain": 50, "_": 57}),
    "avm": Encoder("avm", 0, 255, kind="avm",
                   start={"me_half": 75, "me_quarter": 75, "me_eighth": 75,
                          "fade": 185, "static_repeat": 195,
                          "near_static": 195, "detail_edges": 225,
                          "grain": 215, "_": 200}),
}


class Runner:
    def __init__(self, args):
        self.args = args
        self.out = args.out_dir
        self.work = self.out / "work"
        self.work.mkdir(parents=True, exist_ok=True)
        self.target = args.target_kbps

    # -- encode ---------------------------------------------------------
    def encode_paths(self, probe: str, enc: str, qp: int) -> tuple[Path, Path]:
        d = self.work / probe / enc
        d.mkdir(parents=True, exist_ok=True)
        return d / f"q{qp}.ivf", d / f"q{qp}.json"

    def encode_one(self, probe: str, enc_name: str, qp: int) -> dict:
        enc = ENCODERS[enc_name]
        ivf, meta_path = self.encode_paths(probe, enc_name, qp)
        if meta_path.is_file() and ivf.is_file():
            meta = json.loads(meta_path.read_text())
            if meta.get("bytes") == ivf.stat().st_size:
                return meta
        probe_path = self.args.probe_dir / f"{probe}.y4m"
        t0 = time.perf_counter()
        cmd = enc.command(str(self.args.av2enc), self.args.svt, str(self.args.avm),
                          probe_path, ivf, qp, self.args.frames)
        p = run(cmd, env=enc.env)
        dt = time.perf_counter() - t0
        if p.returncode != 0 or not ivf.is_file():
            raise RuntimeError(
                f"{enc_name} {probe} qp{qp} failed (exit {p.returncode})\n"
                f"cmd: {' '.join(cmd)}\n{p.stdout[-4000:]}\n{p.stderr[-4000:]}")
        sizes = ivf_frame_sizes(ivf)
        meta = {
            "probe": probe, "encoder": enc_name, "qp": qp,
            "bytes": ivf.stat().st_size,
            "kbps": kbps_of(ivf, self.args.frames),
            "frames": len(sizes) if sizes else None,
            "frame_bytes": sizes,
            "encode_seconds": round(dt, 3),
            "encode_fps": round(self.args.frames / dt, 2) if dt > 0 else None,
            "cmd": cmd,
            "env": enc.env,
        }
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        return meta

    def find_bracket(self, probe: str, enc_name: str) -> dict:
        enc = ENCODERS[enc_name]
        points: dict[int, dict] = {}

        def add(q: int) -> dict:
            q = max(enc.qp_lo, min(enc.qp_hi, int(q)))
            if q not in points:
                points[q] = self.encode_one(probe, enc_name, q)
            return points[q]

        q = enc.start_qp(probe)
        add(q)
        for _ in range(14):
            rates = [(p["qp"], p["kbps"]) for p in points.values()]
            up = [r for _, r in rates if r >= self.target]
            lo = [r for _, r in rates if r <= self.target]
            if up and lo:
                break
            if len(rates) >= 2:
                xs = [q for q, _ in rates]
                ys = [math.log(max(r, 1e-6)) for _, r in rates]
                mx = sum(xs) / len(xs)
                my = sum(ys) / len(ys)
                den = sum((x - mx) ** 2 for x in xs)
                slope = (sum((x - mx) * (y - my)
                             for x, y in zip(xs, ys)) / den) if den else 0.0
                if slope < -1e-5:
                    next_q = q + (math.log(self.target) -
                                  math.log(max(points[q]["kbps"], 1e-6))) / slope
                else:
                    next_q = q + (-5 if points[q]["kbps"] > self.target else 5)
                step_cap = 12 if enc.kind == "svt" else 35
                next_q = max(q - step_cap, min(q + step_cap, next_q))
            else:
                next_q = q + (-4 if points[q]["kbps"] > self.target else 4)
            nq = int(round(next_q))
            step = 1 if points[q]["kbps"] > self.target else -1
            while nq in points and enc.qp_lo <= nq + step <= enc.qp_hi:
                nq += step
            nq = max(enc.qp_lo, min(enc.qp_hi, nq))
            if nq in points:
                break
            q = nq
            add(q)

        pts = sorted(points.values(), key=lambda p: p["kbps"])
        above = [p for p in pts if p["kbps"] >= self.target]
        below = [p for p in pts if p["kbps"] <= self.target]
        if above and below:
            # Refine the bracket until it is tight (<= 6% of target) or the
            # two QPs are adjacent.  A wide bracket makes the rate
            # interpolation, not the encoder, dominate the reported K.
            for _ in range(4):
                a = min(above, key=lambda p: p["kbps"])
                b = max(below, key=lambda p: p["kbps"])
                if (a["kbps"] - b["kbps"] <= max(2.0, 0.06 * self.target)
                        or abs(a["qp"] - b["qp"]) <= 1):
                    break
                la = math.log(max(a["kbps"], 1e-6))
                lb = math.log(max(b["kbps"], 1e-6))
                denom = la - lb
                t = ((math.log(self.target) - lb) / denom) if denom else 0.5
                nq = int(round(b["qp"] + t * (a["qp"] - b["qp"])))
                if nq in points:
                    nq = a["qp"] - 1 if a["qp"] - 1 > b["qp"] else b["qp"] + 1
                nq = max(enc.qp_lo, min(enc.qp_hi, nq))
                if nq in points:
                    break
                add(nq)
                pts = sorted(points.values(), key=lambda p: p["kbps"])
                above = [p for p in pts if p["kbps"] >= self.target]
                below = [p for p in pts if p["kbps"] <= self.target]
                if not (above and below):
                    break
            if above and below:
                a = min(above, key=lambda p: p["kbps"])
                b = max(below, key=lambda p: p["kbps"])
                bracketed = True
            else:
                a = above[0] if above else below[-1]
                b = below[-1] if below else above[0]
                bracketed = False
        elif above:
            a = above[0]
            b = above[1] if len(above) > 1 else above[0]
            bracketed = False
        else:
            a = below[-1]
            b = below[-2] if len(below) > 1 else below[-1]
            bracketed = False
        return {
            "probe": probe, "encoder": enc_name, "target_kbps": self.target,
            "bracketed": bracketed,
            "points": sorted(points.values(), key=lambda p: p["qp"]),
            "bracket": sorted([b, a], key=lambda p: p["kbps"]),
        }

    # -- scoring ---------------------------------------------------------
    def score_one(self, probe: str, enc_name: str, point: dict) -> dict:
        qp = point["qp"]
        d = self.work / probe / enc_name
        score_path = d / f"q{qp}.score.json"
        if score_path.is_file():
            return json.loads(score_path.read_text())
        ivf = d / f"q{qp}.ivf"
        probe_path = self.args.probe_dir / f"{probe}.y4m"
        decoded = d / f"q{qp}.dec.y4m"
        if enc_name == "svt":
            n = decode_svt(self.args.dav1d, ivf, decoded)
        else:
            n = decode_av2(str(self.args.avmdec), ivf, probe_path, decoded)
        if n != self.args.frames:
            raise RuntimeError(f"{probe} {enc_name} qp{qp}: decoded {n} frames, "
                               f"expected {self.args.frames}")
        qjson = d / f"q{qp}.quality.json"
        ejson = d / f"q{qp}.extra.json"
        kjson = d / f"q{qp}.kpi.json"
        env = {"AVMDEC": str(self.args.avmdec)}
        if enc_name == "svt":
            qcmd = [sys.executable, str(TOOLS / "quality.py"), "--ref",
                    str(probe_path), "--test", str(decoded), "--json", str(qjson)]
        else:
            qcmd = [sys.executable, str(TOOLS / "quality.py"), "--ref",
                    str(probe_path), "--ivf", str(ivf), "--json", str(qjson)]
        qcmd += ["--encoder", enc_name, "--qp", str(qp)]
        p = run(qcmd, env=env)
        if p.returncode != 0:
            raise RuntimeError(f"quality.py failed for {probe} {enc_name} qp{qp}:\n"
                               f"{p.stdout[-3000:]}\n{p.stderr[-3000:]}")
        p = run([sys.executable, str(TOOLS / "extra_metrics.py"),
                 "--ref", str(probe_path), "--test", str(decoded),
                 "--json", str(ejson)])
        if p.returncode != 0:
            raise RuntimeError(f"extra_metrics.py failed for {probe} {enc_name} "
                               f"qp{qp}:\n{p.stdout[-3000:]}\n{p.stderr[-3000:]}")
        p = run([sys.executable, str(TOOLS / "kpi.py"),
                 "--quality-json", str(qjson), "--extra-json", str(ejson),
                 "--json", str(kjson)])
        if p.returncode != 0:
            raise RuntimeError(f"kpi.py failed for {probe} {enc_name} qp{qp}:\n"
                               f"{p.stdout}\n{p.stderr}")
        q = json.loads(qjson.read_text())
        e = json.loads(ejson.read_text())
        k = json.loads(kjson.read_text())
        out = {
            "probe": probe, "encoder": enc_name, "qp": qp,
            "kbps": point["kbps"], "bytes": point["bytes"],
            "frames": point.get("frames"),
            "frame_bytes": point.get("frame_bytes"),
            "encode_seconds": point.get("encode_seconds"),
            "cvvdp_jod": q["cvvdp_jod"],
            "psnr_y": q["psnr"]["psnr_y"],
            "psnr_avg": q["psnr"]["psnr_avg"],
            "ssim_all": q["ssim"]["ssim_all"],
            "butteraugli": e["butteraugli_mean"],
            "ssimulacra2": e["ssimulacra2_mean"],
            "kpi": k["kpi"],
            "q_cvvdp": k["q_cvvdp"],
            "q_butteraugli": k["q_butteraugli"],
            "q_ssimulacra2": k["q_ssimulacra2"],
            "decoded_frames": n,
        }
        score_path.write_text(json.dumps(out, indent=2) + "\n")
        return out

    # -- interpolation ---------------------------------------------------
    @staticmethod
    def interpolate(points: list[dict], target: float) -> dict:
        pts = sorted(points, key=lambda p: p["kbps"])
        above = [p for p in pts if p["kbps"] >= target]
        below = [p for p in pts if p["kbps"] <= target]
        if above and below:
            a = min(above, key=lambda p: p["kbps"])
            b = max(below, key=lambda p: p["kbps"])
            exact = abs(a["kbps"] - target) < 1e-9
            t = 0.0 if exact or a["kbps"] == b["kbps"] else (
                (target - b["kbps"]) / (a["kbps"] - b["kbps"]))
            interp = {
                "kbps": target,
                "cvvdp_jod": b["cvvdp_jod"] + t * (a["cvvdp_jod"] - b["cvvdp_jod"]),
                "psnr_y": b["psnr_y"] + t * (a["psnr_y"] - b["psnr_y"]),
                "butteraugli": b["butteraugli"] + t * (a["butteraugli"] - b["butteraugli"]),
                "ssimulacra2": b["ssimulacra2"] + t * (a["ssimulacra2"] - b["ssimulacra2"]),
                "kpi": b["kpi"] + t * (a["kpi"] - b["kpi"]),
                "from_qp": [b["qp"], a["qp"]],
                "bracket_kbps": [b["kbps"], a["kbps"]],
                "bracketed": True,
            }
            # Cross-check: K recomputed from interpolated metrics.
            q_c = max(0.0, min(1.0, interp["cvvdp_jod"] / 10.0))
            q_b = 1.0 / (1.0 + max(interp["butteraugli"], 0.0) / 3.0)
            q_s = max(0.0, min(1.0, (interp["ssimulacra2"] + 50.0) / 150.0))
            interp["kpi_from_interp_metrics"] = 100.0 * (
                0.5 * q_c + 0.25 * q_b + 0.25 * q_s)
            return interp
        # Unbracketed: linear extrapolation from the two nearest points.
        ext = pts[:2] if target < pts[0]["kbps"] else pts[-2:]
        if len(ext) < 2:
            ext = pts * 2
        b, a = sorted(ext, key=lambda p: p["kbps"])
        slope = (a["kbps"] - b["kbps"])
        t = (target - b["kbps"]) / slope if slope else 0.0
        out = {
            "kbps": target,
            "cvvdp_jod": b["cvvdp_jod"] + t * (a["cvvdp_jod"] - b["cvvdp_jod"]),
            "psnr_y": b["psnr_y"] + t * (a["psnr_y"] - b["psnr_y"]),
            "butteraugli": b["butteraugli"] + t * (a["butteraugli"] - b["butteraugli"]),
            "ssimulacra2": b["ssimulacra2"] + t * (a["ssimulacra2"] - b["ssimulacra2"]),
            "kpi": b["kpi"] + t * (a["kpi"] - b["kpi"]),
            "from_qp": [b["qp"], a["qp"]],
            "bracket_kbps": [b["kbps"], a["kbps"]],
            "bracketed": False,
        }
        return out

    # -- top-level -------------------------------------------------------
    def run_probe_encoder(self, probe: str, enc_name: str) -> dict:
        try:
            br = self.find_bracket(probe, enc_name)
            scored = [self.score_one(probe, enc_name, p) for p in br["bracket"]]
            at = self.interpolate(scored, self.target)
            return {"probe": probe, "encoder": enc_name,
                    "bracketed": br["bracketed"], "points": scored,
                    "at_target": at}
        except Exception as exc:  # keep the sweep alive
            return {"probe": probe, "encoder": enc_name, "error": str(exc)}

    def run(self) -> dict:
        probes = self.args.only or PROBE_ORDER
        encs = self.args.encoders or list(ENCODERS)
        tasks = [(p, e) for p in probes for e in encs]
        results: dict[tuple[str, str], dict] = {}
        jobs = max(1, self.args.jobs)
        if jobs == 1:
            for p, e in tasks:
                print(f"== {p} / {e}", flush=True)
                results[(p, e)] = self.run_probe_encoder(p, e)
                self._print_result(results[(p, e)])
        else:
            with futures.ThreadPoolExecutor(max_workers=jobs) as pool:
                futs = {pool.submit(self.run_probe_encoder, p, e): (p, e)
                        for p, e in tasks}
                for fut in futures.as_completed(futs):
                    p, e = futs[fut]
                    print(f"== {p} / {e} done", flush=True)
                    results[(p, e)] = fut.result()
                    self._print_result(results[(p, e)])
        out = {
            "target_kbps": self.target,
            "frames": self.args.frames,
            "probe_dir": str(self.args.probe_dir),
            "encoders": {name: {"kind": ENCODERS[name].kind,
                                "cmd_env": ENCODERS[name].env}
                         for name in encs},
            "results": {f"{p}/{e}": v for (p, e), v in sorted(results.items())},
        }
        # A partial --only/--encoders run updates the full report rather than
        # truncating it.
        results_path = self.out / "results.json"
        if results_path.is_file():
            try:
                old = json.loads(results_path.read_text())
                merged = dict(old.get("results", {}))
                merged.update(out["results"])
                out["results"] = {k: merged[k] for k in sorted(merged)}
                out["encoders"] = {**old.get("encoders", {}), **out["encoders"]}
            except Exception:
                pass
        results_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
        (self.out / "summary.md").write_text(self.markdown(out))
        print(self.markdown(out))
        return out

    @staticmethod
    def _print_result(r: dict) -> None:
        if "error" in r:
            print(f"   ERROR: {r['error']}", flush=True)
            return
        a = r["at_target"]
        print(f"   K={a['kpi']:.2f} rate={a['kbps']:.1f} "
              f"CVVDP={a['cvvdp_jod']:.3f} BA={a['butteraugli']:.2f} "
              f"SS2={a['ssimulacra2']:.2f} PSNR={a['psnr_y']:.2f} "
              f"pts={a['bracket_kbps']} bracketed={r['bracketed']}", flush=True)

    @staticmethod
    def markdown(out: dict) -> str:
        rows = {}
        for key, r in out["results"].items():
            probe, enc = key.split("/", 1)
            if "at_target" not in r:
                rows.setdefault(probe, {})[enc] = None
            else:
                rows.setdefault(probe, {})[enc] = r["at_target"]
        order = [e for e in ("nightly", "eff-perc", "svt", "avm")]
        lines = [
            f"Target {out['target_kbps']:.0f} kbps, {out['frames']} frames, "
            "K linearly interpolated in rate.",
            "",
            "| Probe | Encoder | kbps | CVVDP | BA | SS2 | PSNR-Y | K | Bracket |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
        for probe in PROBE_ORDER:
            if probe not in rows:
                continue
            for enc in order:
                a = rows[probe].get(enc)
                if not a:
                    lines.append(f"| {probe} | {enc} | — | — | — | — | — | — | missing |")
                    continue
                br = ", ".join(f"{x:.1f}" for x in a["bracket_kbps"])
                lines.append(
                    f"| {probe} | {enc} | {a['kbps']:.1f} | "
                    f"{a['cvvdp_jod']:.3f} | {a['butteraugli']:.2f} | "
                    f"{a['ssimulacra2']:.2f} | {a['psnr_y']:.2f} | "
                    f"{a['kpi']:.2f} | {br} |")
        return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--av2enc", type=Path, default=DEFAULT_AV2ENC)
    ap.add_argument("--avm", type=Path, default=DEFAULT_AVM)
    ap.add_argument("--avmdec", type=Path, default=DEFAULT_AVMDEC)
    ap.add_argument("--svt", default=DEFAULT_SVT)
    ap.add_argument("--dav1d", default=DEFAULT_DAV1D)
    ap.add_argument("--probe-dir", type=Path,
                    default=ROOT / "build" / "tool-probes")
    ap.add_argument("--out-dir", type=Path,
                    default=ROOT / "build" / "tool-quality")
    ap.add_argument("--target-kbps", type=float, default=100.0)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--only", nargs="*", choices=PROBE_ORDER)
    ap.add_argument("--encoders", nargs="*", choices=list(ENCODERS))
    ap.add_argument("--generate", action="store_true",
                    help="regenerate all probes before encoding")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        print("probes:    " + " ".join(PROBE_ORDER))
        print("encoders:  " + " ".join(ENCODERS))
        return 0
    if args.generate or not (args.probe_dir / "manifest.json").is_file():
        cmd = [sys.executable, str(PROBE_TOOL), "--out-dir", str(args.probe_dir),
               "--frames", str(args.frames)]
        if args.generate:
            cmd.append("--force")
        p = subprocess.run(cmd, cwd=ROOT)
        if p.returncode != 0:
            return p.returncode
    for tool, path in (("av2enc", args.av2enc), ("avm", args.avm)):
        if not Path(path).is_file():
            ap.error(f"{tool} not found: {path}")
    if not Path(args.avmdec).is_file():
        ap.error(f"avmdec not found: {args.avmdec}")
    runner = Runner(args)
    runner.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
