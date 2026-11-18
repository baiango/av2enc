#!/usr/bin/env python3
"""Luma K oracle harness: score a candidate's QP ladder at a matched bitrate.

The oracle is the composite K (CVVDP / Butteraugli / SSIMULACRA2) at a target
bitrate, interpolated across the candidate's own QP ladder.  A candidate is a
group of encodes that differ only in QP; give each point a label of the form
``name@point`` and this tool interpolates K to ``--target-kbps``.

Every stream is decoded with the pinned AVM decoder (``avmdec``) through
``tools/quality.py``.  There is no second decoder here.

Segment ceilings replace one frame range of the decoded stream with the source
(the rate-free ceiling) or with another decoded stream (the audit's
cross-candidate replacement), then re-score the 60-frame sequence.  Segment
rows are pooling bounds, not encodable streams.

Usage:
  tools/oracle.py --ref testdata/nasa_480x240_30fps_60f.y4m \
      --ivf nightly=/tmp/nightly.ivf --target-kbps 100 --json /tmp/n.json

  tools/oracle.py --ref REF \
      --ivf eff@119=/tmp/eff119.ivf --ivf eff@120=/tmp/eff120.ivf \
      --target-kbps 100 --json /tmp/eff.json

  # reproduce docs/luma-audit.md section 6 (SVT replacement)
  tools/oracle.py --ref REF --ivf nightly=/tmp/nightly.ivf \
      --decoded svt=/tmp/svt.decoded.y4m --segment-from svt \
      --target-kbps 100 --json /tmp/audit.json

The JSON report is the hand-off artifact: it carries every stream, the
matched-rate K, the segment rows, and the wall verdict.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

# tools/quality.py is the AVM-scoring oracle shared with the main tree.
# Importing it here keeps one decoder implementation; there is no local copy of
# the avmdec invocation.
import extra_metrics  # noqa: E402
import kpi  # noqa: E402
import quality  # noqa: E402

# The go/no-go walls from docs/oracle-search.md.  K is on the 0-100 axis.
SVT_WALL = 88.8
AVM_WALL = 90.2

SEGMENTS = (
    ("key f0", 0, 0),
    ("f0-32", 0, 32),
    ("f33-43", 33, 43),
    ("f44-59", 44, 59),
)


# --------------------------------------------------------------------------
# labels


def split_label(label: str) -> tuple[str, str]:
    """Split ``candidate@point``.  A label without @ is a one-point candidate."""
    if "@" in label:
        candidate, point = label.rsplit("@", 1)
        if candidate and point:
            return candidate, point
    return label, label


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


# --------------------------------------------------------------------------
# frames and scores


@dataclass
class FrameScores:
    butteraugli: list[float]
    ssimulacra2: list[float]

    def mean(self) -> tuple[float, float]:
        return (
            sum(self.butteraugli) / len(self.butteraugli),
            sum(self.ssimulacra2) / len(self.ssimulacra2),
        )


@dataclass
class Metrics:
    psnr: dict
    ssim: dict
    cvvdp_jod: float
    cvvdp_version: str
    butteraugli: float
    ssimulacra2: float
    kpi: float
    components: dict


@dataclass
class Point:
    label: str
    candidate: str
    point: str
    ivf: Path
    ivf_bytes: int
    frames: int
    kbps: float
    metrics: Metrics
    frame_scores: FrameScores
    decoded: Path

    def as_json(self) -> dict:
        return {
            "label": self.label,
            "point": self.point,
            "ivf": str(self.ivf),
            "ivf_bytes": self.ivf_bytes,
            "frames": self.frames,
            "kbps": round(self.kbps, 4),
            "psnr": self.metrics.psnr,
            "ssim": self.metrics.ssim,
            "cvvdp_jod": self.metrics.cvvdp_jod,
            "cvvdp_version": self.metrics.cvvdp_version,
            "butteraugli": self.metrics.butteraugli,
            "ssimulacra2": self.metrics.ssimulacra2,
            "kpi": self.metrics.kpi,
            "kpi_components": self.metrics.components,
        }


@dataclass
class Candidate:
    name: str
    points: list[Point] = field(default_factory=list)
    matched: dict = field(default_factory=dict)
    segment_base: str | None = None
    segment_base_kbps: float | None = None
    segments: list[dict] = field(default_factory=list)
    segments_skipped: str | None = None

    def as_json(self) -> dict:
        return {
            "candidate": self.name,
            "points": [p.as_json() for p in self.points],
            "matched": self.matched,
            "segment_base": self.segment_base,
            "segment_base_kbps": self.segment_base_kbps,
            "segments": self.segments,
            "segments_skipped": self.segments_skipped,
        }


# --------------------------------------------------------------------------
# metric plumbing


def run_ffmpeg_metrics(ref: Path, test: Path) -> tuple[dict, dict]:
    psnr_out = quality.run([
        "ffmpeg", "-hide_banner", "-v", "info", "-i", str(test),
        "-i", str(ref), "-lavfi", "[0:v][1:v]psnr", "-f", "null", "-",
    ])
    psnr = quality.parse_psnr((psnr_out.stdout or "") + "\n" + (psnr_out.stderr or ""))
    ssim_out = quality.run([
        "ffmpeg", "-hide_banner", "-v", "info", "-i", str(test),
        "-i", str(ref), "-lavfi", "[0:v][1:v]ssim", "-f", "null", "-",
    ])
    ssim = quality.parse_ssim((ssim_out.stdout or "") + "\n" + (ssim_out.stderr or ""))
    return psnr, ssim


def check_metric_tools() -> None:
    missing = [t for t in ("ffmpeg", "butteraugli_main", "ssimulacra2")
               if not extra_metrics.have(t)]
    if missing:
        raise SystemExit(
            "oracle: missing metric tool(s): " + ", ".join(missing) +
            "; install them or put them on PATH")
    # Raises with the CVVDP install hint when neither the CLI nor pycvvdp exists.
    quality.cvvdp_command()


def extract_ref_pngs(ref: Path, out_dir: Path, n: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    got = extra_metrics.extract(ref, out_dir, n)
    if got != n:
        raise SystemExit(
            f"oracle: {ref} yielded {got} PNGs, expected {n}; "
            "does the Y4M frame count match the header?")
    return out_dir


def score_frames(ref_pngs: Path, test: Path, n: int, out_dir: Path) -> FrameScores:
    """Per-frame BA/SS2, averaged exactly as tools/extra_metrics.py does."""
    out_dir.mkdir(parents=True, exist_ok=True)
    got = extra_metrics.extract(test, out_dir, n)
    if got != n:
        raise SystemExit(
            f"oracle: {test} yielded {got} PNGs, expected {n}")
    ba: list[float] = []
    ss: list[float] = []
    for i in range(1, n + 1):
        b, s = extra_metrics.score_pair(
            ref_pngs / f"f{i:04d}.png", out_dir / f"f{i:04d}.png")
        if math.isnan(b) or math.isnan(s):
            raise SystemExit(
                f"oracle: butteraugli/ssimulacra2 failed on frame {i} of {test}")
        ba.append(b)
        ss.append(s)
    return FrameScores(ba, ss)


def score_identical_frames(ref_pngs: Path, n: int) -> FrameScores:
    ba: list[float] = []
    ss: list[float] = []
    for i in range(1, n + 1):
        png = ref_pngs / f"f{i:04d}.png"
        b, s = extra_metrics.score_pair(png, png)
        if math.isnan(b) or math.isnan(s):
            raise SystemExit(
                f"oracle: butteraugli/ssimulacra2 failed on identical frame {i}")
        ba.append(b)
        ss.append(s)
    return FrameScores(ba, ss)


def assemble_metrics(ref: Path, test: Path, frames: FrameScores) -> Metrics:
    psnr, ssim = run_ffmpeg_metrics(ref, test)
    jod, ver = quality.run_cvvdp(ref, test)
    ba, ss2 = frames.mean()
    comps = kpi.components(jod, ba, ss2)
    return Metrics(
        psnr=psnr,
        ssim=ssim,
        cvvdp_jod=jod,
        cvvdp_version=ver,
        butteraugli=ba,
        ssimulacra2=ss2,
        kpi=kpi.composite(jod, ba, ss2),
        components={
            "q_cvvdp": comps[0],
            "q_butteraugli": comps[1],
            "q_ssimulacra2": comps[2],
        },
    )


# --------------------------------------------------------------------------
# rate matching


def match_rate(points: list[Point], target: float) -> dict:
    if not points:
        raise SystemExit("oracle: candidate has no points")
    pts = sorted(points, key=lambda p: (p.kbps, p.label))

    if len(pts) == 1:
        p = pts[0]
        delta = (p.kbps - target) / target * 100.0
        return {
            "target_kbps": target,
            "kpi": p.metrics.kpi,
            "kbps": p.kbps,
            "method": "single-point",
            "bracket": [p.label],
            # A single point cannot bracket the target.  A sub-0.5% rate miss
            # (the 25,008 B pin is 100.032) is still the measured point; a
            # larger miss is only "nearest" and is not a matched-rate oracle.
            "out_of_range": abs(delta) > 0.5,
            "rate_delta_pct": delta,
        }

    for p in pts:
        if abs(p.kbps - target) < 1e-9:
            return {
                "target_kbps": target,
                "kpi": p.metrics.kpi,
                "kbps": p.kbps,
                "method": "endpoint",
                "bracket": [p.label],
                "out_of_range": False,
                "rate_delta_pct": 0.0,
            }

    for a, b in zip(pts, pts[1:]):
        if a.kbps < target < b.kbps:
            frac = (target - a.kbps) / (b.kbps - a.kbps)
            k = a.metrics.kpi + (b.metrics.kpi - a.metrics.kpi) * frac
            return {
                "target_kbps": target,
                "kpi": k,
                "kbps": target,
                "method": "linear",
                "bracket": [a.label, b.label],
                "out_of_range": False,
                "rate_delta_pct": 0.0,
            }

    nearest = min(pts, key=lambda p: (abs(p.kbps - target), p.kbps))
    delta = (nearest.kbps - target) / target * 100.0
    return {
        "target_kbps": target,
        "kpi": nearest.metrics.kpi,
        "kbps": nearest.kbps,
        "method": "nearest",
        "bracket": [nearest.label],
        "out_of_range": True,
        "rate_delta_pct": delta,
    }


def verdict(k: float) -> str:
    if k >= AVM_WALL:
        return "clears AVM wall (>=90.2)"
    if k >= SVT_WALL:
        return "clears SVT wall (>=88.8), below AVM"
    return "NO-GO (<88.8)"


# --------------------------------------------------------------------------
# segments


def frame_range_indices(start: int, end: int, n: int) -> list[int]:
    if n != 60:
        return []
    return list(range(start, end + 1))


def replacement_scores(
    name: str,
    candidates: list[Candidate],
    decoded_scores: dict[str, FrameScores],
    source_scores: FrameScores,
    target: float,
) -> tuple[FrameScores, str]:
    if name == "source":
        return source_scores, "source"
    if name in decoded_scores:
        return decoded_scores[name], name

    exact = [p for c in candidates for p in c.points if p.label == name]
    if exact:
        return exact[0].frame_scores, exact[0].label

    by_name = [p for c in candidates if c.name == name for p in c.points]
    if by_name:
        if len(by_name) > 1:
            pick = min(by_name, key=lambda p: (abs(p.kbps - target), p.kbps))
            print(
                f"oracle: warning: replacement {name!r} has {len(by_name)} "
                f"points; using {pick.label} ({pick.kbps:.3f} kbps)",
                file=sys.stderr,
            )
            return pick.frame_scores, pick.label
        return by_name[0].frame_scores, by_name[0].label

    choices = ["source"] + sorted(decoded_scores) + sorted(
        {p.label for c in candidates for p in c.points})
    raise SystemExit(
        f"oracle: unknown --segment-from {name!r}; choose one of "
        + ", ".join(choices))


def mix_frame_scores(
    base: FrameScores, replacement: FrameScores, indices: list[int], n: int
) -> FrameScores:
    idx = set(indices)
    ba = [replacement.butteraugli[i] if i in idx else base.butteraugli[i]
          for i in range(n)]
    ss = [replacement.ssimulacra2[i] if i in idx else base.ssimulacra2[i]
          for i in range(n)]
    return FrameScores(ba, ss)


def score_segment(
    ref: Path,
    ref_header: str,
    base_frames: list[bytes],
    replacement_frames: list[bytes],
    indices: list[int],
    base_scores: FrameScores,
    repl_scores: FrameScores,
    n: int,
    out_path: Path,
) -> Metrics:
    frames = list(base_frames)
    for i in indices:
        frames[i] = replacement_frames[i]
    if len(frames) != n:
        raise SystemExit(
            f"oracle: segment assembly produced {len(frames)} frames, expected {n}")
    quality.write_y4m(out_path, ref_header, frames)
    mixed = mix_frame_scores(base_scores, repl_scores, indices, n)
    return assemble_metrics(ref, out_path, mixed)


# --------------------------------------------------------------------------
# CLI


def parse_assignment(text: str, flag: str) -> tuple[str, Path]:
    if "=" not in text:
        raise SystemExit(f"oracle: {flag} expects LABEL=PATH, got {text!r}")
    label, raw = text.split("=", 1)
    if not label or not raw:
        raise SystemExit(f"oracle: {flag} expects LABEL=PATH, got {text!r}")
    path = Path(raw).expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"oracle: missing file for {label}: {path}")
    return label, path


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Score an architecture candidate at a matched bitrate with "
                    "the AVM quality oracle.")
    ap.add_argument("--ref", required=True, type=Path,
                    help="reference Y4M (8-bit 4:2:0)")
    ap.add_argument("--ivf", action="append", default=[], metavar="LABEL=PATH",
                    help="candidate IVF (AV02); repeat for a QP ladder. "
                         "A point label may be name@qp.")
    ap.add_argument("--decoded", action="append", default=[],
                    metavar="LABEL=PATH",
                    help="pre-decoded Y4M used only for cross-candidate segment "
                         "replacement (e.g. the SVT audit stream); avmdec does "
                         "not decode AV1")
    ap.add_argument("--segment-from", action="append", default=[],
                    metavar="LABEL",
                    help="also score segment replacement with LABEL "
                         "(source, a scored point, or a --decoded label); "
                         "source rows are always reported")
    ap.add_argument("--no-segments", action="store_true",
                    help="skip segment oracles (fast matched-rate-only run)")
    ap.add_argument("--target-kbps", type=float, default=100.0,
                    help="target bitrate for the matched-rate K (default 100)")
    ap.add_argument("--json", type=Path, help="write the JSON report")
    ap.add_argument("--decode-threads", default=None,
                    help="avmdec thread count (default: AVMDEC_THREADS or decoder default)")
    ap.add_argument("--keep", action="store_true",
                    help="keep decoded/normalized Y4M files and print their directory")
    args = ap.parse_args()

    ref = args.ref.expanduser().resolve()
    if not ref.is_file():
        raise SystemExit(f"oracle: missing ref: {ref}")
    if args.target_kbps <= 0:
        raise SystemExit("oracle: --target-kbps must be positive")
    if not args.ivf:
        raise SystemExit("oracle: give at least one --ivf LABEL=PATH")

    check_metric_tools()
    avmdec = quality.find_avmdec()

    ref_header, ref_frames = quality.read_frames(ref)
    n = len(ref_frames)
    if n == 0:
        raise SystemExit("oracle: reference has no frames")
    w, h, chroma, fps_num, fps_den = quality.y4m_geometry(ref_header)
    if re.search(r"C[0-9A-Za-z]*p(10|12|16)", ref_header):
        raise SystemExit("oracle: the AVM path supports 8-bit Y4M only")
    if not chroma.startswith("420"):
        raise SystemExit("oracle: the AVM path supports 4:2:0 Y4M only")

    assignments = [parse_assignment(x, "--ivf") for x in args.ivf]
    decoded_assign = [parse_assignment(x, "--decoded") for x in args.decoded]
    do_segments = not args.no_segments and n == 60
    segment_from = list(args.segment_from)
    if not segment_from and decoded_assign and do_segments:
        # Passing a decoded stream is the intent to score it as a replacement.
        segment_from = [label for label, _ in decoded_assign]

    tmp = Path(tempfile.mkdtemp(prefix="luma-oracle-"))
    started = time.time()
    try:
        if args.keep:
            print(f"oracle: scratch {tmp}", file=sys.stderr)

        print(f"oracle: ref {ref} ({w}x{h} {chroma} {n}f {fps_num}/{fps_den})",
              file=sys.stderr)
        print(f"oracle: decoder {avmdec}", file=sys.stderr)
        print("oracle: extracting reference PNGs", file=sys.stderr)
        ref_pngs = extract_ref_pngs(ref, tmp / "ref-png", n)

        source_scores: FrameScores | None = None
        if do_segments:
            print("oracle: scoring source-only frames", file=sys.stderr)
            source_scores = score_identical_frames(ref_pngs, n)

        decoded_scores: dict[str, FrameScores] = {}
        decoded_frames: dict[str, list[bytes]] = {}
        for label, path in decoded_assign:
            if do_segments and label not in segment_from:
                print(
                    f"oracle: warning: --decoded {label} is unused; add "
                    f"--segment-from {label}", file=sys.stderr)
            header, frames = quality.read_frames(path)
            rw, rh, rchroma, _, _ = quality.y4m_geometry(header)
            if (rw, rh) != (w, h):
                raise SystemExit(
                    f"oracle: --decoded {label} is {rw}x{rh}, reference is {w}x{h}")
            if not rchroma.startswith("420"):
                raise SystemExit(
                    f"oracle: --decoded {label} is not 4:2:0 ({rchroma})")
            if len(frames) != n:
                raise SystemExit(
                    f"oracle: --decoded {label} has {len(frames)} frames, "
                    f"reference has {n}")
            decoded_frames[label] = frames
            if do_segments:
                print(f"oracle: scoring replacement frames {label}", file=sys.stderr)
                decoded_scores[label] = score_frames(
                    ref_pngs, path, n, tmp / f"png-decoded-{slug(label)}")

        candidates_by: dict[str, Candidate] = {}
        total = len(assignments)
        for i, (label, path) in enumerate(assignments, 1):
            cand_name, point_name = split_label(label)
            cand = candidates_by.setdefault(cand_name, Candidate(cand_name))
            print(f"oracle: [{i}/{total}] decode+score {label} ({path})",
                  file=sys.stderr)
            decoded_path = tmp / f"{i:02d}-{slug(label)}.decoded.y4m"
            quality.decode_avm(avmdec, path, ref, decoded_path,
                               args.decode_threads)
            _, dec_frames = quality.read_frames(decoded_path)
            if len(dec_frames) != n:
                raise SystemExit(
                    f"oracle: {label} decoded {len(dec_frames)} frames, "
                    f"reference has {n}; the oracle needs equal counts")
            frames = score_frames(
                ref_pngs, decoded_path, n, tmp / f"png-{i:02d}-{slug(label)}")
            metrics = assemble_metrics(ref, decoded_path, frames)
            duration = n * fps_den / fps_num
            kbps = path.stat().st_size * 8 / duration / 1000.0
            cand.points.append(Point(
                label=label,
                candidate=cand_name,
                point=point_name,
                ivf=path,
                ivf_bytes=path.stat().st_size,
                frames=n,
                kbps=kbps,
                metrics=metrics,
                frame_scores=frames,
                decoded=decoded_path,
            ))

        candidates = list(candidates_by.values())
        for cand in candidates:
            cand.matched = match_rate(cand.points, args.target_kbps)
            cand.matched["verdict"] = verdict(cand.matched["kpi"])

        if do_segments:
            used_replacements: list[tuple[str, FrameScores]] = [
                ("source", source_scores)  # type: ignore[arg-type]
            ]
            for label in segment_from:
                scores, resolved = replacement_scores(
                    label, candidates, decoded_scores, source_scores,  # type: ignore[arg-type]
                    args.target_kbps)
                if all(resolved != name for name, _ in used_replacements):
                    used_replacements.append((resolved, scores))

            for cand in candidates:
                base = min(cand.points,
                           key=lambda p: (abs(p.kbps - args.target_kbps),
                                          p.kbps))
                cand.segment_base = base.label
                cand.segment_base_kbps = base.kbps
                base_frames = quality.read_frames(base.decoded)[1]
                base_scores = base.frame_scores
                for repl_name, repl_scores in used_replacements:
                    if repl_name == "source":
                        repl_frames = ref_frames
                    elif repl_name in decoded_frames:
                        repl_frames = decoded_frames[repl_name]
                    else:
                        repl_points = [
                            p for c in candidates for p in c.points
                            if p.label == repl_name
                        ]
                        if not repl_points:
                            raise SystemExit(
                                f"oracle: cannot resolve replacement "
                                f"{repl_name!r}")
                        repl_frames = quality.read_frames(
                            repl_points[0].decoded)[1]
                    for seg_name, start, end in SEGMENTS:
                        indices = frame_range_indices(start, end, n)
                        if not indices:
                            continue
                        tag = (f"{slug(cand.name)}-{slug(repl_name)}-"
                               f"{slug(seg_name)}")
                        print(
                            f"oracle: segment {cand.name} / {repl_name} / "
                            f"{seg_name}", file=sys.stderr)
                        out = tmp / f"seg-{tag}.y4m"
                        metrics = score_segment(
                            ref, ref_header, base_frames, repl_frames,
                            indices, base_scores, repl_scores, n, out)
                        cand.segments.append({
                            "segment": seg_name,
                            "frame_range": [start, end],
                            "frame_count": len(indices),
                            "replacement": repl_name,
                            "cvvdp_jod": metrics.cvvdp_jod,
                            "butteraugli": metrics.butteraugli,
                            "ssimulacra2": metrics.ssimulacra2,
                            "psnr": metrics.psnr,
                            "ssim": metrics.ssim,
                            "kpi": metrics.kpi,
                            "delta_k": metrics.kpi - base.metrics.kpi,
                            "base_kpi": base.metrics.kpi,
                            "base_kbps": base.kbps,
                        })
        elif not args.no_segments:
            for cand in candidates:
                cand.segments_skipped = (
                    f"reference has {n} frames; the audit segments are "
                    "defined for the 60-frame fixture")

        report = {
            "schema": "luma-oracle/1",
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_seconds": round(time.time() - started, 1),
            "ref": str(ref),
            "ref_geometry": {
                "width": w, "height": h, "chroma": chroma,
                "fps": f"{fps_num}/{fps_den}", "frames": n,
            },
            "target_kbps": args.target_kbps,
            "decoder": Path(avmdec).name,
            "decoder_path": avmdec,
            "scoring": {
                "quality": str(Path(quality.__file__).resolve()),
                "extra_metrics": str(Path(extra_metrics.__file__).resolve()),
                "kpi": str(Path(kpi.__file__).resolve()),
            },
            "thresholds": {"svt_k": SVT_WALL, "avm_k": AVM_WALL},
            "candidates": [c.as_json() for c in candidates],
        }
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(report, indent=2) + "\n")

        print_markdown(report)
        return 0
    finally:
        if not args.keep:
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            print(f"oracle: kept {tmp}", file=sys.stderr)


# --------------------------------------------------------------------------
# Markdown


def fmt_bytes(n: int) -> str:
    return f"{n:,}"


def print_markdown(report: dict) -> None:
    ref = report["ref_geometry"]
    print("# Luma K oracle")
    print()
    print(f"- ref: `{report['ref']}`")
    print(f"- {ref['width']}x{ref['height']} {ref['chroma']}, "
          f"{ref['frames']} frames @ {ref['fps']}")
    print(f"- target: {report['target_kbps']:.3f} kbps")
    print(f"- decoder: `{report['decoder']}`")
    print(f"- walls: SVT {SVT_WALL:.1f}, AVM {AVM_WALL:.1f}")
    print()

    print("## Streams")
    print()
    print("| candidate | point | bytes | kbps | PSNR-Y | SSIM | JOD | BA | SS2 | K |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for cand in report["candidates"]:
        for p in cand["points"]:
            print(f"| {cand['candidate']} | {p['point']} | "
                  f"{fmt_bytes(p['ivf_bytes'])} | {p['kbps']:.3f} | "
                  f"{p['psnr']['psnr_y']:.3f} | {p['ssim']['ssim_all']:.4f} | "
                  f"{p['cvvdp_jod']:.4f} | {p['butteraugli']:.4f} | "
                  f"{p['ssimulacra2']:.3f} | {p['kpi']:.3f} |")
    print()

    print("## Matched rate")
    print()
    print("| candidate | ladder | matched kbps | K @ target | method | verdict |")
    print("|---|---:|---:|---:|---|---|")
    for cand in report["candidates"]:
        m = cand["matched"]
        bracket = "..".join(m["bracket"])
        print(f"| {cand['candidate']} | {len(cand['points'])} | "
              f"{m['kbps']:.3f} | {m['kpi']:.3f} | "
              f"{m['method']} ({bracket}) | {m['verdict']} |")
    print()
    for cand in report["candidates"]:
        m = cand["matched"]
        if m["out_of_range"]:
            print(f"> {cand['candidate']}: not a matched-rate oracle: the "
                  f"ladder does not bracket {report['target_kbps']:.3f} kbps "
                  f"(nearest {m['kbps']:.3f}, {m['rate_delta_pct']:+.2f}%).")
            print()

    any_segments = any(c["segments"] for c in report["candidates"])
    if any_segments:
        print("## Segment ceilings")
        print()
        print("Source replacement is the rate-free ceiling; another decoded "
              "stream is the audit's cross-candidate replacement. "
              "Segment rows are pooling bounds, not encodable streams.")
        print()
        print("| candidate | base | base kbps | replacement | segment | K | dK |")
        print("|---|---|---:|---|---|---:|---:|")
        for cand in report["candidates"]:
            for s in cand["segments"]:
                print(f"| {cand['candidate']} | {cand['segment_base']} | "
                      f"{cand['segment_base_kbps']:.3f} | {s['replacement']} | "
                      f"{s['segment']} | {s['kpi']:.3f} | "
                      f"{s['delta_k']:+.3f} |")
        print()

    for cand in report["candidates"]:
        if cand["segments_skipped"]:
            print(f"> {cand['candidate']}: segment oracles skipped: "
                  f"{cand['segments_skipped']}")
            print()

    print("## Go/no-go")
    print()
    print(f"- No tuning below SVT {SVT_WALL:.1f}.")
    print(f"- An architecture must clear {SVT_WALL:.1f} to be pursued and "
          f"{AVM_WALL:.1f} to beat AVM.")
    print("- A segment ceiling above a wall is not a claim: only the "
          "matched-rate K is an oracle.")


if __name__ == "__main__":
    raise SystemExit(main())
