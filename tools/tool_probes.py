#!/usr/bin/env python3
"""First-party synthetic probes for av2enc tool quality at 240p/100 kbps.

Every clip is generated from closed-form math and a fixed NumPy seed: no
third-party image, video or code asset is read.  Content is 8-bit 4:2:0 Y4M,
480x240, 30 fps, 60 frames (2 s), so it drops straight into the project's 240p
operating point (480x240 @ 30 fps, 100 kbps).

The probes are deliberately narrow.  Each one isolates a tool family:

  me_half / me_quarter / me_eighth
      A periodic band-limited texture panned by a known global translation
      whose fractional part is 1/2, 1/4 or 1/8 luma sample per frame.  A
      perfect subpel predictor is exactly the previous frame shifted by that
      vector.  Isolates ME search + subpel MC (and, secondarily, the residual
      coder that has to clean up interpolation error).

  fade
      A static texture whose illumination ramps linearly (alpha 1.0 -> 0.45,
      offset 0 -> +15).  A block-adaptive weighted predictor can explain the
      frame from its neighbours; without one every block pays a DC residual.
      Isolates BAWP (and compound/weighted prediction, if the encoder uses it).

  static_repeat
      The same detailed frame repeated exactly.  The optimum is one key frame
      followed by zero-residual zero-MV skips; any drift, non-zero skip MV or
      unnecessary refresh is visible as quality/rate loss.  Isolates the
      skip/residual decision and reference drift.

  near_static
      static_repeat plus deterministic +/-2 LSB temporal noise.  The encoder
      must still prefer skip on most leaves; coding the noise is waste, but a
      non-zero-MV skip resamples the reference and drifts.  Isolates skip
      hysteresis/drift under sensor noise.

  detail_edges
      A static high-frequency texture and hard-edged geometry (checker, bars,
      discs).  The key picture carries almost all the bits, so the probe is an
      intra transform/quantiser/entropy probe; inter tools are idle.  It is
      NOT a clean inter-residual probe.

  grain
      A smooth scene with strong per-frame synthetic grain.  Isolates loop
      filters (deblock/CDEF/CCSO/LR) and the encode-vs-filter decision.  This
      probe is unclean: grain is incompressible at 100 kbps, so every encoder
      trades detail for rate, and the metric choice matters.  Read it as a
      filter sanity check, not an iso-quality tool ranking.

Ground truth is written next to each clip as <name>.json.  For the ME probes it
records the exact per-frame subpel translation; for the others it records the
generator parameters.  The generator is deterministic across hosts because it
uses only integer arithmetic and a fixed PCG64 stream.

Usage:
  tools/tool_probes.py --out-dir /tmp/toolprobes
  tools/tool_probes.py --out-dir /tmp/toolprobes --probes me_eighth fade
  tools/tool_probes.py --list
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import zlib
from pathlib import Path

import numpy as np

WIDTH = 480
HEIGHT = 240
FPS_NUM = 30
FPS_DEN = 1
FRAMES = 60
SEED = 20261118

# Luma limited range, the range the project's NASA fixture uses.
Y_LO, Y_HI = 16, 235
C_LO, C_HI = 16, 240


def _cos_components(rng: np.random.Generator, n: int, fx: tuple[int, int],
                    fy: tuple[int, int], amp: tuple[float, float]) -> list:
    """Random integer-frequency cosines, periodic over the frame."""
    comps = []
    for _ in range(n):
        cx = int(rng.integers(fx[0], fx[1] + 1))
        cy = int(rng.integers(fy[0], fy[1] + 1))
        if cx and rng.random() < 0.5:
            cx = -cx
        if cy and rng.random() < 0.5:
            cy = -cy
        comps.append((cx, cy, float(rng.uniform(*amp)),
                      float(rng.uniform(0.0, 2.0 * math.pi))))
    return comps


def _render(comps: list, w: int, h: int, sx: float = 0.0,
            sy: float = 0.0) -> np.ndarray:
    """Point-sample a periodic continuous texture at (x+sx, y+sy)."""
    x = np.arange(w, dtype=np.float64)[None, :] + sx
    y = np.arange(h, dtype=np.float64)[:, None] + sy
    out = np.zeros((h, w), dtype=np.float64)
    for cx, cy, amp, phase in comps:
        out += amp * np.cos(2.0 * math.pi * (cx * x / w + cy * y / h) + phase)
    return out


def _standardise(img: np.ndarray, mean: float, std: float) -> np.ndarray:
    img = img - float(img.mean())
    s = float(img.std())
    if s > 1e-9:
        img = img * (std / s)
    return img + mean


def _quantise(img: np.ndarray, lo: int, hi: int) -> np.ndarray:
    return np.clip(np.rint(img), lo, hi).astype(np.uint8)


def _chroma_components(rng: np.random.Generator) -> tuple:
    """Two independent low-frequency colour component lists."""
    cu = _cos_components(rng, 8, (1, 6), (1, 4), (0.5, 1.0))
    cv = _cos_components(rng, 8, (1, 6), (1, 4), (0.5, 1.0))
    return cu, cv


def _chroma_pair(cu: list, cv: list, w: int, h: int, sx: float = 0.0,
                 sy: float = 0.0, std: float = 8.0) -> tuple:
    u = _standardise(_render(cu, w, h, sx, sy), 0.0, std)
    v = _standardise(_render(cv, w, h, sx, sy), 0.0, std)
    return u, v


def _y4m_header(w: int, h: int) -> bytes:
    return (f"YUV4MPEG2 W{w} H{h} F{FPS_NUM}:{FPS_DEN} Ip A1:1 C420\n"
            .encode("ascii"))


class Writer:
    def __init__(self, path: Path, w: int, h: int):
        self.path = path
        self.w, self.h = w, h
        self.fh = path.open("wb")
        self.fh.write(_y4m_header(w, h))

    def frame(self, y: np.ndarray, u: np.ndarray, v: np.ndarray) -> None:
        assert y.shape == (self.h, self.w)
        assert u.shape == (self.h // 2, self.w // 2)
        assert v.shape == (self.h // 2, self.w // 2)
        self.fh.write(b"FRAME\n")
        self.fh.write(y.tobytes())
        self.fh.write(u.tobytes())
        self.fh.write(v.tobytes())

    def close(self) -> None:
        self.fh.close()


def _write_meta(path: Path, meta: dict) -> None:
    path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# Probe recipes
# ---------------------------------------------------------------------------

def _me_components() -> tuple:
    """The one ME texture shared by the three subpel probes.

    Fixed seeds and frequencies only; the three clips differ only in their
    translation vector, so their rate/quality difference isolates the subpel
    precision rather than content.
    """
    rng = np.random.default_rng(SEED + 0x4D45)
    comps = _cos_components(rng, 24, (10, 48), (8, 24), (0.45, 1.0))
    comps += _cos_components(rng, 5, (2, 8), (2, 6), (0.6, 1.2))
    comps += _cos_components(rng, 6, (16, 50), (24, 50), (0.3, 0.7))
    cu, cv = _chroma_components(rng)
    return comps, cu, cv


def probe_me(rng: np.random.Generator, out: Path, name: str, shift: tuple,
             frames: int) -> dict:
    """Fractional global translation on a periodic band-limited texture."""
    comps, cu, cv = _me_components()
    wpath = Writer(out / f"{name}.y4m", WIDTH, HEIGHT)
    dx, dy = shift
    for n in range(frames):
        sx, sy = dx * n, dy * n
        y = _standardise(_render(comps, WIDTH, HEIGHT, sx, sy), 134.0, 24.0)
        y8 = _quantise(y, Y_LO, Y_HI)
        # Chroma moves at half the luma rate (4:2:0 sampling grid).
        u, v = _chroma_pair(cu, cv, WIDTH // 2, HEIGHT // 2, sx / 2.0,
                            sy / 2.0, std=8.0)
        u8 = _quantise(128.0 + u, C_LO, C_HI)
        v8 = _quantise(128.0 + v, C_LO, C_HI)
        wpath.frame(y8, u8, v8)
    wpath.close()
    meta = {
        "probe": name,
        "family": "me_subpel",
        "width": WIDTH, "height": HEIGHT,
        "fps": f"{FPS_NUM}/{FPS_DEN}", "frames": frames,
        "translation_per_frame_px": [dx, dy],
        "translation_units_eighth": [round(dx * 8), round(dy * 8)],
        "ground_truth": "periodic texture point-sampled at (x+dx*n, y+dy*n); "
                        "the coded MV (the sample offset into the previous "
                        "frame) should be (+dx,+dy) in 1/8-luma units",
        "clean": True,
        "notes": "chroma is a static low-frequency wash by design; luma carries "
                 "the fractional translation",
    }
    _write_meta(out / f"{name}.json", meta)
    return meta


def probe_fade(rng: np.random.Generator, out: Path, name: str,
               frames: int) -> dict:
    """Linear illumination ramp on a static textured scene."""
    comps = _cos_components(rng, 36, (10, 64), (8, 40), (0.5, 1.0))
    comps += _cos_components(rng, 8, (2, 8), (2, 6), (0.5, 1.0))
    cu, cv = _chroma_components(rng)
    u_field, v_field = _chroma_pair(cu, cv, WIDTH // 2, HEIGHT // 2)
    base = _standardise(_render(comps, WIDTH, HEIGHT), 120.0, 34.0)

    wpath = Writer(out / f"{name}.y4m", WIDTH, HEIGHT)
    alphas, offsets = [], []
    for n in range(frames):
        t = n / max(1, frames - 1)
        alpha = 1.0 - 0.55 * t
        offset = 15.0 * t
        y = (base - Y_LO) * alpha + Y_LO + offset
        y8 = _quantise(y, Y_LO, Y_HI)
        u8 = _quantise(128.0 + u_field * alpha, C_LO, C_HI)
        v8 = _quantise(128.0 + v_field * alpha, C_LO, C_HI)
        wpath.frame(y8, u8, v8)
        alphas.append(alpha)
        offsets.append(offset)
    wpath.close()
    meta = {
        "probe": name,
        "family": "bawp_compound",
        "width": WIDTH, "height": HEIGHT,
        "fps": f"{FPS_NUM}/{FPS_DEN}", "frames": frames,
        "alpha_start": alphas[0], "alpha_end": alphas[-1],
        "offset_start": offsets[0], "offset_end": offsets[-1],
        "ground_truth": "y_n = 16 + (y_0 - 16) * alpha_n + offset_n; "
                        "alpha and offset are linear in n",
        "clean": True,
        "notes": "pure linear fade; a fitted BAWP alpha/beta can explain each "
                 "frame from its reference",
    }
    _write_meta(out / f"{name}.json", meta)
    return meta


def _static_texture(rng: np.random.Generator, mean: float, std: float) -> np.ndarray:
    comps = _cos_components(rng, 34, (10, 80), (8, 50), (0.5, 1.0))
    comps += _cos_components(rng, 8, (2, 8), (2, 6), (0.6, 1.2))
    return _standardise(_render(comps, WIDTH, HEIGHT), mean, std)


def probe_static(rng: np.random.Generator, out: Path, name: str,
                 frames: int, noise_sigma: float = 0.0) -> dict:
    """Repeated frame; optional i.i.d. temporal noise for near-static."""
    base = _static_texture(rng, 132.0, 40.0)
    cu, cv = _chroma_components(rng)
    u_field, v_field = _chroma_pair(cu, cv, WIDTH // 2, HEIGHT // 2)
    noise_rng = np.random.default_rng(SEED + (7 if noise_sigma else 0))

    wpath = Writer(out / f"{name}.y4m", WIDTH, HEIGHT)
    for _ in range(frames):
        y = base
        if noise_sigma > 0.0:
            y = y + noise_rng.normal(0.0, noise_sigma, size=(HEIGHT, WIDTH))
        y8 = _quantise(y, Y_LO, Y_HI)
        u8 = _quantise(128.0 + u_field, C_LO, C_HI)
        v8 = _quantise(128.0 + v_field, C_LO, C_HI)
        wpath.frame(y8, u8, v8)
    wpath.close()
    family = "near_static" if noise_sigma > 0.0 else "skip_drift"
    meta = {
        "probe": name,
        "family": family,
        "width": WIDTH, "height": HEIGHT,
        "fps": f"{FPS_NUM}/{FPS_DEN}", "frames": frames,
        "noise_sigma_lsb": noise_sigma,
        "ground_truth": "the scene is static; the optimum is one key frame "
                        "followed by zero-residual zero-MV skips"
                        + ("; the source itself changes by i.i.d. noise, which "
                           "skip reproduces as zero" if noise_sigma else ""),
        "clean": noise_sigma == 0.0,
        "notes": ("exact repeat" if noise_sigma == 0.0 else
                  "near-static: coding the noise is waste, but a non-zero-MV "
                  "skip resamples and drifts"),
    }
    _write_meta(out / f"{name}.json", meta)
    return meta


def probe_detail(rng: np.random.Generator, out: Path, name: str,
                 frames: int) -> dict:
    """Static high-frequency texture + hard-edge geometry (intra-heavy)."""
    comps = _cos_components(rng, 44, (50, 150), (25, 90), (0.4, 1.0))
    comps += _cos_components(rng, 8, (6, 30), (6, 24), (0.4, 1.0))
    tex = _standardise(_render(comps, WIDTH, HEIGHT), 128.0, 29.0)

    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
    img = tex.copy()
    # Right 42%: hard-edged geometry.  Checker 8x8, vertical bars, discs.
    right = xx >= int(WIDTH * 0.58)
    checker = ((xx // 8 + yy // 8) % 2).astype(np.float64)
    bars = ((xx // 12) % 2).astype(np.float64)
    img = np.where(right, 50.0 + 140.0 * checker, img)
    bar_zone = right & (yy >= int(HEIGHT * 0.62))
    img = np.where(bar_zone, 45.0 + 150.0 * bars, img)
    for cx, cy, rad, val in [(380, 55, 26, 228.0), (255, 190, 20, 30.0),
                             (445, 170, 30, 210.0)]:
        disc = (xx - cx) ** 2 + (yy - cy) ** 2 <= rad * rad
        img = np.where(disc, val, img)
    # A small ramp patch gives the transform search a smooth region too.
    ramp = right & (yy < int(HEIGHT * 0.28))
    img = np.where(ramp, 30.0 + 180.0 * (xx - WIDTH * 0.58) / (WIDTH * 0.42),
                   img)
    img8 = _quantise(img, Y_LO, Y_HI)

    # Chroma follows the geometry with a muted amplitude.
    u = 128.0 + 0.16 * (img - 128.0)
    v = 128.0 - 0.12 * (img - 128.0)
    # Add a low-frequency colour wash so chroma is not a copy of luma.
    cu, cv = _chroma_components(rng)
    wash_u, wash_v = _chroma_pair(cu, cv, WIDTH // 2, HEIGHT // 2, std=6.0)
    u = u.reshape(HEIGHT // 2, 2, WIDTH // 2, 2).mean(axis=(1, 3)) + wash_u
    v = v.reshape(HEIGHT // 2, 2, WIDTH // 2, 2).mean(axis=(1, 3)) + wash_v
    u8 = _quantise(u, C_LO, C_HI)
    v8 = _quantise(v, C_LO, C_HI)

    wpath = Writer(out / f"{name}.y4m", WIDTH, HEIGHT)
    for _ in range(frames):
        wpath.frame(img8, u8, v8)
    wpath.close()
    meta = {
        "probe": name,
        "family": "transform_entropy",
        "width": WIDTH, "height": HEIGHT,
        "fps": f"{FPS_NUM}/{FPS_DEN}", "frames": frames,
        "ground_truth": "static image; P frames should be exact skips, so the "
                        "key picture carries the rate",
        "clean": True,
        "notes": "intra/keyframe transform + quantiser + entropy probe; inter "
                 "tools are idle.  It does not probe inter residual coding",
    }
    _write_meta(out / f"{name}.json", meta)
    return meta


def _blur3(a: np.ndarray) -> np.ndarray:
    """Separable 1-2-1 binomial blur, reflecting edges (numpy only)."""
    p = np.pad(a, 1, mode="reflect")
    b = p[:, :-2] + 2.0 * p[:, 1:-1] + p[:, 2:]
    return (b[:-2, :] + 2.0 * b[1:-1, :] + b[2:, :]) / 16.0


def probe_grain(rng: np.random.Generator, out: Path, name: str,
                frames: int) -> dict:
    """Smooth scene + correlated film-like grain (filter probe, unclean)."""
    low = _cos_components(rng, 10, (1, 5), (1, 4), (0.5, 1.0))
    base = _standardise(_render(low, WIDTH, HEIGHT), 126.0, 30.0)
    cu, cv = _chroma_components(rng)
    u_field, v_field = _chroma_pair(cu, cv, WIDTH // 2, HEIGHT // 2)
    noise_rng = np.random.default_rng(SEED + 99)
    wpath = Writer(out / f"{name}.y4m", WIDTH, HEIGHT)
    for _ in range(frames):
        # 1-2 px correlated grain, normalised to the requested sigma.  Real
        # grain is correlated, which also keeps the peer rate curves usable at
        # 100 kbps (i.i.d. grain makes SVT's CRF curve a cliff).
        n = _blur3(noise_rng.normal(0.0, 1.0, size=(HEIGHT, WIDTH)))
        n = n / (float(n.std()) + 1e-9) * 12.0
        y = base + n
        y8 = _quantise(y, Y_LO, Y_HI)
        u8 = _quantise(128.0 + u_field, C_LO, C_HI)
        v8 = _quantise(128.0 + v_field, C_LO, C_HI)
        wpath.frame(y8, u8, v8)
    wpath.close()
    meta = {
        "probe": name,
        "family": "filters",
        "width": WIDTH, "height": HEIGHT,
        "fps": f"{FPS_NUM}/{FPS_DEN}", "frames": frames,
        "noise_sigma_lsb": 12.0,
        "noise_correlation": "1-2 px separable 1-2-1 binomial",
        "ground_truth": "smooth base + correlated N(0,12) temporal grain",
        "clean": False,
        "notes": "grain is incompressible at 100 kbps; every encoder trades "
                 "detail for rate and the CRF/QP curve is steep.  Filter "
                 "sanity check only",
    }
    _write_meta(out / f"{name}.json", meta)
    return meta


PROBES = {
    "me_half": ("me", (3.5, 1.5)),
    "me_quarter": ("me", (2.25, -1.25)),
    "me_eighth": ("me", (1.125, -0.375)),
    "fade": ("fade", None),
    "static_repeat": ("static", 0.0),
    "near_static": ("static", 2.0),
    "detail_edges": ("detail", None),
    "grain": ("grain", None),
}


def build(name: str, out: Path, frames: int) -> dict:
    kind, arg = PROBES[name]
    probe_seed = SEED + (zlib.crc32(name.encode("ascii")) & 0xFFFF)
    rng = np.random.default_rng(probe_seed)
    if kind == "me":
        return probe_me(rng, out, name, arg, frames)
    if kind == "fade":
        return probe_fade(rng, out, name, frames)
    if kind == "static":
        return probe_static(rng, out, name, frames, float(arg))
    if kind == "detail":
        return probe_detail(rng, out, name, frames)
    if kind == "grain":
        return probe_grain(rng, out, name, frames)
    raise KeyError(name)


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=Path("build/tool-probes"))
    ap.add_argument("--probes", nargs="*", choices=sorted(PROBES),
                    help="subset to generate (default: all)")
    ap.add_argument("--frames", type=int, default=FRAMES)
    ap.add_argument("--list", action="store_true", help="list probe names")
    ap.add_argument("--force", action="store_true", help="regenerate existing clips")
    args = ap.parse_args()

    if args.list:
        for name in sorted(PROBES):
            print(name)
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    names = args.probes or sorted(PROBES)
    manifest = []
    for name in names:
        y4m = args.out_dir / f"{name}.y4m"
        if y4m.exists() and not args.force:
            meta = json.loads((args.out_dir / f"{name}.json").read_text())
        else:
            meta = build(name, args.out_dir, args.frames)
        meta["sha256"] = file_digest(y4m)
        meta["bytes"] = y4m.stat().st_size
        _write_meta(args.out_dir / f"{name}.json", meta)
        manifest.append(meta)
        print(f"{name:14s} {meta['family']:18s} {y4m.stat().st_size:9d} B  "
              f"{meta['sha256'][:16]}")
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
