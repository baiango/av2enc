# M1 results — partition tree, full intra, transforms, CfL, deblock

Status: **bit-exact against `dav2d` f9dafbc**, verified per sample on all three
planes. M1 is opt-in (`--m1`); M0 remains the default until M1 is measured more
broadly.

## Benchmark clip

Source: **[LG QNED 8K MiniLED │Shine 8K HDR 60fps](https://youtu.be/WEwgbKr5R9I)**
(VP9, 3840×2160, 8-bit 4:2:0 SDR, 60 fps). The benchmark frame is a 10 s cut,
scaled to the target resolution. AVM numbers use the same frame and the same
quality tooling.

## Verified streams (QP 90, one frame, one tile)

| Still | IVF | Y / U / V PSNR | Avg | CVVDP | Verify |
|---|---:|---|---:|---:|---|
| 64×64 | 936 B | 42.50 / 45.80 / 45.12 | **43.28 dB** | — | BIT-EXACT |
| 640×360 | 13,355 B | 45.09 / 52.45 / 50.53 | **46.37 dB** | 9.8867 | BIT-EXACT |
| 1080p, native | 61,574 B | 46.36 / 54.12 / 53.15 | **47.73 dB** | 9.8976 | BIT-EXACT |

The earlier 37,856 B / 51.41 dB "1080p" row was a **Lanczos upscale of the
640×360 frame**; it flattered the encoder and is not used for comparisons. The
native 1920×1080 frame (`testdata/shine_1920x1080_10f.y4m`) is authoritative.

## M1 vs M0 vs AVM (same frame, project metrics)

One frame, 8-bit 4:2:0, `dav2d` decode, `tools/quality.py` (CVVDP 0.5.7,
`standard_fhd`, MPS). Bitrate is the one-frame payload normalized over 1/60 s;
it is not comparable to a video bitrate.

**640×360, QP 90:**

| Encoder | IVF | PSNR-Y | PSNR avg | SSIM | CVVDP |
|---|---:|---:|---:|---:|---:|
| av2enc M1 | 13,355 B | 45.09 | 46.37 | 0.9938 | 9.8867 |
| av2enc M0 | 19,751 B | 46.62 | 47.20 | — | — |
| AVM QP 90 | **10,129 B** | **48.98** | **49.99** | **0.9972** | **9.9556** |

AVM is 25% smaller and +2.6 dB PSNR-Y / +0.07 JOD at the same QP.

**1920×1080, native frame:**

| Encoder | IVF | PSNR-Y | PSNR avg | SSIM | CVVDP |
|---|---:|---:|---:|---:|---:|
| av2enc M1 (QP 90) | 61,574 B | 46.36 | 47.73 | 0.9958 | 9.8976 |
| av2enc M0 (QP 90) | 83,228 B | 49.38 | 49.79 | 0.9954 | 9.9152 |
| AVM QP 110 | **25,093 B** | 47.13 | 48.36 | 0.9948 | 9.8135 |
| AVM QP 135 | 12,242 B | 43.31 | 44.60 | 0.9886 | 9.4498 |
| AVM QP 160 | 5,485 B | 39.60 | 40.89 | 0.9746 | 8.8244 |

At comparable perceptual quality (M1 9.90 JOD vs AVM QP 110 9.81 JOD), AVM
spends **2.45× less rate** and has +0.77 dB PSNR-Y. At equal rate AVM would be
ahead on every axis. M1 is 26% smaller than M0 on this frame but gives up 3.0 dB
PSNR-Y and 0.02 JOD.

## What about QBS?

QBS is defined as **max CVVDP subject to a bitrate ceiling and an encode-speed
floor** (`tools/score_tradeoff.py`) and is measured on video, on one host. M1 is
still-picture only and the GPU path is not wired, so a QBS number now would mix
a Mac CPU AVM run with a GPU M1 claim. We therefore report the quality axis
(CVVDP/SSIM/PSNR) and rate, and defer QBS until M1 video exists.

The video reference remains AVM's 2-second 1080p point: **9.77 JOD at
1,033 kbps** (QP 160), and the 720p60 QBS budget of 400 kbps is documented in
`docs/architecture.md`.

## Verification method

`tools/m1_verify/`:

1. encode with the CLI,
2. dump `M1Result.recon_y/u/v` with the *same* `M1Options`,
3. decode the IVF with `dav2d`,
4. compare every sample in all three planes.

`BIT-EXACT` means zero differing samples, not a PSNR threshold. A harness/CLI
option mismatch is caught explicitly (`--expect-depth`, `--expect-deblock`,
`--expect-full-intra`).

Coverage: default DC path at 64×64/640×360/1080p; forced luma modes 0–12;
angle deltas ±1 and ±3 for the eight directional modes; full intra search at all
three sizes.

## Tools enabled

- recursive partition tree (8×8–64×64, rect allowed; `PARTITION_SPLIT` only where legal)
- full luma intra (13 modes, angle deltas, edge filtering, IDIF, wide-angle)
- variable TX sizes via `tx_variants`, including adjusted 64→32 sizes
- CfL with derived alpha
- deblocking filter

## Full intra is opt-in

Full directional intra is bit-exact but not a rate/quality win at QP 90 on this
corpus:

| Still | Default (DC) | Full intra |
|---|---|---|
| 64×64 | 936 B / 43.28 dB | 979 B / 43.26 dB |
| 640×360 | 13,355 B / 46.37 dB | 13,199 B / 46.30 dB |
| 1080p | 37,856 B / 51.41 dB | 38,559 B / 51.28 dB |

The mode search is SSE-only; it needs a rate term (RDO) before it can pay for
the mode signalling. Until then `--m1` stays DC-safe and `--m1-full-intra`
enables the search.

## Unified RDO loop

`src/encoder/m1/rdo_loop/` replaces the SSE-only partition/mode decisions with a
joint `SSE + lambda * coded-bits` search over real coefficient bits
(`lambda = sse_scale * 66^2`, QP 90 step 66). Measured at QP 90, one frame,
`--m1` plus the loop (`AV2ENC_RDO_LOOP=1`, scale 0.0005):

| Still | Heuristic | Unified loop | Delta |
|---|---|---|---|
| 64x64 | 936 B / 43.28 dB / 9.9009 | 918 B / 43.57 dB / 9.9191 | -2.0% B, +0.17 dB, +0.018 JOD |
| 640x360 | 13,355 B / 46.37 dB / 9.8867 | 13,693 B / 49.06 dB / 9.9505 | +2.5% B, +3.03 dB, +0.064 JOD |
| 1080p | 61,574 B / 47.73 dB / 9.8976 | 61,894 B / 51.02 dB / 9.9253 | +0.5% B, +3.68 dB, +0.028 JOD |

This is the first change that improves PSNR **and** CVVDP together on every
fixture. It supersedes the `heuristic + tools` row, which is worse than the
heuristic alone. The loop is default-off until the M1 core calls it; the
patched binary is byte-identical to the reference when the hook is unset.

## Known limits

- Directional/smooth **chroma** is available behind `M1Options::uv_full`
  (`AV2ENC_M1_UV_FULL`; planned CLI `--m1-uv-full`). CPU verify is bit-exact on
  64x64 / 640x360 / native 1080p and on forced D45/D203/D67/SMOOTH/SMOOTH_V,
  including derived `AngleDeltaUV`. Candidates are scored on U+V in this mode
  (the default DC/CfL path keeps its historical U-only DC score). Default
  remains DC/CfL: at QP 90 the SSE-only search measured 13,154 B vs 13,132 B
  (640x360, U -0.31 dB / V +0.11 dB) and 61,690 B vs 61,762 B (1080p,
  U -0.11 dB / V +0.03 dB), i.e. roughly neutral without a rate term. The GPU
  chroma candidate loop is not wired; the GPU module rejects both uv-full flags
  rather than diverging.
- Explicit CfL alpha signalling is not wired; derived alpha is used
  (`cfl_index = 1`).
- The 1080p row uses a Lanczos upscale of the 640×360 test frame; it is a proxy,
  not a natural 1080p corpus.
- Target validation still requires an RTX 3060 enumerating 8192 MiB; the 12 GB
  dev box is a proxy.

## Opt-in feature integration

CDEF, coefficient RDOQ, intra luma TX-type selection, rate-aware intra, full
chroma UV and band-offset CCSO are now wired as opt-in M1 flags, each with a
`dav2d --strict 1` gate and pinned MD5s.  The default `--m1` baseline is
unchanged.  Status, flags, and what is still off (Wiener NS, extended
partitions, 10-bit, GPU ports): `docs/m1-features.md`.
