# Nightly CPU RD mode (`--nightly`)

`--nightly` is the latest measured CPU-only algorithmic RD aggregate. It is
**opt-in and CPU-only**: the default stream, the stable `quality` preset and
every GPU path keep their historical decisions and byte-identical streams.

```sh
av2enc testdata/nasa_480x240_30fps_60f.y4m -o /tmp/nightly.ivf \
    --video --frames 60 --qp 110 --cpu-threads 8 --nightly
```

Expansion (the gate asserts this is byte-identical to typing the flags out):

```text
--video-rd-preset quality
--video-ccso-rd --video-ccso-bands 6 --video-ccso-lambda 22 \
  --video-ccso-refit 2 --video-ccso-unit-bits 2
--video-edge-inter --video-edge-mask
--video-partition-rd --video-partition-rd-lambda 2.75e-6
--video-key-qp-offset -25
--video-rdoq
--video-cdef
--video-bawp --video-bawp-carrier
--video-leaf-rd
--video-subpel-eighth --video-tx-lambda 136
--video-key-intra-txpart
--avg-cdf --adaptive-cdf --video-cdf-per-frame-qp
--video-drift-refresh
```

`--nightly` rejects `--video-gpu` and `--still`. The bottom-edge path requires
the single-tile low-delay P chain (no B/reorder/temporal-layer modes, no
`--video-ref-mvs`); the normal combination validation enforces that.

## Measured RD

Host: Apple M4, `AV2ENC_CPU_FAST=ON`, `--cpu-threads 8`. Clip:
`testdata/nasa_480x240_30fps_60f.y4m`, 60 frames. Quality: `tools/quality.py`
(CVVDP 0.5.7, `standard_fhd`, MPS) with `dav2d-macos`.

The pin is QP110 with the `-25` reference-intro offset; the offset is the
freedom that the adaptive-CDF saving buys, so the sweep below is in offset
rather than QP (bytes and rate both move with it):

| key offset (QP110) | Bytes | kbps | CVVDP | PSNR-Y |
|---:|---:|---:|---:|---:|
| -24 | 24,441 | 97.764 | 9.7148 | 43.6910 |
| **-25** | **24,904** | **99.616** | **9.7069** | **43.8667** |
| -26 | 25,229 | 100.916 | 9.7102 | 43.9231 |

At the established 100 kbps point:

| Stream | Rate | CVVDP | PSNR-Y |
|---|---:|---:|---:|
| default QP189 | 100.036 kbps | 4.8021 | 24.4667 dB |
| `quality` QP190 | 100.564 kbps | 6.3021 | 27.5816 dB |
| previous nightly QP117 (no intra TX part) | 100.088 kbps | 9.5467 | 40.8670 dB |
| previous nightly QP110 (no CDF/drift) | 99.976 kbps | 9.6802 | 42.9799 dB |
| **`--nightly` QP110, key -25 + CDF + drift** | **99.616 kbps** | **9.7069** | **43.8667 dB** |
| at iso-rate 99.976 kbps | 99.976 kbps | **≈9.707** | **≈43.88 dB** |

The keyframe intra luma TX partition with per-part prediction/recon adds
**+0.093 JOD / +1.28 dB** at iso-rate: the three bad 64×16 keyframe leaves gain
~9 dB and the first P frame loses most of its repair cost. The adaptive-CDF
chain plus a lower key offset add a further **+0.028 JOD / +0.90 dB** at
iso-rate (it is one move: the entropy saving pays for the key bits), and the
drift refresh forces the inter residual on the frames whose reference has
slipped. Nightly is **+4.90 JOD** over the default and **+3.40 JOD** over the
stable `quality` preset at the same rate. The stream is 24,904 B, MD5
`5f44c0d1795357b5beec84adaed0fe5b`; `dav2d-macos --strict 1` decodes it and the
60-frame encoder recon payload equals the decoded payload byte for byte
(10,368,000 B).

Encode speed is **7.2 fps** (8.38 s for 60 frames, best of 3) on the M4 at the
QP110 operating point; the CDF chain is the cost, the refresh adds ~2%.

## Per-tool contributions

The aggregate was built by measuring each change at iso-rate and keeping only
the ones that stacked. Approximate contributions at or near the 100 kbps point:

| Change | Contribution |
|---|---|
| Bottom-edge inter (`--video-edge-inter`) | +2.39 JOD over the old quality preset |
| Measured partition probe (`--video-partition-rd`) | +0.41 JOD over edge+CCSO |
| Reference-intro key/first-P QP offset | +0.22 JOD |
| Luma inter RDOQ (`--video-rdoq`) | +0.20 JOD |
| CDEF (bounded search) | +0.20 JOD |
| 64-band/refit/priced-unit CCSO | +0.15 JOD |
| Bottom residual mask (`--video-edge-mask`) | +0.07 JOD |
| BAWP carrier over inter residual | +0.04 JOD |
| Calibrated leaf RD cost (`--video-leaf-rd`) | +0.05 JOD |
| 1/8-pel ME + TX-search λ 136 | +0.04 JOD |
| Keyframe intra luma TX partition | +0.09 JOD |
| Adaptive coefficient CDFs (per-frame reset) | −13.9% rate at identical decisions |
| Key -16 → -25 offset (paid by the CDF saving) + drift refresh | +0.03 JOD |

Positive changes that did **not** stack and are therefore explicit opt-ins,
not part of nightly: the §8.3.2 TX context rate (it needs
`--video-tx-lambda 48` and then loses to this aggregate), the MV mode-RD
candidate term on its own (+0.004 JOD at the operating point; it is one part of
the leaf RD cost below), and the keyframe intra RD re-rank (+0.007 JOD).
Chroma-aware intra/inter RD is a measured null on this achromatic clip.
Per-frame drift refresh (`--video-drift-refresh`,
`docs/video-drift-refresh.md`) is now part of the aggregate: on its own it is
**+0.022 JOD / +0.12 dB at iso-rate 99.976 kbps** (+0.89 dB mean luma on the
refreshed frames 32-40) for +2.6% rate at the same QP, and it stacks with the
CDF/key-QP move above. It is CPU-only, single-tile, cqp and 8-bit 4:2:0, and it
composes with the CDF chain only while the refresh keeps every frame's QP
(`--video-drift-refresh-qp 0`, the default).

## Inter residual RD (`--video-leaf-rd-lambda`)

`--video-leaf-rd` prices a residual block as `coded SSE + lambda(qindex) *
bits` with the anchor constant `c = 125` at QP104. The trace of frame 40 shows
that constant, not the entropy coder, is what makes the drifted P frames skip
every leaf: at QP80 the frame codes 0 of 58 leaves while the best leaf's coded
SSE is 6x smaller than its prediction SSE. `--video-leaf-rd-lambda` exposes the
anchor (default 125, byte-identical). Lowering it to 80 at matched rate gives
+0.019 JOD at -0.13 dB PSNR - a real direction but inside the ripple, so it is
measurement evidence, not a shipped default. Analysis and the per-leaf tables:
`docs/video-inter-residual-rd.md`.

## Leaf RD cost (`--video-leaf-rd`)

One cost replaces the three historical leaf heuristics (inter-candidate
SSE-only choice, the fixed `inter_rd_lambda = 4.0` skip/residual decision, and
the `3.0 + …` / `8.0 + …` intra/inter comparison):

```text
J = coded SSE + lambda(qindex) * bits
lambda(qindex) = 125 * sqrt((qlookup(qindex)/8) / (qlookup(104)/8))
```

The bit terms mirror the signalled codeword shape: YMode set/index/escape, MRL
index + secondary flag, DIP, UV mode and CfL/CCTX, `single_mode`, the NEARMV
DRL chain and the shell-coded NEWMV MVD. The reference lambda was re-measured
on this aggregate (partition probe + RDOQ already price coefficients, so the
leaf needs less rate pressure): 125 beats the standalone 250 by ≈0.04 JOD at
iso-rate, and the measured optimum is flat from 0.4× to 0.6× of the current
value. A `step^2` lambda collapses the P chain into skip-only frames above
QP 110, so the fit follows `sqrt(step)` instead.

## Adaptive CDFs with per-frame QP (`--video-cdf-per-frame-qp`)

`--avg-cdf --adaptive-cdf` used to require one fixed `base_q_idx`: the §8.2.2
CDF image was seeded once from that frame-level quantizer. `--nightly` codes a
per-frame QP plan (key `--qp - 16`, P `--qp`), so the aggregate could not turn
adaptation on.

`--video-cdf-per-frame-qp` opts into a reset-per-frame image instead of a
carry: every frame starts from the §9.3 defaults, and
`disable_cross_frame_cdf_init` stays 1. §8.3.1 selects those defaults from the
**current frame's** `base_q_idx`, so the encoder re-derives the flat default
image for every frame and a plan that crosses a qctx slice is legal. Pinned
`dav2d` accepts that form: it installs exactly the same per-frame §9.3 image
the encoder seeded. The carry form (`disable_cross_frame_cdf_init = 0`) is a
separate story: the pinned decoder does implement §7.4.4, but our post-frame-1
§7.5/§8.2.4 image diverges whenever that frame codes a real tile, and
`--cdf-continuity` stays behind its env gate until that is fixed
(`exp/rd7-floor` has the experiments; note `dav2d` exits 0 after failing
pictures, so check the decoded frame count, not just the exit status).

### Same quality, fewer bytes

At a fixed QP the flag changes no decision: the decoded output is
byte-identical to the frozen stream (MD5 `64be64f05c659f86b52456519ec65330`
for both at QP 110), so CVVDP and PSNR are unchanged while the entropy layer
shrinks. On the QP110 pin the plan is key 94 / P 110, which crosses slices 2
and 3, i.e. exactly the case that used to be rejected.

| Stream | Bytes | kbps | CVVDP | PSNR-Y |
|---|---:|---:|---:|---:|
| `--nightly` QP110 (key 94) | 24,994 | 99.976 | 9.6802 | 42.9799 |
| same, `+ --video-cdf-per-frame-qp` | 21,530 | 86.120 | 9.6802 | 42.9799 |

That is −13.86% rate at identical reconstruction. The QP110 CDF stream is
21,530 B, MD5 `b6220e473ef46b367d27c7925b1e8c23`.

### Spending the saving: the key QP, not the global QP

Freeing 13.9% of the stream does not help by itself, because lowering the
*global* QP buys almost nothing on this clip: the CVVDP loss is carried by the
drifted tail of the P chain (f33-f59 at 41-42 dB), which barely responds to
QP. Measured at iso-rate: QP99 with the CDF is 24,965 B / 99.860 kbps /
9.6703 JOD / 43.4072 dB, i.e. **-0.010 JOD** against the QP110 pin despite
+0.43 dB PSNR.

The saving does pay when it is spent on the **keyframe**, which is 70% of the
bits and the reference for the whole chain. Fixing P at QP110 and moving the
reference-intro offset down:

| Plan (QP110, adaptive CDF) | Bytes | kbps | CVVDP | PSNR-Y |
|---|---:|---:|---:|---:|
| key -16 (no change) | 21,530 | 86.12 | 9.6802 | 42.98 |
| key -24 | 23,981 | 95.92 | 9.6908 | 43.38 |
| **key -26** | **24,516** | **98.06** | **9.7200** | **43.83** |
| key -27 | 24,749 | 99.00 | 9.6927 | 43.57 |
| key -28 | 25,249 | 101.00 | 9.6955 | 43.45 |

The QP-to-QP CVVDP ripple on this clip is ±0.03 JOD, so read the gain as
**+0.01 to +0.02 JOD at iso-rate** (9.694-9.704 interpolated to 99.976 kbps),
with a firmer **+0.6 to +0.9 dB PSNR-Y**. For reference, dropping the key QP by
26 without the CDF costs ~27.9 kB and does not fit the point at all.

Recommended nightly composition: the flag and a `--video-key-qp-offset -26`
retune move together, since the flag is what pays for the lower key QP. It is
not folded into the QP110 pin yet so the in-flight frame-level levers keep a
comparable baseline.

## Where the bits and the loss are
## Where the bits and the loss are

Bit split of the 24,994-byte nightly stream (`--qp 110`), from the encoder's
tile accounting plus the IVF frame sizes:

| Layer | Bytes | Share |
|---|---:|---:|
| Keyframe tile | 17,623 | 70.5% |
| P1 tile (frame 1) | 59 | 0.2% |
| P2-P59 tiles | 4,740 | 19.0% |
| Per-frame OBU headers (60 frames) | 1,820 | 7.3% |
| IVF + sequence header | 752 | 3.0% |

P1 is cheap because its source is **byte-identical to frame 0**, so the
keyframe is its own perfect predictor: the reversed skip decision costs 59
tile bytes (47 inter blocks, 0 intra) and its reconstruction is a copy of the
keyframe's. Frame 1 therefore scores 44.75 dB, the best frame in the stream.

That also bounds how much a P1 refresh could ever be worth. Replacing the
decoded frame with the source (i.e. spending infinite bits on it) and
re-scoring with the same CVVDP model:

| Oracle (decoded frame -> source) | CVVDP | delta |
|---|---:|---:|
| baseline stream | 9.6802 | - |
| frame 1 | 9.6827 | +0.0025 |
| frame 2 | 9.6827 | +0.0025 |
| keyframe | 9.6857 | +0.0055 |
| keyframe + frame 1 | 9.6865 | +0.0063 |
| 10 worst frames | 9.7292 | +0.0490 |
| 30 worst frames | 9.8515 | +0.1713 |

One frame of 60 is worth about +0.0025 JOD no matter which frame it is, so the
duplicated opening pair is not a lever; a real refresh (residual refinement,
938-1,911 B) captures strictly less than its oracle. The loss is spread
uniformly over the whole sequence instead, which is why the remaining gap to
SVT-AV1 shows up on **every** frame at the same rate and byte count:

| Frame | AV2 nightly | SVT preset 0 / CRF 36.5 | delta |
|---|---:|---:|---:|
| 0 (key) | 44.71 dB | 50.15 dB | +5.44 |
| 1 | 44.75 | 50.27 | +5.52 |
| 5 | 43.70 | 50.18 | +6.48 |
| 20 | 43.99 | 50.17 | +6.18 |
| 30 (worst region) | 43.40 | 49.20 | +5.81 |
| 40 (worst) | 41.14 | 47.46 | +6.32 |
| 50 | 42.21 | 48.20 | +5.99 |
| 59 | 43.96 | 47.01 | +3.05 |
| mean Y | 43.11 | 48.98 | +5.87 |

Both encoders spend ~70% of the stream on the keyframe and code the P chain in
about 4.7-4.9 kB; SVT converts the same bytes into ~6 dB on each frame. The
work is per-frame coding efficiency (intra RD and coefficient coding in the
keyframe, residual coding in P), not frame-level rate allocation.

## Verification

- `dav2d-macos --strict 1` decodes the final QP117 stream.
- 60-frame `--video-recon` payload equals the decoded payload byte for byte;
  the hidden padded rows are outside the Y4M payload.
- Default 480x240 QP189 stays **25,009 B / `a1d0dba1...`**.
- `quality` QP190 stays **25,141 B / `3984677b...`**.
- `tools/gate_video_nightly.sh` checks the expansion, both pins (the nightly
  pin is QP117 / 24,939 B / `8f8713da...`), strict decode, visible recon
  equality, BAWP carrier activity and the GPU/`--still` rejections.
- `src/pipeline/video/gate_avg_cdf.sh` adds the `--video-cdf-per-frame-qp`
  case (strict decode + visible recon at key 174 / P 190) and the named
  rejections: a per-frame plan without the
  opt-in, and the opt-in without the CDF chain.
- `tools/gate_video_edge_inter.sh`, `tools/gate_video_rd_preset.sh` and
  `gate_video_any_dims.sh` (including the MI-edge 100x70/640x362 cases) pass.


## Caveats

This is one clip at one quality point on one host. The bottom-edge and BAWP
gains are content-dependent; the partition probe, RDOQ, CDEF and CCSO
contributions should be re-measured on other content and rates. `--nightly` is
a CPU development mode, not the GPU product path, and it is deliberately not
the default.
