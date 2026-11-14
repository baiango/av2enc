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
--video-key-qp-offset -16
--video-rdoq
--video-cdef
--video-bawp --video-bawp-carrier
--video-leaf-rd
--video-subpel-eighth --video-tx-lambda 136
--video-key-intra-txpart
```

`--nightly` rejects `--video-gpu` and `--still`. The bottom-edge path requires
the single-tile low-delay P chain (no B/reorder/temporal-layer modes, no
`--video-ref-mvs`); the normal combination validation enforces that.

## Measured RD

Host: Apple M4, `AV2ENC_CPU_FAST=ON`, `--cpu-threads 8`. Clip:
`testdata/nasa_480x240_30fps_60f.y4m`, 60 frames. Quality: `tools/quality.py`
(CVVDP 0.5.7, `standard_fhd`, MPS) with `dav2d-macos`.

| QP | Bytes | kbps | CVVDP | PSNR-Y |
|---:|---:|---:|---:|---:|
| 110 | **24,994** | **99.976** | **9.6802** | **42.9799** |
| 111 | 25,062 | 100.248 | 9.6808 | 42.9665 |
| 112 | 24,731 | 98.924 | 9.6730 | 42.8777 |

At the established 100 kbps point:

| Stream | Rate | CVVDP | PSNR-Y |
|---|---:|---:|---:|
| default QP189 | 100.036 kbps | 4.8021 | 24.4667 dB |
| `quality` QP190 | 100.564 kbps | 6.3021 | 27.5816 dB |
| previous nightly QP117 (no intra TX part) | 100.088 kbps | 9.5467 | 40.8670 dB |
| **`--nightly` QP110** | **99.976 kbps** | **9.6802** | **42.9799 dB** |
| at iso-rate 100.088 kbps (QP110/111) | 100.088 kbps | **≈9.680** | **≈42.97 dB** |

The keyframe intra luma TX partition with per-part prediction/recon adds
**+0.093 JOD / +1.28 dB** at iso-rate: the three bad 64×16 keyframe leaves gain
~9 dB and the first P frame loses most of its repair cost. Nightly is
**+4.88 JOD** over the default and **+3.38 JOD** over the stable `quality`
preset at the same rate. The stream is 24,994 B, MD5
`bc4a3756d312bd259e97397f9bd86853`; `dav2d-macos --strict 1` decodes it and the
60-frame encoder recon payload equals the decoded payload byte for byte
(10,368,000 B).

Encode speed is **8.1 fps** (7.38 s for 60 frames, best of 3) on the M4 at the
QP110 operating point.

## Per-tool contributions

The aggregate was built by measuring each change at iso-rate and keeping only
the ones that stacked. Approximate contributions at or near the 100 kbps point:

| Change | Contribution |
|---|---|
| Bottom-edge inter (`--video-edge-inter`) | +2.39 JOD over the old quality preset |
| Measured partition probe (`--video-partition-rd`) | +0.41 JOD over edge+CCSO |
| Reference-intro key/first-P QP offset (`-16`) | +0.22 JOD |
| Luma inter RDOQ (`--video-rdoq`) | +0.20 JOD |
| CDEF (bounded search) | +0.20 JOD |
| 64-band/refit/priced-unit CCSO | +0.15 JOD |
| Bottom residual mask (`--video-edge-mask`) | +0.07 JOD |
| BAWP carrier over inter residual | +0.04 JOD |
| Calibrated leaf RD cost (`--video-leaf-rd`) | +0.05 JOD |
| 1/8-pel ME + TX-search λ 136 | +0.04 JOD |
| Keyframe intra luma TX partition | +0.09 JOD |

Positive changes that did **not** stack and are therefore explicit opt-ins,
not part of nightly: the §8.3.2 TX context rate (it needs
`--video-tx-lambda 48` and then loses to this aggregate), the MV mode-RD
candidate term on its own (+0.004 JOD at the operating point; it is one part of
the leaf RD cost below), and the keyframe intra RD re-rank (+0.007 JOD).
Chroma-aware intra/inter RD is a measured null on this achromatic clip.
The per-frame adaptive-CDF path (`--avg-cdf --adaptive-cdf
--video-cdf-per-frame-qp`) is also opt-in: it cuts 13.95% of the rate at
identical decisions, but the exact 100 kbps mapping with the `-16` key offset
crosses a coefficient-qctx slice boundary, so it is not part of this nightly
pin. It is the next rate-efficiency lever once the slice-crossing case is
handled.

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
carry: every frame starts from the §9.3 defaults for its own §8.3.1
coefficient qctx slice, and `disable_cross_frame_cdf_init` stays 1 (the pinned
decoder rejects the carry form). Resetting is only equivalent to one shared
image when every frame lands in the same slice, so a plan that crosses a
slice boundary is rejected by name. At `--qp 117` the key QP is 101 and the P
QP is 117, both in slice 1.

The flag changes no decision: at every tested QP the decoded output is
byte-identical to the frozen stream, so CVVDP and PSNR are unchanged while the
entropy layer is ~14% smaller.

```sh
av2enc testdata/nasa_480x240_30fps_60f.y4m -o /tmp/cdf117.ivf \
    --video --frames 60 --qp 110 --cpu-threads 8 --nightly \
    --avg-cdf --adaptive-cdf --video-cdf-per-frame-qp
```

| QP | frozen bytes | adaptive bytes | rate Δ | CVVDP (both) | PSNR-Y (both) |
|---:|---:|---:|---:|---:|---:|
| 114 | 25,821 | 22,170 | −14.14% | 9.5725 | 41.6478 |
| 115 | 25,485 | 21,944 | −13.89% | 9.5913 | 41.6659 |
| 116 | 25,163 | 21,621 | −14.08% | 9.5642 | 41.5299 |
| **117** | **24,939** | **21,461** | **−13.95%** | **9.5868** | **41.6953** |
| 118 | 24,344 | 20,908 | −14.11% | 9.5547 | 41.2929 |
| 119 | 24,160 | 20,763 | −14.06% | 9.5552 | 41.2985 |
| 120 | 24,021 | 20,650 | −14.03% | 9.5446 | 41.2287 |

The key frame carries ~73% of the tile bits and its coefficient entropy
shrinks by 10.4% (130,984 → 117,408 bits at QP117); the already
skip-dominated P payload shrinks by 29.9%. The tile is 15.6% smaller
(22,315 → 18,842 B) and the OBU/header overhead leaves the IVF 13.95%
smaller. The QP117 stream is 21,461 B, MD5
`cc4bf99f193ff80ac4420d3f6b4ac69b`.

### Iso-rate

- **Same quality:** QP117 is 85.844 kbps instead of 99.756 kbps (−13.95%) with
  identical reconstruction and CVVDP. This is the primary result: the gain is
  a pure rate reduction on this operating point.
- **Same rate:** the lowest nightly point that stays in one slice is QP107
  (key 91 / P 107): 24,301 B / 97.204 kbps / 9.6078 JOD / 42.1715 dB. The
  frozen aggregate interpolated to the same 97.2 kbps (QP118/119) is
  9.5548 JOD / 41.294 dB, so **+0.053 JOD / +0.88 dB at iso-rate**.
- The nominal 99.756 kbps point needs QP≈106.3, where the key frame (90) falls
  in qctx slice 0 and P (106) in slice 1; the encoder rejects that plan by
  name. Interpolating the measured QP107/108 point gives **≈9.616 JOD
  (+0.03)** at 99.756 kbps.

CVVDP is close to saturation on this clip (the working range is ~9.54–9.61
JOD), so the flat JOD curve understates the rate reduction; the PSNR-Y and
same-quality byte deltas are the more legible measures here.

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
  rejections: a plan crossing a qctx slice, a per-frame plan without the
  opt-in, and the opt-in without the CDF chain.
- `tools/gate_video_edge_inter.sh`, `tools/gate_video_rd_preset.sh` and
  `gate_video_any_dims.sh` (including the MI-edge 100x70/640x362 cases) pass.


## Caveats

This is one clip at one quality point on one host. The bottom-edge and BAWP
gains are content-dependent; the partition probe, RDOQ, CDEF and CCSO
contributions should be re-measured on other content and rates. `--nightly` is
a CPU development mode, not the GPU product path, and it is deliberately not
the default.
