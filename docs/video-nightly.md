# Nightly CPU RD mode (`--nightly`)

`--nightly` is the latest measured CPU-only algorithmic RD aggregate. It is
**opt-in and CPU-only**: the default stream, the stable `quality` preset and
every GPU path keep their historical decisions and byte-identical streams.

```sh
av2enc testdata/nasa_480x240_30fps_60f.y4m -o /tmp/nightly.ivf \
    --video --frames 60 --qp 117 --cpu-threads 8 --nightly
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
| 114 | 26,017 | 104.068 | 9.5408 | 41.0384 |
| 115 | 25,481 | 101.924 | 9.5484 | 40.9679 |
| 116 | 25,294 | 101.176 | 9.5358 | 40.9400 |
| **117** | **25,022** | **100.088** | **9.5467** | **40.8670** |
| 118 | 24,728 | 98.912 | 9.5241 | 40.7431 |
| 120 | 23,946 | 95.784 | 9.5175 | 40.6268 |

At the established 100 kbps point:

| Stream | Rate | CVVDP | PSNR-Y |
|---|---:|---:|---:|
| default QP189 | 100.036 kbps | 4.8021 | 24.4667 dB |
| `quality` QP190 | 100.564 kbps | 6.3021 | 27.5816 dB |
| previous nightly QP125 (no leaf RD) | 100.228 kbps | 9.4918 | 40.6553 dB |
| **`--nightly` QP117 (leaf RD)** | **100.088 kbps** | **9.5467** | **40.8670 dB** |
| at iso-rate 100.228 kbps (QP116/117) | 100.228 kbps | **9.5453** | **40.876 dB** |

The calibrated leaf RD cost (`--video-leaf-rd`, §"Leaf RD cost" below) adds
**+0.0535 JOD / +0.22 dB** at the established 100.228 kbps point. Nightly is
**+4.74 JOD** over the default and **+3.24 JOD** over the stable `quality`
preset at the same rate. The stream is 25,022 B, MD5
`b7388311a0e13cd90c7e1c9a2883588e`; `dav2d-macos --strict 1` decodes it and the
60-frame encoder recon payload equals the decoded payload byte for byte
(10,368,000 B).

Across the measured CVVDP range (9.48–9.54 JOD) the PCHIP BD-rate is
**−9.1%** (raw −9.3%) versus the previous aggregate; at iso-CVVDP 9.4918 the
leaf-RD curve reaches ≈91.6 kbps (−8.6% rate).

Encode speed is **9.1 fps** (6.58 s for 60 frames, best of 3) on the M4 at the
QP117 operating point; the previous QP125 aggregate measured 7.7 fps (7.84 s)
on the same host session. The leaf cost replaces residual work with a few syntax
bits, so the higher-rate point is not slower.

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

Positive changes that did **not** stack and are therefore explicit opt-ins,
not part of nightly: the §8.3.2 TX context rate (it needs
`--video-tx-lambda 48` and then loses to this aggregate), the MV mode-RD
candidate term on its own (+0.004 JOD at the operating point; it is one part of
the leaf RD cost below), and the keyframe intra RD re-rank (+0.007 JOD).
Chroma-aware intra/inter RD is a measured null on this achromatic clip.

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

## Verification

- `dav2d-macos --strict 1` decodes the final QP117 stream.
- 60-frame `--video-recon` payload equals the decoded payload byte for byte;
  the hidden padded rows are outside the Y4M payload.
- Default 480x240 QP189 stays **25,009 B / `a1d0dba1...`**.
- `quality` QP190 stays **25,141 B / `3984677b...`**.
- `tools/gate_video_nightly.sh` checks the expansion, both pins (the nightly
  pin is QP117 / 25,022 B / `b7388311...`), strict decode, visible recon
  equality, BAWP carrier activity and the GPU/`--still` rejections.
- `tools/gate_video_edge_inter.sh`, `tools/gate_video_rd_preset.sh` and
  `gate_video_any_dims.sh` (including the MI-edge 100x70/640x362 cases) pass.

## Caveats

This is one clip at one quality point on one host. The bottom-edge and BAWP
gains are content-dependent; the partition probe, RDOQ, CDEF and CCSO
contributions should be re-measured on other content and rates. `--nightly` is
a CPU development mode, not the GPU product path, and it is deliberately not
the default.
