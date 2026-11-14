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
--video-subpel-eighth --video-tx-lambda 136
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
| 116 | 25,163 | 100.652 | 9.5642 | 41.5299 |
| **117** | **24,939** | **99.756** | **9.5868** | **41.6953** |

At the established 100 kbps point:

| Stream | Rate | CVVDP | PSNR-Y |
|---|---:|---:|---:|
| default QP189 | 100.036 kbps | 4.8021 | 24.4667 dB |
| `quality` QP190 | 100.564 kbps | 6.3021 | 27.5816 dB |
| previous nightly QP117 (no 1/8-pel/TX retune) | 100.088 kbps | 9.5467 | 40.8670 dB |
| **`--nightly` QP117** | **99.756 kbps** | **9.5868** | **41.6953 dB** |
| at iso-rate 100.088 kbps (QP116/117) | 100.088 kbps | **≈9.584** | **≈41.66 dB** |

The final 1/8-pel ME refinement plus the RDOQ-tuned TX-search lambda add
**+0.040 JOD / +0.83 dB** at iso-rate. Nightly is **+4.78 JOD** over the default
and **+3.28 JOD** over the stable `quality` preset at the same rate. The stream
is 24,939 B, MD5 `8f8713dac14eaa74af770e9142e5eae9`; `dav2d-macos --strict 1`
decodes it and the 60-frame encoder recon payload equals the decoded payload
byte for byte (10,368,000 B).

Encode speed is **7.6 fps** (7.86 s for 60 frames, best of 3) on the M4 at the
QP117 operating point; the 1/8-pel stage adds ME work but also reduces rate.

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
