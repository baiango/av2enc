# Video coefficient RDOQ (`--video-rdoq`)

Status: **measured positive on the nightly tree; enabled by `--nightly`.**
The flag wires the full coefficient trellis from `src/codec/rate/rdoq/` into
the CPU `--video` luma inter-residual path and prices the luma TX plan search
with the same static AV2 §8.3 coefficient model. It changes coefficient levels
and TX plan choices; the syntax, the sequence/frame headers and the dequantiser
are untouched, so there is no new syntax element and no header change.

Without the flag nothing is read: the default stream (25,009 B /
`a1d0dba1...`) and the `quality` preset stream (25,141 B / `3984677b...`) keep
their bytes. `--nightly` now turns the flag on, so its own stream changes.

```sh
av2enc testdata/nasa_480x240_30fps_60f.y4m -o /tmp/x.ivf \
    --video --frames 60 --qp 117 --cpu-threads 8 --nightly
```

`--video-rdoq-scale X` overrides the SSE-domain lambda calibration
(`lambda = X * (1/quant_factor)^2`, default `0.1`) and implies
`--video-rdoq`.

## Mechanism

1. **Coefficient trellis (`codec/rate/rdoq/`).** Every luma inter-residual TX
   block is re-quantised by the full reverse-scan DP over coefficient levels
   (`rdoq_quantize`). The distortion is the forward-coefficient SSE; the rate
   is the static AV2 §8.3 model from `rdoq_av2_model`, sliced by the frame
   `base_q_idx` (`qctx = slice_for_base_q_idx(base_q_idx)`), which is the CDF
   set the video writer initialises. The trellis returns syntax levels, so the
   writer, the §7.14.4 dequantiser, the inverse transform and the decoder
   consume exactly the same coefficients.
2. **Inter plane contexts.** `rdoq::av2::BlockInfo` carries `is_inter`
   (default `false`, so the still path is byte-identical). For inter luma the
   model selects `txb_skip[1][...]` and eob plane context 1, matching
   `video_syntax.cpp`. `txb_skip_context = dc_sign_context = 0` because the
   block-level transform stage does not carry the above/left cumulative-level
   arrays; this is the documented frozen-context approximation.
3. **AV2-rate-priced TX plan search.** The historical luma TX search prices a
   candidate at `type_bits + 1.0 * nonzero`, which cannot see level
   magnitudes, EOB or the skip symbol. Under `--video-rdoq` every inter luma
   candidate is priced by the same AV2 model (`block_syntax_bits`, already used
   by `--video-tx-context-rate`), scaled by `kModelRateScale = 0.4` for the
   adaptive tile coder, plus the TX-type bits. The scale is the original
   integration's calibration; the `--video-tx-context-rate` path itself stays
   unscaled.
4. **Winner refinement and guard.** After the plan search picks the winning
   plan, each of its leaves is re-quantised by the trellis. A refined leaf is
   kept only when it is codable (an all-zero inter leaf still codes its
   TX_TYPE) and does not increase the plan-search cost
   `sse + tx_lambda * model_bits`; otherwise the scalar leaf is kept.
5. **Luma only, scalar quantiser only.** Chroma stays scalar, and the flag is
   rejected with `--video-qm` (per-coefficient QM step), `--video-tcq` (the TCQ
   trellis owns the levels), segmentation / delta-Q (the model uses the frame
   `base_q_idx` slice), lossless, the CUDA backend and non-8-bit 4:2:0
   formats. It also requires `--video-inter-residual` and the single-tile
   plain P chain.

The two quantisers are wired in both luma inter paths: the TX size/type/
partition search (`src/pipeline/video/video_tx_search.cpp`, used whenever
`--video-tx-search` is on) and `compute_tx` for the plain single-TX path. The
TX search is the path the RD measurements use. The two paths express
`quant_step` differently (the search carries the §7.14.2 `get_q()` step, the
plain path uses `1/quant_factor`), inherited from the original integration;
each is individually consistent with its own quantiser.

## Measured RD

Host: Apple M4 (10 cores, shared), `AV2ENC_CPU_FAST=ON`, `--cpu-threads 8`.
Clip: `testdata/nasa_480x240_30fps_60f.y4m`, 60 frames. Quality:
`tools/quality.py` (CVVDP 0.5.7, `standard_fhd`, MPS) with the pinned
`dav2d-macos` decoder. The reference is `--nightly` **without** RDOQ (the
previous edge + CCSO aggregate), whose operating point is QP 142 at
100.248 kbps / 8.7465 JOD / 35.3754 dB.

`--nightly` (now with RDOQ at the default scale 0.1):

| QP | kbps | CVVDP | PSNR-Y |
|---:|---:|---:|---:|
| 123 | 109.844 | 9.0164 | 36.4020 |
| 124 | 105.396 | 8.9858 | 36.3930 |
| **125** | **103.984** | **8.9613** | **36.2989** |
| 126 | 98.220 | 8.9411 | 36.2173 |
| 127 | 96.952 | 8.9293 | 36.2486 |
| 128 | 95.528 | 8.9051 | 36.0758 |

Iso-rate comparison (linear interpolation on the two bracketing QPs):

| Operating rate | previous nightly JOD | RDOQ nightly JOD | ΔJOD | ΔPSNR-Y |
|---:|---:|---:|---:|---:|
| 100.248 kbps | 8.7465 | **8.9482** | **+0.202** | +0.871 dB |
| 96.840 kbps | 8.6811 | **8.9274** | **+0.246** | +1.066 dB |

At the old operating point (100.248 kbps) the new QP-126 stream is 98.22 kbps
at 8.9411 JOD, i.e. it beats the old point by +0.195 JOD at a *lower* rate;
the QP-125 stream is higher rate and higher quality. The extra gain over the
plain model-priced search is the trellis: with the lambda calibration disabled
(`--video-rdoq-scale 1e-9`, scalar levels in the same model-priced search) the
same iso-rate point is ≈8.77, i.e. only +0.02 over the reference.

### Lambda calibration

Iso-rate (100.248 kbps) CVVDP for each `--video-rdoq-scale`; each row is
interpolated from a QP pair that brackets the operating rate:

| `rdoq_sse_scale` | iso-rate JOD | Δ vs reference |
|---:|---:|---:|
| 1e-9 (trellis off) | ≈8.770 | +0.02 |
| 0.003 (old default) | 8.800 | +0.053 |
| 0.005 | 8.809 | +0.063 |
| 0.01 | 8.865 | +0.119 |
| 0.04 | 8.922 | +0.175 |
| 0.08 | 8.957 | +0.210 |
| **0.1 (new default)** | **8.948** | **+0.202** |
| 0.125 | 8.949 | +0.203 |
| 0.16 | 8.951 | +0.205 |
| 0.32 | 8.910 | +0.164 |

The plateau is broad (0.08–0.16). The QP ladder is not perfectly monotone:
individual QPs flip TX plans, so a single point moves ~0.01–0.02 JOD. 0.1 is
the chosen default because it sits in the middle of the plateau; 0.08 and 0.16
measure within the noise of it.

The old default 0.003 came from the pre-edge `exp/rd-c` tree, where the
operating QP was ~190 and the flag was measured +0.035 JOD. At the current
nightly operating point (QP ~125 after RDOQ) 0.003 leaves most of the gain on
the table.

## Encode time

Interleaved runs on the shared host, 60 frames, `--cpu-threads 8`; the
reference is the explicit previous nightly flags
(`--video-rd-preset quality --video-ccso-rd --video-edge-inter`):

| Build | wall | user CPU |
|---|---:|---:|
| previous nightly (QP 142) | 5.73–6.74 s | 6.29–6.40 s |
| RDOQ nightly (QP 125) | 8.64–9.21 s | 8.24–8.47 s |

≈1.3x user CPU time (≈1.4–1.5x wall on the loaded host): the trellis runs once
per coded luma leaf and the static model rate once per inter candidate. This is
the CPU reference mode, not a speed target.

## Verification

* `dav2d --strict 1` decodes the nightly stream; the decoder payload equals the
  encoder `--video-recon` payload byte for byte (checked by
  `tools/gate_video_nightly.sh` for the expanded nightly flag set, and by
  `tools/gate_video_rdoq.sh` for the standalone `quality --video-rdoq` path).
* Default 480x240 QP189 stays **25,009 B / `a1d0dba1...`**; `quality` QP190
  stays **25,141 B / `3984677b...`**.
* `tools/gate_video_rdoq.sh` checks the usage rejections, both pins, that the
  stream changes and shrinks, `--strict 1`, and the recon equality.
* `tools/gate_video_nightly.sh` checks that `--nightly` expands to the
  documented flags (now including `--video-rdoq`) with byte-identical IVF.

## Caveat

This is one clip at one quality point on one host. The gain is large because
the reference quantiser is the plain scalar quantiser: the trellis removes
coefficients that do not pay for their syntax and spends the saved rate at a
finer QP, which also raises PSNR-Y by ~0.9 dB. Content-dependence and the
exact optimal lambda are not yet mapped; the broad 0.08–0.16 plateau makes the
choice of 0.1 low-risk, but re-check it if the nightly decision set changes.
