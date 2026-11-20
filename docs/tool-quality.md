# Tool-quality probes: which av2enc implementations lose to SVT/AVM

Status: **diagnostic, read-only.** Branch `exp/tool-quality`, base `0bf1d001`.
No encoder source was changed. The branch adds three first-party tools
(`tools/tool_probes.py`, `tools/tool_quality_run.py`,
`tools/tool_probe_evidence.py`) and this document. The clips are generated from
closed-form math and a fixed seed; no third-party asset or code is read.

This is not a BD-rate claim on natural video. It is a per-tool stress test at
the project's operating point: **480x240, 30 fps, 60 frames, 8-bit 4:2:0,
100 kbps**, scored with the project's own stack (AVM decode for AV2 streams,
`dav1d` for the AV1 reference, `tools/quality.py`, `tools/extra_metrics.py`,
`tools/kpi.py`).

## TL;DR

1. **The ME search is not the weakest implementation.** On all three
   fractional-translation probes av2enc finds the exact ground-truth MV for
   78-100% of inter leaves and all three probes use the right global motion.
   Removing the 1/8-pel stage does not hurt (it is slightly better at matched
   rate). The loss is what happens *after* ME: the leaf RD chooses skip and the
   key/residual reconstruction cannot repair the reference.
2. **The largest clean single-tool gap is intra/keyframe coefficient coding.**
   The larger 12-21 K gap on the ME probes decomposes into this key deficit
   plus the skip decision in point 3. On a static high-detail key picture
   av2enc and AVM use almost the same key bytes (22,995 vs 23,500 B), yet
   av2enc is **8.4 dB behind** (38.88 vs 47.31 dB PSNR-Y). On the static repeat
   probe it spends **more** key bytes than SVT (23,101 vs 21,967 B) for
   **4.2 dB less** (39.60 vs 43.82 dB). The key TX histogram is already small
   (82% 16x16), and `--video-rdoq` is inter-only and does not change the key —
   this is a coefficient-quantisation/entropy gap, not a TX-size gap.
3. **The second gap is the fractional-MV skip decision.** Nightly skips
   76/87/91% of inter leaves on the 1/2, 1/4 and 1/8 probes while the P-chain
   loses 2.3/4.0/5.8 dB relative to its own key; SVT and AVM stay flat or gain
   quality through the P chain. Simply disabling `--video-leaf-rd` at matched
   rate gains **+4.1 K** (75.7 vs 71.6) because it codes residual instead of
   skipping. eff-perc's skip flips recover +4.1/+3.9/+7.7 K but are still
   12-17 K behind the best peer.
4. **The third gap is BAWP on a linear fade.** Nightly is 6.5 K behind SVT
   (82.70 vs 89.25). BAWP is applied to 42% of candidates, but **disabling it
   raises K at matched rate** (about 84.4 vs 82.70): the current SSE-only
   acceptance is not rate-aware. AVM, the other BAWP encoder, is also at 82.98,
   so this is not "BAWP missing"; it is BAWP selection/pricing plus the
   P-chain's 96% skip rate.
5. **Exact repeats are fine.** The static-repeat probe is 100% zero-MV skip
   with zero drift in both av2enc presets; its 2.1 K gap to SVT is entirely the
   key picture. The near-static probe is 100% zero-MV skip too; its 4.4 K gap is
   key quality plus the unavoidable source-noise floor, not MV drift.

## 1. What the probes are

All clips are 480x240, 30 fps, 60 frames, C420, limited range (Y 16..235).
Textures are sums of integer-frequency cosines, which makes them periodic over
the frame and exactly translatable at fractional sample positions. Chroma is a
separate low-frequency wash (std 8); on the ME probes it translates at half the
luma rate.

| Probe | Family | Recipe and ground truth | Clean? |
|---|---|---|---|
| `me_half` | ME / subpel | shared mid-detail texture; apparent translation **(-3.5,-1.5) px/frame**, fractional part 1/2. Ground truth MV = `(28,12)` in 1/8-luma units. | Not alone: key share 35-67%; see §4.1 |
| `me_quarter` | ME / subpel | same texture; **(-2.25,+1.25) px/frame**, fractional part 1/4. Ground truth MV = `(18,-10)`. | same |
| `me_eighth` | ME / subpel | same texture; **(-1.125,+0.375) px/frame**, fractional part 1/8. Ground truth MV = `(9,-3)`. | same |
| `fade` | BAWP / compound | static 36-component texture; per-frame `y_n = 16 + (y_0-16)*alpha_n + offset_n`, `alpha` 1.0 -> 0.45, `offset` 0 -> 15, linear. | Cross-codec comparison is not tool-parity (SVT has no BAWP); the within-av2enc ablation is clean. |
| `static_repeat` | skip / drift | one detailed frame repeated exactly 60 times. Optimum: one key + zero-residual zero-MV skips. | Clean for skip/drift; the gap is the key, not the skip. |
| `near_static` | skip / drift | `static_repeat` + i.i.d. N(0,2 LSB) per frame. Optimum: skip (the noise cannot be represented at 100 kbps). | Partly unclean: source-noise floor and metric sensitivity. |
| `detail_edges` | transform / entropy | static image: left 58% dense texture (periods 3-10 px), right 42% 8x8 checker, 12-px bars, discs, a ramp. Key frame carries ~95% of the bits. | Clean for **intra** transform/entropy, not inter; see §4.4 for the SVT anomaly. |
| `grain` | filters | smooth base + 1-2 px correlated grain, sigma 12 LSB, new realisation each frame. | **Unclean.** Grain is incompressible at 100 kbps; CRF/QP curves are cliffs. Not ranked. |

The ME texture is identical across `me_half`, `me_quarter` and `me_eighth`;
only the translation vector changes, so the three clips isolate the fractional
part. The ground-truth MV is written to each `*.json` sidecar. The source is
sampled at `(x + dx*n, y + dy*n)`, so the apparent content motion is negative;
the decoder's sample offset into the previous frame is `(+dx,+dy)`.

Deterministic content hashes (SHA-256 prefixes, first 16 hex):

| Probe | SHA-256 prefix |
|---|---|
| `me_half` | `ac95853d80bb04e1` |
| `me_quarter` | `60a6371d98f88fb8` |
| `me_eighth` | `c3e5a5bd8735662e` |
| `fade` | `374d0f161b706e43` |
| `static_repeat` | `2a47f404290f6662` |
| `near_static` | `40da3ac724386f39` |
| `detail_edges` | `7e06196dd8b0e5bb` |
| `grain` | `528cd6c333620a8c` |

## 2. Encoders and rate matching

| Label | Encoder | Command |
|---|---|---|
| `nightly` | av2enc `--fast-decode` | `av2enc in.y4m -o out.ivf --video --frames 60 --qp Q --cpu-threads 8 --fast-decode` |
| `eff-perc` | av2enc nightly + `AV2ENC_EFF_PERC=1` + the documented drift recipe | same, plus `--video-drift-refresh-threshold 1.5 --video-drift-refresh-budget 3250` |
| `svt` | SVT-AV1 4.3.0 preset 0 | `SvtAv1EncApp -i in.y4m -b out.ivf --preset 0 --crf C --fps-num 30 --fps-denom 1 --keyint -1 --scd 0` |
| `avm` | AVM `e6f99d3` | `avmenc in.y4m -o out.ivf --ivf --end-usage=q --qp Q --cpu-used=6 --threads=8 --limit=60 --psnr=1` |

`svt` uses CRF, not fixed QP; it is the only AV1 encoder here. `--keyint -1
--scd 0` keeps one key picture so the comparison has the same GOP shape as the
AV2 encodes. AV2 streams are decoded by `tools/bin/avmdec`; the SVT stream is
decoded by `dav1d` (AVM cannot decode AV1). Both decoded Y4M files are scored
by the same `tools/quality.py` / `tools/extra_metrics.py` / `tools/kpi.py`.

**Bracketing.** For every encoder/probe the runner searches QP/CRF until two
points bracket 100.0 kbps, then tightens the bracket toward 6% rate width. The
reported point is a linear interpolation of K in rate at 100.0 kbps (the
project's convention). The table lists the actual bracket rates; several are
wider than the ideal because the encoder's rate curve is locally steep (SVT
CRF steps, AVM QP steps, and the one-QP cliff on the grain probe). K computed
from interpolated metrics differs from the interpolated K by <0.01 in every
row.

## 3. Matched-rate results

`K` is the composite from `tools/kpi.py`; higher is better. `BA` lower is
better, `SS2` higher is better. All values at 100.0 kbps by interpolation.

| Probe | Encoder | Rate | CVVDP | BA | SS2 | PSNR-Y | **K** |
|---|---|---:|---:|---:|---:|---:|---:|
| `me_half` | nightly | 100.0 | 9.577 | 14.42 | 66.54 | 38.25 | **71.61** |
| | eff-perc | 100.0 | 9.660 | 5.79 | 63.50 | 37.45 | **75.76** |
| | SVT p0 | 100.0 | 9.959 | 3.09 | 90.73 | 49.28 | **85.58** |
| | AVM | 100.0 | 9.980 | 1.46 | 91.59 | 51.05 | **90.32** |
| `me_quarter` | nightly | 100.0 | 9.349 | 15.55 | 60.31 | 36.72 | **69.41** |
| | eff-perc | 100.0 | 9.539 | 6.66 | 57.04 | 35.70 | **73.30** |
| | SVT p0 | 100.0 | 9.958 | 2.25 | 91.11 | 49.33 | **87.60** |
| | AVM | 100.0 | 9.971 | 1.47 | 91.04 | 50.67 | **90.14** |
| `me_eighth` | nightly | 100.0 | 9.303 | 11.84 | 67.73 | 38.64 | **71.19** |
| | eff-perc | 100.0 | 9.687 | 4.35 | 71.19 | 38.59 | **78.84** |
| | SVT p0 | 100.0 | 9.960 | 1.33 | 91.63 | 49.83 | **90.74** |
| | AVM | 100.0 | 9.966 | 1.26 | 91.30 | 50.71 | **90.98** |
| `fade` | nightly | 100.0 | 9.667 | 2.39 | 72.77 | 43.02 | **82.70** |
| | eff-perc | 100.0 | 9.831 | 1.49 | 79.94 | 43.05 | **87.51** |
| | SVT p0 | 100.0 | 9.755 | 1.26 | 87.18 | 46.92 | **89.25** |
| | AVM | 100.0 | 9.630 | 2.56 | 78.09 | 43.99 | **82.98** |
| `static_repeat` | nightly | 100.0 | 9.840 | 1.33 | 83.84 | 39.60 | **88.84** |
| | eff-perc | 100.0 | 9.842 | 1.33 | 83.80 | 39.49 | **88.84** |
| | SVT p0 | 100.0 | 9.939 | 1.16 | 89.23 | 44.18 | **90.94** |
| | AVM | 100.0 | 9.812 | 1.29 | 85.95 | 42.00 | **89.21** |
| `near_static` | nightly | 100.0 | 9.777 | 1.93 | 81.10 | 36.71 | **85.95** |
| | eff-perc | 100.0 | 9.762 | 1.89 | 80.56 | 36.53 | **85.90** |
| | SVT p0 | 100.0 | 9.924 | 1.32 | 87.34 | 40.03 | **89.89** |
| | AVM | 100.0 | 9.930 | 1.22 | 87.87 | 40.32 | **90.38** |
| `detail_edges` | nightly | 100.0 | 9.593 | 1.88 | 74.87 | 38.91 | **84.14** |
| | eff-perc | 100.0 | 9.605 | 1.86 | 74.91 | 38.88 | **84.26** |
| | SVT p0 | 100.0 | 9.286 | 5.79 | 62.94 | 44.74 | **73.78** |
| | AVM | 100.0 | 9.945 | 0.87 | 89.00 | 47.45 | **92.26** |
| `grain` | nightly | 100.0 | 4.280 | 14.88 | -36.59 | 26.38 | **27.83** |
| | eff-perc | 100.0 | 4.273 | 14.98 | -36.00 | 26.38 | **27.87** |
| | SVT p0 | 100.0 | 4.542 | 12.07 | -22.19 | 26.85 | **32.33** |
| | AVM | 100.0 | 3.699 | 14.20 | -33.77 | 26.37 | **25.56** |

Gap summary (best peer minus av2enc):

| Probe | nightly K | eff-perc K | best peer | nightly gap | eff gap |
|---|---:|---:|---:|---:|---:|
| `me_half` | 71.61 | 75.76 | AVM 90.32 | **-18.71** | -14.56 |
| `me_quarter` | 69.41 | 73.30 | AVM 90.14 | **-20.73** | -16.84 |
| `me_eighth` | 71.19 | 78.84 | AVM 90.98 | **-19.79** | -12.14 |
| `fade` | 82.70 | 87.51 | SVT 89.25 | -6.55 | -1.74 |
| `static_repeat` | 88.84 | 88.84 | SVT 90.94 | -2.10 | -2.10 |
| `near_static` | 85.95 | 85.90 | AVM 90.38 | -4.43 | -4.48 |
| `detail_edges` | 84.14 | 84.26 | AVM 92.26 | -8.13 | -8.00 |
| `grain` (unclean) | 27.83 | 27.87 | SVT 32.33 | -4.49 | -4.46 |

Actual brackets (`QP` and measured kbps at the two points; the target is the
linear interpolation between them). Wide brackets are part of the evidence and
are called out where they matter:

| Probe | nightly | eff-perc | SVT p0 (CRF) | AVM |
|---|---|---|---|---|
| `me_half` | QP142 99.5 / QP141 100.4 | QP153 99.2 / QP145 114.3 | 38 99.1 / 37 102.8 | QP100 95.1 / QP99 101.2 |
| `me_quarter` | QP141 98.8 / QP140 100.6 | QP158 98.8 / QP157 101.5 | 34 98.2 / 32 107.2 | QP102 94.6 / QP101 101.8 |
| `me_eighth` | QP117 99.4 / QP112 107.6 | QP140 98.4 / QP139 100.5 | 37 79.9 / 29 101.5 | QP97 98.5 / QP96 101.0 |
| `fade` | QP119 100.0 / QP117 102.8 | QP125 99.0 / QP122 104.5 | 45 86.3 / 40 100.1 | QP142 99.4 / QP141 101.3 |
| `static_repeat` | QP132 100.0 / QP131 102.2 | QP133 99.7 / QP128 106.9 | 49 96.3 / 48 104.7 | QP187 99.8 / QP178 106.6 |
| `near_static` | QP139 90.3 / QP133 100.9 | QP136 96.3 / QP135 102.7 | 57 67.2 / 49 100.6 | QP158 98.9 / QP157 102.0 |
| `detail_edges` | QP139 100.0 / QP138 102.2 | QP139 99.9 / QP119 133.6 | 51 98.9 / 50 100.6 | QP210 98.9 / QP209 101.6 |
| `grain` (unclean) | QP192 82.1 / QP191 126.7 | QP195 99.9 / QP170 608.2 | 54 85.1 / 53 127.0 | QP234 97.4 / QP233 100.7 |

## 4. Attribution

### 4.1 ME search is correct; the skip decision and reconstruction are not

`AV2ENC_BLOCKDUMP` gives one line per coded leaf. On the ME probes the coded
MV histogram is dominated by the ground-truth vector:

| Probe | inter leaves | exact-MV | skip | non-zero-MV skip | residual leaves | top MV |
|---|---:|---:|---:|---:|---:|---|
| `me_half` nightly | 2,689 | **99.9%** | 75.7% | 2,035 | 654 | `(28,12)` 2,687 |
| `me_half` eff-perc | 2,501 | 99.6% | 42.4% | 1,061 | 1,440 | `(28,12)` 2,492 |
| `me_quarter` nightly | 3,156 | **85.3%** | 87.3% | 2,755 | 401 | `(18,-10)` 2,691; `(17,-9)` 227 |
| `me_quarter` eff-perc | 2,905 | 83.3% | 13.8% | 402 | 2,503 | `(18,-10)` 2,420; `(17,-9)` 150 |
| `me_eighth` nightly | 3,552 | **78.4%** | 91.3% | 3,244 | 308 | `(9,-3)` 2,785; `(9,-2)` 548 |
| `me_eighth` eff-perc | 2,899 | 76.6% | 41.9% | 1,216 | 1,683 | `(9,-3)` 2,221; `(9,-2)` 528 |

The non-exact leaves are almost entirely one 1/8-pel step off (`(17,-9)` or
`(9,-2)`), which is the expected quantisation of a fractional MV, not a failed
search. The global vector is found on every probe.

What is wrong is the follow-through. Decoded luma PSNR per frame:

| Probe | Encoder | key PSNR | P mean | P - key | key share |
|---|---|---:|---:|---:|---:|
| `me_half` | nightly | 40.69 | 38.34 | **-2.35** | 41.9% |
| | eff-perc | 38.57 | 37.37 | -1.20 | 35.3% |
| | SVT p0 | 49.05 | 49.28 | +0.23 | 55.2% |
| | AVM | 51.22 | 51.09 | -0.14 | 66.6% |
| `me_quarter` | nightly | 40.89 | 36.91 | **-3.98** | 42.4% |
| | eff-perc | 37.69 | 35.59 | -2.10 | 32.7% |
| | SVT p0 | 49.30 | 49.32 | +0.02 | 60.4% |
| | AVM | 50.72 | 50.72 | +0.00 | 64.9% |
| `me_eighth` | nightly | 44.87 | 39.09 | **-5.78** | 62.2% |
| | eff-perc | 41.02 | 38.68 | -2.34 | 43.7% |
| | SVT p0 | 49.71 | 49.96 | +0.25 | 69.5% |
| | AVM | 50.61 | 50.74 | +0.13 | 70.1% |

Nightly spends fewer key bits than the peers (41-62% share vs 55-70%), so part
of its key deficit is allocation. But the P-chain is worse than its own key on
every ME probe, while both peers hold or improve. Skip is the mechanism: the
nightly leaf RD chooses skip on 76-91% of inter leaves; the non-zero-MV skip
resamples the reference through subpel MC and passes the blur forward.

Within-av2enc controls at matched rate:

| Control | Rate | K | Read |
|---|---:|---:|---|
| `me_half` nightly | 100.0 | 71.61 | reference |
| `me_half` nightly **without `--video-leaf-rd`** | 100.0 (interp. of 101.7/97.8) | **75.67** | +4.1 K: the calibrated leaf RD is over-skipping |
| `me_half` eff-perc | 100.0 | 75.76 | its skip flips recover most of the same +4.1 K |
| `me_eighth` nightly **quarter-pel, no 1/8 stage** | 100.0 (97.4 measured) | 72.28 | no gain from 1/8 precision; not the lever |

The 1/8-pel branch is therefore not the weak implementation: the exact-MV rate
is already 78%, and removing the eighth stage does not reduce K. The weak
implementation is the leaf skip/residual decision, which prices only this
leaf's distortion and not the reference it is about to degrade.

### 4.2 Intra/keyframe coefficient coding is the largest clean gap

`detail_edges` is a static key picture with all P frames skipped, so its K is
an intra-only measurement. `static_repeat` is the same shape with a different
texture. Key bytes and key PSNR:

| Probe | Encoder | key bytes | key PSNR | key share | total |
|---|---|---:|---:|---:|---:|
| `detail_edges` | nightly | 22,995 | **38.88** | 94.9% | 24,238 |
| | eff-perc | 23,034 | 38.87 | 95.1% | 24,285 |
| | SVT p0 | 22,623 | 44.83 | 92.8% | 24,376 |
| | AVM | 23,500 | **47.31** | 98.0% | 23,974 |
| `static_repeat` | nightly | 23,101 | **39.60** | 95.3% | 24,239 |
| | eff-perc | 22,972 | 39.45 | 95.0% | 24,177 |
| | SVT p0 | 21,967 | 43.82 | 94.2% | 23,339 |
| | AVM | 15,339 | 40.31 | 63.4% | 24,192 |

At the same key bytes, av2enc is 8.4 dB behind AVM on `detail_edges`. On
`static_repeat` it spends more key bytes than SVT for 4.2 dB less. The ME
probes repeat the pattern at a lower key share: 40.7-44.9 dB for av2enc vs
49.1-49.7 (SVT) and 50.6-51.2 (AVM).

This is not a transform-size problem. The key-frame TX histogram (nightly,
`detail_edges`, 335 traced parts) is:

| TX size | 16x16 | 16x32 | 16x64 | 32x16 | 32x32 | 64x16 | 64x32 |
|---|---:|---:|---:|---:|---:|---:|---:|
| share | 82.4% | 5.7% | 6.9% | 3.6% | 0.6% | 0.3% | 0.6% |

The key already selects small transforms. Nor is it deblock/CDEF: on a key-only
`detail_edges` encode at QP139, removing `--video-cdef --video-cdef-rd` changes
the stream from 23,039 B / 38.881 dB to 23,038 B / 38.820 dB, removing
`--video-deblock` gives 23,042 B / 38.880 dB, and `--video-rdoq` is
byte-identical (23,039 B / 38.881 dB). The key is also not simply
under-allocated: `--video-qp-lambda` is active; removing it drops the key to
19,719 B / 30.919 dB, so the QP-scaled lambda is doing its job. The remaining
candidate is the coefficient path:
`video_tx_search.cpp` sets `use_rdoq = rdoq && ... && is_inter`, and the key
decision passes `allow_rdoq=false` (`video_frame_decision.cpp` key path).
`--video-rdoq` is therefore inter-only; the key has no trellis. `docs/luma-audit.md`
already records "intra has no trellis at all" and 97% of key symbol bits in
coefficient syntax.

The `detail_edges` SVT row needs a caveat: SVT has the **highest PSNR** (44.74)
but the **worst K** (73.78) because its decoded picture has much worse
Butteraugli/SSIMULACRA2 (5.79 / 62.94) than av2enc (1.88 / 74.87). That is a
cross-codec perceptual artifact on hard synthetic edges (likely banding from
AV1's in-loop filters at high CRF), not evidence that AV1's intra coder is
worse. Use the key-PSNR/key-byte columns, not K, for the intra comparison; AVM
is the clean AV2-to-AV2 reference.

### 4.3 Fade: BAWP is present, applied, and not paying

`fade` is a static texture with a linear illumination ramp. All MVs are `(0,0)`
and nightly skips 95.8% of inter leaves; BAWP is applied to 42.3% of candidates
in a six-frame trace. The key is 88.5% of the stream, so the P chain has almost
no bits to correct the fade. SVT, which has no normative BAWP, instead spends
35.7% of its bits on P frames and **gains 5.5 dB** over its key. AVM's BAWP is
also weak here (82.98), so this is not a "we lack BAWP" problem.

The within-av2enc ablation is decisive:

| Fade encode | QP | Rate | K |
|---|---:|---:|---:|
| nightly (BAWP on) | 119 | 100.0 | 82.70 |
| nightly flags **minus `--video-bawp --video-bawp-carrier`** | 119 | 97.5 | 83.31 |
| same, one QP lower | 118 | 110.8 | 89.14 |
| no-BAWP interpolated to 100.0 | — | 100.0 | **~84.4** |

Disabling BAWP improves K by about **+1.7** at matched rate. The current BAWP
acceptance test is purely spatial: `bawp_adapt_luma()` is applied when
`adapted_sse < luma_sse`; the `use_bawp` bit and the interaction with the
skip/residual decision are not priced. On a fade whose per-frame ratio is close
to 1, the residual coder (or plain skip) is often cheaper. AVM's similar result
suggests the tool is not a free win; the implementation has to select it.

### 4.4 Static and near-static are not the problem

`static_repeat`: both av2enc presets are **100% zero-MV skip, zero residual,
zero non-zero-MV skip**, and the P-frame PSNR equals the key exactly. The 2.1 K
gap to SVT is entirely the key picture (39.60 vs 43.82 dB) and belongs to §4.2.
`near_static`: also 100% zero-MV skip and zero non-zero-MV skip. Its P-frame
PSNR is 1.6-1.8 dB below the key because the source itself has a different
N(0,2) noise realisation each frame; skipping reproduces the key, not the new
noise. That is a source-noise floor, not MV drift. The 4.4 K gap to AVM/SVT is
mostly key quality plus how much noise the peer chooses to code.

So the skip implementation is **correct on zero motion** and **too eager on
fractional motion**. The fix belongs in the fractional-MV skip decision, not in
skip as a tool.

### 4.5 Grain / filters: do not rank

`grain` is included because the task asks for an optional filter probe, but it
is not a usable ranking:
- the content is incompressible at 100 kbps; every encoder is at PSNR 26-27 dB
  and SSIMULACRA2 is negative;
- SVT's CRF curve jumps from 127 kbps to 85 kbps between adjacent CRF steps,
  and eff-perc jumps from 99.9 to 608 kbps between two QPs;
- K differences (25.6-32.3) track how much grain each encoder chooses to throw
  away, not filter implementation quality.

Keep it as a regression smoke test ("does a grain stream decode and stay
stable?"), not as a tool ranking.

## 5. Ranked implementation fixes

The ranking is by measured gap and by how cleanly the probe attributes it.
All three are changes to existing decisions/algorithms; none adds syntax or a
new tool.

### 1. Intra/keyframe coefficient trellis and context rate (largest clean gap)

**Evidence.** 8.4 dB key-PSNR deficit at equal key bytes on `detail_edges`;
4.2 dB deficit at more key bytes on `static_repeat`; 8-10 dB on the ME keys.
82% of key TX parts are already 16x16, so TX size is not the cause.
`--video-rdoq` is inter-only (`video_tx_search.cpp`: `use_rdoq = rdoq && ...
&& is_inter`); the key passes `allow_rdoq=false`. Disabling CDEF/deblock does
not change the key stream.

**Fix.** Run the existing coefficient trellis/RDOQ on the intra/key plan and
price it with the existing exact §8.3/§8.2.6 context rate (the machinery is
already in `video_rate_interval.h` / `eff_perc`), using the same λ as the TX
search. Quantise each coefficient by minimising
`D(coeff) + λ*R(coeff | live CDF, neighbours)`, not by scalar rounding, and let
the key TX search see that trellis cost. This is the same change the audit's
ranked mechanism #1 requests; the probe now gives it a direct target.

**Expected effect.** Recovering the 4-8 dB key deficit at the same bytes would
close most of `detail_edges` (8.1 K) and `static_repeat` (2.1 K) and a large
part of the ME gap. Gate on key-only PSNR/rate plus the full probe K; do not
accept a change that only shifts error between BA and SS2.

### 2. Drift-aware leaf skip/RD for fractional motion

**Evidence.** Exact MVs on 78-100% of leaves, but nightly skips 76-91% and its
P-chain loses 2.3-5.8 dB relative to its own key. Removing `--video-leaf-rd` at
matched rate gains +4.1 K; eff-perc's skip flips gain +4.1/+3.9/+7.7 K but
still leave 12-17 K to the peers. The quarter-pel control shows precision is
not the lever.

**Fix.** Add the missing future term to the skip decision in
`video_frame_decision.cpp` (the `skip_cost` vs `exact_res_cost` comparison):
for a skip leaf whose MV is non-zero (especially fractional), charge
`D_drift = alpha * mean((pred-src)^2) * H` where `H` is a short horizon (1-2
frames) or a per-superblock reference-quality deficit. A cheaper implementation
is to extend the existing `--video-drift-refresh` selection from frame-level to
superblock/band level and trigger it when the fractional-MV skip share crosses
a threshold. Recalibrate the leaf λ on fractional-motion content; the current
fit was calibrated on the smooth NASA chain.

**Expected effect.** The no-leaf-RD control is a lower bound: +4.1 K on
`me_half` at matched rate. A drift term should capture that without the 23%
rate increase the control pays at fixed QP. Target: P mean within 0.5 dB of the
key on the three ME probes; static/near-static must stay 100% zero-MV skip.

### 3. Rate-aware BAWP selection and refit on fades

**Evidence.** Fade gap 6.5 K to SVT; BAWP applied to 42% of candidates; nightly
skips 95.8% and codes only 117 residual leaves. Disabling BAWP improves K by
~1.7 at matched rate. AVM's BAWP is also at 82.98, so the tool alone is not the
answer; SVT wins by coding P residual.

**Fix.** Make the BAWP decision rate-aware in `video_frame_decision.cpp`: after
`bawp_adapt_luma()` produces an adapted prediction, compare
`plain + residual` against `BAWP + residual` using the exact syntax probe
(`use_bawp` bit plus the coefficient plan), instead of accepting BAWP whenever
`adapted_sse < luma_sse`. Regularise/robustify the template fit (the current
above/left least-squares is sensitive to the 16-sample template) and skip BAWP
when the per-frame fade ratio is near 1 or when the residual is cheaper. This
is a selection/pricing change, not a new prediction mode.

**Expected effect.** +1.7 K immediately from not using BAWP where it loses;
more where BAWP genuinely wins. Gate on `fade` and `static_repeat`; check that
the `use_bawp` rate is not simply zeroed on real fades.

## 6. Reproduction

```sh
# Build (CPU-only is enough; no encoder code changes)
cmake -S . -B build-int -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DAV2ENC_ENABLE_CUDA=OFF -DAV2ENC_ALLOW_CPU_ONLY=ON -DAV2ENC_CPU_FAST=ON
cmake --build build-int -j8

# Generate the 8 clips + ground-truth JSON + manifest
python3 tools/tool_probes.py --out-dir build/tool-probes --frames 60 --force
python3 tools/tool_probes.py --out-dir /tmp/probe-check --frames 60
# (the second run must produce the same SHA-256 manifest)

# Full 4-encoder sweep, 2-QP brackets, cached under build/tool-quality
AVMENC=/path/to/avmenc python3 tools/tool_quality_run.py \
  --av2enc build-int/av2enc --avm "$AVMENC" --jobs 2

# Per-tool evidence: block dump, key/P split, BAWP/leaf-RD/quarter-pel ablations
python3 tools/tool_probe_evidence.py \
  --results build/tool-quality/results.json \
  --av2enc build-int/av2enc --avm "$AVMENC"
```

The runner writes `build/tool-quality/results.json`, `summary.md` and a
`work/<probe>/<encoder>/` cache. The evidence tool writes
`build/tool-quality/evidence.json`. Both are build artifacts, not committed.
Every av2enc probe stream at the scored brackets was also decoded by the pinned
`dav2d --strict 1` (32/32 pass) before its quality number was used.

## 7. Limitations

- **Synthetic content.** Band-limited cosine textures are not natural video.
  They stress high-frequency reconstruction and exact global motion; NASA is
  smooth and understates the keyframe deficit. The numbers rank implementations
  on these components, not a BD-rate on a corpus.
- **One instance per family.** No confidence interval. The three ME probes share
  one texture by design, so their spread is motion, not content.
- **Cross-codec K is not tool parity.** SVT is AV1, AVM is AV2; BAWP, compound
  and the in-loop filters are not the same set. `detail_edges` shows the
  problem clearly: SVT has the best PSNR and the worst K. Use PSNR/key bytes for
  intra efficiency and K only for the end-to-end operating point.
- **Rate control interacts.** SVT uses CRF, av2enc/AVM fixed QP. Some brackets
  are wide (listed in `results.json`); the grain probe has one-QP cliffs and is
  not ranked. K is interpolated in rate, matching the project convention.
- **No peer internals.** SVT/AVM are black boxes. Their key/P split comes from
  IVF frame sizes and decoded-frame PSNR; their MV/TX decisions are not
  inspected. The av2enc block dump is the only decision-level evidence.
- **Single host.** Apple M4, 10 cores. Quality metrics are host-independent
  enough for this comparison; encode speeds are not reported.
