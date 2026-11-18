# Luma audit: where the 3.32 K luma gap actually is

Status: **diagnostic, read-only.** Branch `exp/luma-audit`, base `1b3a132e`.
No encoder source was changed; this commit adds this document only. Streams
were decoded with the pinned `dav2d-macos --strict 1`. Main `a1159bb8` now
scores with `avmdec`; the committed flat-P pins decode identically under it, so
these numbers are unchanged, and AVM is used only as a decoded-quality
reference (it is the B-frame oracle, not an encoder we can copy). K is
`tools/kpi.py`; per-frame Butteraugli/SSIMULACRA2 are
`butteraugli_main`/`ssimulacra2` on extracted PNGs.

## TL;DR

1. **The 44.71 / 60.44 premise is stale or mis-attributed.** 44.71 is the old
   QP110 nightly key (`docs/video-nightly.md`), not the current 25,008 B QP106
   pin; 60.44 does not reproduce as an 8x8 low-pass. On the current pins the
   keyframe is **48.48 dB full-res / 64.75 dB at 8x8 low-pass** against SVT's
   **50.16 / 59.99**: −1.68 dB full-res but **+4.76 dB at 8x8**. The keyframe
   deficit is only below ~3 px (lp2 −1.08, lp3 +0.20, lp4 +0.99): SVT keeps
   more 2-4 px detail, we keep more low-frequency accuracy.
2. **The whole-stream gap is not a 1-4 px gap.** It is −3.94 dB full-res and
   −4.49 dB at lp8. In the fade tail (f44-59) it is −5.06 dB full-res and
   **−8.66 dB at lp8**: the tail gap is low-frequency drift, not fine detail.
3. **K is not keyframe-limited.** Replacing only the decoded keyframe with
   SVT/AVM changes K by +0.00/+0.04. Replacing f44-59 adds **+3.76 K**
   (nightly) and **+1.46 K** (eff-perc). The keyframe is 1/60 of the metric;
   its value is as a predictor and as a bit budget, not as a picture.
4. **77-91% of the P-frame error is in skip leaves**, and the static-section
   drift is residual-free skip with nonzero-MV subpel MC: f5/f7 are all-skip
   with 21/15 non-zero MVs and change the reconstruction by ~1.0 MSE; f3/f6 are
   zero-MV skips and change it by 0.0004. Loop filters off and BAWP off do not
   fix it; subpel off is much worse. This is a decision/reference problem, not
   a filter bug.
5. **The coefficient field is where the key bits are and where the inter
   residual is not shaped.** 79% of the nightly stream is the key tile, 97% of
   its metered symbol bits are coefficient syntax, intra has no trellis at all,
   and the inter residual uses 64x64/64x32/32x16 transforms almost exclusively
   (no TX below 16x16 in 7,492 traced parts). The one decision layer still
   priced by plain SSE is the coefficient trellis.
6. **Ranked mechanisms: (1) frequency-weighted RDOQ/trellis, (2)
   frequency-aware TX size/type search, (3) band-aware drift-refresh
   selection.** Expected combined +0.7-1.8 K at 100 kbps (overlapping, not
   additive), which moves eff-perc 85.1 to ~85.8-86.9 but does not reach SVT
   88.6. Coefficient entropy is ≤0.3 K and learned tools for 95 are not a
   100 kbps lever; both are quantified below.

## 1. Streams measured

| stream | recipe | bytes | kbps | JOD | BA | SS2 | K |
|---|---|---:|---:|---:|---:|---:|---:|
| nightly QP106 | `--qp 106 --nightly` | 25,008 | 100.032 | 9.7334 | 3.4202 | 80.837 | **82.155** |
| drift QP110 | nightly + threshold 1.5 / budget 3250 | 25,146 | 100.584 | 9.7605 | 2.6562 | 81.250 | **83.937** (83.795 @100.0) |
| eff-perc QP119 | drift recipe + `AV2ENC_EFF_PERC=1` | 25,230 | 100.920 | 9.7713 | 2.2630 | 82.441 | **85.180** (85.10 @100.0) |
| SVT-AV1 4.3.0 p0 | preset 0, CRF 36.5 | 24,844 | 99.376 | 9.8262 | 1.5827 | 88.866 | **88.641** (88.8 @100.0) |
| AVM e6f99d3 | QP 124, `--end-usage=q --cpu-used=6` | 24,836 | 99.344 | 9.8616 | 1.2889 | 90.382 | **90.192** (90.2 @100.0) |

SVT's cited 88.83 is the 4.1.0 row; the local binary and the stream used for
the per-frame tables are 4.3.0 (`SvtAv1EncApp` reports 4.3.0). AVM is the clean
upstream `e6f99d3` build. All five streams reproduce their recorded
JOD/BA/SS2 on this host.

Scope: luma only. The decoded-domain hybrid puts 0.61 K of the 3.93 K gap on
chroma and 3.32 K on luma (`docs/eff-perc.md`); chroma is held fixed here, so
the segment and band numbers below are luma Y and the K decomposition is the
whole K with chroma unchanged.

## 2. The premise, re-measured

| quantity | task context | current nightly QP106 | current drift QP110 | SVT p0 |
|---|---:|---:|---:|---:|
| keyframe full-res PSNR-Y | 44.71 | **48.48** | **47.73** | 50.16 |
| keyframe 8x8 low-pass | 60.44 | **64.75** | **63.59** | 59.99 |
| keyframe full-res gap vs SVT | −5.44 | **−1.68** | **−2.43** | — |
| keyframe lp8 gap vs SVT | +0.72 | **+4.76** | **+3.60** | — |
| P-frame full-res gap vs SVT | ≈ −6 | f1-32 −3.24 / f33-43 −4.56 / f44-59 −5.06 | f1-32 −3.71 / f33-43 −4.00 / f44-59 −4.67 | — |

The 44.71/50.15 pair matches the QP110 table in `docs/video-nightly.md`; the
encoder has gained 3.0 dB on that key since (QP-scaled lambdas, CDF carrying,
drift refresh), so those numbers cannot rank today's mechanisms. 60.44 does not
reproduce under a separable 8x8 box on either current stream (the lp8 numbers
are 64.75 vs 59.99); it coincides with the **QP10 full-res** keyframe PSNR in
`bench/keyframe/README.md`, so its provenance should be checked before it is
used again. The qualitative claim survives, but only for the key picture: we
are ahead at scales above ~3 px and behind below it.

The scale probe (separable box, both source and decoded luma, PSNR on the
filtered frame):

| scale | nightly key | SVT key | Δ | nightly all | SVT all | Δ |
|---|---:|---:|---:|---:|---:|---:|
| full | 48.48 | 50.16 | −1.68 | 45.06 | 49.00 | −3.94 |
| lp2 | 53.63 | 54.71 | −1.08 | 48.71 | 53.18 | −4.47 |
| lp3 | 56.79 | 56.59 | +0.20 | — | — | — |
| lp4 | 59.16 | 58.17 | +0.99 | 52.34 | 57.11 | −4.78 |
| lp8 | 64.75 | 59.99 | +4.76 | 55.77 | 60.27 | −4.49 |

Key: the crossover is between lp2 and lp3, so the keyframe's deficit is the
**1-4 px band** (more precisely <3 px). Whole stream: the gap is worst at lp4
and almost as bad at lp8; the tail dominates that.

The old claim also mixed segments. On the current streams the gap grows with
time and changes character:

| segment | ΔPSNR (nightly−SVT) | Δlp2 | Δlp4 | Δlp8 | ΔBA | ΔSS2 |
|---|---:|---:|---:|---:|---:|---:|
| key f0 | −1.68 | −1.08 | +0.99 | **+4.76** | −0.32 | −1.64 |
| f1-32 | −3.24 | −3.43 | −3.19 | −2.23 | +0.69 | −2.87 |
| f33-43 | −4.56 | −5.43 | −5.98 | −5.84 | +2.25 | −8.06 |
| f44-59 | −5.06 | −6.08 | −7.47 | **−8.66** | +3.99 | −18.73 |
| all 60 | −3.94 | −4.47 | −4.78 | −4.49 | +1.84 | −8.03 |

So there are two different luma problems:

* **Intra/keyframe**: a real 2-4 px detail deficit, with better low-frequency
  accuracy than SVT. It is 1/60 of K directly, but it is the predictor for the
  static head of the chain and it is 66-79% of the bits.
* **P chain**: a low-frequency error floor that grows through the fade. It is
  where the K gap lives.

## 3. Per-segment numbers

Arithmetic means over the frames in the segment. `lpN` is the box-N low-pass
PSNR. The 60-row per-frame table is in Appendix A.


| segment | stream | PSNR-Y | lp2 | lp4 | lp8 | BA | SS2 |
|---|---|---:|---:|---:|---:|---:|---:|
| key f0 | nightly QP106 | 48.48 | 53.63 | 59.16 | 64.75 | 1.61 | 89.00 |
| key f0 | drift QP110 | 47.73 | 52.84 | 58.19 | 63.59 | 1.36 | 87.74 |
| key f0 | eff-perc QP119 | 46.05 | 51.28 | 56.80 | 62.45 | 1.58 | 86.70 |
| key f0 | SVT p0 | 50.16 | 54.71 | 58.17 | 59.99 | 1.93 | 90.64 |
| key f0 | AVM QP124 | 52.47 | 57.60 | 61.74 | 64.52 | 1.02 | 92.67 |
| f1-32 | nightly QP106 | 46.54 | 51.07 | 55.80 | 60.34 | 2.09 | 87.75 |
| f1-32 | drift QP110 | 46.07 | 50.60 | 55.33 | 59.91 | 1.93 | 86.83 |
| f1-32 | eff-perc QP119 | 44.92 | 49.73 | 54.76 | 59.68 | 1.93 | 85.62 |
| f1-32 | SVT p0 | 49.78 | 54.51 | 59.00 | 62.57 | 1.40 | 90.62 |
| f1-32 | AVM QP124 | 51.04 | 55.68 | 59.88 | 63.40 | 1.19 | 92.14 |
| f33-43 | nightly QP106 | 43.26 | 46.32 | 49.05 | 51.57 | 3.92 | 80.52 |
| f33-43 | drift QP110 | 43.82 | 47.50 | 51.38 | 55.58 | 2.60 | 82.13 |
| f33-43 | eff-perc QP119 | 43.57 | 47.70 | 52.04 | 56.58 | 2.55 | 81.67 |
| f33-43 | SVT p0 | 47.82 | 51.75 | 55.03 | 57.42 | 1.66 | 88.58 |
| f33-43 | AVM QP124 | 48.72 | 52.50 | 55.67 | 58.20 | 1.49 | 89.45 |
| f44-59 | nightly QP106 | 43.12 | 45.32 | 47.23 | 48.97 | 5.85 | 66.72 |
| f44-59 | drift QP110 | 43.51 | 45.96 | 48.22 | 50.29 | 4.23 | 69.08 |
| f44-59 | eff-perc QP119 | 44.28 | 48.07 | 51.98 | 56.04 | 2.77 | 76.35 |
| f44-59 | SVT p0 | 48.18 | 51.40 | 54.70 | 57.63 | 1.86 | 85.45 |
| f44-59 | AVM QP124 | 50.65 | 53.63 | 56.00 | 57.92 | 1.38 | 87.36 |
| all 60 | nightly QP106 | 45.06 | 48.71 | 52.34 | 55.77 | 3.42 | 80.84 |
| all 60 | drift QP110 | 45.00 | 48.83 | 52.76 | 56.61 | 2.66 | 81.25 |
| all 60 | eff-perc QP119 | 44.52 | 48.94 | 53.55 | 58.19 | 2.26 | 82.44 |
| all 60 | SVT p0 | 49.00 | 53.18 | 57.11 | 60.27 | 1.58 | 88.87 |
| all 60 | AVM QP124 | 50.53 | 54.58 | 58.10 | 61.00 | 1.29 | 90.38 |

The requested `f0-32` aggregate (key plus static head):

| stream | PSNR-Y | lp2 | lp4 | lp8 | BA | SS2 |
|---|---:|---:|---:|---:|---:|---:|
| nightly QP106 | 46.60 | 51.15 | 55.91 | 60.47 | 2.08 | 87.79 |
| drift QP110 | 46.12 | 50.67 | 55.41 | 60.02 | 1.91 | 86.86 |
| eff-perc QP119 | 44.95 | 49.77 | 54.82 | 59.77 | 1.92 | 85.65 |
| SVT p0 | 49.79 | 54.51 | 58.97 | 62.49 | 1.42 | 90.62 |
| AVM QP124 | 51.08 | 55.74 | 59.94 | 63.44 | 1.18 | 92.16 |

All PSNR values in this section are arithmetic means of per-frame PSNR. The
stream-level `tools/quality.py` value is MSE-weighted over the sequence:
44.67 (nightly), 44.79 (drift), 44.46 (eff-perc), 48.83 (SVT), 50.27 (AVM).
Use the arithmetic rows to compare segments and the weighted number for stream
reports.


| segment | ΔPSNR (N−SVT) | Δlp2 | Δlp4 | Δlp8 | ΔBA (N−SVT) | ΔSS2 |
|---|---:|---:|---:|---:|---:|---:|
| key f0 | -1.68 | -1.08 | +0.99 | +4.76 | -0.32 | -1.64 |
| f1-32 | -3.24 | -3.43 | -3.19 | -2.23 | +0.69 | -2.87 |
| f33-43 | -4.56 | -5.43 | -5.98 | -5.84 | +2.25 | -8.06 |
| f44-59 | -5.06 | -6.08 | -7.47 | -8.66 | +3.99 | -18.73 |
| all 60 | -3.94 | -4.47 | -4.78 | -4.49 | +1.84 | -8.03 |

Rate/payload by segment (sum of per-picture IVF frame sizes; the 32-byte file
header and the sequence-header OBU are excluded, so "key share" is the share of
the coded picture payload, not of the whole file):

| stream | key | f1-32 | f33-43 | f44-59 | key share of payload |
|---|---:|---:|---:|---:|---:|
| nightly QP106 | 19,708 | 1,326 | 1,456 | 1,766 | 81.2% |
| drift QP110 | 18,545 | 1,889 | 2,534 | 1,426 | 76.0% |
| eff-perc QP119 | 16,138 | 1,433 | 2,971 | 3,936 | 65.9% |
| SVT p0 | 17,546 | 2,619 | 2,187 | 1,740 | 72.8% |
| AVM QP124 | 18,112 | 6,308 | — | — | 74.2% of coded payload |

On the whole IVF file the key share is 78.8% (nightly), 73.8% (drift), 64.0%
(eff-perc) and 70.6% (SVT); the TL;DR's "79%" is that whole-file nightly share.

AVM's IVF has 32 coded pictures plus show-existing units, so its per-segment
payload is not comparable frame-for-frame; it is included for quality only.
The allocation signature is clear: eff-perc moves **3.6 kB out of the key and
2.2 kB into f44-59** versus nightly at almost the same total. SVT sits between:
a smaller key than ours and a much larger static head (f1-32, 2.6 kB) than
either of ours. Our static head is under-spent, not over-spent.

## 4. Where the bits go

### 4.1 Keyframe

The nightly key tile is 19,673 B. The symbol meter attributes 12,402 B to
symbols, of which 12,072 B (97%) is the residual/coefficient family; partition
is 29 B and intra mode is 301 B. The literal sign/escape bits are outside the
meter, so the coefficient family is ~61% of the tile bytes and 97% of the
metered symbol bits. Within the 256 key leaves:

* TX partition: 118 SPLIT, 53 HORZ, 22 VERT, 63 NONE.
* TX parts: **464 8x8**, 106 16x8, 44 8x16, 39 16x16, 11 32x16, 8 64x64,
  5 64x32, 4 64x16, 3 32x32, 1 16x32 (685 parts). **Zero 4x4.** 8x8 is 68%
  of parts.
* TX types: DCT_DCT 248, ADST_DCT 29, DCT_ADST 21, ADST_ADST 123, other 2-D
  37, IDTX 20 (this is the QP120 accounting from
  `docs/keyframe-bit-accounting.md`; the current QP79 key is DCT-dominant).

This matches the previous accounting: the key is a coefficient field, not a
signalling problem. Note the instrumentation limit: the branch has the
symbol-family meter (`AV2ENC_SYMBOL_COST`) but not the per-band
`coeff_tile`/`coeff_band` blocks described in `docs/keyframe-bit-accounting.md`;
the per-band **bit** split would require adding encoder code, which this
read-only audit does not do. Section 5 reports the per-band **error** energy
instead. A 5-10% cut in key bytes is 1.0-2.0 kB, i.e. 4-8% of the
whole stream. That is the only large, low-risk budget source left.

### 4.2 P frames

Symbol-meter split for the nightly stream (metered symbol bytes, not tile
bytes; literal sign/escape bits are excluded):


| stream | segment | tile B | info B | residual B | residual share |
|---|---|---:|---:|---:|---:|
| nightly QP106 | key | 19,673 | 12,402 | 12,072 | 97% |
| nightly QP106 | f1-32 | 959 | 817 | 496 | 61% |
| nightly QP106 | f33-43 | 1,179 | 982 | 690 | 70% |
| nightly QP106 | f44-59 | 1,211 | 1,011 | 556 | 55% |
| eff-perc QP119 | key | 16,098 | 10,724 | 10,362 | 97% |
| eff-perc QP119 | f1-32 | 1,073 | 882 | 546 | 62% |
| eff-perc QP119 | f33-43 | 2,687 | 2,135 | 1,924 | 90% |
| eff-perc QP119 | f44-59 | 3,437 | 2,748 | 2,303 | 84% |


The key is 97% residual; every P segment is >55% residual *of the metered
symbols*. But the P tile budget is small: 959 B over 32 static frames, 1,179 B
over the fade onset, 1,211 B over the tail. The drift-refresh frames are the
only frames that force residual on every leaf; the rest are skip-dominated.

### 4.3 Skip / residual / intra attribution

Error energy is the reconstruction error `src − decoded`, binned by the
covering leaf from `AV2ENC_BLOCKDUMP`; `skip-0MV` is a skip leaf with a zero
motion vector, `skip-nzMV` a skip leaf with a non-zero vector. "Error in skip"
is the share of the segment's luma error energy in skip leaves.

| stream | segment | leaves | skip (0MV / nzMV) | residual | error in skip | error in residual |
|---|---|---:|---:|---:|---:|---:|
| nightly | f1-32 | 1,446 | 1,114 (1,039 / 75) | 332 | **77%** | 23% |
| nightly | f33-43 | 593 | 449 (417 / 32) | 144 | **77%** | 23% |
| nightly | f44-59 | 867 | 793 (707 / 86) | 74 | **91%** | 9% |
| eff-perc | f1-32 | 1,400 | 1,101 (965 / 136) | 299 | **77%** | 23% |
| eff-perc | f33-43 | 530 | 0 | 530 | 0% | **100%** |
| eff-perc | f44-59 | 792 | 377 (339 / 38) | 415 | 31% | **69%** |

This is the pivot of the whole audit. Nightly's P error is mostly **uncoded
prediction error**; eff-perc's mid/tail error is mostly **coded residual
error**. The mechanism that moved K from 82.15 to 85.10 was flipping skip to
residual (the gate reports `skip_flips=558`), not better coding of the same
residual. Once those leaves are coded, the next limit is the coefficient
decision inside them. There are **zero intra leaves in the P frames** of both
streams (and the key is all intra), so the intra/inter decision is not a
factor on this clip; the decision axis is skip vs residual and the TX shape.

### 4.4 Inter TX size

Trace of the nightly stream (`--video-trace`, 2,906 inter leaves, 7,492 TX
parts; the histogram includes the would-be plan of leaves that finish as skip):

| TX size | all parts | residual leaves | skip leaves |
|---|---:|---:|---:|
| 64x64 | 21% | 19% | 22% |
| 64x32 | 20% | 21% | 20% |
| 32x16 | 18% | 19% | 18% |
| 64x16 | 12% | 13% | 12% |
| 32x64 | 11% | 9% | 11% |
| 32x32 | 5% | 7% | 5% |
| 32x8 / 16x32 / other | 13% | 12% | 12% |
| ≤16x16 | **<2%** | **<2%** | **<2%** |

A 480x240 P residual is never coded with a transform smaller than 16x16 in the
emitted nightly stream. TX types are DCT_DCT 82%, ADST_DCT 9%, DCT_ADST 7%,
IDTX 1%. Large transforms are cheap for the smooth drift residual, but they
cannot localise a 2-4 px correction; the key search, in contrast, chose 8x8 for
68% of its TX parts. If the P-chain detail is to improve, the inter TX search
has to be given a reason to choose small transforms in the detail leaves.

### 4.5 Static-section drift forensics

The first eight P frames are all-skip. If skip were a pure copy, their PSNR
would stay at the key's 48.48. It does not:

| f | nzMV skip leaves | residual leaves | recon MSE vs f−1 | source MSE vs f−1 | PSNR-Y |
|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0 | — | 0.074 | 48.48 |
| 2 | 2 | 0 | 0.041 | 0.074 | 48.30 |
| 3 | 0 | 0 | **0.0004** | 0.075 | 48.21 |
| 4 | 2 | 0 | 0.038 | 0.358 | 47.20 |
| 5 | 21 | 0 | **0.956** | 0.306 | 46.08 |
| 6 | 0 | 0 | **0.0006** | 0.000 | 46.07 |
| 7 | 15 | 0 | **1.074** | 0.277 | 46.52 |
| 8 | 2 | 0 | 0.096 | 0.252 | 45.95 |
| 9 | 0 | 46 (refresh) | 0.725 | 0.326 | 46.95 |

The zero-MV skips are exact copies (0.0004-0.0006 MSE). The non-zero-MV skips
resample the reference through subpel MC and change the picture by ~1 MSE with
no residual. That is the static-section drift mechanism. Controls:

| control | f1-32 PSNR-Y | bytes | read |
|---|---:|---:|---|
| nightly QP106 | 46.54 | 25,008 | — |
| `--video-loop-filters-off` | 45.45 | 28,406 | filters are not the drift cause (and the stream is larger) |
| `--video-bawp-off` | 46.54 | 25,051 | static head identical, tail −0.22 dB; not the static drift cause |
| subpel-eighth removed (explicit expansion) | 44.75 | 25,213 | removing subpel is much worse; do not |
| `--video-mv-precision quarter` | 44.71 | 25,215 | K 79.73; worse |
| `--video-mv-precision half` | — | 25,174 | decoder produced 36/60 pictures in this harness; invalid |

A skip-specific penalty on non-zero MVs (only when the leaf would be skip) is
therefore a cheap, well-targeted probe, unlike global MV coarsening. It is not
in the top three because the measured K headroom is in the tail, not the static
head, but it is a valid secondary experiment for section 8.3.

## 5. Where the error goes in frequency

DCT ring bands on 8x8 blocks, defined as in `docs/svt-diff.md`: ring 1-2
(periods 4-8 px), ring 3-5 (1.6-2.7 px), ring ≥6 (<1.6 px), plus DC. Values are
error energy / source energy in the same ring; "low+DC share" is the fraction
of the segment's error energy in ring 1-2 plus DC.


| segment | stream | ring 1-2 | ring 3-5 | ring ≥6 | DC | low+DC share |
|---|---|---:|---:|---:|---:|---:|
| key f0 | nightly QP106 | 6.75e-04 | 2.40e-03 | 8.28e-03 | 4.16e-06 | 19% |
| key f0 | drift QP110 | 8.20e-04 | 2.81e-03 | 9.84e-03 | 5.80e-06 | 19% |
| key f0 | eff-perc QP119 | 1.16e-03 | 4.24e-03 | 1.45e-02 | 7.00e-06 | 18% |
| key f0 | SVT p0 | 4.44e-04 | 1.45e-03 | 5.19e-03 | 1.30e-05 | 26% |
| key f0 | AVM QP124 | 2.47e-04 | 8.41e-04 | 3.50e-03 | 4.72e-06 | 21% |
| f1-32 | nightly QP106 | 1.29e-03 | 3.76e-03 | 1.15e-02 | 1.46e-05 | 25% |
| f1-32 | drift QP110 | 1.43e-03 | 4.15e-03 | 1.29e-02 | 1.40e-05 | 24% |
| f1-32 | eff-perc QP119 | 1.71e-03 | 5.48e-03 | 1.76e-02 | 1.40e-05 | 22% |
| f1-32 | SVT p0 | 5.16e-04 | 1.71e-03 | 5.80e-03 | 1.13e-05 | 25% |
| f1-32 | AVM QP124 | 4.20e-04 | 1.24e-03 | 4.54e-03 | 6.78e-06 | 24% |
| f33-43 | nightly QP106 | 4.28e-03 | 7.87e-03 | 2.18e-02 | 1.40e-04 | 42% |
| f33-43 | drift QP110 | 3.97e-03 | 8.04e-03 | 2.22e-02 | 4.86e-05 | 33% |
| f33-43 | eff-perc QP119 | 3.72e-03 | 8.88e-03 | 2.63e-02 | 3.52e-05 | 28% |
| f33-43 | SVT p0 | 1.13e-03 | 3.01e-03 | 1.01e-02 | 3.49e-05 | 31% |
| f33-43 | AVM QP124 | 9.65e-04 | 2.37e-03 | 8.00e-03 | 3.34e-05 | 34% |
| f44-59 | nightly QP106 | 1.20e-02 | 1.65e-02 | 3.58e-02 | 3.94e-04 | 53% |
| f44-59 | drift QP110 | 1.16e-02 | 1.62e-02 | 3.63e-02 | 2.86e-04 | 49% |
| f44-59 | eff-perc QP119 | 9.13e-03 | 1.67e-02 | 4.69e-02 | 7.13e-05 | 31% |
| f44-59 | SVT p0 | 3.85e-03 | 6.83e-03 | 1.55e-02 | 5.24e-05 | 37% |
| f44-59 | AVM QP124 | 1.72e-03 | 3.10e-03 | 8.77e-03 | 7.46e-05 | 48% |
| all 60 | nightly QP106 | 2.91e-03 | 6.02e-03 | 1.62e-02 | 9.44e-05 | 40% |
| all 60 | drift QP110 | 2.92e-03 | 6.29e-03 | 1.73e-02 | 6.13e-05 | 35% |
| all 60 | eff-perc QP119 | 2.82e-03 | 7.43e-03 | 2.26e-02 | 2.64e-05 | 26% |
| all 60 | SVT p0 | 9.65e-04 | 2.56e-03 | 7.72e-03 | 2.19e-05 | 30% |
| all 60 | AVM QP124 | 6.48e-04 | 1.66e-03 | 5.65e-03 | 2.18e-05 | 33% |


Read this table carefully: the absolute error energy is high-frequency
dominated (35-43% in ring ≥6 for the key, 18-35% for P), but the **relative**
gap to SVT is largest at low frequencies in the P chain. Per-band ratio to SVT:


| segment | band | nightly/SVT | eff-perc/SVT | AVM/SVT |
|---|---|---:|---:|---:|
| key f0 | ring 1-2 | 1.52× | 2.61× | 0.56× |
| key f0 | ring 3-5 | 1.65× | 2.92× | 0.58× |
| key f0 | ring ≥6 | 1.60× | 2.80× | 0.67× |
| key f0 | DC | 0.32× | 0.54× | 0.36× |
| f1-32 | ring 1-2 | 2.50× | 3.32× | 0.81× |
| f1-32 | ring 3-5 | 2.20× | 3.21× | 0.72× |
| f1-32 | ring ≥6 | 1.98× | 3.03× | 0.78× |
| f1-32 | DC | 1.29× | 1.24× | 0.60× |
| f33-43 | ring 1-2 | 3.80× | 3.30× | 0.86× |
| f33-43 | ring 3-5 | 2.62× | 2.96× | 0.79× |
| f33-43 | ring ≥6 | 2.17× | 2.61× | 0.80× |
| f33-43 | DC | 4.00× | 1.01× | 0.96× |
| f44-59 | ring 1-2 | 3.11× | 2.37× | 0.45× |
| f44-59 | ring 3-5 | 2.41× | 2.44× | 0.45× |
| f44-59 | ring ≥6 | 2.31× | 3.03× | 0.57× |
| f44-59 | DC | 7.51× | 1.36× | 1.42× |
| all 60 | ring 1-2 | 3.02× | 2.92× | 0.67× |
| all 60 | ring 3-5 | 2.35× | 2.90× | 0.65× |
| all 60 | ring ≥6 | 2.10× | 2.93× | 0.73× |
| all 60 | DC | 4.32× | 1.21× | 1.00× |


* **Key**: ours/SVT is ~1.5-1.65× in every AC ring and 0.32× at DC. The key is
  uniformly noisier in AC, with much better DC. That is a coefficient-field
  efficiency gap, not a DC/offset bug.
* **f1-32**: 2.5× low, 2.2× mid, 2.0× high, 1.3× DC. Both detail and a small
  drift component.
* **f33-43**: 3.8× low, 2.6× mid, 2.2× high, 4.0× DC. The fade onset is
  low-frequency dominated.
* **f44-59**: 3.1× low, 2.4× mid, 2.3× high, **7.5× DC**. The tail is a
  low-frequency/DC error floor: our reconstruction lags the illumination ramp.
  eff-perc fixes most of the DC ratio (1.36×) but leaves the AC gap (2.4-3.0×).

High-band share of error (ring ≥6):

| stream | key | f1-32 | f33-43 | f44-59 |
|---|---:|---:|---:|---:|
| nightly QP106 | 40% | 35% | 25% | 18% |
| drift QP110 | 40% | 36% | 29% | 20% |
| eff-perc QP119 | 41% | 37% | 32% | 31% |
| SVT p0 | 37% | 37% | 33% | 25% |
| AVM QP124 | 43% | 39% | 32% | 23% |

Nightly's tail error is only 18% high-band; SVT's is 25% and eff-perc's 31%.
The tail is not a detail deficit. Dropping high-frequency coefficients there
would save bits but barely move BA/SS2; the useful target is the 2-8 px band.

## 6. What the K gap decomposes into

Segment replacement oracles: take the decoded frames of one stream and replace
one segment with SVT's (or AVM's) decoded frames, then re-score the assembled
sequence with the same tools. This is a K pooling bound, not an encodable
stream; the mixing ripple is ±0.1 K.

| base | segment replaced | K | Δ vs base | share of gap |
|---|---|---:|---:|---:|
| nightly (82.155) | key f0 | 82.15 | +0.00 | 0% |
| nightly | f0-32 | 83.06 | +0.91 | 14% |
| nightly | f33-43 | 83.26 | +1.11 | 17% |
| nightly | f44-59 | 85.91 | **+3.76** | **58%** |
| nightly | f0-43 | 84.23 | +2.08 | 32% |
| eff-perc (85.180) | key f0 | 85.19 | +0.01 | 0% |
| eff-perc | f0-32 | 86.42 | +1.24 | **36%** |
| eff-perc | f33-43 | 85.90 | +0.72 | 21% |
| eff-perc | f44-59 | 86.64 | **+1.46** | **42%** |
| eff-perc | f0-43 | 87.14 | +1.96 | 57% |
| nightly | AVM f0 | 82.19 | +0.04 | — |
| nightly | AVM f0-32 | 83.57 | +1.42 | — |
| nightly | AVM f33-43 | 83.34 | +1.19 | — |
| nightly | AVM f44-59 | 86.34 | **+4.19** | — |

The eff-perc gap to local SVT (88.641) is 3.46 K and decomposes almost
additively into 1.24 + 0.72 + 1.46 = 3.42 K. There is no single segment to
fix; the early chain and the tail are comparable. For nightly, the tail is 58%
of the 6.49 K gap, which is why the drift recipe was the only flag pair to move
K.

This also kills the idea that a better keyframe is the K lever by itself. The
key is 1/60 of the metric; a key-only oracle is +0.00. The key matters because
it is 66-79% of the bit budget and the predictor for the static head. A better
key that costs the same frees bits for the tail; a better key that costs more
does not pay.

## 7. What is already disproven (do not repeat)

| mechanism | measured result | verdict |
|---|---|---|
| Coefficient entropy/context | The coder is 1.015× its own per-context bound; pooled contexts cost 39% more (`docs/keyframe-bit-accounting.md`). `--video-tx-context-rate` is neutral to worse (QP120 15,764 B / 44.13 dB vs 16,130 B / 44.90 dB; QP60 36,285 B / 47.32 dB at λ=136, 52.75 vs 55.94 dB at the best λ=8) | ≤0.3 K; no |
| Adaptive quantisation (`--video-aq`) | Best strength +0.05 K at iso-rate, inside the ±1.5 K QP ripple; under `--nightly` it forces `--video-cdf-per-frame-qp` and is −0.47 K | no |
| QM / coring / dead zone | Five families at QP104-108: best iso-rate −0.75 K, worst −1.78 K. The fixed-QP "win" is a decision lottery | no |
| Perceptual source weights | At matched rate every D-shape is neutral or negative; the surviving piece is the error CSF, not a source mask | no |
| Explicit frame allocators (`AV2ENC_PIPE_LPL_REFRESH`) | 83.9 K at 240p, at the 83.80 wall; flat at 720p | no |
| Intra mode RD (`--video-key-intra-rd`) | −0.0014 JOD at iso-rate; iso-quality needs +0.92% rate; the gain was TX re-pricing already in `--video-tx-search` | no |
| Non-causal B / alt-ref | K 36.85 at 100.0 kbps (−45.3 vs the 82.15 pin); the 10 extra keys cost more than the future reference saves | no |
| Show-existing duplicate frames | Net −322 B, +0.0058 JOD at iso-rate; 44/60 decoded pictures change | no |
| Deep anchors at the key QP | Envelope 84.02 K; all-inter-at-key-depth ceiling 87.69 at 2.4× rate; short of 90 | no |
| Past-only multi-reference | refs2+drift 83.56 (−0.24), refs3+drift 83.77 (−0.03), refs4+drift 83.56 (−0.24); the refs-3 curve alternates sign inside the ripple | no |
| Compound prediction | ~79 K on the B carrier (−4.7 vs the bar); the P carrier is skip-only and not a quality tool | no |
| Wiener / PC-Wiener / chroma LR | Wiener −1.35 K at the drift point; the whole LR family decodes now but is negative at 240p | no |
| GDF | −0.012 JOD at 2.7× the CDEF CPU cost | no |
| Global MV / warp | Model fits IDENTITY at 240p (bit-level no-op); the escape-hatch warp is unproven and −0.75 K at 720p | no |
| TIP / SEF / RAS / OLK as quality levers | Syntax work is gated but none of it changes the 240p flat-P decision set | out of scope |
| Key QP offset as a rate knob | −21 with budget 3250 is 82.54 K at 99.9 kbps (vs drift 83.80); −24 is 82.96 at 105.3 (~81.5 iso). Key quality is not slack | no |
| Quarter/half MV precision | Quarter 79.73 K; half failed to decode 60/60 in this harness | no |
| Loop filters off | f1-32 45.45 dB vs 46.54, +3.4 kB; filters are not the drift | no |
| BAWP off | Static head identical, tail −0.22 dB, +43 B; mildly positive but not the drift lever | no |
| More/earlier full-PSNR refresh | threshold 1.0 / budget 3250: 26,819 B, K 82.43 at 107.3 kbps (≈80.5 iso); under-repairs the tail | no |

The one thing that did move K is the **decision metric**: eff-perc's
error-CSF + rate refit + top-2 rescore moved 83.80 → 85.10 at iso-rate by
changing which leaves code residual and which coefficients survive. That is the
class of change to continue.

## 8. Ranked implementation brief

The three mechanisms below are ranked by expected K per unit of implementation
risk. They overlap: all three change the coefficient/leaf decision set, so do
not add their ranges. Each has a kill criterion and a gate that keeps the
existing pins byte-identical when off.

### 8.1 Rank 1: frequency-weighted coefficient trellis (intra + inter RDOQ)

**What is missing.** Every major decision is now priced with the eff-perc
error CSF (`distortion_weight` + `lowpass_gain` in the TX search), **except the
coefficient trellis**. `rdoq_quantize()` minimizes unweighted forward-
coefficient SSE + λ·R (its `Coefficient` struct has no distortion weight); the
caller sets `level_context = 0`. And intra has no trellis at all:
`allow_rdoq=false` at every intra call site. So the one layer that decides
which individual coefficients survive is still optimizing plain SSE, exactly
the metric that over-values the high-frequency tail that Butteraugli/SS2
penalize least.

**Why it should pay.** In the eff-perc tail, 69% of the error is in coded
residual leaves and 61% of that error energy is in ring ≥6 while the useful
BA/SS2 band is 2-8 px. A trellis that weights ring 1-5 above ring ≥6 can drop
high-band ±1s (they are ~45% of nonzeros and cost ~4 bits each) and keep the
mid-band coefficients that BA sees. It also makes the residual cheaper, which
lets the leaf RD flip more skip→residual in the drift at the same λ — the
mechanism that produced the eff-perc gain. For the key, intra RDOQ is a pure
budget source: 97% of the key's metered symbol bits are coefficient syntax and
every 1 kB saved is ~0.3-0.8 K if spent on the tail (measured drift slope:
2.05 kB bought 1.64 K).

**Expected K: +0.4 to +0.9.** Inter shaping +0.2..0.4, intra budget
+0.2..0.5. Confidence: medium. If the key saving cannot be spent usefully (the
eff-perc tail budget is already at its measured plateau), count only the
inter-shaping half; the low end of the range assumes that.

**Cost: M, 3-5 days, ~300-500 LOC.**
Coordination: an in-flight `exp/luma-rdoq` worktree already has uncommitted
high-frequency-weight changes in `rdoq.cpp`/`rdoq.h`/`video_tx_search.cpp`
(plus a new `video_rdoq_hf.h`). Reconcile with that arm before opening a
second implementation; the estimate here is for that mechanism.
Exact files/functions:

* `src/codec/rate/rdoq/rdoq.h` — add `double distortion_weight = 1.0` to
  `Coefficient` (or a per-position weight span); default 1.0 keeps every
  existing test and caller bit-identical.
* `src/codec/rate/rdoq/rdoq.cpp` — `rdoq_quantize()`: use
  `weight * (original - level*step)^2` in the DP and in the distortion stop
  rule; report the weighted distortion.
* `src/pipeline/video/video_tx_search.cpp` — the `use_rdoq` block (~line 624):
  fill the weight from the scan position (same 8x8/16x16 ring map as
  `eff_perc::csf_gain()`); allow `use_rdoq` when `is_inter` **or** the new
  intra option is on and the TX class is 2-D.
* `src/pipeline/video/video_tx_search.h` — `Options`: add
  `intra_rdoq`/`rdoq_band_gain`; keep defaults off.
* `src/pipeline/video/video_frame_intra.cpp` — `encode_intra_luma_tx_partition`
  (line 292) and the two `search_luma_tx` calls at 437/510 currently pass
  `allow_rdoq=false`; pass the gate and the key qctx.
* `src/pipeline/video/video_frame_decision.cpp` — non-txpart intra call at
  ~2324, same change; the inter call at ~2693 already passes `enable_rdoq`.
* `src/cli/nightly.cpp` / `parse.cpp` — an env gate first
  (`AV2ENC_RDOQ_CSF`), CLI only after it measures positive.
* Gates: extend `src/pipeline/video/gate_video_rdoq.sh` with an intra case;
  `tools/gate_video_nightly.sh` must stay byte-identical with the gate off.

**Experiment.** Keyframe-only RD at QP106/100/95/90 (current key RD: 48.48 dB
@19.7 kB, 50.45 @23.2, 51.39 @25.0; SVT 50.16 @17.5) plus a full 60-frame
iso-rate pair. Success: ≥0.3 dB at lp2 for ≤2% key rate, and iso-rate K
≥ +0.2. Kill: key lp2 does not move, or iso-rate K < +0.15 over two QPs.

### 8.2 Rank 2: frequency-aware TX size/type search (detail TX)

**What is missing.** The key TX search chooses 8x8 for 68% of parts (464/685
parts) and never 4x4; the inter search chooses 64x64/64x32/32x16 for 59% of
parts and never a part narrower or shorter than 16. A 64x64 transform cannot
localise a 2-4 px correction at all. The current search prices every candidate
with the same SSE/CSF but a single rate term, so on a sparse drift residual the
large transform wins; in a detail leaf it can still win because the rate saving
dominates a distortion the leaf metric barely sees.

**Why it should pay.** The key's only deficit is below ~3 px; the early P chain
is 2-3 dB behind at lp2 and lp4. A TX candidate set that includes 8x8/4x4 for
leaves whose source band energy is concentrated at 2-4 px, and a 16x16 cap on
drift-refresh frames, is the direct test. It is also a natural ablation of Rank
1: if the trellis weights are right but the TX cannot represent the band, the
gain saturates.

**Expected K: +0.2 to +0.5.** Confidence: low-medium (the simple global TX
lambda sweeps are already negative; only the frequency-aware form is untested).

**Cost: M, 2-4 days, ~150-300 LOC.**
Exact files/functions:

* `src/pipeline/video/video_tx_search.h` — `Options::max_tx_size` exists but is
  only a post-check; make it restrict candidate enumeration, and add a
  per-leaf `detail_gain`/band mask.
* `src/pipeline/video/video_tx_search.cpp` — candidate enumeration at
  ~899 (`partition_leaf` / TX size sets); add 4x4/8x8 when the leaf's 2-4 px
  source energy is above a threshold and the rate permits.
* `src/pipeline/video/video_frame_intra.cpp` — `encode_intra_luma_tx_partition`
  (292): allow a second partition level for 16x16→8x8→4x4 when the detail
  metric selects it; the current single level is why 4x4 is unreachable.
* `src/pipeline/video/video_frame_internal.h` — `search_luma_tx` (~1568) pass
  the cap/weight; `video_frame_decision.cpp` inter call ~2693 set the cap on
  drift-refresh frames.
* No syntax change: TX sizes/types are already signalled and the decoder
  accepts every size in the set.

**Experiment.** Keyframe-only at QP106 and the full 60-frame point. Success:
lp2 +0.2 dB at ≤2% key rate, or tail BA −0.1 at iso-rate. Kill: >3% rate for
<0.2 dB lp2.

### 8.3 Rank 3: band-aware drift-refresh selection (not past-only compound)

**What is missing.** The drift trigger uses full-frame luma PSNR against the
key anchor and fires only after the reference has lost 1.5 dB (f9, f20, f25,
f26, f27, f30, f31, f33...). By then the low band has already drifted; the
refresh then spends its budget at the same QP (`drift_refresh_qp=0`) and can
under-repair. The measured no-go is important: simply refreshing **more**
(threshold 1.0, budget 3250) costs +1.8 kB and is K 82.43 at 107.3 kbps
(~80.5 iso). The lever is **which** frames at a fixed 3,250 B budget, not how
many.

**Why it might pay.** 77-91% of nightly P error is in skip leaves; the tail
oracle is +1.46 K for eff-perc. A trigger on the ring-1-2 + DC error of the
reference (or on a 4x4/8x8 block-mean MSE) would fire at f5-f7, where the
non-zero-MV skips first damage the picture, and skip frames whose full PSNR is
low only because of high-frequency noise. At the same budget it can repair more
of the error that BA/SS2 actually see. The skip-specific non-zero-MV penalty
(section 4.5) is the cheapest companion probe: it targets the static drift
directly, without touching global MV precision.

**Why not past-only two-hypothesis / OBMC.** It is measured negative:
refs2+drift 83.56 (−0.24), refs3+drift 83.77 (−0.03), refs4+drift 83.56
(−0.24); the refs-3 curve alternates sign inside the ripple. AV2 v1.0.0 has no
normative OBMC (`src/codec/inter/advanced/obmc.h` is an encoder-only
experiment), so it cannot change the decoder's prediction. On this clip the
source is a monotonic global fade; averaging two past references lags the fade
instead of extrapolating it. A future reference or a two-sided pair is the
prediction fix, and that is the measured-negative B/alt-ref family. Do not
spend the 1-2 weeks here.

**Expected K: +0.1 to +0.4** (trigger + skip-MV probe). Confidence:
medium-low.

**Cost: S-M, 2-3 days, ~80-150 LOC** for the trigger; 1-2 weeks for refs and
not recommended.
Exact files/functions:

* `src/pipeline/video/video_frame.cpp` — `encode_video_sequence_impl`, the
  drift block around lines 2574-2720 (`drift_anchor_psnr`, `ref_psnr`,
  `drift_refresh_frame`, `drift_refresh_budget`). Replace/augment `ref_psnr`
  with a low-band metric computed on the reference plane vs source.
* `src/pipeline/video/video_frame.h` — `VideoOptions` fields 100-130
  (`drift_refresh_threshold_db`, `drift_refresh_budget`); add an env-gated
  metric selector first, no CLI.
* `src/pipeline/video/video_frame_decision.cpp` — the skip/non-skip leaf RD
  (~2640-2740): add a small penalty to a non-zero MV on a leaf that finishes
  as skip, gated off by default.
* `src/cli/parse.cpp` — only if a CLI knob is kept after the measurement.

**Experiment.** Same 60-frame QP106/QP110 pair, budget fixed at 3,250 B;
compare the refresh frame set and iso-rate K. Success: K +0.15 at iso-rate with
the same bytes. Kill: the selected frames are the same, or K is inside the
±0.3 ripple.

### 8.4 Ranked table (all candidates)

| rank | mechanism | expected K | confidence | cost | files |
|---:|---|---:|---|---|---|
| 1 | Frequency-weighted RDOQ/trellis (intra+inter) | **+0.4..0.9** | medium | 3-5 d | `rdoq.h/.cpp`, `video_tx_search.*`, `video_frame_intra.cpp` |
| 2 | Frequency-aware TX size/type search | **+0.2..0.5** | low-med | 2-4 d | `video_tx_search.*`, `video_frame_intra.cpp`, `video_frame_internal.h` |
| 3 | Band-aware drift refresh + skip-MV probe | **+0.1..0.4** | med-low | 2-3 d | `video_frame.cpp/.h`, `video_frame_decision.cpp` |
| 4 | Past-only two-hypothesis / OBMC | **−0.2..0.0** | high (negative) | 1-2 wk | refs/compound carrier; not recommended |
| 5 | Coefficient entropy/context | **≤0.3** | high (bounded) | 1-2 d | no syntax room; do not |
| 6 | Learned tools for the 95 wall | **0 at 100 kbps** | high | months | not a luma-gap mechanism; see §9 |

## 9. What would close the whole gap

The top three mechanisms are expected to move eff-perc from 85.1 to roughly
**85.8-86.9** at 100 kbps. That is a real 0.7-1.8 K, and it is the right next
step because each is local, gated, and measurable against the existing pins.
It is not enough to reach SVT's 88.6, and nothing in the measured toolset is:

* The remaining 1.7-2.8 K is per-frame coding efficiency, spread 36% early
  (f0-32), 21% mid, 42% tail. There is no single segment or single tool that
  owns it.
* The only measured mechanism class that closes a segment is a prediction
  class change: SVT's tail uses two-sided prediction and deep anchors. Our
  future-reference/B/SEF family is negative because the key cost and schedule
  are wrong, not because the idea is impossible. A hidden-ARF pyramid with a
  proper key budget is the one untested path to >88, but it is a 1-2 week
  experiment with a hard gate (iso-rate K > 86.0) and it should wait until the
  coefficient mechanisms are exhausted, because it is not additive with them.
* **95 is a different program.** The K oracle needs JOD 9.90, BA ≈0.44, SS2 92.
  AVM e6f99d3 measures 90.19 with BA 1.29 / SS2 90.38; even the reference AV2
  encoder is 4.8 K short. AV2 v1.0.0 has no normative neural prediction or
  filter. Encoder-side learned predictors cannot be reproduced by the decoder
  without coding the residual, so they cannot lower rate; the only learned
  components available are (a) learned RD/metric models, bounded by the same
  toolset that eff-perc already exhausts at 85.18, and (b) learned/solved
  coefficients for the normative loop filters (Wiener/GDF/PC-Wiener), which
  measure negative at 240p. A learned tool for 95 is a new-codec project, not a
  100 kbps luma lever.

## 10. Reproduce

All commands are on `testdata/nasa_480x240_30fps_60f.y4m`, 60 frames, 8-bit
4:2:0, `--cpu-threads 8`.

```sh
# the five streams
build-dev/av2enc REF -o /tmp/nightly.ivf --video --frames 60 --qp 106 --nightly
build-dev/av2enc REF -o /tmp/drift.ivf --video --frames 60 --qp 110 --nightly \
  --video-drift-refresh-threshold 1.5 --video-drift-refresh-budget 3250
AV2ENC_EFF_PERC=1 build-dev/av2enc REF -o /tmp/eff.ivf --video --frames 60 \
  --qp 119 --nightly --video-drift-refresh-threshold 1.5 \
  --video-drift-refresh-budget 3250
SvtAv1EncApp -i REF -b /tmp/svt.ivf --preset 0 --crf 36.5 \
  --fps-num 30 --fps-denom 1 --progress 0
avm_latest/build/avmenc --end-usage=q --cpu-used=6 --threads=8 \
  --limit=60 --qp=124 -o /tmp/avm.ivf REF

# decode and score
tools/bin/dav2d-macos --strict 1 -i /tmp/X.ivf -o /tmp/X.y4m -q
DAV2D=tools/bin/dav2d-macos python3 tools/quality.py --ref REF \
  --ivf /tmp/X.ivf --json /tmp/X.q.json
python3 tools/extra_metrics.py --ref REF --test /tmp/X.y4m --json /tmp/X.e.json
python3 tools/kpi.py --quality-json /tmp/X.q.json --extra-json /tmp/X.e.json
```

Per-frame PSNR/lpN: a scratch numpy harness (not committed). Method:
`scipy.ndimage.uniform_filter(frame, N, mode="nearest")` on both source and
decoded luma, then PSNR on the filtered frames (`N=1` is full-res). DCT bands:
orthonormal 8x8 DCT on non-overlapping blocks, ring index
`sqrt(u^2 + v^2)`; `ring 1-2` is `1 <= r < 3`, `ring 3-5` is `3 <= r < 6`,
`ring >= 6` is the rest; DC is `(0,0)`. Block/TX attribution:

```sh
AV2ENC_BLOCKDUMP=/tmp/blocks.txt AV2ENC_SYMBOL_COST=1 \
  build-dev/av2enc REF -o /tmp/x.ivf --video --frames 60 --qp 106 --nightly
./build-dev/av2enc REF -o /tmp/x.ivf --video --frames 60 --qp 106 --nightly \
  --video-trace 2>/tmp/trace.log
```

`AV2ENC_BLOCKDUMP` records `mi_row mi_col bsize partition is_inter mode ref
drl mv_x mv_y skip eob_y eob_u eob_v digest`, one line per leaf, sorted by
`mi_row`; a frame boundary is a row reset. `--video-trace` prints
`video tx: leaf WxH type=T` for inter leaves and `intra tx part ... partition=P
parts=N` for the keyframe's TX-partition decision. Neither changes the
bitstream: the instrumented nightly stream is still 25,008 B /
`0639b43630abb454f7be5ac494d2efb4`.

## Appendix A: per-frame table

Nightly vs SVT (the gap definition), 60 frames. `N` = nightly QP106,
`S` = SVT p0, `lp8` = box-8 low-pass PSNR.


| f | N dB | N lp8 | N BA | N SS2 | S dB | S lp8 | S BA | S SS2 | E dB | E BA | A dB | A BA |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 48.48 | 64.75 | 1.61 | 89.00 | 50.16 | 59.99 | 1.93 | 90.64 | 46.05 | 1.58 | 52.47 | 1.02 |
| 1 | 48.48 | 64.75 | 1.61 | 89.00 | 50.27 | 61.17 | 1.93 | 90.75 | 46.05 | 1.57 | 52.47 | 1.02 |
| 2 | 48.30 | 64.23 | 1.57 | 88.91 | 50.31 | 62.04 | 1.66 | 90.77 | 45.94 | 1.58 | 52.30 | 1.01 |
| 3 | 48.21 | 63.47 | 1.81 | 88.86 | 50.22 | 62.13 | 1.46 | 90.79 | 45.89 | 1.92 | 52.16 | 1.29 |
| 4 | 47.20 | 61.26 | 1.96 | 88.46 | 49.97 | 60.14 | 1.52 | 90.57 | 45.30 | 2.02 | 51.95 | 0.88 |
| 5 | 46.08 | 59.62 | 1.84 | 87.68 | 50.12 | 61.48 | 1.48 | 90.56 | 44.62 | 1.91 | 51.85 | 1.27 |
| 6 | 46.07 | 59.62 | 1.84 | 87.68 | 50.15 | 61.88 | 1.48 | 90.61 | 44.63 | 1.91 | 52.08 | 0.84 |
| 7 | 46.52 | 60.91 | 1.84 | 88.01 | 48.76 | 59.59 | 1.76 | 90.10 | 44.90 | 1.97 | 50.79 | 1.28 |
| 8 | 45.95 | 59.26 | 1.81 | 87.70 | 50.29 | 62.59 | 1.23 | 90.75 | 44.58 | 1.86 | 52.26 | 1.23 |
| 9 | 46.95 | 61.64 | 1.58 | 88.18 | 49.91 | 63.05 | 1.30 | 90.70 | 45.03 | 1.87 | 51.13 | 1.54 |
| 10 | 46.97 | 61.64 | 1.63 | 88.18 | 50.11 | 64.61 | 1.27 | 90.92 | 45.08 | 1.85 | 51.69 | 1.27 |
| 11 | 46.96 | 61.64 | 1.63 | 88.16 | 50.08 | 64.34 | 1.27 | 90.89 | 45.08 | 1.85 | 51.57 | 1.32 |
| 12 | 46.97 | 61.35 | 1.70 | 88.13 | 50.19 | 65.23 | 1.31 | 90.94 | 45.07 | 2.83 | 51.41 | 1.12 |
| 13 | 46.98 | 61.32 | 2.00 | 88.10 | 50.07 | 65.25 | 1.21 | 90.85 | 45.13 | 2.26 | 51.29 | 0.92 |
| 14 | 46.91 | 60.96 | 2.13 | 88.08 | 50.15 | 65.69 | 1.24 | 90.92 | 45.09 | 2.07 | 51.60 | 0.96 |
| 15 | 46.78 | 60.43 | 2.36 | 87.99 | 50.10 | 65.17 | 1.23 | 90.93 | 45.02 | 1.92 | 51.54 | 0.91 |
| 16 | 46.77 | 60.43 | 2.36 | 87.98 | 50.14 | 65.59 | 1.23 | 90.99 | 45.02 | 1.92 | 51.54 | 0.91 |
| 17 | 46.70 | 60.71 | 2.53 | 87.98 | 49.96 | 65.06 | 1.23 | 90.93 | 44.92 | 2.31 | 51.11 | 1.17 |
| 18 | 46.76 | 60.76 | 2.92 | 88.04 | 49.92 | 64.88 | 1.28 | 90.89 | 45.03 | 2.08 | 51.45 | 1.62 |
| 19 | 45.88 | 59.05 | 3.24 | 87.54 | 49.45 | 62.37 | 1.62 | 90.36 | 44.52 | 1.80 | 50.87 | 1.17 |
| 20 | 46.77 | 60.74 | 1.74 | 87.89 | 50.10 | 64.65 | 1.24 | 90.93 | 45.14 | 1.66 | 51.22 | 1.05 |
| 21 | 46.77 | 60.74 | 1.74 | 87.87 | 50.09 | 64.35 | 1.24 | 90.92 | 45.14 | 1.67 | 51.35 | 1.00 |
| 22 | 46.67 | 60.23 | 2.04 | 87.80 | 50.03 | 64.29 | 1.71 | 90.78 | 45.09 | 1.75 | 51.18 | 1.26 |
| 23 | 46.73 | 60.33 | 2.14 | 87.82 | 50.14 | 64.33 | 1.37 | 90.88 | 45.11 | 2.03 | 51.11 | 0.99 |
| 24 | 45.68 | 58.31 | 2.63 | 87.16 | 49.94 | 61.70 | 1.40 | 90.73 | 44.49 | 2.17 | 50.28 | 1.00 |
| 25 | 45.94 | 59.59 | 2.28 | 87.23 | 49.67 | 62.30 | 1.37 | 90.54 | 44.51 | 1.87 | 50.11 | 1.18 |
| 26 | 45.95 | 59.61 | 2.31 | 87.28 | 49.68 | 62.32 | 1.37 | 90.52 | 44.53 | 1.87 | 50.08 | 1.18 |
| 27 | 45.99 | 59.52 | 2.43 | 87.06 | 49.88 | 62.37 | 1.32 | 90.50 | 44.55 | 1.93 | 50.14 | 1.14 |
| 28 | 46.13 | 59.76 | 2.49 | 87.22 | 49.94 | 62.25 | 1.32 | 90.65 | 44.54 | 1.93 | 50.25 | 1.17 |
| 29 | 45.70 | 58.58 | 2.57 | 86.82 | 49.71 | 61.15 | 1.33 | 90.46 | 44.59 | 1.81 | 49.84 | 1.45 |
| 30 | 45.98 | 59.51 | 1.85 | 87.13 | 49.17 | 59.87 | 1.52 | 90.26 | 44.54 | 1.83 | 49.30 | 1.60 |
| 31 | 46.03 | 59.53 | 1.84 | 87.07 | 49.24 | 60.02 | 1.52 | 90.28 | 44.58 | 1.80 | 49.30 | 1.60 |
| 32 | 43.66 | 51.28 | 2.46 | 84.88 | 45.29 | 50.37 | 1.48 | 88.07 | 43.58 | 1.93 | 47.93 | 1.64 |
| 33 | 44.45 | 57.12 | 2.14 | 85.84 | 47.03 | 54.32 | 1.39 | 88.34 | 43.70 | 2.04 | 48.71 | 1.20 |
| 34 | 44.14 | 56.63 | 2.25 | 85.31 | 47.23 | 55.47 | 1.66 | 88.68 | 43.48 | 2.25 | 46.30 | 1.83 |
| 35 | 43.49 | 51.84 | 2.60 | 83.30 | 47.88 | 59.11 | 1.44 | 89.08 | 43.43 | 2.34 | 48.93 | 1.26 |
| 36 | 43.69 | 52.79 | 2.95 | 83.70 | 47.89 | 59.05 | 1.44 | 89.10 | 43.49 | 2.37 | 48.89 | 1.26 |
| 37 | 43.04 | 50.17 | 4.50 | 80.34 | 48.18 | 59.37 | 1.64 | 88.68 | 43.55 | 2.42 | 49.11 | 1.15 |
| 38 | 42.89 | 50.46 | 4.24 | 80.20 | 47.79 | 58.67 | 1.83 | 88.65 | 43.45 | 2.79 | 48.80 | 1.33 |
| 39 | 42.82 | 49.64 | 5.13 | 79.21 | 47.94 | 57.48 | 1.81 | 88.73 | 43.62 | 2.72 | 48.97 | 1.49 |
| 40 | 42.81 | 49.65 | 5.05 | 77.54 | 47.50 | 53.38 | 1.84 | 88.83 | 43.53 | 2.75 | 48.61 | 2.56 |
| 41 | 43.14 | 50.43 | 4.02 | 78.21 | 48.04 | 57.39 | 1.70 | 87.75 | 43.64 | 2.75 | 48.70 | 1.36 |
| 42 | 42.73 | 49.78 | 4.54 | 77.06 | 48.42 | 60.53 | 1.85 | 88.93 | 43.63 | 3.08 | 49.13 | 1.60 |
| 43 | 42.65 | 48.80 | 5.71 | 75.01 | 48.06 | 56.82 | 1.71 | 87.57 | 43.74 | 2.58 | 49.76 | 1.30 |
| 44 | 42.30 | 48.15 | 6.07 | 72.19 | 48.14 | 58.28 | 1.54 | 88.23 | 43.73 | 2.57 | 50.25 | 1.27 |
| 45 | 42.39 | 47.69 | 6.19 | 72.58 | 47.62 | 58.57 | 1.73 | 87.38 | 43.79 | 2.66 | 50.36 | 1.29 |
| 46 | 42.87 | 48.85 | 5.86 | 73.10 | 47.60 | 58.49 | 1.73 | 87.38 | 43.85 | 2.66 | 50.30 | 1.29 |
| 47 | 42.75 | 48.74 | 6.55 | 72.07 | 47.28 | 57.76 | 1.74 | 86.50 | 43.90 | 2.61 | 50.56 | 1.33 |
| 48 | 42.76 | 49.19 | 6.25 | 70.43 | 46.75 | 57.40 | 1.82 | 86.25 | 43.84 | 2.93 | 50.93 | 1.24 |
| 49 | 42.61 | 48.55 | 6.58 | 69.31 | 47.87 | 58.43 | 1.78 | 85.92 | 44.03 | 2.48 | 50.91 | 1.28 |
| 50 | 42.63 | 48.47 | 6.98 | 67.40 | 48.34 | 57.37 | 1.66 | 85.95 | 44.04 | 2.79 | 50.56 | 1.60 |
| 51 | 43.06 | 49.80 | 4.65 | 68.89 | 48.43 | 57.80 | 1.66 | 85.79 | 44.16 | 2.79 | 51.32 | 1.32 |
| 52 | 43.03 | 49.03 | 5.78 | 67.16 | 48.93 | 58.46 | 1.81 | 86.56 | 44.18 | 3.22 | 51.07 | 1.24 |
| 53 | 42.99 | 48.38 | 5.89 | 64.59 | 49.03 | 58.28 | 1.93 | 85.67 | 44.21 | 3.08 | 50.67 | 1.31 |
| 54 | 43.20 | 48.65 | 6.79 | 63.15 | 48.75 | 56.74 | 1.93 | 85.22 | 44.38 | 2.56 | 51.48 | 1.20 |
| 55 | 43.63 | 49.81 | 5.69 | 63.82 | 48.45 | 57.50 | 2.02 | 84.06 | 44.63 | 2.81 | 51.49 | 1.41 |
| 56 | 43.93 | 50.33 | 5.26 | 65.22 | 48.45 | 57.57 | 2.02 | 84.22 | 44.81 | 2.53 | 51.59 | 1.35 |
| 57 | 43.87 | 49.53 | 5.11 | 62.71 | 49.67 | 58.16 | 1.46 | 86.59 | 44.67 | 2.82 | 51.48 | 1.36 |
| 58 | 43.77 | 49.11 | 5.25 | 57.02 | 48.21 | 57.42 | 2.04 | 83.27 | 44.97 | 3.06 | 51.79 | 1.34 |
| 59 | 44.14 | 49.29 | 4.73 | 57.91 | 47.38 | 53.86 | 2.96 | 78.21 | 45.30 | 2.78 | 45.65 | 2.15 |


eff-perc and AVM, same frames:


| f | E dB | E lp8 | E BA | E SS2 | A dB | A lp8 | A BA | A SS2 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 46.05 | 62.45 | 1.58 | 86.70 | 52.47 | 64.52 | 1.02 | 92.67 |
| 1 | 46.05 | 62.45 | 1.57 | 86.66 | 52.47 | 64.52 | 1.02 | 92.67 |
| 2 | 45.94 | 62.04 | 1.58 | 86.56 | 52.30 | 65.50 | 1.01 | 92.65 |
| 3 | 45.89 | 61.82 | 1.92 | 86.52 | 52.16 | 66.15 | 1.29 | 92.73 |
| 4 | 45.30 | 60.19 | 2.02 | 86.21 | 51.95 | 64.64 | 0.88 | 92.49 |
| 5 | 44.62 | 58.88 | 1.91 | 85.64 | 51.85 | 62.19 | 1.27 | 92.11 |
| 6 | 44.63 | 58.90 | 1.91 | 85.63 | 52.08 | 62.68 | 0.84 | 92.42 |
| 7 | 44.90 | 59.90 | 1.97 | 85.81 | 50.79 | 61.17 | 1.28 | 92.12 |
| 8 | 44.58 | 59.03 | 1.86 | 85.56 | 52.26 | 64.13 | 1.23 | 92.63 |
| 9 | 45.03 | 60.23 | 1.87 | 85.91 | 51.13 | 62.59 | 1.54 | 92.21 |
| 10 | 45.08 | 60.43 | 1.85 | 85.96 | 51.69 | 64.36 | 1.27 | 92.49 |
| 11 | 45.08 | 60.43 | 1.85 | 85.95 | 51.57 | 64.04 | 1.32 | 92.36 |
| 12 | 45.07 | 60.21 | 2.83 | 85.89 | 51.41 | 64.43 | 1.12 | 92.22 |
| 13 | 45.13 | 60.36 | 2.26 | 85.88 | 51.29 | 64.33 | 0.92 | 92.28 |
| 14 | 45.09 | 60.34 | 2.07 | 85.84 | 51.60 | 65.51 | 0.96 | 92.45 |
| 15 | 45.02 | 60.07 | 1.92 | 85.78 | 51.54 | 65.25 | 0.91 | 92.44 |
| 16 | 45.02 | 60.07 | 1.92 | 85.79 | 51.54 | 65.26 | 0.91 | 92.44 |
| 17 | 44.92 | 59.54 | 2.31 | 85.74 | 51.11 | 64.86 | 1.17 | 92.34 |
| 18 | 45.03 | 60.05 | 2.08 | 85.86 | 51.45 | 64.70 | 1.62 | 92.31 |
| 19 | 44.52 | 58.69 | 1.80 | 85.51 | 50.87 | 63.38 | 1.17 | 91.94 |
| 20 | 45.14 | 59.75 | 1.66 | 85.62 | 51.22 | 64.14 | 1.05 | 92.29 |
| 21 | 45.14 | 59.75 | 1.67 | 85.61 | 51.35 | 64.85 | 1.00 | 92.32 |
| 22 | 45.09 | 59.65 | 1.75 | 85.55 | 51.18 | 64.14 | 1.26 | 92.29 |
| 23 | 45.11 | 59.16 | 2.03 | 85.52 | 51.11 | 63.71 | 0.99 | 92.32 |
| 24 | 44.49 | 58.32 | 2.17 | 85.14 | 50.28 | 62.16 | 1.00 | 91.98 |
| 25 | 44.51 | 58.46 | 1.87 | 85.06 | 50.11 | 62.99 | 1.18 | 91.76 |
| 26 | 44.53 | 58.56 | 1.87 | 85.08 | 50.08 | 62.61 | 1.18 | 91.72 |
| 27 | 44.55 | 58.95 | 1.93 | 84.96 | 50.14 | 62.50 | 1.14 | 91.76 |
| 28 | 44.54 | 58.92 | 1.93 | 84.98 | 50.25 | 61.20 | 1.17 | 91.95 |
| 29 | 44.59 | 59.02 | 1.81 | 85.08 | 49.84 | 61.64 | 1.45 | 91.64 |
| 30 | 44.54 | 58.94 | 1.83 | 84.89 | 49.30 | 60.94 | 1.60 | 91.30 |
| 31 | 44.58 | 59.23 | 1.80 | 84.98 | 49.30 | 60.81 | 1.60 | 91.24 |
| 32 | 43.58 | 57.49 | 1.93 | 84.62 | 47.93 | 57.46 | 1.64 | 90.69 |
| 33 | 43.70 | 57.16 | 2.04 | 84.29 | 48.71 | 58.84 | 1.20 | 90.75 |
| 34 | 43.48 | 56.74 | 2.25 | 83.28 | 46.30 | 50.08 | 1.83 | 89.60 |
| 35 | 43.43 | 56.25 | 2.34 | 82.03 | 48.93 | 59.72 | 1.26 | 89.35 |
| 36 | 43.49 | 56.49 | 2.37 | 82.54 | 48.89 | 59.31 | 1.26 | 89.35 |
| 37 | 43.55 | 56.57 | 2.42 | 82.11 | 49.11 | 58.96 | 1.15 | 89.69 |
| 38 | 43.45 | 56.31 | 2.79 | 81.30 | 48.80 | 59.04 | 1.33 | 89.55 |
| 39 | 43.62 | 56.66 | 2.72 | 81.20 | 48.97 | 58.45 | 1.49 | 89.04 |
| 40 | 43.53 | 56.25 | 2.75 | 80.95 | 48.61 | 59.05 | 2.56 | 89.27 |
| 41 | 43.64 | 56.82 | 2.75 | 81.05 | 48.70 | 59.24 | 1.36 | 89.45 |
| 42 | 43.63 | 56.52 | 3.08 | 80.36 | 49.13 | 59.77 | 1.60 | 89.55 |
| 43 | 43.74 | 56.63 | 2.58 | 79.25 | 49.76 | 57.70 | 1.30 | 88.35 |
| 44 | 43.73 | 56.59 | 2.57 | 79.11 | 50.25 | 58.92 | 1.27 | 88.37 |
| 45 | 43.79 | 56.45 | 2.66 | 78.68 | 50.36 | 59.11 | 1.29 | 88.60 |
| 46 | 43.85 | 56.40 | 2.66 | 78.71 | 50.30 | 58.81 | 1.29 | 88.59 |
| 47 | 43.90 | 56.10 | 2.61 | 78.41 | 50.56 | 58.53 | 1.33 | 88.31 |
| 48 | 43.84 | 55.96 | 2.93 | 77.25 | 50.93 | 59.51 | 1.24 | 88.55 |
| 49 | 44.03 | 56.02 | 2.48 | 77.22 | 50.91 | 58.18 | 1.28 | 88.76 |
| 50 | 44.04 | 55.88 | 2.79 | 76.92 | 50.56 | 55.36 | 1.60 | 88.38 |
| 51 | 44.16 | 55.77 | 2.79 | 77.00 | 51.32 | 59.68 | 1.32 | 88.43 |
| 52 | 44.18 | 55.75 | 3.22 | 76.27 | 51.07 | 58.51 | 1.24 | 87.34 |
| 53 | 44.21 | 56.25 | 3.08 | 76.59 | 50.67 | 57.09 | 1.31 | 87.21 |
| 54 | 44.38 | 56.04 | 2.56 | 76.20 | 51.48 | 59.41 | 1.20 | 88.18 |
| 55 | 44.63 | 55.50 | 2.81 | 74.96 | 51.49 | 59.13 | 1.41 | 87.09 |
| 56 | 44.81 | 56.39 | 2.53 | 75.15 | 51.59 | 58.97 | 1.35 | 87.31 |
| 57 | 44.67 | 56.00 | 2.82 | 73.09 | 51.48 | 59.39 | 1.36 | 86.65 |
| 58 | 44.97 | 55.52 | 3.06 | 72.95 | 51.79 | 58.96 | 1.34 | 86.27 |
| 59 | 45.30 | 56.09 | 2.78 | 73.05 | 45.65 | 47.19 | 2.15 | 79.71 |


## Appendix B: key RD and tail efficiency

Keyframe RD on the current encoder (`--nightly`, key offset −27; bytes are the
single-picture IVF size, PSNR the decoded luma):

| QP | key qidx | bytes | PSNR-Y | lp2 | lp8 |
|---:|---:|---:|---:|---:|---:|
| 106 | 79 | 19,708 | 48.48 | 53.63 | 64.75 |
| 100 | 73 | 21,739 | 49.40 | 54.59 | 65.83 |
| 95 | 68 | 23,179 | 50.45 | 55.84 | 67.01 |
| 90 | 63 | 24,965 | 51.39 | 56.74 | 67.95 |
| SVT p0 | 25 | 17,546 | 50.16 | 54.71 | 59.99 |

At SVT's key PSNR (50.16) our key needs ~22.8 kB, **1.30× SVT's key rate**.
At SVT's key bytes (17.5 kB) we are ~1.7 dB behind. The key is not uniformly
worse: at 23.2 kB we beat SVT's lp2 by 1.1 dB and lp8 by 7.0 dB. The 1.3×
rate gap is real coding efficiency, and the band shape says the 2-4 px band is
where the first 0.5-1.0 kB should go.

Tail efficiency:

| segment | stream | bytes | PSNR-Y | BA | SS2 |
|---|---|---:|---:|---:|---:|
| f44-59 | eff-perc | 3,936 | 44.28 | 2.77 | 76.35 |
| f44-59 | SVT | 1,740 | 48.18 | 1.86 | 85.45 |
| f33-43 | eff-perc | 2,971 | 43.57 | 2.55 | 81.67 |
| f33-43 | SVT | 2,187 | 47.82 | 1.66 | 88.58 |

In the tail eff-perc spends 2.3× SVT's bytes and is 3.9 dB / 9.1 SS2 behind.
More tail bits alone do not fix that: the all-refresh QP106 probe (57 refresh
frames, 31,455 B, 125.8 kbps) is K 85.84, i.e. +3.69 K for +26% rate, which is
≈81 K at iso-rate. The tail needs better residual/prediction decisions, not
more of the same residual.
