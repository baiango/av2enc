# Keyframe bit accounting and the keyframe RD-weight gap

Where a 480x240 keyframe's bits actually go, what the coefficient coder costs
against its own entropy bound, which transforms the keyframe selects, and the
one measured lever that accounts for most of the published gap to SVT.

Diagnosis only: every number below comes from instrumentation that is
env-gated and bitstream-neutral, or from an existing CLI flag. The pinned
streams are unchanged (re-verified after the last change; 60/60 pictures
decoded by `dav2d --strict 1`):

| stream | bytes | MD5 |
|---|---:|---|
| default `--qp 189`, 60 f | 25,009 | `a1d0dba1259b20a4b628d4cd1e14c33c` |
| `quality --qp 190`, 60 f | 25,141 | `3984677b9b665952984b07b7cf75c6ea` |
| nightly `--qp 110`, 60 f | 24,822 | `41d90595489d4aa9a86a6b3e9132a84f` |

`tools/gate_video_nightly.sh --bin build-t/av2enc` passes (expansion, pins,
strict decode, visible recon, BAWP), including the CDEF params-row case
24,885 → 24,822 B.

## Object under test

```sh
cmake -S . -B build-t -DAV2ENC_ENABLE_CUDA=OFF -DAV2ENC_ALLOW_CPU_ONLY=ON \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build-t -j1

AV2ENC_SYMBOL_COST=1 build-t/av2enc testdata/nasa_480x240_30fps_60f.y4m \
    -o /tmp/kf.ivf --video --frames 1 --qp 120 --nightly
```

`--qp 120` resolves to `base_q_idx` 93 (nightly applies the −27 keyframe
offset). One tile, one keyframe, 16,162 B IVF / 16,074 B tile payload
(128,592 bits), 189 intra leaves, decoded PSNR-Y 44.90 dB.

## 1. Tile accounting (QP120)

`AV2ENC_SYMBOL_COST=1` now prints the tile summary plus three detail blocks.
The first line is unchanged; `coeff_syntax`, `coeff_tile` and `coeff_band` are
new (see §5).

| group | symbols | bits | share of tile |
|---|---:|---:|---:|
| partition (`do_split`, `do_square_split`, `rect_type`) | 500 | 293.4 | 0.23 % |
| intra mode + MRL + intra TX type | 1,162 | 1,568.6 | 1.22 % |
| TX partition + intra TX type symbols (misc) | — | 377.0 | 0.29 % |
| **coefficient syntax (all three planes)** | | **126,193.4** | **98.14 %** |
| CDEF index / params | 189 | 0 | 0 % |
| CCSO block symbols | 189 | 22.0 | 0.02 % |
| delta-Q | 0 | 0 | 0 % |
| §8.2.4 marker, padding, un-metered trailer | — | 137.7 | 0.11 % |

The coefficient syntax splits as:

| element | n | bits | share of tile |
|---|---:|---:|---:|
| `txb_skip` | 847 | 428.8 | 0.33 % |
| EOB prefix symbol | 1,264 | 1,687.8 | 1.31 % |
| EOB escape/extra **literal** bits | — | 2,416 | 1.88 % |
| `coeff_base` (incl. the `_lf`/`_uv` arrays) | 48,029 | 67,126.8 | 52.20 % |
| `coeff_base_eob` | 662 | 242.1 | 0.19 % |
| `coeff_br` | 8,214 | 14,393.7 | 11.19 % |
| `read_quant` escape **literal** bits | — | 13,495 | 10.49 % |
| `dc_sign` symbol | 371 | 373.2 | 0.29 % |
| sign **literal** bits (one per remaining nonzero) | — | 25,778 | 20.05 % |
| `cctx_type` | 0 | 0 | — |
| other literal bits (`write_literal` outside the groups above) | — | 252 | 0.20 % |
| **total** | | **126,193.4** | **98.14 %** |

So: partition + mode/MRL = **1.45 %** of the keyframe. This is not a
signalling-overhead problem; it is a residual problem. And the residual is not
an entropy-coder problem: the arithmetic coder plus the syntax is at 98 % of
the tile and, as §2 shows, within 1.5 % of its own bound.

Two of the elements are pure per-nonzero overhead and scale with the *number*
of coded coefficients, not with the picture: the 25,778 sign bits (20.1 % of
the tile, exactly one bit per nonzero outside the DC position) and the 13,495
escape bits (10.5 %).

## 2. Is the coder wasteful, or is the quantisation?

The meter keeps a per-CDF-instance symbol histogram, so each syntax element can
be compared against two bounds:

* `bits` — what the tile actually spent, `-log2 p` against the live CDF;
* `ctxH` — the summed ideal static entropy of the *same* CDF-instance
  partition (a lower bound for this syntax decomposition at this context
  granularity); `bits - ctxH` is adaptation lag plus per-instance estimation
  noise;
* `pooledH` — the entropy of the pooled histogram, i.e. what the element would
  cost if the contexts were thrown away.

| element | bits spent | `ctxH` | bits/ctxH | `pooledH` |
|---|---:|---:|---:|---:|
| `txb_skip` | 428.8 | 350.7 | 1.22× | 641.4 |
| EOB prefix | 1,687.8 | 1,571.7 | 1.07× | 3,223.6 |
| `coeff_base` | 67,126.8 | 66,373.3 | **1.011×** | 93,027.8 |
| `coeff_base_eob` | 242.1 | 194.5 | 1.24× | 279.7 |
| `coeff_br` | 14,393.7 | 14,153.3 | 1.017× | 15,258.4 |
| `dc_sign` | 373.2 | 366.8 | 1.02× | 370.0 |
| all symbol families | 84,252.4 | 83,010.3 | **1.015×** | — |

The literal groups have no entropy bound to beat: the escape code is the
normative `read_quant` escape, the sign is a raw bit per nonzero, and the EOB
extra bits are the normative suffix.

Interpretation:

* The coder is **1.5 % above its own per-context entropy bound**. It is not
  wasting bits on the symbol stream it is given.
* The contexts are doing real work: the same symbols cost **39 % more if
  pooled** (`coeff_base` 93.0 k vs 66.4 k), i.e. 39 % above the per-context
  bound. Most of that is the significance/EOB/zero modelling, not a structure
  the writer could drop.
* Therefore the rate is the *cost of the coefficient field the quantiser
  produced*, not entropy-stage overhead. Any real saving has to come from
  coding fewer or cheaper coefficients.

### The coefficient field

| | luma | U | V |
|---|---:|---:|---:|
| transform blocks | 469 | 189 | 189 |
| skipped (`eob=0`) | 59 | 59 | 67 |
| EOB sum | 42,138 | 3,661 | 2,892 |
| coded positions (inside EOB) | 42,138 | 3,661 | 2,892 |
| of which **zero** | 18,339 | 2,382 | 1,821 |
| nonzero levels | 23,799 | 1,279 | 1,071 |
| Σ abs(level) | 80,131 | 1,654 | 1,422 |
| coded coefficient area | 80,896 | 28,800 | 28,800 |

Derived:

* 26,149 nonzero levels for the frame, mean |level| **3.18**, i.e.
  **3.99 bits per nonzero** (base + `coeff_br` + escape + sign + share of
  `coeff_base_eob`).
* 22,542 of the 48,691 coded positions (46.3 %) are zeros inside the EOB.
  Those positions cost **17,358 bits = 13.5 % of the tile** just to say
  "this coefficient is zero" (`coeff_tile … zero_pos=… zero_bits=…`).
* Luma coefficient density is **29.4 %** of the coded transform area
  (23,799 / 80,896) and the mean EOB covers **52.1 %** of the area — half of
  every transform is never coded, the other half is half full of nonzeros.
* Levels are broad, not dominated by ±1: |1| 10,794, |2| 4,300, |3| 2,349,
  |4| 1,561, |5| 1,062, |6| 781, |7| 560, ≥8 2,392 (luma). 45 % of the
  nonzeros are ±1, but 27 % are ≥4, which is why 10.5 % of the tile is
  `read_quant` escapes.

Practical consequence: every coefficient the encoder declines to code saves
its base symbol + sign (+ escape) — the sign alone is 1 bit, and the average
nonzero costs 4 bits. A 25 % cut in coded nonzeros is worth ~25 % of 98 % of
the frame at a fixed quantiser.

## 3. Transform selection (is it coarse-only?)

QP120 luma transform sizes (TX_SIZE index: count), from `coeff_tile`/`tx_size_hist`:

| size | count | | size | count |
|---|---:|---|---|---:|
| 8x8 (1) | **200** | | 32x8 (16) | 44 |
| 8x16 (7) | 50 | | 8x32 (15) | 2 |
| 16x8 (8) | 90 | | 32x64 (11) | 1 |
| 16x16 (2) | 46 | | 64x32 (12) | 5 |
| 32x32 (3) | 4 | | 64x16 (18) | 4 |
| 64x64 (4) | 8 | | 64x8 (22) | 6 |
| 16x32 (9) | 2 | | 32x16 (10) | 7 |

Luma TX types: DCT_DCT 248, ADST_DCT 29, DCT_ADST 21, ADST_ADST 123, other
2-D types 37, IDTX 20. Chroma is DCT_DCT only (`enable_chroma_dctonly = 1`),
8x8 118 of 189 blocks.

Coding-block sizes: 16x16 118, 16x32 14, 32x16 22, 32x32 6, 32x64 1, 64x32 5,
64x64 8, 64x16 15.

So the transform is **not** coarse-only, and the tree does split. 8x8 is the
modal luma transform (43 % of blocks) and 16x16/16x8/8x16 make up most of the
rest. 4x4 (TX_SIZE 0) is *legal* — an 8x8 coding block with
`TX_PARTITION_SPLIT` yields four 4x4 transforms
(`tx_partition_geometry(kSplit)` → `find_tx_size(4,4) = k4x4`) — but it is
selected **zero** times at QP120. Lowering the coding-block floor
(`--video-min-block 4`) still produces no 4x4 transform; the search prefers
16x4/32x4 shapes (156 / 64 blocks). That is a preference, not a restriction,
and it is consistent with a rate model that under-prices coefficients (the
search charges 1.0 bit/nonzero against the writer's measured 3.99; see §5).

## 4. Where the rate goes: the keyframe RD weight

`--nightly` sets the TX-search rate weight to a **constant**
`--video-tx-lambda 136` (SSE per bit). It is passed unchanged to
`search_luma_tx()` for every keyframe leaf and to
`encode_intra_luma_tx_partition()`'s NONE/SPLIT/HORZ/VERT comparison. A
constant SSE-per-bit weight cannot be right across a QP range: the RD-optimal
weight scales with the quantiser step squared, which is a factor of ~32 between
`base_q_idx` 3 and 93 in the table the encoder uses.

Measured on the same keyframe, only the weight changed:

| `--qp` | frame QP | λ=136 (nightly) | λ=8 | Δ |
|---:|---:|---|---|---|
| 120 | 93 | 16,130 B / 44.90 dB | 16,385 B / **45.64 dB** | +0.74 dB, +1.6 % rate |
| 110 | 83 | 18,913 B / 46.24 dB | 19,199 B / 47.55 dB | +1.31 dB |
| 100 | 73 | 22,013 B / 46.76 dB | 22,495 B / **49.29 dB** | +2.53 dB, +2.2 % rate |
| 90 | 63 | 25,368 B / 47.77 dB | 25,934 B / 51.38 dB | +3.61 dB |
| 80 | 53 | 29,113 B / 48.48 dB | 29,695 B / **52.95 dB** | +4.47 dB, +2.0 % rate |
| 70 | 43 | 32,940 B / 48.98 dB | 33,653 B / 54.58 dB | +5.60 dB |
| 60 | 33 | 37,033 B / 49.30 dB | 37,877 B / **55.94 dB** | +6.64 dB, +2.3 % rate |
| 50 | 23 | 41,201 B / 49.57 dB | — | |
| 40 | 13 | 45,551 B / **49.67 dB** | — | asymptote |
| 30 | 3 | 49,548 B / 47.60 dB | — | anomalous: more rate, lower PSNR than QP40 |

λ=4 and λ=2 are within 0.1–0.2 dB of λ=8 at every QP tested, so λ≤8 is a
plateau, not a cliff.

It is not only a keyframe effect. On the pinned nightly operating point
(480x240, 60 frames, `--qp 110 --nightly`), changing only the weight:

| λ | bytes | frames | mean PSNR-Y | key PSNR-Y |
|---:|---:|---:|---:|---:|
| 136 (pinned) | 24,822 | 60/60 | 44.090 | 46.24 |
| 8 | 24,915 | 60/60 | **44.650** | 47.55 |
| 4 | 25,470 | 60/60 | 44.715 | 47.72 |

+0.56 dB of mean-Y for +0.37 % rate on the very point the constant was tuned
for (the calibration used JOD on a larger benchmark, so this is not a
like-for-like re-measurement — but the direction and size are clear). Changing
it moves the pinned stream, so it is **not** applied here.

The published operating point follows from this table:

* with λ=136 the keyframe **saturates at 49.67 dB at 45.5 kB** — it cannot
  reach the 49.72 dB that SVT reaches at 15.4 kB at *any* rate;
* with λ=8 the curve reaches 49.72 dB at **≈23.2 kB** (interpolating QP100/90),
  i.e. the iso-PSNR rate gap falls from **2.4×** to **1.50×**;
* at the CRF15 point (12,389 B / 47.16 dB) the same interpolation gives
  ≈18.6 kB for λ=8, a **1.50×** gap.

Mechanism, measured rather than assumed. At QP60 the two λ settings give
`pred_sse` identical to 3 significant figures (95,901,921 vs 95,904,778 — the
prediction is chosen before the TX search) while `recon_sse` is 86,990 vs
17,946. The partition histogram moves by only 7 leaves, the transform-size
histogram barely moves, and the coded coefficient count/Σ|level| moves by 4 %
(40,293 → 41,948 nonzeros, 477,861 → 483,615 Σ|level|). With λ=136 the
residual SSE is concentrated: 53 % of the frame's SSE sits in 6 of the 450
16x16 regions at the bottom/right of the frame, with per-block MSE 15–44. λ=8
spends the missing ~850 B on those leaves and every block lands at MSE ≤ 0.5.
A handful of expensive leaves abandoned by an over-large weight is exactly
what a small PSNR drop spread over *every* measurement scale looks like after
box averaging — which matches the band-resolved deficit that motivated this
investigation (box1 +4.8 dB, box8 +6.4 dB).

The same constant is used by the inter TX search, so the mis-calibration is not
keyframe-only — on the 60-frame nightly point above it is +0.56 dB for
+0.37 % rate. But the *scale* is QP-dependent: at the QP 110-120 operating
point the keyframe difference is only 0.7–1.3 dB, while at QP 60 it is 6.6 dB.
That is exactly the signature of a weight that should track the quantiser
step: the calibration point is nearly right and the rest of the QP range is
not, which is why a 60-frame 100 kbps aggregate never exposed it.

## 5. Negative and control results

* **Dead-zone rounding is not free money.** `AV2ENC_Q_DEADZONE=<0..0.5>` is a
  diagnosis-only rounding offset added at the two scalar quantisers
  (`txv::quantize` and the TX-search quantiser, kept identical). At QP120:
  dead-zone 0.15 → 14,805 B / 43.88 dB, 0.25 → 14,058 B / 42.82 dB, against
  16,130 B / 44.90 dB for `lround`. Interpolated at 44.90 dB the dead-zone form
  needs ~16.4 kB, i.e. it is ~1.6 % **worse** than the pinned rounding. The
  45 % ±1 coefficients are not waste in the RD sense; a trellis (RDOQ) is the
  only untested way to drop them.
* **Intra luma has no RDOQ.** `allow_rdoq=false` at every intra call site
  (`video_frame.cpp` keyframe TX part search, whole-block `compute_tx`, and the
  chroma planes); `use_rdoq` additionally requires `is_inter`. So every
  keyframe coefficient is plain `lround` at the frame step, while the inter
  path runs the full trellis (`--video-rdoq`).
* **The true §8.3 rate model does not close the gap.**
  `--video-tx-context-rate` (which prices candidates with `txb_skip`, EOB,
  `coeff_base`/`_br`, escape and sign from the static CDF slice instead of
  1 bit/nonzero) is neutral-to-worse on this keyframe: QP120 15,764 B /
  44.13 dB, QP60 36,285 B / 47.32 dB (λ=136), and it *hurts* at the best
  weight (QP60 λ=8: 52.75 dB vs 55.94 dB). The mismatch between the search's
  1.0 bit/nonzero and the writer's measured 3.99 bits/nonzero is therefore not
  the dominant error.
* **Partition probes are inert on the keyframe.** `--video-partition-rd-lambda`
  2.75e-7…2.75e-4 and `--video-partition-rd-depth` 1…3 all produce the
  identical QP120 stream: the measured probe is P-frame-only, so the keyframe
  tree still runs on the variance proxy + 6-bit leaf estimate.
* **Prediction is not obviously broken, and not the explained part of the
  gap.** For the 165 leaves that code a residual (72,960 luma px, the other 24
  leaves are eob=0 palette/IBC-exact), the chosen prediction SSE is 788 per
  pixel (std 28.1) while the source's own within-block (block-mean) variance is
  1,009 and a plain "copy the row above / column to the left" predictor gives
  878. Reconstruction MSE on the same leaves is 4.29. So the search leaves
  only ~22 % less energy than DC, and it is better than a trivial copy, but a
  28-level residual at this rate is what fills the coefficient field.
* **External reference (same frame, SVT-AV1 preset 0 via `libsvtav1`, keyint
  1).** CRF20 10,474 B / 45.18 dB; CRF25 8,978 B / 43.49 dB; CRF30 7,520 B /
  41.52 dB. At 45.18 dB our best-λ keyframe needs ≈16.0 kB, so we are ~1.5×
  behind AV1 as well as ~1.5× behind the reported AV2 number. Whatever is left
  after the RD weight is a per-frame coding-efficiency gap, not an AV2-vs-AV1
  difference.

## 6. Ranked conclusions

1. **The keyframe RD weight is the largest measured defect (~40 % of the
   published rate gap).** A constant `--video-tx-lambda 136` SSE/bit against a
   weight that must scale as the quantiser step squared: +6.6 dB at +2.3 %
   rate at QP60, and at iso-PSNR 49.72 dB the keyframe rate falls 37 kB → 23 kB
   (2.4× → 1.5× vs SVT). It is a *decision* defect, not a coder defect: the
   prediction never changes, a handful of expensive leaves are abandoned, and
   the same coefficient syntax then costs 4 bits for each retained level. On
   the pinned nightly point the same change is +0.56 dB mean-Y for +0.37 %
   rate, so this is not only a "high quality" artefact.
2. **The rate is the coefficient field: 98.1 % of the keyframe.** 26,149
   nonzeros at 3.99 bits each, 22,542 zero positions inside the EOBs (13.5 % of
   the tile), one literal sign bit per nonzero (20.1 %) and 10.5 % escape bits.
   The arithmetic coder is 1.5 % above its own per-context entropy bound and
   the contexts are worth 26 % over pooling, so there is nothing left to win in
   the entropy stage. Every coefficient not coded is worth ~4 bits (1 of them
   the sign); intra luma has no RDOQ to make that decision, and the measured
   dead-zone substitute is RD-neutral — the trellis is the untested lever here.
3. **A residual ~1.5× gap remains unexplained by any lever measured here**
   (vs SVT-AV1 preset 0 and vs the reported SVT-AV2 point). It is not: the
   entropy coder (1.5 % over bound), the transform sizes (8x8 is modal, 4x4
   legal but never chosen), the coefficient rate model (`--video-tx-context-rate`
   neutral/worse), the dead-zone (RD-neutral), the partition probe (inert on
   keyframes), or a broken predictor (the search beats DC variance by 22 % and
   a trivial copy by 10 %). The remaining suspects are the intra quantiser
   decision (trellis) and the intra mode/TX-type RD models, which no
   measurement in this task isolates.

## 7. Instrumentation

All new output is behind `AV2ENC_SYMBOL_COST=1` and never touches the
bitstream (verified: QP120 stream MD5 identical with the meter on and off, and
`tools/gate_video_nightly.sh` passes).

* `coeff_syntax <element> n= bits= ctxH= pooledH= inst=` — per syntax element,
  with the per-CDF-instance ideal static entropy (`ctxH`) and the pooled
  entropy (`pooledH`).
* `coeff_total sym_bits= literal_bits= all_metered= tile_payload_bits=` — the
  reconciliation against the payload.
* `coeff_tile plane= tb= tb_nz= tb_skipped= eob_sum= nz= abs_sum= zero_pos=
  zero_bits= eob_hist= lvl_hist=` — the level/EOB statistics per plane.
* `coeff_band plane= … bN=bits/positions/nonzeros/Σ|level|` — the eight
  position bands (band 0 = DC, band 7 = far corner); bits include
  per-position base/br/sign/escape.
* `tx_size_hist`, `block_size_hist` — the §5.18.8 transform and §5.20.3.1
  coding-block histograms.
* `AV2ENC_Q_DEADZONE=<0..0.5>` — diagnosis-only quantiser rounding offset,
  applied identically in `txv::quantize()` and the TX-search quantiser so the
  search prices what the writer emits. Unset = pinned `lround`.
* `AV2ENC_VIDEO_TRACE=1` (`--video-trace`) now prints `psse=` on the
  `m2 intra` leaf line: the SSE of the *chosen prediction* before the
  residual, next to the existing reconstruction `sse=`.
