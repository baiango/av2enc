# Tool-improvement queue 3: the key/intra gap

Status: **measured ledger of the third loop.** Branch `exp/key-gap`, base
`832e1a8b` (`main`). The loop starts from `docs/tool-quality.md` §4.2 and
`docs/tool-improvement-loop.md`, so it targets the key/intra deficit and
nothing else. No encoder source was changed in the final tree: every patch
below is one contained, default-off env-gated experiment that was measured and
then reverted (this loop kept no code).

K is `tools/kpi.py` interpolated linearly in rate to 100.000 kbps across the
candidate's own bracket, scored by `tools/oracle.py` with the AVM decoder.
Natural content is `testdata/nasa_480x240_30fps_60f.y4m` (60 frames, 30 fps,
8-bit 4:2:0). Probe content is `build/tool-probes/` from
`tools/tool_probes.py`. The baseline ladder (nightly QP104/106/108) reproduces
`docs/tool-improvement-queue-2.md`: 26,158 / 25,008 / 24,291 B, K 80.734 /
82.155 / 81.604, matched **82.149**.

## 0. TL;DR

1. **The gap is not prediction.** On the `detail_edges` key the intra
   prediction removes almost nothing: the chosen-mode residual is 1,399
   LSB²/sample (16.7 dB "prediction PSNR") versus a source variance of 1,004,
   and the best predictor that is even approximately signalable (a per-16×16
   least-squares 4-tap fit, which AV2 cannot code) reaches only
   518 LSB²/sample. DC is 851, one-tap copy 1,749/1,881, gradient 3,577. There
   is at most ~2-3 dB of prediction gain on this content and most of it is
   outside the normative mode set.
2. **The gap is a uniform effective-quantiser gap.** Our key error energy is
   6.7-7.2× AVM's in *every* 8×8 DCT ring (DC 10.3×), and 94% of it sits in
   the dense-texture region. A prediction-shape defect would be band-selective;
   a flat ratio across bands means the same picture is being coded at ~2.6×
   coarser step for the same bytes.
3. **The coefficient field, not the entropy coder, is the difference.** With
   instrumentation on the real stream, the key's luma field at QP112 is 33,710
   nonzeros over 107,520 positions with a per-position order-0 entropy of
   17.5 kB; the emitted tile is 22.9 kB (1.31×). At matched quality (key
   QP68, 47.8 dB) the field is 48,873 nonzeros / 325,517 Σ|level|, order-0
   30.1 kB, emitted 44.7 kB. AVM's *whole* key is 23.5 kB at that quality, so
   at equal distortion our quantised field alone carries ~1.3× AVM's entire
   budget.
4. **A real pricing defect was found and it is not enough.** The intra TX
   search prices every nonzero coefficient at **1.0 bit**
   (`txtsearch::Options::coefficient_bits`, never overridden; the key does not
   run `--video-tx-context-rate`). The key's `rate_bits` therefore totals
   4,406 B for a field that really costs 22.9 kB — a **5.2×** underprice, so
   the key's TX size/type/partition search is effectively distortion-only.
   Making it honest (exact §8.3 model, or a scalar calibration) improves the
   probe's key RD by only +0.16..+0.35 dB at iso-rate, **inverts sign on
   natural content**, and costs Butteraugli: at equal key PSNR the learned
   trade is BA +0.66 on the natural clip. The trellis loops failed for the
   same reason; the pricing is not the 8 dB.
5. **The K gate cannot see a key-only quality win at fixed rate.** The audit's
   key-replacement oracle is +0.00 K. A key patch can only pay through (a)
   rate saved at *equal perceptual* quality, or (b) a cheaper key that is a
   better reference. Every mechanism measured in this loop produced (a) with a
   Butteraugli cost or a P-chain regression, so nothing passed the
   sign-consistency rule. The best probe-side key saving (CfL, −4.75% at
   strictly better chroma) is −0.50 K on natural.

## 1. Diagnosis in detail

### 1.1 Setup

`detail_edges` key-only (`--frames 1`, nightly, key QP = base − 27). Our RD
curve and AVM's, same source, same decoder, same PSNR tool:

| key QP | av2enc bytes / PSNR-Y | | AVM QP | avmenc bytes / PSNR-Y |
|---:|---|---|---:|---|
| 10 | 84,483 / 57.00 | | 60 | 29,983 / 51.02 |
| 30 | 69,141 / 54.40 | | 78 | 23,722 / 47.35 |
| 60 | 49,753 / 49.79 | | 90 | 20,358 / 44.97 |
| 80 | 37,977 / 45.72 | | 120 | 13,062 / 38.68 |
| 90 | 32,576 / 43.59 | | 150 | 7,094 / 32.03 |
| 100 | 27,954 / 41.41 | | 180 | 2,795 / 25.58 |
| 112 | 23,039 / 38.88 | | | |

At equal bytes ours is 1.6-1.9× the rate of AVM across 39-57 dB. It is a
constant factor, not a floor (at QP10 we still reach 57.0 dB). The published
probe point reproduces: 23,039 B / 38.881 dB (key tile 22,933 B) versus AVM
23,722 B / 47.349 dB (IVF) at the same bytes.

### 1.2 Prediction is measured, not inferred

Per-leaf instrumentation (source-driven prediction SSE, then coded SSE, both
relative to the same block) on the key at QP112, 335 leaves:

| region | leaves | pred SSE/sample | coded SSE/sample | coded PSNR |
|---|---:|---:|---:|---:|
| left dense texture | 248 | 952 | 13.2 | 36.8 dB |
| right structured half | 87 | 2,041 | 2.4 | 47.2 dB |
| whole frame | 335 | 1,399 | 8.74 | 38.7 dB |

The source has variance 1,004 on the left texture, so prediction leaves 95% of
it. Externally measured simple predictors on the same region (residual energy
per sample): DC 851, copy-left 1,749, copy-above 1,881, gradient 3,577,
per-16×16 least-squares 4-tap 518. Our chosen mode (952) is no better than the
plain DC estimate (851) on this region, and the *unavailable* LS optimum is
2.3 dB below that. This is why the
mode-search ablation (DC-only key) is worth only 353 B / 0.024 dB: the texture
has ρ(0,1)=0.50, ρ(1,0)=0.27, and no directional or two-tap predictor can do
meaningfully better. Prediction was eliminated as the primary lever by
measurement, not assumption.

### 1.3 The error is uniform across bands

8×8 DCT ring error energy at the matched point (ours QP139 vs AVM QP78,
23.0 vs 23.7 kB):

| ring | source var | ours err | AVM err | ratio |
|---|---:|---:|---:|---:|
| DC | 112,080 | 15.91 | 1.54 | 10.3× |
| 1-2 | 3,080 | 11.14 | 1.55 | 7.2× |
| 3-5 | 871 | 10.18 | 1.43 | 7.1× |
| 6+ | 200 | 6.09 | 0.91 | 6.7× |

A flat ratio means "same picture, coarser step", not "wrong spectrum" and not
"wrong prediction". Region split: ours 36.79 dB left / 47.21 dB right, AVM
45.00 / 65.86; 94% of our error energy is in the left region (99% for AVM).

### 1.4 The coefficient field

Real-stream instrumentation of the key's luma TX parts (levels in scan order,
`eob`, per-leaf prediction/coded SSE):

| key QP | parts | nonzeros / positions | Σ\|level\| | mean EOB | positions coded | order-0 entropy | emitted tile |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 112 | 556 | 33,710 / 107,520 (31.4%) | 90,722 | 126 | 70,598 | 17.5 kB | 22.9 kB |
| 68 (47.8 dB) | 513 | 48,873 / 104,448 | 325,517 | 150 | 77,483 | 30.1 kB | 44.7 kB |

("order-0 entropy" = Σ over TX positions of the empirical entropy of the
signed quantised level at that position; it is a *lower* bound for the
spec's coarser context classes, but both encoders pay that coarsening, so it
compares like with like.)

Two facts follow. First, our writer is 1.3-1.5× above its own field's
order-0 bound — normal for class-based contexts. Second, **at matched
distortion our field's entropy alone (30.1 kB) exceeds AVM's total key
(23.5 kB)**. The difference is in the field: at the same step AVM produces
fewer, smaller levels. That is a tool-set difference (transform choice,
dependent quantisation, explicit level selection), not one bug.

### 1.5 The one concrete defect: a 5.2× coefficient-rate underprice

`src/pipeline/video/video_tx_search.cpp` prices a candidate leaf as

```
rate = type_bits + coefficient_bits * nonzero      (coefficient_bits = 1.0)
```

unless `use_context_rate` (`--video-tx-context-rate`) or the inter-only RDOQ
model is active. `--fast-decode` does not enable either on the key, and no caller
ever changes `coefficient_bits`. Measured consequence: the key's summed
`rate_bits` is **4,406 B** while the same field costs **22,933 B** in the
stream — 1.05 vs 5.4 bits per nonzero, a 5.2× underprice. The TX
size/type/partition search on the key is therefore nearly distortion-only
(the rate term is ~13% of the objective), which is consistent with the dense
±1 field (16,002 of 33,710 nonzeros are ±1) and with the EOB sitting at half
the block.

Making the price honest does *not* recover the gap: exact pricing improves the
probe key by +0.26..+0.38 dB at iso-rate and costs 1.7% of natural key bytes at
equal PSNR, and the natural ladder loses 1.05 K because the error re-shaping
costs Butteraugli (BA 3.42 → 4.08 at QP106, equal PSNR). This is the same
class of result as the two prior trellis attempts, but for a different reason:
the level *field* is the limit, not the price of a level.

### 1.6 Chroma is not the gap

AVM encodes this key monochrome at 22,483 B / 47.348 dB and 4:2:0 at 23,678 B /
47.349 dB (per-frame sizes from `avmenc`), so AVM's chroma costs 1,195 B (5.0%)
with no measurable luma loss.
Our key's chroma costs about the same: CfL removes 1,094 B (4.75%) from the
23,039 B key at *identical* luma PSNR and +0.74/+0.68 dB U/V. Chroma is a
probe-visible 4.7% and a natural-content 0.4%.

## 2. Ledger

| # | target | attempt | probe (detail_edges key unless noted) | natural 240p K @100 | ΔK | verdict |
|---:|---|---|---|---:|---:|---|
| 1 | intra prediction | key CfL chroma (`--video-cfl`) | 21,945 B / 38.881 dB (−4.75% B, U+0.74, V+0.68 dB) | 81.652 (82.149) | −0.50 | negative |
| 2 | residual/entropy | exact §8.3 candidate rate on the key (`--video-tx-context-rate`, on CfL) | 22,418 B / 38.795 dB (−2.7%, −0.086 dB) | 81.099 | −1.05 | negative |
| 3 | intra prediction | CCTX joint U/V (`--video-cctx`) | — | 81.350 | −0.80 | negative |
| 4 | transform | IST (`--video-ist`) | 23,048 B / 38.881 (inert, +9 B) | 82.140 | −0.01 | negative |
| 5 | intra prediction | CfL + CCTX | 22,046 B / 38.881 dB | 80.455 | −1.69 | negative |
| 6 | transform | IDTX 2-D intra (`--video-idtx`, probe on CfL+ctx) | 21,245 B / 38.803 dB | 81.930 | −0.22 | negative (signed +0.85/−0.20/+0.26 at QP104/106/108) |
| 7 | intra prediction | DIP (`--video-dip`) | 23,083 B / 38.831 dB | — | — | negative (probe only) |
| 8 | intra prediction | MRL (`--video-mrl`) | 23,875 B / 38.759 dB | — | — | negative (probe only) |
| 9 | transform/quant | TCQ, RDOQ off (`--video-tcq`) | 23,270 B / 38.735 dB | — | — | negative (probe only) |
| 10 | transform | coarser key tree/TX (depth 3) | 23,964 B / 36.831 dB (matched bytes −2.0 dB) | — | — | negative (probe only) |
| 11 | transform | depth 2 / depth 1 key tree | 21,128 B / 30.75 dB; 22,190 B / 31.29 dB | — | — | negative (probe only) |
| 12 | key RD | key TX λ 4/16/32 | 23,460/22,780/22,492 B at 38.911/38.843/38.752 dB | — | — | negative (flat) |
| 13 | intra prediction | DC-only key (mode-search ablation) | 21,670 B / 38.819 dB vs 21,317 / 38.795 (with CfL+ctx) | — | — | no bug: search is worth ~+0.15 dB iso-rate |
| 14 | residual/entropy | `AV2ENC_KEY_COEFF_BITS` k=3 (implemented) | 22,711 B / 38.804 dB (+0.16 dB iso-rate) | key-only: +1.9% B at equal PSNR | — | negative (sign flips vs probe) |
| 15 | residual/entropy | `AV2ENC_KEY_COEFF_BITS` k=4 (implemented) | 22,579 B / 38.796 dB | key-only: +1.7% B at equal PSNR | — | negative |
| 16 | key RD/allocation | `AV2ENC_KEY_CSF` low-band distortion weight 0.5 (implemented) | 23,068 B / 38.900 dB | 81.726, per-QP −0.11/+0.35, BA spike at QP104 | −0.42 | negative (inconsistent) |
| 17 | key RD/allocation | `AV2ENC_KEY_CSF` 1 / 2 / 4 | 23,061/23,141/23,121 B at 38.877/38.861/38.850 dB | — | — | negative (probe flat) |
| 18 | residual/entropy | exact rate + key CSF 1 (composition) | — | 81.579 (BA 3.80 vs 3.42 at QP106) | −0.57 | negative |
| 19 | allocation | chroma cost split (AVM mono vs 4:2:0) | AVM chroma 1,195 B (5.0%); ours ≈1,094 B | — | — | diagnostic: not the gap |
| 20 | allocation | key RD curve vs AVM, 7 QPs + 5 AVM QPs | constant 1.6-1.9× rate at equal PSNR | — | — | diagnostic: the blocker |

Non-negotiables held: 4:2:0 only; gate off reproduces the 25,008 B /
`0639b43630abb454f7be5ac494d2efb4` nightly QP106 pin; `tools/test.sh --build
build-release` 10/10; warning-free `Release` build. No preset, CLI default,
pin, syntax or decoder path changed. AVM/SVT were used as black-box binaries
only.

## 3. What was implemented and reverted

Two contained, default-off, env-only patches were written, measured and
reverted (the tree is clean at HEAD):

* `AV2ENC_KEY_COEFF_BITS=<bits>` — replaces the intra TX search's one-bit
  coefficient price with a measured per-nonzero cost, for keyframe luma only.
  `0`/unset is byte-identical.
* `AV2ENC_KEY_CSF=<gain>` — adds a low-frequency error weight
  (`txtsearch::Options::lowpass_gain`) to the keyframe intra TX candidate cost
  so the TX choice keeps the band Butteraugli sees. `0`/unset is byte-identical.

Both move the probe in the intended direction and both fail the natural gate.
A third, diagnosis-only instrument (`AV2ENC_KEY_STATS` / `AV2ENC_KEY_LEVELS`:
per-key-leaf prediction/coded SSE, TX shape and quantised levels) produced the
§1.2/§1.4 tables and was also reverted.

## 4. The remaining blocker

* **There is no single key mechanism left with 2× headroom.** The measured
  decomposition is: ~0 prediction headroom inside the syntax, ~0.2-0.3 dB from
  honest coefficient pricing, ~20-25% from the largest transforms (which the
  key tree already rejects at equal rate), and the rest is the accumulated
  intra toolset (dependent quantisation, secondary transform selection,
  explicit level selection, partition RD) that AVM applies and we do not.
* **A key patch pays K only if it is rate-cheaper at equal *perceptual*
  quality.** Every rate saving measured here (CfL, exact pricing, coefficient
  calibration) moved Butteraugli the wrong way at equal PSNR (BA +0.04..+0.66)
  and the P chain amplified it. The probe's 8.4 dB is real but it is a
  PSNR/bytes statement; the 100 kbps K gate is a different currency.
* **The next experiment should be a level-field one that is not a trellis and
  not a dead zone:** a key-only dependent-quantiser state machine (two
  interleaved reconstruction levels) with the level state priced by the
  existing §8.3 model, gated on BA-neutral-at-equal-PSNR on the probe *and*
  natural key. `--video-tcq` is the normative version of this and the current
  implementation loses 0.15 dB at fixed QP when RDOQ is off; the missing piece
  is the level *decision*, which is exactly what this loop could not reach
  without repeating the trellis.
