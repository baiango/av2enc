# 720p still: where the detail deficit actually is (2026-11-24)

Measured on `~/Downloads/src720.png` (1280x720 waterfall), one frame, to
settle the question `docs/luma-audit.md` §6a left open: AVM keeps a different
coefficient set at the same rate — *which* set, and is it reachable?

**Answer: it is not a coefficient-set difference, and it is not a high-band
(mosquito-zone) problem. Our error is uniformly ~2x across every frequency
band, the deficit shrinks monotonically as rate rises, and at QP90 we overtake
AVM.** The gap is low-rate coding efficiency, not a missing tool.

## Setup (all recipes reproduce the documented pins)

| arm | command | bytes | luma PSNR |
|---|---|---:|---:|
| ours | `av2enc src720.y4m --video --frames 1 --qp 145 --cpu-threads 8 --nightly-b` | 127,324 | 31.3875 |
| AVM | `avmenc --end-usage=q --cpu-used=6 --threads=8 --limit=1 --qp=127 -w 1280 -h 720 --i420` | 130,529 | 34.0856 |
| SVT-AV1 p0 | `SvtAv1EncApp --preset 0 --crf 35` | 130,718 | 33.749 |

`--qp 145` and AVM `--qp 127` reproduce the §6a table exactly (127,324 and
124,467 for the y4m-fed AVM run). `dav2d --strict 1` is the decoder for our
streams; it matches `--video-recon` bit-for-bit.

### Harness traps (both produced plausible wrong numbers — do not repeat)

1. **`avmenc` fed a y4m silently encodes the wrong frame.** It exits 0 and the
   output decodes, but block MSE against the source is ~1134 (PSNR 19 dB).
   Feed raw I420 plus `-w/-h`; the same source then gives 34.09 dB.
2. **`avmdec -o` writes raw I420, not y4m.** A scorer that skips a y4m header
   unconditionally reads 12 bytes out of alignment and reports ~17.5 dB (a
   14 dB phantom error).

## The gap by source activity

Per-8x8 luma, bucketed by source variance (`blockdiff.py`):

| source activity | blocks | ours | AVM | ratio |
|---|---:|---:|---:|---:|
| flat (var < 50) | 2,898 | 6.75 | 5.91 | **1.14x** |
| mid (var < 500) | 5,378 | 43.16 | 25.61 | 1.69x |
| texture (var >= 500) | 6,124 | 69.99 | 34.40 | **2.03x** |

In PSNR: flat **39.84 vs 40.42** (+0.58 dB, near parity), mid 31.78 vs 34.05,
texture **29.68 vs 32.77** (-3.09 dB).

Only **322 blocks (2.2 %)** are more than 3x worse than AVM, and they carry
**5.7 %** of the total excess MSE. There is no outlier population to patch:
the whole textured distribution is ~2x off.

## The error spectrum is unchanged

8x8 DCT-II error energy on textured blocks (`spectrum.py`):

| arm | DC | low(1) | low(2) | mid | high |
|---|---:|---:|---:|---:|---:|
| ours | 5.0 % | 4.2 % | 5.4 % | 39.7 % | 45.7 % |
| AVM | 4.7 % | 4.7 % | 6.2 % | 37.3 % | 47.2 % |
| SVT-AV1 | 4.3 % | 4.8 % | 6.2 % | 38.3 % | 46.5 % |

All three put essentially the same fraction of error in each band. Our error is
*larger* everywhere by a constant factor, not concentrated in the high band.
This directly contradicts the §6a reading that the loss is "in 2-4 px detail
structure": our shape matches the references, it is their magnitude we miss.

## Rate curve — the deficit shrinks and reverses

Textured-block PSNR vs bytes (`eqrate.py`):

| texture PSNR | ours | AVM | our rate overhead |
|---:|---:|---:|---:|
| 33.0 dB | 176,382 | 134,420 | **+31.2 %** |
| 34.0 dB | 191,599 | 150,977 | +26.9 % |
| 36.0 dB | 225,148 | 185,718 | +21.2 % |
| 38.0 dB | 259,765 | 224,890 | +15.5 % |
| 40.0 dB | 296,016 | 266,225 | +11.2 % |
| 41.5 dB | 323,727 | 297,226 | **+8.9 %** |

At **QP90 we overtake AVM on every band**: all 42.82 vs 42.43 dB, texture
42.09 vs 41.69, flat 45.72 vs 45.28 (334,590 vs 301,153 B).

## Verdict

- It is **not** a coefficient-set or mosquito-zone problem. The error spectrum
  is the same shape as both reference codecs.
- It is **not** fixable by spending rate: our overhead per dB of texture is
  ~2x AVM's, so more rate buys less.
- It **is** a low-rate efficiency gap that converges at high rate. That is the
  signature of a header/context/prediction-side inefficiency that matters when
  the budget is small, not a missing transform or quantiser mode.
- SVT-AV1 reaches 32.45 dB texture at the matched 130 kB, only 0.31 dB behind
  AVM, so AVM is not an outlier. Two independent codecs land where we do not.

## What to measure next

The remaining unexplained fact is *why* our textured residual is uniformly ~2x
more expensive. Two candidates the data points at, neither yet tested:

1. **Per-block rate, not error.** Everything above measures distortion at fixed
   rate. The complement — bits per textured block at fixed error — would show
   whether we spend more bits for the same block (prediction/context) or the
   same bits for less (quantiser). This is the one measurement that separates
   the two candidates.
2. **Intra prediction on textured 4x4/8x4 leaves.** 4x4 is the most-used
   emitted size on this still (28,586 parts) and rect parts are 24 % of all
   parts (`key-tx-rd-shape: square=29790 rect=9352`). If those leaves are
   mostly DC-predicted, the residual is larger than it needs to be.

Do not re-try: TX/partition shape widening (`--video-max-depth` 5/6,
`--video-min-block 4` are byte-inert; `--video-ext-partitions` is rejected with
the B schedule), TX lambda, keyframe RDOQ, or AC-spend knobs — all measured
inert in §6a and re-confirmed here.

## Per-block rate at fixed error (2026-11-24)

The measurement the previous section asked for. `AV2ENC_LEAFBITS=<path>` dumps
the exact bits each coded leaf cost, read from the coder's cumulative
`prefix_bits` across `write_leaf()` — emitted bits, not a model estimate. The
stream is byte-identical with the dump on. Per-leaf bits are spread over the
8x8 blocks the leaf covers and bucketed by source variance, the same buckets
as the distortion tables above.

**Our marginal cost per dB is cheapest on texture:** QP145 -> QP132 buys
+2.76 dB on texture, +2.24 on mid and only +1.34 on flat, for the same +40,475
bytes — 14,686 B/dB on texture against 30,183 B/dB on flat. We are not
over-spending on detail; we get the most per bit exactly where the gap is.

Allocation is also roughly proportionate to where the error sits:

| band | share of our bits | share of our MSE | our vs AVM error |
|---|---:|---:|---:|
| flat | 5.2 % | 2.8 % | 1.13x |
| mid | 32.0 % | 33.5 % | 1.68x |
| texture | **62.8 %** | **63.7 %** | **2.02x** |

**Conclusion: the bits are in the right place.** The deficit is not
misallocation and not over-spending — the same bytes produce ~2x the error on
texture. That rules out both candidates in the previous section (prediction
bloat *and* quantiser inefficiency as allocator problems) and points at the
cost of coding a textured residual itself.

Practical consequence: allocator-shaped fixes (rate shaping, band weighting,
AQ, per-band lambdas) cannot close this, which is consistent with every one of
them measuring inert in `docs/negative-results.md`. What remains is how
expensively we represent textured coefficients — the intra mode / prediction
side of those 4x4 and 8x4 leaves (28,586 + 7,367 of 39,142 parts) is the
untested variable.

### Per-band spend, exact

| QP | leaf total | flat B/blk | mid B/blk | texture B/blk |
|---:|---:|---:|---:|---:|
| 145 | 123,190 | 1.24 | 3.75 | 6.29 |
| 135 | 153,415 | 1.54 | 4.97 | 7.86 |
| 132 | 163,726 | 1.59 | 5.41 | 8.37 |

(Leaf totals are below the IVF size by ~4 kB of headers and CDF image.)
