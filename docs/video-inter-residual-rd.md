# Inter residual RD: where the P-chain loss actually is

Diagnostic for the f32-f59 trough that keeps the nightly stream ~0.12 JOD
behind SVT-AV1. It rules out three candidates and lands on the leaf residual
price.

## Method

`--video-trace` now prints one line per inter leaf:

```text
ileaf WxH at X,Y mv=dy,dx ref=R pred=P res=S rbits=B parts=N eob=E \
      zero=Z want=W [part/eob]...
```

`pred` is the prediction SSE, `res` the SSE after residual coding, `rbits` the
rate used by the skip decision, `parts` the chosen transform-partition parts.
Frame N is isolated by diffing the trace of `--frames N` against `--frames N+1`.

## f40, QP110 (the nightly pin)

51 inter leaves, 9 code a residual, 42 skip. The skipping leaves hold **71.7%
of the frame's prediction SSE** (479,051 of 668,240). 13 leaves select a
transform partition (2+ parts), so inter TX partitioning is already in use, and
42 of 51 leaves keep a zero MV (the clip has no global motion).

The worst leaf, 64x32 at (320,96):

| leaf | pred SSE | coded SSE | rate | verdict |
|---|---:|---:|---:|---|
| 64x32 @ 320,96 | 45,169 | 34,895 | 240 bits | skip |
| 64x32 @ 0,160 | 42,441 | 18,861 | 153 bits | code |

## f40, QP80: the smoking gun

At QP80 the frame codes **0 of 58 leaves**, even though coding would collapse
the error:

| leaf | pred SSE | coded SSE | rate | λ(80) | decided |
|---|---:|---:|---:|---:|---|
| 64x32 @ 384,192 | 38,319 | 6,330 | 548 | 88.4 | skip |
| 64x32 @ 0,160 | 33,281 | 6,048 | 487 | 88.4 | skip |
| 64x32 @ 320,96 | 32,584 | 14,790 | 1,765 | 88.4 | skip |

The skip rule is `coded SSE + λ·bits < prediction SSE + 2λ`, so 6,330 +
88.4·548 = 54,817 loses to 38,496. The residual is not judged too expensive by
the entropy coder - it is priced out by the **leaf λ**, which is anchored at
`c = 125` at QP104 and scaled by `sqrt(step)`.

## `--video-leaf-rd-lambda`

The anchor is now a knob (default 125, byte-identical when unset). At QP110,
key -25, all other nightly flags fixed:

| c | bytes | kbps | CVVDP | PSNR-Y |
|---:|---:|---:|---:|---:|
| 125 (default) | 24,904 | 99.616 | 9.7069 | 43.8667 |
| 100 | 25,293 | 101.172 | 9.7159 | 43.9461 |
| 80 | 25,675 | 102.700 | 9.7283 | 44.0372 |
| 60 | 26,111 | 104.444 | 9.7192 | 44.0466 |

Matched rate (c=80 with `--video-key-qp-offset -22`, 24,880 B / 99.520 kbps):
**9.7258 JOD / 43.7442 dB** versus the pin's 9.7069 / 43.8667. That is
**+0.019 JOD for -0.13 dB PSNR**.

## What this means

The result repeats the pattern from the refresh-budget experiment (+0.025 JOD
at -0.12 dB): **CVVDP prefers trading average PSNR for the worst frames**, and
every MSE-calibrated decision in the encoder (leaf λ, skip, key offset, refresh
budget) under-serves them. Each nudge in that direction is worth ~+0.02 JOD and
none of them is individually outside the ±0.03 ripple.

So the next algorithmic lever is not another tool, it is a **CVVDP-shaped
allocator**: per-frame/per-region bit allocation driven by the pooled
worst-case error instead of MSE. The three independent +0.02 results
(refresh budget, refresh QP depth, leaf λ) are three partial samples of it.

The λ knob stays opt-in at 125: the single-point gain is inside the ripple and
the PSNR cost is real, so it is calibration evidence, not a shipped default.
