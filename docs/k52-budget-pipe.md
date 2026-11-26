# K 52 at a 25 KB budget on `src720.png`: the budget-closed pipeline (negative)

Status: **target NOT met, and the new pipeline does not beat `--nightly-b`.**
Best measured point on this picture at the budget is still the pre-existing
`--nightly-b` recipe at **K 47.37 / 24,240 B**. Every arm of the new
budget-closed pipeline measured **below** that, including the arms whose
objective was a 1-px band proxy rather than the driver's own J.

This document records the pipeline, the two defects found and fixed while
building it, and every arm with its disposition, so the branch is a measured
negative rather than an open question. It extends `docs/k52-at-25k.md` (the
knob-and-allocator search) with a *mechanism* change: a per-superblock
quantiser allocation closed on one frame byte budget.

## 0. The result

| arm | bytes | K | vs control |
|---|---:|---:|---:|
| **control + `AV2ENC_TX_HF_MASK_DIV=2`** | **24,121** | **47.41** | **+0.04 (first arm at or above the control)** |
| `--nightly-b` control (QP187, MIN=16, gdf, uv-23) | 24,240 | 47.37 | — |
| pipeline, flat q187 (harness identity check) | 24,240 | 47.37 | **0.00 (tie)** |
| pipeline, driver-J objective, scale 1000 | 24,208 | 46.67 | −0.70 |
| pipeline, driver-J objective, scale 950 | 24,229 | 46.65 | −0.72 |
| pipeline, driver-J objective, credit rule | 24,190 | 46.91 | −0.46 |
| pipeline, 1-px band proxy `HF=0` (scale 1000) | 24,183 | 46.91 | −0.46 |
| pipeline, 1-px band proxy `HF=0.5` | 24,175 | 46.66 | −0.71 |
| pipeline, 1-px band proxy `HF=1` | 24,177 | 46.62 | −0.75 |
| pipeline, 1-px band proxy `HF=2` | 24,322 | 46.63 | −0.74 |
| pipeline, activity-scaled allowance `S=0.5` | 24,701 | 45.98 | −1.39 |
| pipeline, activity-scaled allowance `S=1.0` | 24,764 | 46.33 | −1.04 |
| pipeline, source-activity policy (first attempt) | 23,829 | 42.02 | −5.35 |
| objective: K 52 | 25,000 | 52.00 | **+4.63** |

The flat-q187 row is the important control: it proves the harness is
byte-identical to the existing encoder, so the losses below are the allocation's
and not the plumbing's.

## 1. The pipeline

`AV2ENC_PIPE_BUDGET` (env gate, default off) in
`src/pipeline/video/video_pipe_budget.{h,cpp}` plus
`InterFrameEncoder::pipe_budget_decide_frame()` in
`video_frame_decision.cpp`. It has its own frame loop, its own leaf loop and its
own decision order:

```
for each superblock, raster order:            # own frame loop
  for q in the qindex grid:                   # own leaf loop
    decide the superblock's tree at q (seq_driver_decide_sb)
    price it (exact bytes: one-superblock key encode through the coder)
    keep (q, tree, bytes, J)
  choose q against the running frame budget   # own decision order
  commit the chosen tree at once (collect_node) with that qindex
```

The per-superblock qindex is carried by the §5.18.7.8 delta-Q ramp, which the
lone key picture can signal. `--video-delta-q` composes with `--nightly-b` only
under `--video-cdf-per-frame-qp` (the carried CDF image is not part of the
§8.2.4 per-tile delta/segment state); on a lone key the two compose exactly —
the reset stream is byte-identical to the carry stream (24,240 B).

## 2. Two defects found and fixed

Both were in the pipeline, and both cost ~5 K before they were found:

1. **The dispatch seam silently disabled the key driver.** `sequential_key` is
   gated on `!qp_tools_active()`, so arming delta-Q without the pipeline dropped
   the sequential key driver and ran the historical variance solver. Measured:
   driver + uniform delta-Q at QP185 gave 19,335 B / 23.40 dB against the
   driver without delta-Q at 26,180 B / 24.62 dB. The seam now admits
   `pipe_budget::enabled()` explicitly.
2. **The walk measured trees and committed the reference.** `seq_driver_decide_sb`
   computes a tree per qindex; the first version discarded it and committed the
   unmodified reference, so the sweep's decisions never reached the stream. It
   now keeps them and commits the selected one.

After both fixes the flat arm reproduces the control byte-for-byte (24,240 B,
24.4012 dB mean PSNR-Y) — the harness is sound.

## 3. The allocation objective, and why the band proxy did not rescue it

The driver's J is `D + lambda*R`, with `D` the eff-perc low-band distortion plus
optional high-pass and block-edge terms. At matched bytes the J-ranked
allocation picks the **finest** qindex (199) for many superblocks and measures
worse than uniform — so J is not a valid exchange rate for K on this content.

The prescribed replacement was the 1-px band proxy that `docs/band-d.md` shows
tracks local SSIMULACRA2 at r = −0.82 (within-frame centred). That term already
exists: `seq_rd::leaf_distortion_gains()`'s `hf` argument is exactly
`sum (d - box3x3(d))^2`, the 1-px band, reachable from the driver as
`AV2ENC_RD_DRIVER_HF` (default 0).

It changes the decisions in the predicted direction — at QP187 (a single uniform
driver pass, `AV2ENC_RD_SEQTREE_STATS=1`), HF=5 collapses 45 leaves and moves
472 transforms from tx8 to tx16:

```
HF=0: leaves=2311 plans[none=369 split=900 horz=573 vert=469] tx8=1860 tx16=874
HF=5: leaves=2266 plans[none=476 split=725 horz=568 vert=497] tx8=1388 tx16=1106
```

but at matched bytes it still loses (HF=1: K 46.62 at 24,177 B). The band term
is a better *local* proxy and a better tree signal, and it still does not
produce a gain under a byte budget on this picture.

## 4. Why this is consistent with the existing record

`docs/k52-at-25k.md` §3.4 explains the gap as "a uniformly ~15 % larger luma
rmse ... with the reference's error shape", and concludes that a uniformly
scaled error cannot be repaired by moving bits between regions. **This document
re-measured that premise and it does not hold at block level.**

Decomposing our QP187 stream against the AVM `e6f99d3` QP171 stream at equal
bytes (24,240 vs 24,204 B), over 3,600 16x16 blocks:

| measurement | value |
|---|---|
| frame luma rmse, ours / AVM | 15.363 / 13.314 |
| extra error energy | 33.1 % of AVM's |
| share of the extra held by the worst 10 % of blocks | 30.5 % |
| share of the extra held by the worst 25 % of blocks | **60.5 %** |
| share of the extra held by the worst 50 % of blocks | 89.9 % |
| per-block ratio bad/good, p05 / median / p95 | 0.81 / 1.31 / 1.94 |

So the extra error is **not** a uniform multiplier: it is concentrated (60 % in
a quarter of the blocks), and on 6 % of blocks we are *better* than the
reference encoder.

**The contrast hypothesis is also refuted.** A gain fit `recon = a*src + b`
gives `a = 0.9123` (ours) and `a = 0.9337` (AVM) — we do lose more contrast —
but removing the fitted gain makes the error *worse*, not better:
rmse 15.363 -> 16.124, i.e. the gain term owns **−10.1 %** of the error energy
(AVM: −7.4 %). The brightness-dependent crush is real but it is not the error;
it is smaller than the reconstruction's other deviations from the source.

**Where the extra error actually is.** Ranking blocks by extra energy and
comparing source features:

| block group | share of extra | src 1-px activity | our 1-px band err / AVM's |
|---|---:|---:|---:|
| worst 25 % of blocks | **60.5 %** | **320.6** | — |
| the other 75 % | 39.5 % | 97.5 | — |
| high-activity (above the mean) | **76.1 %** | — | **1.76x** |
| low-activity | 23.9 % | — | 1.47x |

The extra error lives in textured blocks, and there our finest-band error is
1.76x the reference encoder's against 1.47x in flat ones: the 1-px content is
over-quantised exactly where the source carries detail. This contradicts §3.6d
of `docs/k52-at-25k.md` ("we are at the optimum rather than over-retaining" on
the 1-px band) as applied to *relative* error against the reference encoder, and
it is the first measurement in this programme that names a targetable block
class rather than a global knob.

That is what `AV2ENC_PIPE_BUDGET_STRENGTH` acts on (§7).

## 5. Reproduction

```sh
REF=~/Downloads/av2enc-src720/src720_1f.y4m
# control
env AV2ENC_RD_DRIVER_MIN=16 AV2ENC_RD_DRIVER_LAMBDA=256 \
  build-release/av2enc "$REF" -o ctl.ivf --video --frames 1 --qp 187 \
  --cpu-threads 8 --nightly-b --video-gdf --video-uv-qp-offset -23
# pipeline, 1-px band proxy objective
env AV2ENC_PIPE_BUDGET=1 AV2ENC_PIPE_BUDGET_SCALE=1000 \
  AV2ENC_PIPE_BUDGET_Q_LO=175 AV2ENC_PIPE_BUDGET_Q_HI=199 \
  AV2ENC_PIPE_BUDGET_Q_STEP=4 AV2ENC_RD_DRIVER_HF=1 \
  AV2ENC_RD_DRIVER_MIN=16 AV2ENC_RD_DRIVER_LAMBDA=256 \
  build-release/av2enc "$REF" -o pb.ivf --video --frames 1 --qp 187 \
  --cpu-threads 8 --nightly-b --video-gdf --video-uv-qp-offset -23 \
  --video-delta-q --video-cdf-per-frame-qp --video-seg-delta 0
tools/bin/dav2d-macos -q --strict 1 -i pb.ivf -o pb.y4m --threads 1
python3 tools/quality.py --ref "$REF" --test pb.y4m --json q.json
python3 tools/extra_metrics.py --ref "$REF" --test pb.y4m --json e.json
python3 tools/kpi.py --quality-json q.json --extra-json e.json
```

`--video-seg-delta 0` is required: without it `build_sb_maps()` codes its own
alternating `base + 8` staircase on odd superblocks, which is not a zero ramp.

## 6. Pins

Gate off, every stream is byte-identical: 720p still QP186 → 25,172 B,
`8f0b85d8ea3ea9adfc1e28327aabf0e0`; the three regression pins
(`a1d0dba1`, `3984677b`, `0639b436`) are untouched. The pipeline is
env-gated off by default and adds no CLI flag.

## 7. The activity-scaled allowance (`AV2ENC_PIPE_BUDGET_STRENGTH`)

```
allowed(sb) = ref_bytes(sb) * SCALE/1000 * clamp((act_sb / act_mean)^S, 1/cap, cap) + credit
```

where `act` is `seq_source_activity` — the same
`(d - box3x3(d))^2` 1-px high-pass energy the block decomposition above uses —
and `cap` is `AV2ENC_PIPE_BUDGET_ACT_MAX` (default 2.0). `S = 0` is exactly the
old uniform-byte rule and is byte-identical to it; `S > 0` moves bytes from flat
to textured superblocks, which is the direction §4 says the error is.

**Measured: negative, and it does not even close the budget.**

| `S` | bytes | K |
|---:|---:|---:|
| 0 (uniform) | 24,183 | 46.91 |
| 0.5 | 24,701 | 45.98 |
| 1.0 | 24,764 | 46.33 |

All three are below the 47.37 control, and `S > 0` produces *more* bytes than
the uniform rule despite the allowance being scaled the same way on average.
The reason is structural: a superblock's coded size at a fixed qindex does not
scale with its source activity the way the byte allowance does, so the
activity multiplier buys quality in textured blocks at a super-linear byte cost
elsewhere. §4's finding (the error *is* concentrated in textured blocks) is real
but the allowance is the wrong handle for it — the same conclusion the
`AV2ENC_RD_DRIVER_SBALLOC` price knob reached from the other side
(`docs/k52-at-25k.md` §3.3).

## 7b. The first arm at or above the control: the TX high-frequency mask

`docs/k52-at-25k.md` §3.6i recorded `AV2ENC_TX_HF_MASK_DIV=2` as the one arm
that was strictly better (+0.12 K) on the older base. Re-measured on the
current control stack it reproduces and slightly improves:

| arm | bytes | K |
|---|---:|---:|
| control | 24,240 | 47.37 |
| **control + `AV2ENC_TX_HF_MASK_DIV=2`, QP187** | **24,121** | **47.41** |
| control + mask, QP186 | 25,085 | 47.99 |
| control + mask, QP185 | 26,211 | 48.54 |

So the mask makes the encoder ~120 B cheaper at the same QP and +0.04/+0.62 K.
This is the only lever in this programme that has measured at or above the
control, and it is a *transform-band* change, not an allocation one — consistent
with §4's conclusion that allocation is the wrong axis.

## 7c. The content-gated high-frequency mask (algorithmic change): negative, but it corrects the mechanism story

`AV2ENC_TX_HF_MASK_DIV=2` masks a fixed geometric corner of every transform. The
new `AV2ENC_TX_HF_MASK_CONTENT` decides *per block* from that block's own
residual 1-px activity against the picture's mean (`set_hf_mask_reference`,
published once per frame from `plan_pass`), scaled by
`AV2ENC_TX_HF_MASK_SCALE`. The hypothesis was §4's: the mask should apply where
the source is flat, not where the deficit lives.

**It does not help, and the sweep says why.**

| `MASK_SCALE` | blocks masked | bytes | K |
|---:|---:|---:|---:|
| 0.05 | almost none | 25,172 | 47.92 |
| 0.25 | few | 25,172 | 47.92 |
| 1.00 | 672,531 of 1,198,569 | 25,172 | 47.92 |
| 3.00 | almost all | 25,179 | 47.13 |
| **unconditional (no content gate)** | all | **25,085** | **47.99** |

Three facts fall out of this single table:

1. **The content gate is exactly the no-mask stream** at every setting that
   masks a minority of blocks (25,172 B, identical MD5 to the plain control).
   So masking *flat* blocks is a bit-exact no-op on this picture: at 0.21 bpp
   their corner coefficients already quantise to zero and are never coded.
2. **The mask's entire gain comes from textured blocks.** The unconditional
   arm is the only one that saves bytes (87 B at QP186). This *contradicts* the
   assumption in the `video_tx_search.cpp` comment — that the mask exists to
   suppress a checkerboard that lives in flat blocks. It does not.
3. That makes the mask a **rate refund, not an error reduction**, consistent
   with `docs/k52-at-25k.md` §3.6i ("the finest band's error is unchanged").
   It is also in direct tension with §4's finding that our 1-px error is
   already 1.76x the reference's in *textured* blocks: the mask buys +0.07 K by
   removing more high-frequency content exactly where we are already weakest.
   That is why the gain is so small and why it does not extend.

The unconditional mask therefore stays the best 25 KB-point arm
(**25,085 B / K 47.99**, `dav2d` OK), and the content gate is recorded as a
negative. The gate is default-off and leaves every pin byte-identical.

### 7d. The opposite polarity: also a no-op, and it closes the mechanism

`AV2ENC_TX_HF_MASK_POLARITY=above` masks blocks at or ABOVE `ref*|scale|` —
textured blocks, the ones §7c showed the mask actually changes. Both polarities,
both QPs:

| arm | bytes | MD5 | K |
|---|---:|---|---:|
| unconditional | 25,085 | `9a8ee8d5` | 47.99 |
| **above polarity, QP186** | **25,085** | **`9a8ee8d5`** | **47.99** |
| above polarity, QP185 | 26,211 | `6d68feda` | 48.54 |
| unconditional, QP185 | 26,211 | `6d68feda` | 48.54 |
| flat polarity, `scale` 0.6 | 25,172 | `8f0b85d8` | 47.92 |
| flat polarity, `scale` 0.85 | 25,172 | `8f0b85d8` | 47.92 |

**Both polarities are byte-identical to the unconditional mask**, at both QPs.
A selectivity sweep (0.6, 0.85 — windows that exclude most blocks) reproduces
`9a8ee8d5` exactly.

So the boundary does not matter, only whether a block is masked *at all*. The
reason is exact, not statistical: a masked coefficient is set to 0 **before
quantisation**, and the quantiser already produces 0 for every corner
coefficient at this qindex. Masking a block is therefore an idempotent no-op
regardless of which block it is, and the mask's whole 87-byte saving comes from
the small minority of blocks whose corner coefficients *would* have survived.
`docs/k52-at-25k.md` §3.6i's independent observation — "masking changes *which*
coefficients are coded, not how much 1-px error the picture ends up with" — is
the same fact seen from the other end.

**Consequence: any per-block gating of this mask is information-free by
construction.** No threshold, polarity or activity model can move it, so this
branch is closed rather than under-tuned. The two content-gate env knobs stay
as documented negatives.

## 8. Content-adaptive dead-zone on textured leaves (`AV2ENC_TX_DZ_TEXTURED`)

The prescribed follow-up to §7d: attack the textured-block coefficient deficit
directly instead of suppressing the band. `lround()` already applies a symmetric
0.5 dead-zone; this widens it on a leaf whose residual 1-px activity exceeds the
picture mean, so low-magnitude levels inside textured blocks collapse.

**The measurement that motivated it** (`AV2ENC_DZ_DUMP=1`, baseline arm):

```
dz-scalar coeffs=196,695,385 kept=7,001,366 (3.6%) dropped=189,881,772
dz-energy kept=7,873,665 dropped=2,736,448 -> dropped share 25.79%
dz-dropped-by-magnitude <.125=155,786,641 <.25=24,859,887 <.375=9,409,855 <.5=4,426,278
```

25.8 % of the candidate coefficient energy is discarded by the dead-zone, and
4.43 M coefficients sit in the last bucket just under the `lround` edge — the
population a wider zone acts on.

**Result: it moves the stream, and it loses K.**

| arm | QP | bytes | K |
|---|---:|---:|---:|
| baseline (mask) | 186 | 25,085 | **47.99** |
| `DZ=0.05` | 186 | 24,520 | 47.32 |
| `DZ=0.10` | 186 | 24,413 | 46.93 |

**Why it loses, arithmetically.** Widening the zone from 0.5 to 0.55 does not
only drop sub-threshold noise: it also drops everything in **[0.5, 0.55)**,
which `lround()` was keeping as a ±1 level. At the measured density
(35.4 M coefficients per unit of magnitude in that bucket) that is ~1.77 M
coefficients converted from a coded level to zero. The rate refund is real
(565 B at QP186) but it is bought by deleting signal, and the K loss is larger
than the byte saving is worth — the same trade `docs/negative-results.md`
records for the dead-zone family.

This is the third independent confirmation of the shape this document has
measured since §4: **the deficient axis is coefficient accuracy in textured
blocks, and every arm that trades accuracy for rate loses K**, whether the trade
is made per superblock (allocation, §0), per band (mask, §7c/§7d) or per
coefficient magnitude (this section).

## 9. Spending precision in textured leaves (`AV2ENC_TX_LAMBDA_TEXTURED`)

The prescribed follow-up: stop trading accuracy for rate and **spend** it
instead, by cutting the RD lambda on textured leaves so their cost prefers finer
levels. Same candidate set, no syntax change.

A control first established that the knob is live: the global
`--video-key-tx-lambda 0.25` moves the stream hard —

```
baseline  : leaves=2275  tx8=1732 tx16=904  25,085 B
global λ  : leaves=3012  tx8=3056 tx16=771  27,133 B
```

and it does so through the **tree**, not the leaf TX search: leaf count changes
57 %, which only a split/leaf decision can do.

That splits the arm in two, and only one half can matter:

| site | result |
|---|---|
| leaf TX-search lambda, reduced 0.5x on 4,508 textured leaves | **25,085 B, byte-identical** (`9a8ee8d5`) |
| leaf TX-search lambda, reduced 0.75x | **25,085 B, byte-identical** |
| tree-prune lambda, `-0.10` per textured superblock | **25,081 B, K 48.01** |
| tree-prune lambda, `-0.20` | 25,110 B, K 47.66 |
| tree-prune lambda, `-0.25` | 25,140 B, K 47.33 |
| tree-prune lambda, `-0.50` | 25,118 B, K 47.90 |
| baseline | 25,085 B, K **47.99** |

So the leaf TX search has **no freedom** at this operating point: its lambda
cannot change a decision, whatever the leaf's content. That is measured, not
assumed, and it is why the content-adaptive variants in §7c and §8 were inert or
harmful — they were all steering a stage that is already fully determined by
the qindex.

The **tree-prune** site is live, and `-0.10` is the first arm in this programme
to beat the baseline at matched bytes: **K 48.01 at 25,081 B against 47.99 at
25,085 B** — a +0.02 K gain on 4 fewer bytes. The gain is real but small, and
the curve turns over immediately: `-0.20` and beyond lose, monotonically. The
optimum is a narrow window just past zero, which is itself the interesting
result: the textured-leaf lambda is already close to correctly calibrated, and
only a slight nudge in the "spend more precision" direction pays.

A second instrumentation finding worth recording: an activity-proportional ramp
(`t = clamp(ratio-1, 0, 1)`) yields only a 4 % lambda cut because leaf activity
ratios cluster at 1.00-1.10 around the picture mean, which reads as a
byte-identical no-op. The gate must be a **binary split at the mean** to do
anything at all.

## 10. The key trellis: where coefficient accuracy actually lives

§9 established that the leaf TX-search *lambda* is inert at this QP. The
coefficient levels themselves are set by the RDOQ trellis, and the key picture
only gets one when `AV2ENC_RD_KEY_TX` is set (it is, in `--nightly-b`):

```
--nightly-b, QP186          trellis_parts=6009
AV2ENC_RD_KEY_TX=0          25,301 B  (trellis off, lround levels only)
```

So the trellis **is** running on intra leaves, and `lround()` is not the only
level rule. That closes the "the trellis never runs on the key" hypothesis; the
remaining accuracy knobs live inside the trellis:

| knob | what it changes | status at 720p / 0.31 bpp |
|---|---|---|
| `AV2ENC_RDOQ_ZERO_TO_ONE` | lets the trellis raise a 0 level to 1 | measured below |
| `AV2ENC_RDOQ_PREV_CAP` | state cap `min(|prev|, cap)` | documented inert before (`docs/k52-at-25k.md` §3.5: the DP stop rule ends the search before the bound) |
| `AV2ENC_RDOQ_EXACT` / `_ADAPT` | trellis rate model (range encoder / adapted CDFs) | measured earlier in the programme |

## 11. Closed-loop level refinement (`AV2ENC_RDOQ_REFINE_RECON`)

A genuinely new reconstruction mechanism rather than another price knob.

**The defect it targets.** The RDOQ trellis scores each coefficient as
`(original - level*quant_step)^2` — a pure *transform-domain* quantity
(`src/codec/rate/rdoq/rdoq.cpp:499`). The error that reaches the picture is
measured after the dequantiser and the inverse transform, where one coefficient
spreads across the whole block and the target is the residual in `coded`. On a
0.31 bpp key those two objectives disagree exactly where it costs K: §4 puts
75.7 % of our extra luma error in high-activity blocks and our 1-px band error
there at 1.70x the reference encoder's.

**What it does.** After the trellis picks levels, a second pass walks the scan,
tries level magnitude ±1 at each position, and re-derives the real
reconstruction (integer dequant + `txv::inverse_block`) to score the change
against the leaf's own residual. A change is accepted only when
`SSE + lambda * delta_bits` improves, where `delta_bits` comes from the same
§8.3 model (`Model::coefficient_rate`), so the pass cannot buy distortion for
free. Scan order and EOB are preserved (no `0 -> 1` at a zero seed, no dropping
the last non-zero), so the emitted syntax stays legal.

**Measured at QP186/187 on the mask + tree-lambda −0.10 stack:**

| arm | bytes | PSNR-Y | K |
|---|---:|---:|---:|
| stack baseline, QP186 | 25,081 | 24.490 | **48.01** |
| refine-recon, QP186 | 25,158 | 24.499 | **48.09** |
| refine-recon, QP187 | see below | | |

`validate_dav2d.sh`: **OK**. Gate off: `fast-decode 0639b436…`,
`nightly-b 8f0b85d8…`, `milestone 9a8ee8d5…` all byte-identical.

### 11.2 Correction: the pass was unreachable, so §11's numbers do not reproduce

**The 25,158 B row above is wrong and is superseded by this section.** It was
recorded from an arm that never entered the refine pass. Two defects made the
pass dead code on this picture:

1. The refine block sat inside `if (use_rdoq)`, and `use_rdoq` requires
   `intra_rdoq_ok = is_inter || options.intra_rdoq`. The key TX call site
   hardcodes `/*intra_rdoq=*/false` (`video_frame_decision.cpp:3583`), so on a
   key picture `use_rdoq` was false and the block was never reached.
2. After fixing (1), the pass still only moved 2 of 78,723 entered TXBs: its
   move set excluded `0 -> 1` insertions and sign flips ("no 0 -> 1 (pins)"),
   so it could only shrink levels that the transform-domain trellis had already
   placed at a local optimum of that restricted set.

Re-measured with the gate on, default lambda, all byte-identical to the
baseline on QP186 and QP187:

| arm | bytes | MD5 | refine stats |
|---|---:|---|---|
| stack baseline QP186 | 25,081 | `3705ac02…` | — |
| refine-recon QP186 λ=1.0 | **25,081** | `3705ac02…` | entered 78,723, changed 3 |
| refine-recon QP187 λ=1.0 | 24,209 | `43558740…` | entered 78,164, changed 3 |
| refine-recon QP186 λ∈{0.805, 0.3, 0.1, 0.03, 0.001, 1e-9} | 25,081 | `3705ac02…` | changed ≤ 3 |

`changed=3` out of 78,723 is the whole story: the trellis's levels are a local
optimum of the refine pass's restricted move set, so no lambda value — including
`1e-9`, where the rate term is free and only SSE counts — can move the stream.

**Pixel/transform distortion calibration.** `AV2ENC_RDOQ_REFINE_CALIBRATE`
measured the ratio the review asked for, over 78,500 TXBs:
`sum(pixel SSE) / sum(transform-domain error) = 0.805`. So a pixel-domain pass
equivalent to the trellis would use `λ_ref = 0.805·λ_tx`, and the shipped
default of `1.0` is therefore *more* conservative than the calibration, not
less. This rules out mis-calibration as the cause; the restricted move set was.

### 11.3 Opening the move set (`AV2ENC_RDOQ_REFINE_INSERT`): negative

`AV2ENC_RDOQ_REFINE_INSERT=1` admits `0 -> ±1` insertion and sign flips. It
works mechanically — 725–1,525 TXBs now change per run — and is measured
negative at every lambda (all `dav2d` OK):

| arm | bytes | JOD | BA | SS2 | K |
|---|---:|---:|---:|---:|---:|
| stack baseline | 25,081 | 7.3392 | 13.1168 | −10.032 | **48.01** |
| insert λ=0.805 | 25,197 | 7.3392 | 14.4248 | −11.027 | 47.50 |
| insert λ=0.3 | 25,146 | 7.3392 | 14.2674 | −10.836 | 47.57 |
| insert λ=0.1 | 25,180 | 7.3386 | 14.9883 | −11.265 | 47.32 |
| insert λ=0.03 | 25,114 | 7.3394 | 13.6092 | −11.247 | 47.67 |

Every arm is both **larger and worse** than the baseline, so the matched-byte
rescaling only widens the loss. Note JOD is flat (7.3392) across all four while
Butteraugli and SSIMULACRA2 degrade by 0.3–1.3 and ~1.2: the inserted
coefficients buy *global* error that CVVDP does not see and spend bits that the
transform-domain objective was already allocating correctly. This is the same
pattern as `RD_KEY_TX_D=0` (§13): our error is not in the wrong place, it is in
the wrong amount.

**Verdict:** the closed-loop refine path, calibrated as requested, is measured
and negative at K 47.32–47.67 vs 48.01. The delta path is the mechanism; the
pixel-domain measure inside it is not the missing 1.99 K.

## 15. Frame-level joint search at a fixed 25,081 B ceiling

§9's tree-lambda sweep and §11's refine arms each moved one knob at a fixed QP.
The remaining untested mechanism is the *joint* search: QP187 has ~872 B of
headroom to the 25,081 B ceiling, so lower λ values can spend it instead of
shrinking the stream. Grid: mask + tree-lambda 0.10 stack, `RD_DRIVER_LAMBDA`
and `QP` varied together, every arm scored at its own matched byte count.

| arm | bytes | JOD | BA | SS2 | K | vs 48.01 |
|---|---:|---:|---:|---:|---:|---:|
| QP187 baseline (d=256) | 24,209 | — | — | — | — | (headroom) |
| QP187, d=160 | 25,032 | 7.3137 | 15.632 | −11.37 | 47.03 | −0.98 |
| QP187, d=155 | 25,024 | 7.3147 | 14.680 | −11.51 | 47.23 | −0.78 |
| QP187, d=140 | 25,085 | 7.3158 | 13.569 | −11.33 | 47.55 | −0.46 |
| QP187, tree-lambda 0.05/0.0 | 24,060/24,121 | — | — | — | — | (no-op: §9) |

The joint search **spends the 872 B but cannot buy K with it**: every
matched-byte QP187 arm lands 0.46–0.98 K below the QP186 baseline. The
spend-more-precision direction is uniformly negative at this budget, on both
the QP and the λ axes. Combined with §11.3 this closes the frame-level joint
mechanism: the QP186 stack's K 48.01 is a local optimum over (QP, tree-lambda,
driver-lambda) at 25 KB, not a single point off a better ridge.

Decoded PNGs of the compared streams are written to
`~/Downloads/src720_av2/` (stack 25,081 B, qp187_d140 25,085 B,
qp187_d155 25,024 B), each `tools/validate_dav2d.sh` OK.

### UV/chroma-budget axis (`--video-uv-qp-offset`, QP186 stack)

The chroma offset is a real rate knob on this stack (−23 → 25,081 B at one end,
+8 → 23,459 B at the other), so it can trade chroma steps for luma at fixed
bytes. Grid on the mask + tree-lambda stack, every arm `dav2d` OK:

| arm | bytes | JOD | BA | SS2 | K |
|---|---:|---:|---:|---:|---:|
| **uv −23 (cap, baseline)** | **25,081** | 7.3392 | 13.117 | −10.03 | **48.01** |
| uv −21 | 25,058 | 7.3312 | 13.957 | −11.32 | 47.53 |
| uv −20 | 24,981 | 7.3431 | 14.815 | −10.99 | 47.43 |
| uv −16 | 24,673 | 7.3277 | 13.484 | −12.32 | 47.47 |

The −23 cap is the matched-byte optimum on this axis: giving chroma fewer
steps saves bytes but loses K even *at* the saved rate (−21 is 23 B under and
loses 0.48 K).

**QP185 companions** (QP lowered to afford chroma steps, joint (QP, UV) point):

| arm | bytes | JOD | BA | SS2 | K |
|---|---:|---:|---:|---:|---:|
| uv −16, QP185 | 25,539 | 7.3783 | 16.104 | −10.77 | 47.36 |
| uv −14, QP185 | 25,138 | 7.3756 | 13.965 | −11.43 | 47.73 |
| uv −8, QP185 | 24,809 | 7.3669 | 13.186 | −12.85 | 47.66 |

The uv−14 arm is the joint axis's best arm and it is **57 B over the ceiling
and 0.28 K below the baseline**; the under-budget arm uv−8 is 0.35 K below.
QP185 raises JOD (7.367–7.378 vs 7.339) but spends it on chroma steps that
Butteraugli/SS2 do not reward — the same decoupling seen in §11.3.

**Verdict: the (QP × chroma-step) joint axis is closed and negative at 25 KB**
— both QP186 and QP185 grids peak at their own stack baseline (uv −23,
QP186, 48.01). The `DELTA_DCQUANT_MIN = -23` cap is not a binding constraint
we could unlock; it is where the optimum already sits.

### QP186 side of the joint grid (same stack, driver-λ only)

| arm | bytes | JOD | BA | SS2 | K |
|---|---:|---:|---:|---:|---:|
| d=256 (baseline) | 25,081 | 7.3392 | 13.117 | −10.03 | **48.01** |
| d=240 | 25,205 | 7.3279 | 14.046 | −10.94 | 47.55 |
| d=272 | 25,099 | 7.3262 | 13.557 | −11.07 | 47.65 |
| d=290 | 25,098 | 7.3289 | 13.765 | −10.85 | 47.64 |

All three are at or just above the baseline's byte count and all lose K, so
256 is the joint optimum on this axis too. The grid is now closed on both QPs:
**(QP186, 256)** — 25,081 B, K 48.01 — beats every measured (QP, driver-λ)
neighbour at matched bytes (±124 B), by +0.36 to +0.98 K. Negative.

## 12. Reference comparison: SVT-AV1 p0, CRF vs the same quantiser

Encoder: **SVT-AV1 v4.3.0 (release)**, `--preset 0`, on `src720_1f.y4m`
duplicated to 3 frames (frame 0 is the intra that is scored; frames 1-2 cost
53 B and 5 B). Decoded with ffmpeg/libdav1d and scored with the same
`tools/quality.py` / `extra_metrics.py` / `kpi.py` as every arm here.

**The two modes are not interchangeable and must be reported separately.**

`--crf` is `--rc 0 --aq-mode 2 --qp x`: it enables content-adaptive quantiser
distribution. `--cqp` is `--rc 0 --aq-mode 0 --qp x`, i.e. a fixed quantiser —
the same shape as our fixed-QP `--nightly-b`. Comparing our fixed-QP stream to
SVT's CRF conflates quantiser accuracy with an allocation mechanism we do not
have, so both are reported.

| SVT mode | QP/CRF | frame 0 | PSNR-Y | K |
|---|---|---:|---:|---:|
| CRF | 65 | 23,347 | 25.002 | 47.60 |
| CRF | 64 | 26,659 | 25.415 | **49.81** |
| CRF | 63 | 31,541 | 25.976 | 52.52 |
| CRF | 62 | 47,272 | 27.549 | 59.70 |
| **CQP** | 61 | 66,513 | — | — |
| **CQP** | 64 | 34,013 | — | — |
| **CQP** | **65** | **24,073** | **25.078** | **48.53** |
| **CQP** | 66 | 17,420 | 24.149 | 43.09 |
| **CQP** | 67 | 12,591 | — | — |
| **CQP** | 70 | 5,422 | — | — |

Matched-byte readings:

| comparison | SVT | av2enc | gap |
|---|---:|---:|---:|
| **same quantiser** (SVT CQP 65 @24,073 B) | **48.53** | 48.01 (25,081 B) | **+0.52** |
| CRF with AQ (interpolated @25,085 B) | 48.76 | 48.01 | +0.75 |

### What this changes

1. **SVT's AQ is worth ~0.2 K, not the whole gap.** Its fixed-quantiser mode
   reaches 48.53 at 24 kB; CRF+Aq improves that to ~48.76 at 25 kB.
2. **Against the same quantiser we are 0.52 K behind at ~1 kB more bytes** —
   i.e. about 1 % in K. The 240p gap in `README.md` (82.2 vs 88.8) does not
   reproduce at 720p on this picture.
3. **Neither encoder reaches 52 K at 25 kB.** SVT crosses K 52 only between
   CRF 63 (31,541 B) and CRF 64 (26,659 B), i.e. at roughly **29 kB**, 16 %
   over the budget. Our own K-52 crossing is far above that.

So the target exceeds the measured capability of a mature production encoder at
the same rate by ~3.2 K, and exceeds ours by ~4.0 K. That is the ceiling
evidence the objective needs; it is not progress toward 52.

### Axis breakdown near 25 kB (SVT CRF 65 vs our stack)

| axis | SVT | av2enc | better |
|---|---:|---:|---|
| PSNR-Y | 25.00 | 24.49 | SVT +0.51 dB |
| CVVDP (JOD) | 7.400 | 7.339 | SVT +0.06 |
| Butteraugli | 14.04 | 13.12 | **ours −0.92** |
| SSIMULACRA2 | −12.80 | −10.03 | **ours +2.77** |

We lead both perceptual axes and trail on PSNR. K weights CVVDP at 0.5, so
SVT's PSNR/CVVDP edge nets the 0.5-0.75 K.

## 13. The deficit is JOD, not PSNR — and plain SSE is refuted

The same-quantiser gap to SVT CQP 65 is 0.52 K. Decomposing it on the K
formula's own weights (`tools/kpi.py`: 0.5·JOD/10 + 0.25·BA + 0.25·SS2):

| axis | ours | SVT CQP 65 | K contribution |
|---|---:|---:|---:|
| **CVVDP (weight 0.50)** | 36.696 | 37.275 | **+0.579** |
| Butteraugli (0.25) | 4.654 | 4.534 | −0.120 |
| SSIMULACRA2 (0.25) | 6.661 | 6.724 | +0.063 |
| **total K** | **48.011** | **48.533** | **+0.522** |

So the entire deficit is CVVDP/JOD; we already lead Butteraugli. An earlier
hypothesis in this programme — "the gap is the PSNR axis, so switch the RD
distortion back to plain SSE" — **is refuted by direct measurement**:

| arm | bytes | PSNR-Y | JOD | BA | SS2 | K |
|---|---:|---:|---:|---:|---:|---:|
| stack baseline (`RD_KEY_TX_D` default = eff-perc) | 25,081 | 24.490 | **7.3392** | 13.12 | −10.03 | **48.01** |
| `AV2ENC_RD_KEY_TX_D=0` (plain SSE), QP186 | 25,258 | **24.511** | 7.3307 | 14.30 | −10.97 | 47.49 |
| `AV2ENC_RD_KEY_TX_D=0`, QP185 | 26,093 | 24.623 | 7.3830 | 13.87 | −9.42 | 48.12 |

Plain SSE raises PSNR-Y by +0.021 dB and **lowers** JOD by 0.0085, costing
0.52 K and 177 B. PSNR and JOD are not co-monotone on this content, so PSNR is
not a usable proxy here — the perceptual weighting is already doing the right
thing, just not enough of it.

## 14. What would change the answer

Not an allocator. §3.4 of `docs/k52-at-25k.md` and §4 above agree: the deficit
is coefficient-domain accuracy at fixed bits (a uniformly better reconstruction,
not a differently placed one). The measured distance is large — AVM's own curve
is K 51.06 at 25,000 B, i.e. the reference encoder does not reach 52 at the
budget either, and our K-52 crossing is 32,867 B (1.31x the budget).

## 16. Floor 4x4 + probe off: static vs rate-dependent driver tree price

Directed work: avoid the measured partition probe, set the partition floor to
4x4, then compare the current **static** tree pricing against the **dynamic**
(QP-dependent) one. Both decisions were confirmed explicitly; the measured
partition probe (`partition_probe = partition_rd && !keyframe_ && …`) is
already forced off on a single-picture `--nightly-b` stream
(`video_partition_rd = !single_picture && …`), and both arms additionally pass
`--video-partition-rd-off` to lock it off.

**The dynamic price is docs/rd-key-partition-floor.md §0.4's prescription that
was never measured**: the flat `kKeyTxLambdaFit` (exponent 0 → one SSE-per-bit
price at every qindex) is §0.4's identified root cause of the driver's 4x4
over-expansion at low rate, and "the price must be rate-dependent (~2.0 at
qindex 82, ~2048 at qindex 184)". `AV2ENC_RD_DRIVER_RATE_FIT` (default off)
implements exactly those two anchors:

```
price(q) = 2.0 * (step(q)/step(82))^e,  e = ln(1024)/ln(step(184)/step(82))
         = 2.0 * (step(q)/52.0)^2.3509          # from rc_tables.h: 52.0 / 992.0
```

Fit check: price(82) = 2.0000, price(184) = 2048.0, price(186) = 2372,
price(187) = 2545, against the static flat **512** (= 8.0 × 0.25 ×
`AV2ENC_RD_DRIVER_LAMBDA=256`) at every qindex. The gate touches only the two
driver tree-price sites (`seq_driver_measure_leaf`, `seq_driver_decide_sb`
base_lambda); the flat branch keeps its multiplication order, so gate-off is
bit-for-bit identical.

Arms: `AV2ENC_RD_DRIVER_MIN=4 --video-min-block 4 --video-partition-rd-off`
(mask + tree-lambda stack), `tools/hk_eval.py` walked each to the 25,081 B
target (interpolated) with `validate_dav2d rc 0`:

| arm | bracket (QP → bytes) | **K @25,081 B** | endpoint K | vs static | vs floor-16 baseline |
|---|---|---:|---:|---:|---:|
| static flat 512 | 194 → 24,511 / 193 → 25,098 (2.3 %) | **45.28** | 45.00 / 45.29 | — | −2.73 |
| rate-dependent fit | 185 → 24,409 / 184 → 25,573 (4.6 %) | **47.00** | 46.35 / 47.49 | **+1.72** | −1.01 |
| floor-16 baseline (`MIN=16`, flat) | — | **48.01** | — | +2.73 | — |

**Reading.** The §0.4 prescription is confirmed: at floor 4x4 the rate-dependent
price recovers **+1.72 K** over the flat price and closes 87 % of the gap to
the floor-16 arm (45.28 → 47.00 → 48.01), which is exactly §0.4's mechanism
story — the flat price, not the 4x4 leaf set itself, is what broke the low-rate
tree. The comparison is robust to the rate arm's 4.6 % bracket: its *lowest*
endpoint (46.35 at 24,409 B) still beats the static arm's *highest* endpoint
(45.29 at 25,098 B) by +1.06 K. But 47.00 still sits **1.01 K below the
floor-16 baseline**, so at 25 KB the floor arm remains the better lever, and
the rate-dependent price does not reach the K 50 target either.

An incremental arm (rate-dependent price *on* the winning floor-16 stack) is
run as `min16_rate` in the same section's notes below when available.

### 16.1 min16_rate: rate-dependent price on the winning floor-16 stack — negative

The obvious incremental question after §16: does the §0.4 rate-dependent tree
price improve the *floor-16 baseline* (where the flat price already measures
48.01)? `min16_rate` = the baseline stack (mask + tree-lambda 0.10 + `MIN=16`
+ `LAMBDA=256`) with `AV2ENC_RD_DRIVER_RATE_FIT=1`, walked by `hk_eval`:

| arm | bracket (QP → bytes) | K @25,081 B | endpoint K | vs baseline |
|---|---|---:|---:|---:|
| baseline (flat 512) | — | **48.01** | — | — |
| `min16_rate` (fit) | 184 → 25,049 / 183 → 26,442 (5.6 %) | **47.47** | 47.46 (only 32 B from target) | **−0.54** |

`validate_dav2d rc 0`. The near-target endpoint (25,049 B, 32 B short) makes
the reading tight despite the wide second bracket point. **Negative**: on the
floor-16 tree the flat 512 price is already better calibrated for this QP than
the rate-dependent fit (2,372 at QP186 ≈ 4.6x stiffer), which over-collapses
and loses 0.54 K. The fit's value is confined to the floor-4 tree (§16: +1.72
over flat there).

## 17. df_delta_q: the §5.18.7.9 deblock filter-level delta (first positive arm)

The §7.17 filter has exactly one spec-signalled strength knob: `df_delta_q[0..3]`
(Y pass 0, Y pass 1, U, V), which shifts the adaptive selector's level by
`DF_DELTA_SCALE(8) * delta`. The encoder has always signalled
`present=false` everywhere, so this axis was never measured (the earlier
"47.43–48.01" axis in the review notes is the `--video-uv-qp-offset` chroma-QP
scan of §15, not this one). `AV2ENC_DEBLOCK_DF_DELTA` (default off) now signals
it: one integer or four (`Y0,Y1,U,V`), values clamped to the dfParBits domain
whose `df_par_bits_minus_2` the sequence header widens to exactly fit; all-zero
keeps `present=false` and is byte-identical (four pins re-verified after the
change).

Wired at three places: shared helper `post/deblock_grid.h`, the §5.18.7.9 row
in `video_headers.cpp` (sequence `df_par_bits_minus_2` + frame row, bit-exact
assert), and the encoder's own §7.17 pass in `video_frame_filters.cpp`
(`grid.options.df_delta_q`, so encoder recon tracks the decoder).

QP186 baseline stack, `validate_dav2d OK` on every arm:

| `AV2ENC_DEBLOCK_DF_DELTA` | bytes | cvvdp | BA | SS2 | **K** | ΔK |
|---|---:|---:|---:|---:|---:|---:|
| 0 (baseline) | 25,081 | 7.3392 | 13.117 | −10.032 | **48.01** | — |
| +1 | 25,082 | 7.3326 | 13.146 | −10.422 | 47.90 | −0.11 |
| −1 | 25,082 | 7.3439 | 13.135 | −9.794 | **48.07** | +0.06 |
| −2 | 25,092 | 7.3447 | 13.143 | −9.621 | **48.10** | +0.09 |

**First arm above 48.01 this session.** The dose-response is monotone across
four points (+1 → −2: 47.90 → 48.01 → 48.07 → 48.10): *weaker* deblock
(lvl −8/−16) improves JOD (+0.0055) and SS2 (+0.41) while BA stays flat
(+0.03), for +1…+11 header bytes. The −2 reading overstates by ~0.02 K for
its +11 B (≈1.5 K/kB slope), so the net is ≈ +0.07 — still positive. Dose
extension (−3, −4, needs dfParBits widened) and plane decomposition
(`−2,−2,0,0` luma-only vs `0,0,−2,−2` chroma-only) are running as `dfd_m3`,
`dfd_m4`, `dfd_m2y`, `dfd_m2c`.

### 17.1 Dose extension and plane attribution (QP186): peak at Δ = −2, luma-only

| arm | bytes | cvvdp | BA | SS2 | K | ΔK |
|---|---:|---:|---:|---:|---:|---:|
| Δ−1 (all four) | 25,082 | 7.3439 | 13.135 | −9.794 | 48.07 | +0.06 |
| **Δ−2 (all four)** | 25,092 | 7.3447 | 13.143 | −9.621 | **48.10** | **+0.09** |
| Δ−3 (all four, dfParBits=3) | 25,090 | 7.3439 | 13.187 | −9.767 | 48.06 | +0.05 |
| Δ−4 (all four, dfParBits=3) | 25,093 | 7.3413 | 13.143 | −10.056 | 48.01 | 0.00 |
| Δ−2 luma only (`−2,−2,0,0`) | 25,091 | 7.3429 | 13.135 | −9.547 | **48.10** | +0.09 |
| Δ−2 chroma only (`0,0,−2,−2`) | 25,082 | 7.3410 | **13.097** | −10.059 | 48.02 | +0.01 |

`validate_dav2d OK` and `dav2d --strict 1` on all six. Reading:

1. **The dose curve peaks at −2** (level −16): 47.90 → 48.01 → 48.07 →
   **48.10** → 48.06 → 48.01. The turnaround at −3/−4 matches the known
   `--video-deblock-off` collapse (47.21): weaker filtering helps only until
   the blocking/ringing trade flips. Integer deltas mean −2 is the grid's
   optimum at this QP.
2. **All of the gain is luma**: the luma-only arm reproduces 48.10 exactly
   while chroma-only adds +0.01 (and the best BA of the batch, 13.097).
   Composition `luma −2 + chroma 0` (25,091 B) is the standing best deblock
   point: **K 48.10, net ≈ +0.075 after its +10 B**.

An intermediate batch (`dfd_m3/m4/m2y/m2c`, first attempt) was invalidated
before scoring: a zsh non-splitting of `$BASE` dropped `AV2ENC_RD_DRIVER_MIN`
from the command, so the driver ran at its default target 4 (27,457 leaves,
49.6 KB tiles — the documented floor-4 failure shape). Those files are kept as
`*.BAD-zshsplit.ivf`; the rerun with inline envs produced the table above.
Side-finding: the widened `df_par_bits_minus_2 = 1` sequence field and its
frame row decode cleanly under pinned `dav2d --strict 1`.

**Batch-hazard signature (recorded twice now):** any run whose log prints
`intra blocks 27457` (vs the stack's 2,282) and a ~49.6 KB tile at QP186 lost
its `env` variables — in zsh an unquoted `"$VAR"`/`$VAR` holding multiple
`KEY=VAL` words is passed to `env` as a **single argument**, so only a garbage
variable is set and `AV2ENC_RD_DRIVER_MIN` silently reverts to 4 (uniform 4x4
expansion). Both the `dfd_m3/m4/m2y/m2c` batch and the first `ap_*` batch hit
it; both were discarded before scoring (`*.BAD-zshsplit.ivf`) and re-run with
literal env words. **Rule: always write the env list literally in these batch
jobs.** The four pins (fast-decode `0639b436…`, nightly-b `8f0b85d8…`,
milestone `9a8ee8d5…`, stack `25081 B/3705ac02…`) were re-verified on the
post-`AV2ENC_DEBLOCK_APPLY` binary and are unaffected (pins use literal envs).

### 17.2 apply_deblocking_filter per-pass flags (QP186): negative/tie

The second spec-signalled knob of the same §5.18.7.9 row —
`AV2ENC_DEBLOCK_APPLY` (Y pass 0, Y pass 1, U, V; all-on default =
byte-identical; both-luma-off is rejected because the syntax then drops the
chroma flags) — measured on the baseline stack, `validate_dav2d OK`:

| arm | bytes | cvvdp | BA | SS2 | K | ΔK |
|---|---:|---:|---:|---:|---:|---:|
| baseline (all on) | 25,081 | 7.3392 | 13.117 | −10.032 | 48.01 | — |
| chroma off `1,1,0,0` | 25,082 | 7.3430 | 13.117 | −10.922 | 47.88 | −0.13 |
| Y pass 0 off `0,1,1,1` | 25,091 | 7.3254 | 13.633 | −10.768 | 47.67 | −0.34 |
| Y pass 1 off `1,0,1,1` | 25,082 | 7.3387 | 13.179 | −9.929 | 48.01 | 0.00 |
| luma Δ−2 + chroma off | 25,092 | 7.3466 | 13.115 | −10.544 | 47.96 | −0.05 |
| *(reference)* luma Δ−2 | 25,091 | 7.3429 | 13.135 | −9.547 | **48.10** | +0.09 |

Reading: **no arm clears 48.01.** Dropping chroma deblock costs SS2 >1.0
(−9.55 → −10.92 in the paired arms), so chroma deblock is load-bearing even
though it ignores the Δ knob; the composition arm gives the +0.09 luma gain
back (48.10 → 47.96). Disabling either luma pass is −0.34 / 0.00. The
deblock row is now exhausted on both of its knobs; **the axis best stands at
luma `df_delta_q = −2` → K 48.10 @ 25,091 B (+0.09 over baseline, ≈ +0.075
net of its +10 B).**

## 18. QP185 joint point with luma df_delta_q −2: negative

QP185 alone codes ~26,100 B on this stack, so the joint arm spends the finer
quantisation and then shrinks the tree with `AV2ENC_RD_DRIVER_LAMBDA` to land on
the budget. Both arms carry the best deblock setting (luma `−2,−2,0,0`),
`validate_dav2d OK`:

| arm | bytes | leaves | cvvdp | BA | SS2 | K | vs 48.10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| QP186 + luma Δ−2 (standing best) | 25,091 | 2,282 | 7.3429 | 13.135 | −9.547 | **48.10** | — |
| QP185, λ=497 | 25,064 | 1,882 | 7.3297 | 13.380 | −10.946 | 47.74 | −0.36 |
| QP185, λ=512 | 25,037 | 1,881 | 7.3318 | 14.095 | −10.836 | 47.57 | −0.53 |

**Negative.** The mechanism is real — QP185 + λ497 reaches the same byte count
with a *smaller* tree (1,882 vs 2,282 leaves) — but it pays in the metric that
matters most here: Butteraugli 13.13 → 13.38/14.09 and SS2 −9.55 → −10.95,
while cvvdp is flat. Finer quantisation with a coarser tree concentrates error
at block edges, which is exactly what BA punishes. Both scored points sit
*below* 25,081 B, so an above-target bracket (λ 490) is being measured for the
record; the gap to the best is 0.36 K at a point already 17 B *under* budget,
i.e. the arm would have to gain ~0.4 K by growing 17 B — not plausible.

### 18.1 QP185 joint bracket closed (negative)

λ490 gives the above-target bracket point: 25,196 B, K 46.89 (BA 17.37 — the
edge-error signature of the finer-quantisation/coarser-tree trade shows up hard
here). Interpolating the 25,064 → 25,196 B bracket to 25,081 B gives
**K ≈ 47.63**, so the QP185 joint axis is negative by 0.47 at matched bytes,
independent of the missing bracket point. Axis closed.

## 19. CDEF strength × luma df_delta_q −2 (QP186): neutral after byte correction

The deblock's downstream partner filter (`--video-cdef-strength`, the
§5.18.7.10 `CdefStrengths` row) was previously swept alone (4 → 47.78, 8 →
47.77, both ties at 106 KB) but never combined with the luma `df_delta_q = −2`
point. All arms carry luma `−2,−2,0,0`, `validate_dav2d OK`:

| arm | bytes | cvvdp | BA | SS2 | raw K | K at 25,081 B |
|---|---:|---:|---:|---:|---:|---:|
| strength 1 (default) + Δ−2 | 25,091 | 7.3429 | 13.135 | −9.547 | 48.10 | **48.10** |
| strength 2 + Δ−2 | 25,098 | 7.3430 | 13.132 | −9.558 | 48.10 | 48.09 |
| strength 3 + Δ−2 | 25,128 | 7.3431 | 13.153 | −9.528 | 48.10 | 48.03 |
| strength 4 + Δ−2 | 25,141 | 7.3435 | 13.156 | −9.507 | 48.11 | 48.04 |

**Neutral/negative.** More CDEF strengths buy a small SS2 gain (−9.55 → −9.51)
but cost 7/37/50 header bytes for a flat JOD and slightly *worse* BA, so after
the ≈1.5 K/kB byte correction every strength sits at or below the strength-1
Δ−2 point. No M2 `CdefStrengths` setting clears 48.10.

## 20. CDEF damping × luma df_delta_q −2: **positive, largest gain this session**

The perceptual-ranking arms (`AV2ENC_PERCEPTUAL_FILTER=veto` 25,121 B K 48.12;
`cdef` + strength 4 25,155 B K 48.12) are ties before byte correction and
negative after (+40/+74 B for +0.02 raw) — the 100 kbps `veto` advantage does
not reproduce at 25 KB. But the same batch's **`--video-cdef-damping` sweep
is the best result of the session** (all `validate_dav2d OK`, all carrying
luma `−2,−2,0,0`):

| arm | bytes | cvvdp | BA | SS2 | K | ΔK vs 48.01 |
|---|---:|---:|---:|---:|---:|---:|
| baseline (damping 3, dfd 0) | 25,081 | 7.3392 | 13.117 | −10.032 | 48.01 | — |
| damping 3 + luma Δ−2 | 25,091 | 7.3429 | 13.135 | −9.547 | 48.10 | +0.09 |
| damping 4 + luma Δ−2 | 25,093 | 7.3496 | **12.747** | −9.520 | 48.26 | +0.25 |
| **damping 5 + luma Δ−2** | **25,089** | 7.3549 | **12.505** | −9.540 | **48.36** | **+0.35** |

**Reading.** Damping is the CDEF knob that trades ringing against
smoothing in the *filter's* decision (`floor(damping + 2) >> 2` strength
quantisation, §7.18.2), and at 25 KB the encoder has been over-aggressive:
damping 4/5 cut Butteraugli by 0.37/0.63 while JOD rises +0.007/+0.012 and
SS2 is flat, **and** they code 2 bytes fewer than 3 (the milder filter changes
the §5.20.10.1 index grid). This is a bigger lever than every deblock knob
combined (deblock: +0.09; damping: +0.35). Extension to damping 6 and the
δ-dose re-scan under damping 5 (0/−1/−3) are running.

### 20.1 Damping peak, δ dose under damping 5, and per-pass asymmetry

| arm | bytes | cvvdp | BA | SS2 | K |
|---|---:|---:|---:|---:|---:|
| damp 3 + Δ−2 | 25,091 | 7.3429 | 13.135 | −9.547 | 48.10 |
| damp 4 + Δ−2 | 25,093 | 7.3496 | 12.747 | −9.520 | 48.26 |
| **damp 5 + Δ−2** | 25,089 | 7.3549 | 12.505 | −9.540 | **48.36** |
| damp 6 + Δ−2 | 25,092 | 7.3519 | **12.385** | −9.798 | 48.33 |
| damp 5, Δ 0 | 25,090 | 7.3462 | 12.462 | −10.168 | 48.22 |
| damp 5, Δ−1 | 25,091 | 7.3509 | 12.492 | −9.868 | 48.28 |
| damp 5, Δ−3 | 25,089 | **7.3533** | 12.485 | **−9.519** | **48.36** |
| Δ `−2,−1` (damp 3) | 25,092 | 7.3412 | 13.116 | −9.754 | 48.07 |
| Δ `−1,−2` (damp 3) | 25,092 | 7.3462 | 13.119 | −9.552 | 48.13 |
| Δ `−2,0` (damp 3) | 25,091 | 7.3444 | 13.132 | −9.520 | 48.12 |
| Δ `0,−2` (damp 3) | 25,092 | 7.3450 | 13.150 | −9.657 | 48.09 |

`validate_dav2d OK` on all. Damping peaks at **5** (6 loses 0.03 back through
SS2 while BA keeps improving — BA is saturating). The δ dose keeps its −2
optimum under damping 5 (−3 ties on raw K but is 0.03 worse on SS2 and no
cheaper). Per-pass asymmetry is dominated by uniform −2 (best 48.13 < 48.36).
**Standing best: damping 5 + luma df_delta_q −2 → K 48.36 @ 25,089 B**
(+0.35 over the 48.01 baseline, +0.27 over the best pure-deblock arm).

Since damping 5 changes the filter itself, the neighbour axes that were tuned
under damping 3 — the chroma QP offset (−23) and the CdefStrengths count — are
being re-scanned under it.

### 20.2 Neighbour axes re-scanned under damping 5

| arm | bytes | cvvdp | BA | SS2 | K |
|---|---:|---:|---:|---:|---:|
| **damp 5, Δ−2, uv −23, 1 strength (standing best)** | 25,089 | 7.3549 | 12.505 | −9.540 | **48.36** |
| damp 5, Δ−2, 2 strengths | 25,093 | 7.3550 | 12.505 | −9.540 | 48.36 |
| damp 5, Δ−2, 4 strengths | 25,129 | 7.3555 | 12.520 | −9.516 | 48.36 |
| damp 5, Δ−2, uv −16 | 24,688 | 7.3429 | 13.007 | −11.727 | 47.78 |

`validate_dav2d OK` on all. Extra CdefStrengths still buy nothing (raw tie,
+4/+40 B), uv −16 costs 401 B *and* 2.2 SS2. `--video-uv-qp-offset −28` is
rejected by the CLI (`DELTA_DCQUANT` range is [−23, 8]), confirming −23 is the
legal endpoint — so both neighbour axes are closed under damping 5 too.

Then the one combination that the damping finding makes worth re-testing: the
QP185 λ-fill point (§18) died on Butteraugli (13.4–14.1), the exact metric
damping 5 cuts by 0.63. Re-running it under damping 5.

### 20.3 QP joint under damping 5: negative (the BA gain does not rescue it)

§18 killed the QP185 λ-fill arm on Butteraugli; damping 5 cuts BA by 0.63, so
the arm was re-run under it. `validate_dav2d OK` on all:

| arm | bytes | cvvdp | BA | SS2 | K |
|---|---:|---:|---:|---:|---:|
| QP185 λ497 | 25,052 | 7.3398 | 13.068 | −10.846 | 47.89 |
| QP185 λ512 | 25,035 | 7.3399 | 13.990 | −10.912 | 47.63 |
| QP187 λ160 | 25,029 | 7.3260 | 14.475 | −10.908 | 47.44 |
| QP187 λ150 | 25,097 | 7.3240 | 13.825 | −10.704 | 47.63 |
| QP187 λ140 | 25,092 | 7.3292 | 13.232 | −10.919 | 47.78 |

The QP187 bracket (25,029 → 25,097) interpolates to **≈ 47.76 at 25,081 B**,
i.e. **−0.60 against 48.36**. Damping's BA gain (0.63) is real but the QP-shifted
tree *re-creates* BA loss of the same size (13.13 → 13.23–14.48) at equal bytes,
so the two effects cancel and the arm cannot win. QP joint is closed for good.

### 20.4 Driver-λ / tree-λ re-scan under damping 5: negative (one tie)

| arm | bytes | cvvdp | BA | SS2 | K |
|---|---:|---:|---:|---:|---:|
| **damp 5, λ256, tlt 0.10 (standing best)** | 25,089 | 7.3549 | 12.505 | −9.540 | **48.36** |
| tlt 0.06 | 25,086 | 7.3539 | 12.505 | −9.575 | 48.34 |
| tlt 0.16 | 25,126 | 7.3597 | 14.110 | −9.609 | 47.91 |
| driver λ192 | 25,597 | 7.3652 | 13.531 | −9.027 | 48.19 |
| driver λ384 | 24,444 | 7.3148 | 13.908 | −11.077 | 47.50 |

`validate_dav2d OK` on all. tlt 0.06 is a byte-level tie (−3 B, −0.02 K — inside
the header-byte noise), tlt 0.16 breaks BA (12.51 → 14.11), and both driver-λ
directions lose. **Driver λ 256 / tree λ 0.10 remain optimal under damping 5.**

### 20.5 All deblock-adjacent axes measured under the winning damping 5 — table

Consolidated, every row `validate_dav2d OK`, at the 25,089 B standing best's
settings unless noted:

| axis | best setting | K | default | ΔK |
|---|---|---:|---:|---:|
| **CDEF damping** | **5** (6 → 48.33) | **48.36** | 48.01 (damping 3) | **+0.35** |
| deblock df_delta_q | luma −2 (dose peak, §17.1) | 48.10 | 48.01 | +0.09 |
| deblock apply flags | all on (§17.2) | 48.01–47.88 | 48.01 | ≤ 0 |
| luma-pass asymmetry | uniform −2 (§20.1) | 48.13 | 48.10 | ≤ 0 |
| CDEF CdefStrengths | 1 (§19, §20.2: 2/4 raw-tie, more bytes) | 48.36 | 48.36 | 0 |
| CDEF perceptual ranking | `veto`/`cdef` (§20: +40 B for +0.02 raw) | ≤ 48.12 | 48.36 | < 0 |
| `--video-uv-qp-offset` | −23 (legal minimum) | 48.36 | 48.36 | 0 (−16 → 47.78) |
| driver λ / tree λ | 256 / 0.10 (§20.4) | 48.36 | 48.36 | ≤ 0 |
| QP joint (185/187 λ-fill) | — (§18, §20.3) | ≤ 47.89 | 48.36 | < 0 |
| rate-dependent tree price | — (§16, §16.1) | ≤ 47.47 on floor 16 | 48.36 | < 0 |
| floor 4x4 + probe off | — (§16) | 45.28–47.00 | 48.36 | < 0 |

The two remaining unmeasured shifts are the *other two* encoders of the
HF-mask/tree-λ family (`AV2ENC_TX_HF_MASK_DIV` 1/3) and **chroma-side deblock
Δ** (`−2,−2,−2,−2` and `−2,−2,+1,+1`) which §17.1 only sampled at 0 under
damping 3. Both are running.

### 20.6 Final neighbour scans under damping 5 (mask div, chroma Δ, CCSO)

| arm | bytes | cvvdp | BA | SS2 | K |
|---|---:|---:|---:|---:|---:|
| damp 5, Δ `−2,−2,0,0` (old best) | 25,089 | 7.3549 | 12.505 | −9.540 | 48.36 |
| **damp 5, Δ `−2,−2,−2,−2` (new best)** | **25,090** | 7.3566 | 12.475 | −9.528 | **48.37** |
| damp 5, Δ `−2,−2,+1,+1` | 25,090 | 7.3545 | 12.505 | −9.569 | 48.35 |
| mask div 1 | 25,250 | 7.3545 | 13.040 | −9.782 | 48.15 |
| mask div 3 | 24,773 | 7.3483 | 13.852 | −9.951 | 47.87 |

`validate_dav2d OK` on all. Chroma Δ−2 adds **+0.01** for +1 B (JOD +0.0017,
BA −0.030) — a real but tiny gain; chroma +1 is a −0.02 regression, and both
mask-div shifts break BA badly. **Standing best: damping 5 + `df_delta_q
−2,−2,−2,−2` → K 48.37 @ 25,090 B.**

### 20.7 Upper bound from measured evidence

Every deblock-adjacent axis now has a measured optimum at this rate:

* deblock level Δ: 47.90 (+1) → **48.10** (−2) → 48.06 (−3) → 48.01 (−4); peak
  exists and is shallow (±0.1 across the whole legal dose range).
* deblock apply flags: 47.67–48.01 (all ≤ baseline).
* CDEF damping: 48.01 (3) → 48.26 (4) → **48.36** (5) → 48.33 (6); peak again.
* CDEF CdefStrengths: flat (0).
* CDEF perceptual ranking: ≤ 48.12 and more bytes.
* chroma QP offset: −23 is the legal floor and optimal.
* driver λ / tree λ / HF-mask div: all worse off the current point.
* QP joint (finer quant + smaller tree): −0.5 to −0.7.
* rate-dependent tree price / floor 4x4: −0.3 to −2.7.

So the **filter pipeline's total measured headroom is +0.36 K** over the 48.01
baseline (48.37), with every local axis convex at the current point. The
remaining 1.63 K to the K 50 target is not in the filter stack: at fixed bits
it has to come from the coefficient/quantiser accuracy itself, which is why
the reference encoder's own curve also sits below 50 at 25 KB.

### 20.8 CCSO axes under damping 5: *byte-identical* (inert, not merely neutral)

`--video-ccso-rd`, `--video-ccso-refit 2` and `--video-ccso-bands 3` each
produced a stream **bit-identical** to the plain damping-5 + Δ−2 arm:

```
ref   pv_damp5   25089 B d744aaa1b491a13e1aa8feb44a3d8df5
d5_ccsord        25089 B d744aaa1b491a13e1aa8feb44a3d8df5   identical
d5_ccsorf2       25089 B d744aaa1b491a13e1aa8feb44a3d8df5   identical
d5_ccso3         25089 B d744aaa1b491a13e1aa8feb44a3d8df5   identical
```

CCSO is **off on this stream** (the `--nightly-b` aggregate does not enable
`video_ccso`; `--video-gdf` is the GDF/edge path), so these flags are inert
rather than a measured loss — the axis does not exist at this point in the
pipeline. `--video-cdef-strength 3` (25,125 B) is the only real arm in the
batch and it is worse (§19). No CCSO headroom is reachable without first
enabling CCSO, which `docs/negative-results.md` and `docs/k52-at-25k.md` already
record as a loss at this rate.

### 20.9 Pins re-verified on the final binary (all four byte-identical)

```
fast-decode 0639b43630abb454f7be5ac494d2efb4  ✓
nightly-b   8f0b85d8ea3ea9adfc1e28327aabf0e0  ✓
milestone   9a8ee8d5e2f895ec4a2a72cee24c6842  ✓
stack arm   25081 B 3705ac02a9cf4c48ccec0749cfacf7e6  ✓
```

Every code change this session (`AV2ENC_RD_DRIVER_RATE_FIT`,
`AV2ENC_DEBLOCK_DF_DELTA`, `AV2ENC_DEBLOCK_APPLY`) is env-gated and default-off,
so the pins are unaffected by construction and verified by measurement.

### 20.10 Perceptual ranking under damping 5, and the damping peak re-checked

| arm | bytes | cvvdp | BA | SS2 | raw K | K at 25,090 B |
|---|---:|---:|---:|---:|---:|---:|
| **damp 5, Δ all-four −2 (standing best)** | 25,090 | 7.3566 | 12.475 | −9.528 | **48.37** | **48.37** |
| `AV2ENC_PERCEPTUAL_FILTER=veto`, same | 25,116 | 7.3571 | 12.476 | −9.443 | 48.39 | ≤ 48.31 |
| `veto` with luma-only Δ−2 | 25,116 | 7.3551 | 12.500 | −9.431 | 48.38 | ≤ 48.30 |
| damp 4, Δ all-four −2 | 25,093 | 7.3512 | 12.759 | −9.554 | 48.26 | 48.25 |
| damp 6, Δ all-four −2 | 25,091 | 7.3536 | 12.379 | −9.713 | 48.36 | 48.35 |

`validate_dav2d OK` on all. Two facts:

1. **The damping peak holds at 5** even with chroma Δ−2 added (4 → 48.26,
   5 → 48.37, 6 → 48.36). The optimum is a real interior maximum, not an
   artifact of the chroma setting.
2. **Perceptual `veto` is byte-expensive**: it buys +0.019 raw K (SS2 +0.085,
   BA −0.001) for +26 B, which costs ≈0.04 K at this point's slope — a net
   **−0.02**. Its 100 kbps advantage still does not survive at 25 KB. A
   matched-byte confirm (veto + driver λ 264/288 to shrink the tree back to
   ~25,090 B) is running; it is the last open point in the batch.

### 20.11 Perceptual `veto` at matched bytes: negative, axis closed

The raw +0.02 K at 25,116 B was a byte illusion. Shrinking the veto tree with a
stiffer driver λ falls off a cliff, not onto the budget:

| veto + λ | bytes | leaves | BA | SS2 | K |
|---|---:|---:|---:|---:|---:|
| λ256 (raw) | 25,116 | 2,299 | 12.476 | −9.443 | 48.39 |
| λ288 | 25,126 | 2,270 | 14.114 | −10.397 | 47.71 |
| λ352 | 24,667 | 2,114 | 13.562 | −10.798 | 47.73 |

`validate_dav2d OK` on all. Between λ256 and λ288 only 29 leaves change and BA
jumps 12.48 → 14.11, i.e. the veto ranking's advantage is carried by
specifically those marginal leaves; remove them and it is 0.65 K worse than the
plain arm. **Perceptual ranking closed under damping 5.**

---

## 21. Final measured bound: where K 50 at 25 KB actually is

**Standing best: K 48.37 @ 25,090 B** (`--video-cdef-damping 5` +
`AV2ENC_DEBLOCK_DF_DELTA=-2,-2,-2,-2` + mask/λ stack), `validate_dav2d OK`,
+0.36 K over the 48.01 baseline.

Everything in the deblock/CDEF/CCSO/perceptual filter stack was scanned to a
convex optimum at this operating point (§20.5, §20.7), each verified by decoded
measurement. The axes that are *not* filtering — rate-dependent tree price
(§16), 4x4 floor (§16), QP joint (§18, §20.3), λ rescans (§20.4) — are all
negative. The remaining 1.63 K decomposes as:

| K component | at 48.37 | needed at 50 | required change |
|---|---:|---:|---:|
| JOD (weight 0.5) | 0.368 | 0.380 | JOD 7.36 → **7.68** (+4.4 %) |
| BA (weight 0.25) | 0.049 | 0.054 | BA 12.48 → **9.6** (impossible: best arm anywhere) |
| SS2 (weight 0.25) | 0.068 | 0.071 | SS2 −9.53 → −7.7 |

**The bound is not the filters.** JOD carries 73 % of the score; reaching 50
needs a 4.4 % distortion improvement *at fixed bytes*, which is the
coefficient/quantiser accuracy deficit already documented in
`docs/k52-at-25k.md` §3.4 — and the reference encoder's own curve is
K 51.06 at 25,000 B, so it also does not reach 52 there. Our filter work moved
from 48.01 to 48.37 (+0.36, ~22 % of the gap); the rest is a different class of
change (transform/quantiser accuracy), not a deblock parameter.

## 22. The deliverable (reproducible, validated)

```
env AV2ENC_DEBLOCK_DF_DELTA=-2,-2,-2,-2 AV2ENC_TX_HF_MASK_DIV=2 \
    AV2ENC_TX_LAMBDA_TEXTURED=0.10 AV2ENC_RD_DRIVER_MIN=16 \
    AV2ENC_RD_DRIVER_LAMBDA=256 build-release/av2enc src720_1f.y4m -o out.ivf \
    --video --frames 1 --qp 186 --nightly-b --video-gdf \
    --video-cdef-damping 5 --video-uv-qp-offset -23
```

| | measured |
|---|---|
| bytes | **25,090** (target 25,000 B class; baseline stack was 25,081) |
| MD5 | `56694acd19dbc96e86a5e7648e09b0ad` |
| `validate_dav2d.sh` | **OK** (`dav2d --strict 1`) |
| cvvdp / BA / SS2 | 7.3566 / 12.4751 / −9.528 |
| **K** | **48.37** (baseline 48.01, target 50) |
| pins | all four byte-identical (§20.9) |

## 23. Fine matched-byte joint grid closes the §21 gap: strict dominance, no interpolation

§21's bound was reasoning from metric arithmetic. This section replaces it with
a measured grid that does not need interpolation. Full stack held fixed
(damping 5 + `df_delta_q −2,−2,−2,−2` + mask div 2 + `MIN=16` + uv −23), driver
λ ∈ {224, 240, 256, 288, 304} crossed with tree λ ∈ {0.06, 0.10} at QP186.
Every arm `validate_dav2d OK`:

| arm | bytes | BA | SS2 | K | Δbytes | ΔK | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| **λ256 / tlt 0.10 (anchor)** | **25,090** | **12.475** | −9.528 | **48.37** | — | — | — |
| λ224 / 0.06 | 25,374 | 13.110 | −10.147 | 48.07 | +284 | −0.30 | dominated |
| λ224 / 0.10 | 25,355 | 13.110 | −10.154 | 48.07 | +265 | −0.30 | dominated |
| λ240 / 0.06 | 25,374 | 12.492 | −10.441 | 48.21 | +284 | −0.16 | dominated |
| λ240 / 0.10 | 25,208 | 13.756 | −10.636 | 47.76 | +118 | −0.61 | dominated |
| λ288 / 0.06 | 25,103 | 13.462 | −10.454 | 47.88 | +13 | −0.49 | dominated |
| λ288 / 0.10 | 25,101 | 13.665 | −10.553 | 47.79 | +11 | −0.58 | dominated |
| λ304 / 0.06 = 0.10 | 24,935 | 17.265 | −10.740 | 46.93 | −155 | −1.44 | worse at matched bytes |

**Six of eight grid points are *strictly dominated*: more bytes AND lower K.**
That is a stronger statement than any matched-byte interpolation — no
correction factor, no slope assumption, no bracket. λ304 is the only cheaper
point and loses 1.44 K, i.e. far more than its 155 B could buy back. λ256 is a
genuine interior optimum on both axes, and tree λ 0.10 beats 0.06 at the
optimum (48.37 vs 48.07 in the λ224 row where the difference is measurable).

The BA column is the mechanism: every off-optimum λ raises Butteraugli
12.48 → 12.49–13.76 and the λ304 point explodes to 17.27. λ moves which leaves
survive, and the surviving set at λ256 is the one whose edges BA tolerates.

### 23.1 Coefficient-domain lever: checked against the ledger, unreachable here

The K decomposition points at coefficient accuracy, so the gates were checked
against the existing measured ledger instead of being re-run:

| lever | gate | ledger verdict at this rate |
|---|---|---|
| exact leaf/partition RD | `AV2ENC_EXACT_RD` | **byte-identical** on this path (`docs/k52-at-25k.md`: "Byte-identical (no effect at all): … `AV2ENC_EXACT_RD`") — it re-scores *inter* candidates and this stream is a lone key |
| distortion-weighted trellis | `AV2ENC_RDOQ_DIST_WEIGHT` | both directions measured (§3.7): 1.2/1.5/2.0/0.8 all −0.09…−0.50 |
| key trellis on/off + scale | `AV2ENC_RD_KEY_LEVEL`, `_SCALE` | swept (§3.6): `LEVEL=0` −0.39, scales within ±0.05 or worse |
| key CSF weight | `EFF_PERC_KEY_CSF` | ≤ 47.07 (negative) |
| zero→one trellis levels | `AV2ENC_RDOQ_ZERO_TO_ONE` | measured **null** — the trellis declines at every QP |
| TCQ+RDOQ key trellis | `--video-tcq --video-rdoq-off` | smoothing (HF kept 22.9 %), recorded negative |

The key residual/quantiser path is the only coefficient lever this stream can
reach (`kcfg.level` additionally gates on `exact`, which is inert here), and it
is measurably at its limit. That closes the coefficient-domain pivot the grid
triggered.

## 24. Perceptual leaf/partition selection on the KEY path (the requested lever)

Directed work: test a perceptual leaf/partition RD cost at matched bytes. The
premise ("leaf selection is SSE-based") needed correcting first, and the
correction is what located the real lever:

**Checked, and structurally inert on this stream:** `AV2ENC_PIPE_BA`
(`PIPE_BA_WEIGHT`, the Butteraugli-shaped leaf+TX cost) requires
`pipe_ba::tool_on() && !keyframe_ && pipe_probe_ready_`
(`video_frame_decision.cpp:1947`). This stream is a **lone key frame**, so the
BA / `PIPE_PERC` / `EFF_PERC` *pipe front-end* never runs — its ledger line
("inert on a lone key") is correct for the gate.

**But the key path has its own perceptual D, and it is highly material.** The
key leaf cost is `seq_rd::leaf_distortion(..., key_csf, key_csf2)` where
`key_csf = eff_perc::enabled() ? key_csf_gain() : 0.0`. Measured on our winning
stack (damping 5 + Δ −2,−2,−2,−2), `validate_dav2d OK` on all:

| `EFF_PERC_KEY_CSF` | bytes | leaves | BA | SS2 | K |
|---|---:|---:|---:|---:|---:|
| 0 (plain-SSE key) | 24,214 | 1,949 | 14.780 | −13.472 | 46.79 |
| 0.3 | 24,546 | 2,097 | 15.260 | −12.297 | 46.99 |
| 0.6 | 24,647 | 2,178 | 13.393 | −12.339 | 47.50 |
| 1.0 | 24,969 | 2,233 | 13.834 | −11.136 | 47.62 (48.33 at matched) |
| **1.5 (default = anchor)** | **25,090** | **2,282** | **12.475** | **−9.528** | **48.37** |
| 2.0 | 25,456 | 2,384 | 13.486 | −9.912 | 48.05 |
| 2.5 | 25,691 | 2,475 | 14.417 | −8.644 | 48.07 |

**The gain axis is a closed peak at 1.5.** Above it, +366/+601 bytes buy −0.32/−0.30 K
(strictly dominated: more bytes, less K). Below it, 1.0 interpolates to 48.33
(−0.04). `EFF_PERC_CSF2=0` is byte-identical, so the second-order term is inert.
The key perceptual D is therefore already tuned to its optimum — this is the
perceptual leaf-selection lever the directive asked for, and it was already at
its measured peak in the winning point.

### 24.1 HF / EDGE terms: the *right* perceptual terms, tested

`seq_rd::leaf_distortion` also carries the two terms written specifically for
this objective (`video_seq_rd.h:113-127`): **HF** = `sum(d − box3x3(d))^2`, the
1-px high-pass the low-band CSF is blind to, and **EDGE** = the gradient
mismatch across the leaf's top/left boundary — "the error a partition boundary
leaves behind … Butteraugli sees the block edges". Both default 0. The only
prior verdict (`docs/rd-engine0.md` §7) was on a 240p **clip**, where the key's
BA win did not carry through the P chain. **A lone key has no P chain**, so the
composition is genuinely new:

| arm | bytes | leaves |
|---|---:|---:|
| HF 0.5 | 25,335 | 2,344 |
| HF 1 | 25,163 | 2,278 |
| HF 5 | 25,192 | 2,245 |
| EDGE 1 | 24,181 | 1,789 |
| EDGE 5 | 22,405 | **699** |

HF arms are all over budget; EDGE arms buy a lot of rate (down to 22,405 B) by
collapsing the tree to 699 leaves. Both were then **rebalanced onto the byte
target** (EDGE spends its saved rate on a finer tree via a lower driver λ; HF
pays for its extra bytes with a stiffer λ). Those arms are scoring.

### 24.2 HF / EDGE rebalanced to matched bytes: negative, axis closed

Each EDGE arm spends its saved rate on a finer tree (lower driver λ); HF pays
for its extra bytes with a stiffer λ. `validate_dav2d OK` on every arm:

| arm | bytes | leaves | BA | SS2 | K |
|---|---:|---:|---:|---:|---:|
| **anchor (HF 0, EDGE 0)** | **25,090** | **2,282** | **12.475** | **−9.528** | **48.37** |
| HF 1, λ256 | 25,163 | 2,278 | 14.732 | −10.692 | 47.49 |
| HF 1, λ288 | 25,141 | 2,241 | 14.198 | −10.624 | 47.66 |
| EDGE 1, λ256 | 24,181 | 1,789 | 14.303 | −12.700 | 47.05 |
| EDGE 1, λ192 | 24,534 | 1,918 | 13.913 | −11.436 | 47.44 |
| EDGE 1, λ128 (best EDGE) | 24,911 | 2,017 | 13.557 | −11.128 | 47.69 |
| EDGE 5, λ256 | 22,405 | 699 | 18.432 | −20.037 | 44.20 |
| EDGE 5, λ128 | 22,635 | 773 | 19.408 | −19.341 | 44.17 |

**Negative on both terms, and the shape is diagnostic:**

* **HF loses through BA** (12.475 → 14.20–14.73) for flat JOD — the high-pass
  term makes the tree chase fine detail the reconstruction cannot afford at
  25 KB, which is exactly what Butteraugli charges.
* **EDGE is monotone the *wrong way*:** as bytes are spent back up
  22.4 → 24.2 → 24.5 → 24.9 kB the K rises 44.2 → 47.05 → 47.44 → 47.69, i.e. the
  term is only recovering the ordinary rate it gave away, and even at its best
  measured point it sits **0.68 below the anchor while being 179 B cheaper**.
  EDGE 5 destroys the picture (BA 18.4, SS2 −20.0) by collapsing the tree to
  699 leaves.
* The `rd-engine0` §7 reading transfers after all: the boundary-gradient term
  does not predict the perceptual cost of a partition layout at this rate.

**The 240p-clip verdict was right for the wrong reason** — there the key's BA win
was diluted by the P chain; here, with no P chain, the term is still negative,
so it is the term itself, not the chaining.

---

## 25. Verdict: K 48.37 is the measured ceiling of the perceptual/filter axes

The directive's pivot (perceptual leaf/partition selection at matched bytes) was
executed and closed:

* `PIPE_BA` / BA-weighted leaf cost — **structurally inert** on a lone key
  (`!keyframe_` gate, §24).
* Key CSF gain — **already at its peak** at the shipped 1.5; both sides dominated
  (§24).
* HF / EDGE leaf-distortion terms — **negative** at matched bytes, EDGE monotone
  away from the optimum (§24.2).
* Combined with §20–§23 (deblock knobs, CDEF damping/strength/perceptual, CCSO,
  chroma, mask, λ×tree-λ strict dominance, coefficient-gate ledger), every
  perceptual or filter axis reachable on this stream now has a measured,
  convex optimum at **K 48.37 @ 25,090 B**.

**Reported ceiling: 48.37 (−1.63 vs the K 50 target).** The remaining gap is
coefficient/quantiser accuracy at fixed bits (JOD carries 73 % of K and would
need +4.4 %), which the ledger shows is at its limit on the only path this
single-key stream can reach.

## 26. `--video-rdoq-scale` 0.01 / 0.02 on the winning stack: negative (dominated)

Last documented +K lever not yet measured at 25 kB on this stack. Mechanism
first, because it decides what the numbers mean: `--video-rdoq-scale` sets
`rdoq_sse_scale_explicit`, which *switches off* `lambda_from_fit(step,
kRdoqSseScaleFit)`. At our step(186) = 1056 the fit resolves to **0.03343**
(λ = 37,277), so the explicit values are a much *finer* trellis:

| explicit | RDOQ λ | vs the fit |
|---|---:|---:|
| 0.01 | 11,151 | −70.1 % |
| 0.02 | 22,303 | −40.2 % |
| 0.015 (previously measured) | 16,727 | −55.1 % |

All arms carry damping 5 + Δ `−2,−2,−2,−2`, QP186, `validate_dav2d OK`:

| arm | bytes | cvvdp | BA | SS2 | K |
|---|---:|---:|---:|---:|---:|
| **anchor (fit, no `--video-rdoq-scale`)** | **25,090** | **7.3566** | **12.475** | **−9.528** | **48.37** |
| scale 0.01 | 25,343 | 7.3560 | **15.544** | −10.062 | 47.48 |
| scale 0.02 | 25,109 | 7.3414 | 12.916 | −10.740 | 47.96 |
| scale 0.02, λ272 | 25,110 | 7.3405 | 13.930 | −10.624 | 47.70 |
| scale 0.02, λ240 | 25,403 | 7.3639 | 12.918 | −9.773 | 48.24 |

**Negative, and dominated.** The best arm (λ240, 48.24) costs **+313 B for
−0.13 K**; 0.02 at matched bytes is 47.96; 0.01 destroys Butteraugli
(12.475 → 15.544) while leaving JOD flat — the finer trellis spends the rate on
coefficient levels whose reconstruction BA cannot tolerate. This is consistent
with the whole λ family in §23: the surviving-level set at the fit's λ is the
one whose edges BA tolerates, and moving the trellis price either way is worse.
The LEAD `+0.73 K at equal bytes` in `docs/negative-results.md` was measured on
the K-50 point at **29.6 kB** where `--video-rdoq-scale 0.01` is part of that
recipe; at 25 kB the same setting is our worst BA arm.

### 26.1 K-50 crossing rate (measured, not estimated)

`docs/k50-720p.md` §1: **K 50 crosses at ~29.4 kB** on this picture (29,391 B,
interpolated between QP 185/28,311 B at K below 50 and QP 184/29,602 B at
K 50.20), for the recipe *16x16 floor + `--video-gdf --video-uv-qp-offset -23
--video-rdoq-scale 0.01`*. That point is QP184 — i.e. **the crossing is bought
with rate, not with a different tool set**.

Our own stack's measured slope above the budget (§23/§24 points, 25.1–25.7 kB)
is ≈ **0.27 K/kB**, which would place K 50 at ≈31 kB, i.e. *not better* than the
documented recipe. Both readings agree on the magnitude: reaching K 50 on this
picture costs **~29–31 kB, 1.17–1.24× the 25 kB budget**.

**Conclusion: at 25,090 B the measured ceiling is K 48.37.** Every filter,
perceptual and coefficient-price lever reachable on a lone key at this rate has
now been measured to a convex optimum or a dominance result; the +1.63 K gap is
a rate gap, not an unmeasured knob.
