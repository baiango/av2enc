# RD key partition floor — the driver's 4x4 expansion at low rate

Status: **measured KEEP at one operating point; env-only, default off; the
ladder is not run.** Flooring the key driver's prune at 16x16
(`AV2ENC_RD_DRIVER_MIN=16`, default 4) moves a single 720p still at **25,010 B**
from **K 31.84 to 46.94** (+15.10, closing 81.9 % of the gap to the AVM
reference) and cuts the encode from **69 s to 16 s**. Unset the variable and
every pinned stream is byte-identical. This is a measurement at 0.21 bpp, not a
promotion: the 240p `--nightly-b` ladder has not been run.

Provenance: measured 2026-11-24/25 on `~/Downloads/src720.png` (1280x720 RGB)
converted with ffmpeg to `src720_1f.y4m` (1 frame, yuv420p, limited range,
bt709, md5 `80708336148219e2d20c502a43675f94`, 1,382,485 B). Encoder
`build-pub/av2enc`, built 2026-11-24 21:59 from the tree at `7beb8543`
(`50491f73` landed later and is byte-identical per its own pin record); CPU
`--video`, `--cpu-threads 8`, Apple M4. Harness, every stream, every decode and
every score live outside the tree in `~/Downloads/av2enc-src720/`
(`sweep.py`, `rescore.py`).

## 0. Answer first

1. **The defect is the key driver's expansion floor, not the quantiser.**
   `seq_driver_decide_sb` expands each reference leaf to a uniform `target`
   (`AV2ENC_RD_DRIVER_MIN`, default 4) and only then prunes. At 0.21 bpp that
   puts **23,569 leaves** on a 1280x720 frame (98 per superblock) and
   **61.0 % of the tile is per-leaf syntax** — intra mode 47.5 % + partition
   13.5 % — against **38.7 %** coefficients. At the same image's high-rate
   key (`docs/keyframe-bit-accounting.md`, QP120, 1.116 bpp) the identical
   meter reads **1.45 %** syntax and 98.1 % coefficients: a **42x** separation.
2. **Flooring the prune at 16x16 inverts the split.** Same image, same
   25,010 B: coefficients **81.4 %**, syntax 18.3 %, leaves 2,935, driver
   trials 139,107 -> 7,543, TX mix `tx4` 17,540 -> **0**. The same bytes now
   buy **QP187 instead of QP229**.
3. **It is worth +15.10 K and 4.2x speed.** K 31.84 -> **46.94** (AVM p6
   reference 50.27); wall 69 s -> **16 s**.
4. **It is not a price you can tune away.** `AV2ENC_RD_DRIVER_LAMBDA` scales
   only the driver's tree lambda (not the TX search): x4 -> +0.87 K, x16 ->
   +2.28 K, moving the 25 KB point QP229 -> 219 while the floor moves it to
   187. Going the other way (`--video-force-single`, 64x64) gives **43.60**,
   3.34 K *below* the 16x16 floor. The win is specifically *floor at 16 and
   then let the driver decide*, not *coarser is better*.
5. **One operating point.** 1280x720, one frame, 0.216 bpp. The ladder's key
   sits near 1.1 bpp where the meter says syntax is 1.45 %, so this arm's
   trigger should not fire there — but that is an inference from the meter,
   not a measurement.

## 1. The knob

| Variable | Default | Meaning |
|---|---|---|
| `AV2ENC_RD_DRIVER_MIN` | 4 | deepest uniform leaf the key prune expands to (px), clamped to `[4, 64]`; unset or out of range reads 4 |
| `AV2ENC_RD_DRIVER_LAMBDA` | 1.0 | scale on the driver's own tree lambda; measured weak (see §3) |

One call site: `rd_driver::prune_target()` is read only at
`src/pipeline/video/video_frame_decision.cpp:6453`, inside
`seq_driver_decide_sb`. It does not touch the P-frame driver, the historical
solver, the TX search or any writer path. No CLI, preset, pin or CMake change;
the variable is documented in `docs/rd-engine0.md` §5 as "the coarser
ablation".

The expansion, verbatim:

```text
target    = rd_driver::prune_target()                 // AV2ENC_RD_DRIVER_MIN
max_depth = max(options_.max_partition_depth, seq_depth_for_target(target))
...
if (node is a leaf && bw <= target && bh <= target) return node;
return seq_build_uniform(..., target, max_depth, kMinBlockPixels, can_split);
```

Two consequences worth recording because they explain the prior art in §5:

* `seq_depth_for_target(4) == 8` (64 -> 4 is four halvings at `depth += 2`
  each), so with the default floor the driver's depth cap is
  `max(--video-max-depth, 8)`. **`--video-max-depth 0..7` cannot lower the
  driver's expansion**; it can only raise the cap above 8.
* `--video-min-block` is not read anywhere in the driver (it sets
  `search_min_block` for the historical solver only). Neither flag reaches
  this knob.

## 2. What was measured

Recipe, one frame each:

```sh
# baseline
build-pub/av2enc src720_1f.y4m -o out.ivf --video --frames 1 --qp 229 \
                 --cpu-threads 8 --nightly-b                      # 25,010 B
# the arm
AV2ENC_RD_DRIVER_MIN=16 build-pub/av2enc src720_1f.y4m -o out.ivf \
                 --video --frames 1 --qp 187 --cpu-threads 8 --nightly-b  # 24,886 B
```

`--nightly-b` on a single picture takes `apply_video_nightly_b`'s
`single_picture` path: BAWP, compound, TIP, CWP depth, the P partition probe
and the joint solve are all **off** (they need a chain), so this measures the
key encoder alone.

### 2.1 Rate curve

| arm | QP -> bytes |
|---|---|
| baseline | 200: 36,902 · 210: 31,898 · 220: 27,845 · 228: 25,144 · **229: 25,010** · 230: 24,812 · 231: 24,111 · 232: 23,595 · 234: 23,452 · 236: 22,998 |
| `MIN=16` | 185: 26,765 · 186: 26,040 · **187: 24,886** · 188: 24,123 · 189: 23,080 · 197: 17,454 · 205: 13,004 · 213: 9,865 · 221: 7,617 · 229: 6,006 |

The floor costs the baseline **42 QP steps** at fixed rate.

### 2.2 Arms, all interpolated to 25,010 B

Interpolation is linear **in bytes** between the two points that bracket the
target, then every metric is interpolated and K recomputed from the
interpolated triple (`rescore.py`). Arms are `--nightly-b` plus the one change
in the arm column.

| arm | QP | JOD | BA | SS2 | PSNR-Y | **K** | dK |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline (`MIN=4`) | 229 | 5.877 | 27.56 | -62.09 | 21.886 | **31.84** | — |
| `+ --video-uv-qp-offset 0` | 229 | 5.885 | 27.61 | -62.30 | 21.897 | 31.88 | +0.04 |
| `+ --video-tx-context-rate` | 229 | 5.877 | 27.56 | -62.09 | 21.886 | 31.84 | 0.00 |
| `AV2ENC_RD_DRIVER_LAMBDA=4` | 226 | 6.017 | 25.60 | -59.91 | 22.118 | 32.71 | +0.87 |
| `AV2ENC_RD_DRIVER_LAMBDA=16` | 219 | 6.279 | 24.56 | -53.78 | 22.553 | 34.12 | +2.28 |
| `AV2ENC_RD_DRIVER_MIN=8` | 198 | 7.007 | 15.27 | -26.66 | 23.818 | 43.03 | +11.19 |
| **`AV2ENC_RD_DRIVER_MIN=16`** | 187 | **7.309** | 14.20 | **-13.82** | 24.417 | **46.94** | **+15.10** |
| `AV2ENC_RD_DRIVER_MIN=32` | 184 | 7.317 | 14.85 | -14.85 | 24.437 | 46.65 | +14.81 |
| `--video-force-single` (64x64, driver off) | 182 | 7.191 | 23.45 | -21.18 | 24.176 | 43.60 | +11.76 |
| `--fast-decode --video-key-qp-offset-off` | 187 | 7.239 | 14.18 | -22.19 | 24.422 | 45.20 | +13.36 |
| *AVM `e6f99d3` p6 reference* | 171 | *7.653* | *13.78* | *-4.76* | *25.644* | *50.27* | *+18.44* |

Coarseness is monotone up to a plateau and then falls off: 4 -> 31.84,
8 -> 43.03, **16 -> 46.94**, 32 -> 46.65 (inside the ripple), 64 -> 43.60. The
plateau is 16..32; both edges are measured.

`--video-tx-context-rate` produced a **byte-identical** stream at QP229
(same size, same metrics), so it is inert on this path.

### 2.3 Gap decomposition

| component | baseline -> AVM | share | `MIN=16` -> AVM | share |
|---|---:|---:|---:|---:|
| CVVDP/JOD | 8.88 | 48 % | 1.72 | 52 % |
| Butteraugli | 2.02 | 11 % | **0.11** | **3 %** |
| SSIMULACRA2 | 7.54 | 41 % | 1.51 | 45 % |
| **total K** | **18.44** | | **3.34** | |

Butteraugli is essentially closed (`q_ba` 4.36 against AVM's 4.47) and SS2 is
off its -50 floor. What is left is residual fidelity, not artifact structure.

### 2.4 Speed

`/usr/bin/time -p`, two reps each, `-o /dev/null`, 8 threads:

| arm | wall | driver leaf trials |
|---|---:|---:|
| baseline QP229 | 70.76 / 67.09 s | 139,107 |
| `MIN=16` QP187 | **16.32 / 16.24 s** | 7,543 |

4.2x faster. The 18x trial reduction does not map 1:1 to wall time because the
floor arm does far more real coding per byte (QP187 against QP229, 81 %
coefficients against 39 %) — yet it still finishes 4.2x sooner, which is how
much of the baseline's 69 s is the 4x4 expansion search rather than coding.

## 3. Mechanism — where the bits go

`AV2ENC_SYMBOL_COST=1` (stream-neutral; verified byte-identical with and
without, §6). `raw_bits` are the residual-family literals (EOB escape/extra,
coefficient magnitude, sign — `fine_for()` only tags families whose coarse tag
is `kCostResidual`), so coefficients = `residual` + `raw_bits`.

| | baseline QP229 (tile 24,826 B) | `MIN=16` QP187 (tile 24,723 B) |
|---|---:|---:|
| `info_bits` | 198,441 | 197,530 |
| `intra_mode` | 94,259 (**47.5 %**) | 34,173 (17.3 %) |
| `partition` | 26,741 (**13.5 %**) | 1,974 (1.0 %) |
| `residual` (coefficient CDFs) | 18,868 (9.5 %) | 116,831 (59.1 %) |
| `raw_bits` (coefficient literals) | 57,928 (29.2 %) | 43,941 (22.2 %) |
| `ccso_blk` | 645 (0.3 %) | 610 (0.3 %) |
| **coefficients** | **38.7 %** | **81.4 %** |
| **syntax (mode + partition)** | **61.0 %** | **18.3 %** |
| coded leaves | 23,569 | 2,935 |
| `key-tx-rd` TX mix | tx4 17,540, tx8 1,951 | tx4 **0**, tx8 2,904 |
| `rd-driver` leaf trials | 139,107 | 7,543 |

At baseline the syntax line is **121,000 bits = 15,125 B = 60.5 % of the
25,010 B stream** (5.13 bits of mode+partition per leaf x 23,569 leaves).
Flooring the prune does not make each leaf cheaper — per-leaf syntax actually
rises to 12.3 bits — it makes there be 8x fewer of them, which is what frees
the budget for coefficients and for a 42-step finer quantiser.

This is the opposite regime from the one every other key measurement in the
tree was taken in: `docs/keyframe-bit-accounting.md` reads 98.14 % coefficients
/ 1.22 % mode / 0.23 % partition at 1.116 bpp. The two operating points differ
by 42x in syntax share, which is the whole reason a fixed floor can be right at
one and wrong at the other (§7).

## 4. Conformance

* `tools/validate_dav2d.sh dmin16_q187.ivf src720_1f.y4m` -> **exit 0**,
  `decoder: f9dafbc`, `nb_read_frames=1`, PSNR y 24.406091, tool pin
  `d17e7cf7…` (ok). This is a single-picture stream, not a chained multi-GOP
  one, so it is inside the `dav2d` gate as `docs/validation.md` writes it; the
  `avmdec` exception is for chained streams only.
* `avmdec --i420 --summary` -> 1 decoded frame; a second `dav2d --strict 1`
  decode is byte-identical to the Y4M that was scored.
* Baseline reproduces **byte-identical across independent runs**:
  `q229.ivf` and the separately instrumented `sym_q229.ivf` are both
  `c927f2be04713674b26b628092baeaf9`. The arm likewise:
  `dmin16_q187.ivf` and `sym_min16_q187.ivf` are both
  `6e0f1c9c9e8a6e35dfb89ed1abe71d31`.
* `AV2ENC_RD_DRIVER_MIN` unset is the current behaviour; no pin, preset, CLI,
  `CMakeLists.txt` or `main.cpp` edit was made. **Not re-run this session:**
  `ctest`, the five pinned streams and the `AGENTS.md` reference bitstreams
  (`c9294fa4…`, `0d90c958…`) — they should be run before any promotion, since
  the knob is env-only but non-default-inert only when unset.
* Not run, and the blocking gate for promotion: the 240p `--nightly-b` ladder
  (88.2 K), validated with `tools/validate_avmdec.py` — pinned `dav2d` f9dafbc
  has the §7.21.6 drain defect on chained multi-GOP streams.

## 5. Prior art — do not repeat

| source | what it measured | why it is not this |
|---|---|---|
| `docs/tool-improvement-queue-3.md` rows 10-11 | "coarser key tree/TX (depth 3)" 23,964 B / 36.831 dB and "depth 2 / depth 1" 21,128 B / 30.75 dB — **negative (probe only)** | `detail_edges` probe key at ~1.1 bpp via `--video-max-depth`, which per §1 **cannot lower the driver's expansion** (`max(max_partition_depth, seq_depth_for_target(4)) = max(D, 8)`), and no natural-content K was measured (that column is `—`). It did not measure `AV2ENC_RD_DRIVER_MIN`. What `--video-max-depth` did change on that probe is **not established here** and is an open question, not a contradiction. |
| `docs/rd-key-dp.md` | full §5.20.3 tree DP on the key driver: **Phase 1 KILL**, 0/32 superblocks improve, its own tree +10.71 % | that tests *shape reorientation* (HORZ vs VERT at a node). This arm changes *how deep the expansion goes*, which the DP inherited unchanged. |
| `docs/rd-joint-dp.md` | P-frame full DP, KILL at 0.155 % J | P frames; this is the key driver. |
| `docs/rd-engine0.md` §4, `AV2ENC_RD_DRIVER_ACT` | source high-pass gate that *stops* sub-16 descent in flat regions | a per-block activity screen; measured default-off. Complementary, not the same control. |
| `docs/negative-results.md` | `AV2ENC_RD_HEAD_PART`, forced band descent, `AV2ENC_RD_TAIL_ACT` | all force *finer* trees or open the **P** probe's clamp; this arm forces *coarser* on the **key** driver. |

## 6. Harness traps (both produced plausible wrong numbers)

1. **Interpolate in bytes, never in QP order.** Pairing "last QP at or below
   target" with "first QP at or above target" *after sorting by QP* pairs the
   two far ends whenever the walk jumps: the `MIN=16` first pass paired
   6,006 B with 26,765 B and reported K 45.97 instead of 46.94. Sort by size,
   take the largest `<= target` and the smallest `>= target`. This is why
   `rescore.py` exists separately from `sweep.py`.
2. **A walk that never brackets must fail loudly, not report its nearest
   point.** The first `--fast-decode` pass ran out of iterations at 15,081 B
   and emitted a row labelled 25,010 B; it was 40 % under target. The harness
   now sets `"bracketed": false` and prints a warning.
3. `avmenc` fed a y4m is a documented trap (`docs/detail-gap-720p.md`); it did
   **not** fire here — the raw-I420 feed with `-w 1280 -h 720` gives the same
   24,204 B and bit-identical PSNR (25.644366 / 31.242321 / 36.631637 /
   27.035779) as the y4m feed, so both AVM arms are valid.

## 7. Design — a rate-adaptive floor (not implemented)

A fixed 16 is right at 0.21 bpp and, on the evidence in §5, wrong near 1.1 bpp.
The goal is a control that is *coarse when syntax dominates and neutral when it
does not*, with `AV2ENC_RD_DRIVER_MIN` kept as the explicit override.

### 7.1 The constraint any design must satisfy

1. 0.21 bpp (this still): behave like `MIN=16` — K >= 46.9, tile syntax share
   <= ~0.19, `tx4` count 0.
2. ~1.1 bpp (ladder key, queue-3 probe): behave like `MIN=4` — **byte-identical**
   to today's pins.
3. Off by default; unset byte-identical.

### 7.2 The trigger signal already exists

`AV2ENC_SYMBOL_COST=1` meters the exact split with no new instrumentation, and
the driver already separates the two terms inside
`seq_driver_measure_leaf`:

```text
syntax = intra_luma_syntax_bits(...) + intra_chroma_syntax_bits(...)
       + (tx_part_count ? 2 : 0) + lambda * 2.0   // per split
coeff  = the calibrated plan rate over the emitted levels
cost   = D + lambda * (coeff + syntax)
```

so a share is computable from quantities the prune already has. The two regimes
are separated by **42x** (61.0 % against 1.45 %), so any threshold in roughly
`[0.05, 0.55]` selects this arm and leaves the high-rate regime with
`kappa = 1` exactly. `S_max = 0.30` sits 20x above the high-rate value and 2x
below the low-rate value — not a knife edge.

### 7.3 Experiment 0 — is it only price? (cheapest falsification, ~10 min)

`AV2ENC_RD_DRIVER_LAMBDA` scales *only* the driver's tree lambda, so it is a
clean tree-only price with no TX side effects. Measured x4 -> +0.87 K and
x16 -> +2.28 K. Sweep **64 / 256 / 1024** at 25,010 B.

* If one reaches >= 46, the defect is the `key_tx_rd_lambda * 0.25`
  calibration (`docs/rd-engine0.md` §3, fitted at QP112) and the fix is a
  recalibration — no new policy, no new state, smallest possible change.
* If it plateaus below the hard floor — plausible, because monotone lambda
  drives the tree toward the reference/collapse, which measures 43.60 — then
  price alone cannot do it and §7.4 is required.

Run this first: it decides between "calibrate" and "new policy" for the cost of
one ladder-free sweep.

### 7.4 D1 — syntax-share surcharge in the prune (recommended if 7.3 fails)

Keep the driver's measured `D + lambda*R` and tax only the syntax half once the
tree's syntax share exceeds a budget:

```text
share  = sum_syntax / (sum_syntax + sum_coeff)     // over committed leaves, raster order
kappa  = 1 + alpha * max(0, share - S_max) / S_max
cost   = D + lambda * (coeff + kappa * syntax)
```

* `S_max` default **0.30**, `alpha` default **2.0**; both env-gated, default
  off -> byte-identical.
* **Causal, one pass:** the walk is already sequential in raster order, so the
  share of the leaves committed so far in the superblock is available at the
  node being decided. Prefer this over a two-pass meter-then-fix version; if
  the running share proves too noisy, fall back to a cheap first pass that only
  meters (the arm is already 4.2x faster than baseline, so 2x is still ~2x
  net).
* **Why a surcharge rather than a hard floor:** the floor blocks an 8x8 even
  where its syntax is paid for. The surcharge keeps the driver's measured
  choice intact and only taxes the regime that is broken, so it degrades
  gracefully instead of cliffing — which is exactly what a fixed 16 cannot do
  at the ladder's rate.

### 7.5 D2 — recalibrate the per-leaf mode/partition price

Feed `kPartitionSignalBits` and `intra_luma_syntax_bits()` a price measured
from `AV2ENC_SYMBOL_COST` at the leaf's own qindex (5.13 bits/leaf at QP229
against 1.45 % of the tile at high rate). Ranked **below** D1: scaling the
whole R by 16 only bought +2.28 K, which argues the *relative* D:R balance is
off rather than R's scale.

### 7.6 D3 — qindex/resolution heuristic (fallback, not recommended)

`floor = 16 if qindex >= T else 4`. Reproduces both measured points with one
comparison, but encodes a constant as a function, has no mechanism, and will
misclassify a 720p still at qindex 90 or a 240p clip at qindex 200. Keep only
as a stopgap if no policy change can land.

### 7.7 Kill criteria (all of them)

* unset/off: the five pinned streams byte-identical (`25,009 a1d0dba1`,
  `25,141 3984677b`, `23,690 1b99127a`, `24,113 7f03eb6e`,
  `36,572 c4ba2813`) and the `AGENTS.md` reference bitstreams
  (`c9294fa4…`, `0d90c958…`).
* the 240p `--nightly-b` ladder holds 88.2 within the +/-0.05 gate, validated
  with `tools/validate_avmdec.py` (not `dav2d`, per `docs/validation.md`).
* no regression on the `detail_edges` key probe of
  `docs/tool-improvement-queue-3.md`.
* every stream passes `tools/validate_dav2d.sh` (flat/single) or
  `tools/validate_avmdec.py` (chained).
* promotion touches `src/cli/nightly.cpp` and the presets — shared roots, left
  to an integrator per `AGENTS.md`.

### 7.8 Measurement plan

1. §7.3 lambda sweep — decides calibrate vs policy, ~10 min, no code.
2. If policy: D1 behind an env gate; calibrate `S_max`/`alpha` at 0.21 bpp to
   K >= 46; confirm the 1.1 bpp probe is byte-identical (`kappa` never fires);
   then the ladder.
3. Only then propose promotion, through an integrator.

## 8. Files

| file | role |
|---|---|
| `src/pipeline/video/video_frame_internal.h` | `prune_target()` — the knob, default 4, clamp `[4,64]` |
| `src/pipeline/video/video_frame_decision.cpp:6453` | the single call site, inside `seq_driver_decide_sb`; `seq_depth_for_target` at `:6273`; the split price `lambda * 2.0` at `:6618` |
| `docs/rd-engine0.md` §5 | the existing knob table this measurement extends |
| `~/Downloads/av2enc-src720/sweep.py`, `rescore.py` | the harness (outside the tree) |

## 9. Verdict

**KEEP at one operating point.** Flooring the key driver's expansion at 16x16
is **+15.10 K and 4.2x faster** on the 720p still at 25,010 B, with the
mechanism measured rather than inferred: **61.0 % of the baseline stream is
per-leaf syntax** and the floor moves it to **81.4 % coefficients**. It closes
81.9 % of the gap to AVM, almost all of the Butteraugli half of it.

It is **not** a promotion. One operating point, the ladder not run, and the
standing probe negatives in `tool-improvement-queue-3` rows 10-11 are at a
42x different syntax share through a different knob. What would change the
verdict: §7.3 showing that a lambda recalibration reaches the same K (which
would make this a calibration bug rather than a policy question), or §7.4's
share-gated version holding 88.2 on the ladder while keeping K >= 46 here.

The transferable result is the **measurement**, not the constant: at 0.21 bpp
the key driver spends most of the stream on signalling the tree instead of the
picture, and `AV2ENC_SYMBOL_COST=1` reads that directly from any stream in one
encode.
