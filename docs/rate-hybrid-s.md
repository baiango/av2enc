# Exact inter syntax + analytic Laplacian coefficients (`AV2ENC_RATE_HYBRID_S`)

Branch `exp/rate-hybrid-s` (R09).  Env-gated, default off.  No CLI flag.

## What the historical leaf decision charges

`--video-leaf-rd` compares candidates and modes with

```
J = D + lambda * R
R = 2.0 (is_inter + skip base)
  + inter_syntax_bits(mode, drl, mvd)      // fixed 1.7 / +1.0 / +0.7 drl /
                                           // 4.0 + floor_log2(shell)
  + luma_plan_bits / estimate_tx_bits      // 8 EOB + 4 type + 2.5*nz + 0.5*eob
```

The syntax constants are a rough average over all contexts; they never read
the adaptive CDF the writer is about to use.  The coefficient constants are
two measured averages.

## What the gate changes

`AV2ENC_RATE_HYBRID_S=1` replaces both with models computed at decision time:

1. **Inter syntax: exact `-log2(p)` from the live CDF.**
   `SyntaxProbe` seeds a copy of the frame's §8.2 CDF image
   (`cdf_seed_`, or the §9.3 defaults when the frame has none), mirrors the
   writer's symbol sequence for `is_inter`, `skip_flag`, `comp_mode`,
   `single_ref`, `single_mode`, `drl_idx`, the eighth-pel raw-shell MVD
   (`shell_set` / class / offset / column / signs) and the BAWP flags, and
   derives the same §8.3 contexts from its own neighbour grid
   (`is_inter_context`, `skip_context`, `count_refs`, `comp_mode_context`,
   `new_mv_context` / `derive_new_mv_context`).  After each emitted leaf it
   adapts exactly those CDF rows with `av2::adapt_cdf` and advances the grid,
   in coded order, so later leaves see the same CDF state the writer will.
   The per-symbol cost is `-log2(freq / 32768)` from the inverse-CDF
   boundaries, the same quantity `AV2ENC_SYMBOL_COST` reports.

2. **Coefficients: analytic two-sided Laplacian.**
   For every transform block the EOB/TX-type overhead stays (8 bits, +4 for
   inter luma; the existing constants), and the magnitude term replaces
   `2.5 * nonzero` with the discrete-Laplacian entropy of the block's own
   coded scan prefix:

   ```
   b = mean(|level|) over the nonzeros      (ML scale of a two-sided geometric)
   p = exp(-1/b)
   H = nonzero * log2(e * (1 + p) / (1 - p))
   bits = 8 + (inter_luma ? 4 : 0) + 0.5 * eob + H
   ```

   `b` comes from the quantised block, so there is no fitted constant; the
   existing `0.5 * eob` position/EOB term is kept.  An iid full-prefix variant
   (all `eob` positions under the same geometric, no separate position term)
   was measured first: it predicted 11.327 bits/leaf against 5.756 actual
   (bias +5.57) versus 8.215 against 5.756 (bias +2.46) for the shipped form,
   so the shipped form replaces the magnitude constant only.

3. **Skip flag.**  The fixed residual/skip pair `3.0` vs `2.0` becomes the
   exact skip delta: `-log2 P(skip=1) - (-log2 P(skip=0))`.  Both options pay
   the same `is_inter`/mode syntax, so only the skip delta changes the
   comparison.  The intra/inter comparison charges `is_inter` to both sides
   (`-log2 P(1)` vs `-log2 P(0)`) instead of the shared `2.0` constant.

### Ablation switches (env only)

The point of R09 is to separate the two error sources, so both halves can be
disabled independently while the master gate stays on:

| Variable | Default | Effect |
|---|---|---|
| `AV2ENC_RATE_HYBRID_S` | off | master gate: model drives the decisions |
| `AV2ENC_RATE_HYBRID_S_SYNTAX` | on | `0` keeps `inter_syntax_bits` |
| `AV2ENC_RATE_HYBRID_S_COEFF` | on | `0` keeps `estimate_tx_bits` |
| `AV2ENC_RATE_HYBRID_S_MEASURE` | off | keep the historical decisions (stream byte-identical) and only record the report |

`gate=1 syntax=0 coeff=0` is the historical decision path plus the probe
(measurement only).

## Model error first

Every leaf records, under `AV2ENC_RATE_HYBRID_S[_MEASURE]`, four decision-side
predictions plus the measured range-coder bits.  The writer takes
`enc_.prefix_bits` deltas around three groups per leaf:

* `act_is_inter_skip`: `code_is_inter` + `code_skip_flag`
* `act_mode_info`: the whole `write_inter_mode_info()` call (ref/mode/DRL/MVD/
  BAWP)
* `act_residual`: the whole `write_inter_residual()` call (TX/EOB/coefficients)

CDEF/CCSO/delta-Q bits sit between the groups and belong to neither.  Each
symbol group also accumulates its write-time `-log2(p)`, which is the term
breakdown (the range coder only exposes one per-leaf total).  Samples are
grouped by model, mode and block size; the report prints median, p90 of
|error| and signed bias of `predicted - actual`, and the number of fallback
leaves (compound / TIP / AMVD / flex-MV-resolution, none at the measured point).

### 240p `--fast-decode` QP110, 60 frames (measurement arm, 2,893 inter leaves)

| Scope | Term | Predicted | Actual | Median | p90 abs | Bias |
|---|---|---:|---:|---:|---:|---:|
| all inter | syntax old (`2.0 + inter_syntax_bits`) | 4.258 | 1.577 | +3.70 | 3.70 | +2.68 |
| all inter | syntax new (probe) | 1.559 | 1.577 | +0.07 | 0.80 | **-0.02** |
| all inter | coeff old | 7.723 | 5.756 | +0.00 | 3.00 | +1.97 |
| all inter | coeff new | 8.215 | 5.756 | +0.00 | 3.67 | +2.46 |
| residual leaves (437) | coeff old | 51.127 | 38.103 | +6.50 | 27.00 | +13.02 |
| residual leaves (437) | coeff new | 54.384 | 38.103 | +10.53 | 35.39 | +16.28 |
| skipped leaves (2,456) | syntax new | 1.392 | 1.413 | +0.08 | 0.81 | -0.02 |

Per syntax term (probe vs the same symbols' write-time `-log2(p)`):

| Term | Predicted | Actual | Bias |
|---|---:|---:|---:|
| `is_inter` | 0.005 | 0.005 | +0.00 |
| `skip` | 0.203 | 0.203 | +0.00 |
| `single_mode` | 0.814 | 0.814 | +0.00 |
| ref (`comp_mode` + `single_ref`) | 0.000 | 0.000 | +0.00 |
| `drl_idx` | 0.116 | 0.116 | +0.00 |
| MVD shell | 0.286 | 0.213 | +0.07 |
| BAWP | 0.135 | 0.135 | +0.00 |

Per mode / block size (new syntax model, same measurement run):

| mode | size | n | pred | actual | median | p90 abs | bias |
|---|---|---:|---:|---:|---:|---:|---:|
| GLOBALMV | 32x16 | 277 | 0.607 | 0.679 | +0.08 | 0.85 | -0.07 |
| GLOBALMV | 32x64 | 292 | 0.595 | 0.658 | +0.06 | 0.83 | -0.06 |
| GLOBALMV | 64x32 | 744 | 0.640 | 0.636 | +0.08 | 0.82 | +0.00 |
| GLOBALMV | 64x64 | 793 | 0.431 | 0.425 | +0.08 | 0.79 | +0.01 |
| GLOBALMV | 64x16 | 137 | 0.596 | 0.620 | +0.08 | 0.71 | -0.02 |
| NEARMV | 32x16 | 81 | 3.501 | 3.469 | -0.03 | 0.84 | +0.03 |
| NEARMV | 32x64 | 88 | 2.813 | 2.886 | +0.05 | 0.72 | -0.07 |
| NEARMV | 64x32 | 193 | 3.159 | 3.176 | +0.02 | 0.73 | -0.02 |
| NEARMV | 64x64 | 33 | 3.101 | 3.212 | +0.05 | 0.68 | -0.11 |
| NEARMV | 64x16 | 40 | 3.241 | 3.250 | +0.03 | 0.73 | -0.01 |
| NEWMV | 32x16 | 48 | 9.511 | 9.604 | -0.01 | 0.68 | -0.09 |
| NEWMV | 32x64 | 37 | 10.248 | 10.351 | -0.04 | 0.71 | -0.10 |
| NEWMV | 64x32 | 84 | 8.078 | 8.083 | -0.06 | 0.67 | -0.01 |
| NEWMV | 64x64 | 21 | 8.594 | 8.476 | +0.07 | 0.53 | +0.12 |
| NEWMV | 64x16 | 25 | 8.059 | 8.120 | -0.11 | 0.47 | -0.06 |

Every group is within |median| <= 0.12 bits and p90 <= 0.85 bits.

**Result: the syntax model removes the decision-time syntax error (bias +2.68
-> -0.02 bits/leaf, median +3.70 -> +0.07).  The analytic Laplacian
coefficient model does not reduce the coefficient error at this point: the
context-coded coder is ~30% cheaper than the iid marginals on residual leaves,
so the old tuned 2.5 bits/nonzero was closer to the coder than the fitted
Laplacian.  This is a finding about where the decision error lives: the syntax
term was the large, systematic one; the coefficient term is context-dominated
and an iid marginal model cannot beat the tuned average.**

## Decisions changed

Counted against the historical formula recomputed at the same point (not
against a re-encode):

| Arm | QP | candidate winner | skip/residual | intra/inter |
|---|---|---:|---:|---:|
| full (syntax + coeff) | 105 | 2707 | 49 | 0 |
| full | 106 | 2690 | 33 | 9 |
| full | 107 | 2715 | 27 | 1 |
| syntax only | 105 | 2716 | 12 | 0 |
| syntax only | 106 | 2747 | 9 | 3 |
| syntax only | 107 | 2697 | 7 | 2 |
| coeff only | 105 | 0 | 17 | 0 |
| coeff only | 106 | 0 | 28 | 1 |
| coeff only | 107 | 0 | 28 | 0 |

Most mover is the candidate ranking.  The exact probe prices GLOBALMV (the
dominant zero-MV mode) at 0.43-0.64 bits instead of the flat 1.7, and gives
NEARMV/NEWMV their real single_mode/DRL/shell costs, so leaves whose
predictions are nearly SSE-tied re-rank.

## K at the bars

K is the protocol's `tools/kpi.py` composite (CVVDP 0.50 / Butteraugli 0.25 /
SSIMULACRA2 0.25).  Every arm is bracketed and interpolated linearly in K vs
kbps between the two encoder QPs that bracket the bar's rate; cells are
`kbps / K`.

### 240p plain (bar: K 82.15 at 100.0 kbps)

`--fast-decode`, QPs 105/106/107:

| Arm | QP105 | QP106 | QP107 | K @ 100.0 kbps |
|---|---|---|---|---|
| baseline (no gate) | 101.368 / 81.81 | 100.032 / 82.15 | 99.540 / 80.51 | **82.04** |
| R09 full (syntax + coeff) | 100.316 / 79.82 | 98.684 / 80.65 | 97.548 / 81.65 | **79.98** (-2.06) |
| ablation: syntax only | 100.636 / 80.55 | 99.800 / 81.77 | 97.948 / 80.37 | **81.48** (-0.56) |
| ablation: coeff only | 101.304 / 81.38 | 98.416 / 81.64 | 98.088 / 81.03 | **81.50** (-0.54) |

The baseline reproduces the bar (82.15 measured at QP106; the interpolated
82.04 sits inside the documented +-1.6 K adjacent-QP ripple).  Every R09 arm
is at or below it.  The full model is **-2.06 K**, beyond the ripple; each
half alone is inside it.  The loss is superadditive because the skip decision
moves on both terms at once: the exact skip delta makes `skip_flag = 1`
cheaper and the Laplacian makes the residual more expensive, so the combined
arm flips 27-49 skip decisions per QP versus 7-12 for the syntax half and
17-28 for the coefficient half.  At matched rate the extra skipping costs
visible quality (full QP106: Butteraugli 4.08 vs baseline 3.42, CVVDP 9.689
vs 9.733, SSIMULACRA2 79.75 vs 80.84).

### 240p drift (bar: K 83.80 at 100.0 kbps)

`--fast-decode --video-drift-refresh-threshold 1.5 --video-drift-refresh-budget 3250`,
QPs 109/110/111:

| Arm | QP109 | QP110 | QP111 | K @ 100.0 kbps |
|---|---|---|---|---|
| baseline | 104.020 / 83.21 | 100.584 / 83.94 | 99.800 / 83.75 | **83.80** |
| R09 full | 101.644 / 83.94 | 100.192 / 83.19 | 99.416 / 83.38 | **83.24** (-0.56) |

The baseline reproduces the bar exactly; the full model is 0.56 K below it,
inside the adjacent-QP ripple.  The drift point uses the same per-leaf model
(QP110: candidate=2696, skip=19, intra=7 of 2,871 leaves).

### 720p pin (bar: K 45.61 / JOD 7.3236 at 146.288 kbps)

`--qp Q --fast-decode --video-key-qp-offset -57 --video-leaf-rd-lambda 250
--video-ext-partitions`, QPs 216/217/218 (an extra QP215 point brackets the
full arm at the pin rate because it codes ~4.7% smaller at the same QP):

| Arm | QP215 | QP216 | QP217 | QP218 | @ 146.288 kbps |
|---|---|---|---|---|---|
| baseline | — | 150.196 / 45.89 | 146.288 / 45.61 | 143.776 / 45.24 | **45.61 K / 7.3236 JOD** (pin) |
| R09 full | 151.780 / 46.50 | 142.832 / 45.56 | 139.440 / 45.24 | 137.728 / 44.69 | **45.92 K / 7.3627 JOD** (+0.31 / +0.039) |

At the 720p pin the model is nominally **+0.31 K / +0.039 JOD** at matched
rate, but the bracket is 3.5-5.5 kbps wide on each side and the movement is
inside the ripple.  This is the one point where the model does not lose
quality; the 240p plain point carries the clear negative.

### Verdict

R09 does not clear any bar by more than the ripple.  The 240p plain point is a
clear **-2.06 K**, the drift point -0.56 K, the 720p pin +0.31 K.  The point
of the branch was to see whether exact syntax pricing (without a coefficient
CDF walk) recovers the leaf decision error.  The answer from the model-error
table is that the syntax error is removed almost exactly, but the coefficient
half is the larger remaining term and the analytic Laplacian does not improve
it; trading those two against the fixed SSE lambda moves the encoder toward
more skipping and loses K at 240p.  This is a finding about the decisions
(SSE + fixed lambda), not a reason to keep the model.

## Reproduce

```sh
REF=testdata/nasa_480x240_30fps_60f.y4m
BIN=build-dev/av2enc

# model error first: historical decisions, stream byte-identical
AV2ENC_RATE_HYBRID_S_MEASURE=1 "$BIN" "$REF" -o /tmp/m.ivf --video --frames 60 \
    --qp 110 --cpu-threads 8 --fast-decode

# model on (and the 2x2 ablation)
AV2ENC_RATE_HYBRID_S=1 "$BIN" ... --fast-decode
AV2ENC_RATE_HYBRID_S=1 AV2ENC_RATE_HYBRID_S_COEFF=0 "$BIN" ... --fast-decode
AV2ENC_RATE_HYBRID_S=1 AV2ENC_RATE_HYBRID_S_SYNTAX=0 "$BIN" ... --fast-decode
```

The one-line summary and the per-term/per-mode tables print to stderr at
process exit.  `leaves=` is the sample count, `fallbacks=` the leaves the probe
could not model (zero on every measured point).

## Encode cost

60-frame 240p, QP106 `--fast-decode`, same build, interleaved runs:

| Threads | Arm | user CPU (s) | wall (s) |
|---|---|---:|---:|
| 1 | baseline | 11.88, 11.20 | — |
| 1 | R09 full | 11.92, 11.33 | — |
| 8 | baseline | 11.54, 12.09 | 39.6, 37.3 |
| 8 | R09 full | 11.82, 13.79 | 44.4, 49.2 |

The probe costs about **+1% user CPU at one thread** (11.54 -> 11.63 s
two-run mean); the Laplacian is O(eob) like the heuristic it replaces.  At
eight threads the *work* is similar but the wall time rises ~15-30%: the live
CDF/grid probe advances in coded order, so the superblock wavefront is
disabled while the gate is active (`wavefront_allowed()`).  The host was
running several other encode/metric jobs during this table, so read the wall
column as a scheduling-bound estimate, not a product timing.  Gate-off runs
keep the wavefront and every pin.

## Verification

- Gate off: default `25,009 a1d0dba1`, quality `25,141 3984677b`, nightly
  `23,690 1b99127a`, explicit-λ control `24,113 7f03eb6e`, 720p recipe
  `36,572 c4ba2813` — all byte-identical.
- Gate on (`=1`): 60/60 `dav2d --strict 1`, `--video-recon` payload equals the
  decoded payload byte for byte.
- `tools/test.sh --build build-dev` 10/10.
- `tools/gate_video_nightly.sh` and
  `src/pipeline/video/gate_video_qp_lambda.sh` pass with the gate unset (they
  assert the pins, so a gated run is not expected to pass them by design).

## Limits / why this does not ship

- The probe only models the measured nightly surface: single prediction, no
  TIP, eighth-pel raw-shell MVD, no AMVD/flex-MV-resolution/inter-intra.
  A frame that selects one of those leaves falls back to
  `inter_syntax_bits` and stops advancing the probe for the rest of the tile;
  the fallback count is part of the report.  Every measured leaf scored
  `fallbacks=0`.
- For TIP frames, ext-partition SDP keyframes, and tiles the report is still
  valid but the model is only exact where the probe is used.
- The coefficient model is the R09/R06 analytic Laplacian; the measurement
  above shows it is not a better predictor than the tuned average on this
  coder.  The branch is therefore reported as a syntax-vs-coefficient
  decomposition, not as a shipping model.
