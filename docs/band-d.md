# Band-D: texture-preserving residual coding

Status: **week-1 census done; block-level target found; ladder running.**
Project start 2026-11-20. Goal: recover SS2 on the tail at fixed JOD and rate
without the BA regression that closed the dead-zone, sharp-filter and
source-weight attempts.

## Why

The K gap is BA (−2.82 K) and SS2 (−1.06 K) at equal JOD. The tail's SS2 falls
85.2 → 76.1 over f33–59 while PSNR rises 44.5 → 47.0 dB, so the deficit lives in
detail, not in error magnitude. Four earlier attempts failed for different
reasons (coefficient suppression, source-activity weighting, global predictor
sharpness, block variance); none measured **which error band SS2 charges for**.

## Week 1 — band census

The error is decomposed into the same nested box-difference bands the
`AV2ENC_PIPE_BA` mask uses: 1 px, 2–4 px, 4–8 px, 8+ px. Two tests:

**Per frame** (`tools/lane-band/band_census.py`). Regress per-frame SS2 / BA on
the source bands (content control) and the error bands.

| window | SS2 R² content | + error | ΔR² | BA ΔR² |
|---|---:|---:|---:|---:|
| head 1–32 | 0.699 | 0.992 | **+0.294** | **+0.457** |
| tail 33–59 | 0.975 | 0.982 | +0.007 | +0.110 |

In the head the 1-px band's partial correlation with SS2 is **−0.964**
(error up → SS2 down) after controlling for content. In the tail the
frame-to-frame variance is content: no band adds signal (ΔR² ≤ 0.007, even
with p90/p99, edge-masked and flat-masked variants). Per-frame is the wrong
level for the tail.

**Within a frame** (120×120 tiles, f50/f55/f59). Local SS2 against the local
1-px band error:

| | r |
|---|---:|
| raw (24 tiles) | −0.750 |
| within-frame centred | **−0.819** |

Tiles with more 1-px error score far worse SS2 in the same frame — flat tiles
96, textured tiles 82–87. That is the level the leaf D sees, and it is the
target.

Caveat: both tests are on the final stream, where the refresh schedule has
already acted. The block-level result is the one that matters for a per-leaf
objective.

## Week 2 — the band term already exists

`seq_rd::leaf_distortion_gains()` already carries the high-pass error term
(`AV2ENC_RD_TAIL_HF`, the driver's `HF` weight), used by the exact-rate TX plan
search. The head-TX arm's objective is the same function, so one gate covers
the head and the tail. The key's HF sweep (`docs/rd-engine0.md` §7) is not this
experiment: it spent rate on the key and propagated; the tail has no downstream.

Ladder: merged opt-in base (driver + head/tail exact-rate) at QPs 114/116/118,
arms `opt`, `AV2ENC_RD_TAIL_HF=20`, `=50`, oracle at 100 kbps.
Script: `tools/lane-band/ladder.sh`.

Result: **pending** (this section is updated when the ladder completes).

## Week 3 — only if the ladder is positive

If a non-zero HF arm clears +0.15 K with BA not worse, the next step is a
two-parameter fit (1-px weight plus the existing low-band gain) on a held-out
QP, then a full 6-QP ladder. If the arm is flat or trades SS2 for BA, the
project stops: the band term is already priced and the tail deficit is coding
efficiency, not the objective.

## Week 2 result — NEGATIVE, project closed

`tools/lane-band/ladder.sh /tmp/rd-band "114 116 118" "opt hf20 hf50"` on the
merged opt-in base, oracle at 100 kbps:

| arm | QP | bytes | kbps | JOD | BA | SS2 | K |
|---|---:|---:|---:|---:|---:|---:|---:|
| opt | 114 | 25,073 | 100.292 | 9.8320 | 2.0858 | 83.974 | **86.236** |
| opt | 116 | 23,798 | 95.192 | 9.8237 | 2.0786 | 83.647 | 86.161 |
| hf20 | 114 | 25,563 | 102.252 | 9.8389 | 2.1265 | 83.883 | 86.138 |
| hf20 | 116 | 24,293 | 97.172 | 9.8229 | 2.0979 | 83.694 | 86.109 |
| hf50 | 114 | 25,672 | 102.688 | 9.8297 | 2.0479 | 83.825 | 86.310 |
| hf50 | 116 | 24,476 | 97.904 | 9.8269 | 2.1033 | 83.496 | 86.080 |

Matched K at 100 kbps: **opt 86.232, hf20 86.125 (−0.107), hf50 86.181
(−0.051)**. The arms sit inside the ±0.2 QP ripple and are not leads.

At a fixed QP the high-pass weight costs **+2.0 % / +2.6 %** rate and leaves
SS2 flat (hf20 QP116: 83.694 vs 83.647) while improving BA (2.098 vs 2.079) —
the same BA/SS2 trade the sharp-filter and dead-zone experiments measured. The
segment ceilings show where it goes: f0–32 improves (+5.447 vs +5.160) and
f33–43 improves (+1.878 vs +1.805), but **f44–59 falls** (3.209 vs 3.460), the
exact band the K gap lives in.

**Verdict: stop. Do not fit a two-parameter band-D or wire band1 into the leaf
/quantiser objective.** The block-level finding is real — local SS2 tracks the
1-px error at r = −0.82 — but the rate needed to move that error exceeds what
the metric pays back in this encoder. The late tail needs a content-aware
decision (which blocks are worth detail), not a frequency weight.

The gate stays default off (`AV2ENC_RD_TAIL_HF=0`); the pins and the merged
opt-in point are unchanged.
