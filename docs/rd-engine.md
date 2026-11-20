# RD engine rewrite

Status: **active program**. Replaces the ad-hoc decision costs (variance
partition proxy, calibrated leaf RD, separate key/P/chroma paths) with one
engine. Base K = 85.634 at 240p/100 kbps (UV + eff-perc + drift + tail-TX);
AVM = 90.2. The gap decomposition is Butteraugli 64%, SSIMULACRA2 28%,
CVVDP 7%, i.e. fine-detail error at the same rate.

## Cost

```
J = D_perc + lambda(qindex) * R_exact
```

- `D_perc`: the eff-perc perceptual distortion (per-picture Q8 weight maps,
  3x3 CSF gain, chroma plane weight, visible-row clip, source-extent clamp).
  One D for every candidate, key and P.
- `R_exact`: the information bits the §8.2 coder charges, per family:
  partition, mode/MV, EOB, txb_skip, coeff_base, coeff_br, coeff_mag
  (golomb/escape), sign, tx_type, tx_partition, filter and header rows.
  Use the live/§8.3 CDF `-log2 p` or the exact model; no SSE-based estimates.
- `lambda`: per site, calibrated at the operating point. The key exact-cost
  work found eff-perc over-zeroes the key by ~4x; assume every site is
  miscalibrated until measured.

## Search

Every decision is a candidate set with the same J:

- partition: recursive tree, 4x4/8x8/16x16/32x32/64x64 (no 16x16 floor),
- intra mode: 61 luma + full UV/CfL,
- inter: GLOBALMV/NEARMV/NEWMV/compound, skip vs residual,
- TX: size/type/partition + coefficient level trellis,
- two-stage: cheap screen, exact J on the top-K; the engine may spend 5-10x
  the current CPU time (AVM is ~5x slower and speed is not scored).

## Invariants

- Default off, env-gated; gate-off pins: QP106 25,008 B / `0639b436...`;
  QP110 23,690 B / `1b99127a...`.
- No syntax changes; spec-only; 4:2:0; decoder 60/60; `--video-recon` equal to
  the decoded payload.
- `AV2ENC_BIT_CENSUS` / `AV2ENC_DECISION_DUMP` for every result.
- Release build passes with `-Werror`.
- One clip, one base recipe, 5 interleaved QPs; a lane result is a KEEP only at
  >= +0.15 K, a LEAD if the mechanism is right but K is inside the ripple.

## Lanes

| Lane | Owns | Milestone |
|---|---|---|
| L1 cost/rate | `video_rd_engine.*`, exact rate, lambda calibration | one J validated per family within a few %, used by the key leaf |
| L2 key partition/mode | key recursive tree + mode RD | measured key tree, 4x4/8x8 available |
| L3 key TX/levels | TX size/type/partition + level trellis | full key TX/level RD with J |
| L4 inter | inter partition/mode/skip/residual | inter RD with J, skip cost includes propagation |
| L5 chroma | UV modes + CfL + chroma TX/levels | chroma RD with J |

Lanes are isolated worktrees; the integrator merges and measures the
composition. A lane that cannot separate its work from the driver reports the
dependency instead of rewriting the pipeline.
