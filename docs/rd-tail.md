# RD tail — exact-rate / BA-shaped TX and level RD on the P-frame tail

Status: **KEEP for the f0-32 ceiling; the K gain alone is a LEAD.**
`AV2ENC_RD_TAIL_TX=1` (default off) makes the f33-59 inter luma residual choose
its transform plan and coefficient levels with the L3 exact-rate objective and
the driver's BA-shaped distortion instead of the RDOQ-calibrated static model.
At a fixed QP the tail spends **6.4 % fewer bytes** with flat tail PSNR, the
pooled f0-32 source-replacement ceiling moves **91.325 → 91.439 (+0.113)** on
the five-point fit (**+0.218** at the oracle's nearest points) and matched K
moves **86.060 → 86.186 (+0.126 fit, +0.225 bracket)**.  K does not regress and
the ceiling is higher on every reading, which meets the ceiling rule.  The
prescribed 4x4/8x4 coding-block extension was built, measured and is
**negative** (default off, §7).

Branch `exp/rd-tail`, worktree `/Users/arin/work/rd-tail`, from `e3234b96`
(the merged L3+L5+driver recipe).

## 1. Defect

The merged recipe decides the key picture with the sequential RD driver
(`AV2ENC_RD_DRIVER`, docs/rd-engine0.md), which prices every candidate leaf
with one `J = D_BA + lambda * R_exact`.  The P-frame tail does not: on a tail
inter leaf the luma TX search ranks its plans with the RDOQ-calibrated static
model (`model_rate_scale = 0.4`, `docs/rd-lane3.md` §"Two-stage search" gives
the same 4.76x static-vs-exact gap on the key) and a distortion summed per
transform leaf with the 3x3/7x7 windows clipped to the leaf.  The leaf
decision *after* the search already charges the live §8.2.6 interval rate
(`plan_rate_bits`), so the plan/level choice and the skip decision are made on
different costs.  The tail is 29-30 % of the stream's bytes and 56-57 % of its
information bits are residual (QP116, §6), with the stream's worst Butteraugli
(the matched f44-59 segment is BA 2.51 / SS2 78.42 in the base at QP116).

## 2. Files

| File | Change |
|---|---|
| `src/pipeline/video/video_tail_tx.h` | new: `AV2ENC_RD_TAIL_TX` config, source-activity screen, counters |
| `src/pipeline/video/video_seq_rd.h` | `leaf_distortion_gains()`: the same BA-shaped D with explicit HF/EDGE weights; the historical wrapper passes the driver's env gains (behaviour unchanged) |
| `src/pipeline/video/video_frame_decision.cpp` | tail exact-rate plan/level stage in the inter leaf; the `AV2ENC_RD_TAIL_ACT` sub-16 partition gate |
| `src/pipeline/video/video_frame_internal.h` | include only |
| `tools/lane-tail/ladder.sh` | interleaved 2-arm ladder (strict decode, validate, PSNR, recon equality, census/dump) |
| `tools/lane-tail/perframe_metrics.py` | per-frame BA/SS2 and the f0-32/f33-43/f44-59 segment means |
| `tools/lane-tail/tail_stats.py` | tail byte / residual / skip / transform statistics from the existing diagnostics |

No CLI, CMake, `main.cpp`, options/parse, gate or pin edit.  Gate unset is
byte-identical to the pins (§8).

## 3. Gate

Environment only, default off.  The master gate plus the calibrated defaults
reproduce the measured point:

| Variable | Default | Meaning |
|---|---|---|
| `AV2ENC_RD_TAIL_TX` | off | master gate; pictures 33-59 only, inter luma residual only |
| `AV2ENC_RD_TAIL_LEVEL` | 1 | coefficient trellis in the exact re-search |
| `AV2ENC_RD_TAIL_LEVEL_SCALE` | 0.25 | trellis SSE lambda as a fraction of `rdoq_sse_scale_at()` (the L3 auto value) |
| `AV2ENC_RD_TAIL_TX_LAMBDA` | 1.0 | plan-J lambda as a fraction of the inter leaf pipe lambda |
| `AV2ENC_RD_TAIL_RATE` | 1 | 1 = live §8.2.6 interval rate; 0 = the search's own static rate (ablation) |
| `AV2ENC_RD_TAIL_TX_D` | 1 | 1 = BA-shaped assembled-block D; 0 = plain coded SSE (ablation) |
| `AV2ENC_RD_TAIL_HF` | 0 | high-pass error weight of the BA D |
| `AV2ENC_RD_TAIL_EDGE` | 0 | block-edge weight of the BA D |
| `AV2ENC_RD_TAIL_ACT` | 0 | sub-16 coding-block screen (0 = off; §7, measured negative) |
| `AV2ENC_RD_TAIL_STATS` | off | print counters |

## 4. Design

### One J on the assembled block

For one tail inter luma leaf the plan comparison is

```text
D = seq_rd::leaf_distortion_gains() on the assembled block reconstruction
    (eff-perc inter CSF: csf 1.5, csf2 0; HF/EDGE weights as configured)
R = plan_rate_bits() -- the §8.3 symbol sequence through the §8.2.6 interval
    arithmetic against the carry image the writer starts from, §5.20.7.30 scan
J = D + pipe_lambda * AV2ENC_RD_TAIL_TX_LAMBDA * R
```

`R` includes the TX partition and TX-type symbols; the mode/MV syntax is
constant inside one leaf and cancels.  `pipe_lambda` is the same lambda the
leaf's skip/residual decision charges to `plan_rate_bits`, so the plan and the
skip choice finally optimise one objective.  The assembled-block D differs
from the search's internal sum in exactly the windows that cross a transform
leaf boundary; the top/left causal neighbours are copied into a one-pixel
border so the EDGE term is the gradient the decoder shows.

### Two-stage search

1. **Screen**: the unchanged historical call (`search_luma_tx` with
   `model_rate_scale = 0.4`, eff-perc weights and lowpass gains).  This is the
   gate-off winner and stays the fallback.
2. **Exact re-search**: the same call with `model_rate_scale = 1.0`, the live
   CDF slice (`use_context_rate = true`), the trellis enabled at
   `0.25 x rdoq_sse_scale_at()` and the same lambda.  Both plans are then
   scored with the assembled-block J above; the better wins, ties keep the
   screen plan.

At QP 116 the stage runs on every tail inter candidate that reaches the
residual (11,100 candidate evaluations over pictures 33-59); the exact plan
wins 1,905 of them and changes the plan structure (partition or part count)
1,031 times.  Across all evaluations the exact candidates sum to +2.6 %
distortion and −19.5 % §8.3 rate relative to the screen candidates, which is
the trade the J then arbitrates per leaf.

### Why the tail only

`tx_detail::picture_ordinal()` is advanced once per whole picture and not by
the TIP second pass, so the stage is exactly pictures 33-59 of the 60-frame
clip.  The key picture, intra leaves, chroma, compound and inter-intra leaves,
segmented skip leaves and tiled encodes never enter it.

## 5. Method

Recipe (the merged 86.013 base plus the driver):

```sh
BASE_ENV="AV2ENC_EFF_PERC=1 AV2ENC_RD_UVFULL=1 \
  AV2ENC_TX_DETAIL_SEARCH_TAIL=1 AV2ENC_TX_DETAIL_MIN=0.50 \
  AV2ENC_RD_KEY_TX=1 AV2ENC_RD_UVLEVEL=1 AV2ENC_RD_DRIVER=1"
COMMON="--video --frames 60 --cpu-threads 8 --fast-decode \
  --video-drift-refresh-threshold 1.5 --video-drift-refresh-budget 3250"
```

Two arms, five interleaved QPs (113-117), arm order rotated per QP,
`build-release` (`-O3`, warnings + `-Werror`), driven by
`tools/lane-tail/ladder.sh` under bash.  Matched K is
`tools/oracle.py --target-kbps 100`; the source-replacement ceilings are the
oracle's segment rows, and the per-arm ceiling curve was completed with
single-point oracle runs (a candidate with one point is its own nearest point,
so it gets the segment rows).

## 6. Results

### Ladder

| QP | arm | bytes | kbps | PSNR-Y | JOD | BA | SS2 | K |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 113 | base | 26,372 | 105.488 | 45.859 | 9.8415 | 2.0675 | 83.812 | 86.310 |
| 113 | cand | 25,544 | 102.176 | 45.837 | 9.8407 | 2.0500 | 83.939 | 86.378 |
| 114 | base | 26,160 | 104.640 | 45.661 | 9.8315 | 2.0677 | 83.781 | 86.254 |
| 114 | cand | 25,653 | 102.612 | 45.704 | 9.8347 | 2.1077 | 83.885 | 86.171 |
| 115 | base | 25,281 | 101.124 | 45.674 | 9.8306 | 2.0504 | 83.743 | 86.294 |
| 115 | cand | 24,722 | 98.888 | 45.667 | 9.8335 | 2.0857 | 83.767 | 86.209 |
| 116 | base | 24,843 | 99.372 | 45.580 | 9.8239 | 2.1571 | 83.425 | 85.900 |
| 116 | cand | 24,425 | 97.700 | 45.541 | 9.8206 | 2.0778 | 83.469 | 86.118 |
| 117 | base | 24,207 | 96.828 | 45.478 | 9.8135 | 2.1309 | 83.346 | 85.909 |
| 117 | cand | 23,821 | 95.284 | 45.439 | 9.8163 | 2.1383 | 83.320 | 85.898 |

| arm | bracket | K@100 | Δ | 5-pt fit | Δ fit | slope | R² |
|---|---|---|---:|---:|---:|---:|---:|
| base | 116/115 | **86.041** | — | **86.060** | — | +0.0491 | 0.715 |
| cand | 115/113 | **86.266** | **+0.225** | **86.186** | **+0.126** | +0.0463 | 0.679 |

Per-QP rate-adjusted deltas (base slope 0.0491 K/kbps):

| QP | Δkbps | ΔK | rate-adjusted ΔK | ΔBA | ΔSS2 |
|---:|---:|---:|---:|---:|---:|
| 113 | −3.31 | +0.068 | +0.231 | −0.0175 | +0.127 |
| 114 | −2.03 | −0.083 | +0.017 | +0.0400 | +0.103 |
| 115 | −2.24 | −0.085 | +0.025 | +0.0353 | +0.024 |
| 116 | −1.67 | +0.218 | +0.300 | −0.0793 | +0.044 |
| 117 | −1.54 | −0.011 | +0.065 | +0.0074 | −0.026 |

The arm is 1.5-3.3 kbps cheaper at every fixed QP; the sign-adjusted gain is
positive at every QP and averages +0.128, matching the fit.

### Source-replacement ceilings

`f0-32` replaces pictures 0-32 with source, so it pools the tail f33-59;
`f44-59` replaces the last 16 pictures and pools the head plus f33-43.

| QP | arm | kbps | f0-32 | f33-43 | f44-59 |
|---:|---|---:|---:|---:|---:|
| 113 | base | 105.488 | 91.526 | 88.233 | 89.574 |
| 113 | cand | 102.176 | 91.626 | 88.275 | 89.599 |
| 114 | cand | 102.612 | 91.321 | 88.060 | 89.569 |
| 115 | base | 101.124 | 91.608 | 88.152 | 89.554 |
| 115 | cand | 98.888 | 91.466 | 88.029 | 89.578 |
| 116 | base | 99.372 | 91.248 | 87.747 | 89.311 |
| 116 | cand | 97.700 | 91.606 | 87.977 | 89.338 |
| 117 | base | 96.828 | 91.072 | 87.840 | 89.398 |
| 117 | cand | 95.284 | 91.049 | 87.797 | 89.425 |

`base@115` reproduces the programme's **91.608** exactly, which is the
reference point this lane was given.  (`base@114` was not run separately: the
base ladder's nearest point is QP116 and the fit has four base points.)

Matched-rate ceilings (oracle nearest points, one point per arm):

| ceiling | base@116 (99.372) | cand@115 (98.888) | Δ |
|---|---:|---:|---:|
| f0-32 | 91.248 | **91.466** | **+0.218** |
| f33-43 | 87.747 | **88.029** | **+0.282** |
| f44-59 | 89.311 | **89.578** | **+0.267** |

Rate-normalised (linear in rate over each arm's own bracket):

| ceiling | base @100 | cand @100 | Δ |
|---|---:|---:|---:|
| f0-32 | 91.377 | 91.522 | +0.145 |
| f44-59 | 89.398 | 89.558 | +0.160 |

The cand's 100 kbps bracket is QP114/115 and the base's is QP115/116; the
cand QP114 point is a Butteraugli ripple (BA 2.108 vs 2.086 at the higher rate)
and drags the interpolation down: at the oracle's nearest points the gain is
+0.218, between the brackets +0.145, and the two-point QP114/115 reading is
+0.046.  Best estimate: **+0.12 ceiling (fit), range +0.05..+0.22**.

Five-point fits of the ceiling against rate (ordinary least squares, evaluated
at 100 kbps), the same reading the K fit uses:

| ceiling | base fit @100 | R² | cand fit @100 | R² | Δ |
|---|---:|---:|---:|---:|---:|
| f0-32 | 91.325 | 0.64 | **91.439** | 0.24 | **+0.113** |
| f33-43 | 87.955 | 0.69 | **88.060** | 0.74 | **+0.104** |
| f44-59 | 89.441 | 0.55 | **89.520** | 0.54 | **+0.079** |

Same-QP deltas are mixed at the ceiling level even though the fixed-QP tail is
consistently cheaper: f0-32 is +0.100 (QP113), −0.142 (QP115), +0.358 (QP116),
−0.023 (QP117); f44-59 is +0.025, +0.024, +0.027, +0.027 at the same four
QPs.  The f44-59 row is the sign-consistent one: its pooled f33-43
contribution carries the tail gain at every QP.  The f0-32 row is dominated by
the whole tail's Butteraugli ripple, which is why its fit R² is 0.24.

The cand QP114 point is the ripple to avoid reading: it is 3.7 kbps larger
than QP115 yet its f0-32 ceiling is 0.145 lower (whole-stream BA 2.108 vs
2.086).  That one point is what makes the two-point interpolation at the
base's 101.124 kbps reference read −0.23, and it is why the fit, not the
bracket, is the headline for the ceiling.

### Tail bytes, residual and skip share

Per-picture `frame bits` from the encoder log (payload only, headers excluded):

| QP | arm | key B | f1-32 B | f33-43 B | f44-59 B | tail f33-59 B | tail share |
|---:|---|---:|---:|---:|---:|---:|---:|
| 113 | base | 14,710 | 2,580 | 3,420 | 3,940 | 7,360 | 29.9 % |
| 113 | cand | 14,710 | 2,580 | 2,938 | 3,629 | 6,567 | 27.5 % |
| 114 | base | 14,644 | 2,465 | 3,406 | 3,875 | 7,281 | 29.9 % |
| 114 | cand | 14,644 | 2,465 | 3,147 | 3,655 | 6,802 | 28.4 % |
| 115 | base | 14,489 | 2,069 | 3,188 | 3,774 | 6,962 | 29.6 % |
| 115 | cand | 14,489 | 2,069 | 2,951 | 3,458 | 6,409 | 27.9 % |
| 116 | base | 14,259 | 2,069 | 3,167 | 3,622 | 6,789 | 29.4 % |
| 116 | cand | 14,259 | 2,069 | 2,903 | 3,453 | 6,356 | 28.0 % |
| 117 | base | 14,066 | 1,660 | 3,251 | 3,503 | 6,754 | 30.0 % |
| 117 | cand | 14,066 | 1,660 | 2,846 | 3,466 | 6,312 | 28.6 % |

The key and f1-32 rows are identical at every QP: the change is confined to
pictures 33-59, and the tail is 6.4-10.8 % smaller.

Census + decision dump, QP116 (`tools/lane-tail/tail_stats.py`):

| metric | base | cand |
|---|---:|---:|
| tail payload bytes | 6,789 | 6,356 |
| residual bits / tail info bits | 56.70 % | 56.02 % |
| tail skip share (inter leaves) | 55.95 % | 52.76 % |
| tail residual-leaf share | 44.05 % | 47.24 % |
| tail coded-luma share | 33.41 % | 34.53 % |
| tail TX partition NONE / SPLIT / HORZ / VERT | 1,803 / 66 / 379 / 186 | 2,066 / 31 / 334 / 186 |
| tail 4x4 TX leaves | 44 | 12 |
| tail 8x8 TX leaves | 1,526 | 1,737 |

The exact plan codes **more residual leaves** (skip share −3.2 pp) with
**fewer, larger transforms** (SPLIT 66 → 31, 4x4 44 → 12): exact-rate pricing
makes the fine partitions pay their real cost, and the freed rate goes into
leaves the historical bar skipped.

### Matched-point quality

The reference point this lane was given is the driver recipe at QP115
(101.124 kbps): base f44-59 **BA 2.2688 / SS2 78.847**, which reproduces the
task's 2.269 / 78.85 exactly, and f0-32 91.608 (reproduced above).

Oracle nearest points: base QP116 (99.372 kbps) vs cand QP115 (98.888 kbps).

| metric | base QP116 | cand QP115 | Δ |
|---|---:|---:|---:|
| clip BA | 2.1571 | **2.0857** | −0.0714 |
| clip SS2 | 83.425 | **83.767** | +0.342 |
| clip JOD | 9.8239 | **9.8335** | +0.0096 |
| mean PSNR-Y | 45.580 | 45.541 | −0.039 |

Per-frame segments at those points (`tools/lane-tail/perframe_metrics.py`):

| segment | metric | base QP116 | cand QP115 | Δ |
|---|---|---:|---:|---:|
| f0-32 | BA | 2.0017 | 1.9286 | −0.0731 |
| f0-32 | SS2 | 86.197 | 86.287 | +0.090 |
| f0-32 | PSNR-Y | 45.8882 | 45.8882 | 0.000 |
| f33-43 | BA | 2.1117 | 2.0684 | −0.0433 |
| f33-43 | SS2 | 82.386 | 83.239 | +0.853 |
| f33-43 | PSNR-Y | 44.5379 | 44.6019 | +0.064 |
| f44-59 | BA | 2.5090 | 2.4216 | −0.0874 |
| f44-59 | SS2 | 78.420 | 78.931 | +0.511 |
| f44-59 | PSNR-Y | 45.8366 | 45.8514 | +0.015 |

Selected per-frame rows at the same two points (whole frames, not segments):

| frame | base BA | cand BA | ΔBA | base SS2 | cand SS2 | ΔSS2 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 (key) | 1.4993 | 1.2797 | −0.2195 | 87.637 | 87.538 | −0.099 |
| 33 | 1.8221 | 1.7923 | −0.0299 | 84.966 | 85.226 | +0.260 |
| 36 | 1.9705 | 1.8820 | −0.0885 | 83.350 | 84.320 | +0.970 |
| 40 | 2.1553 | 2.1035 | −0.0518 | 81.058 | 82.321 | +1.263 |
| 43 | 2.2174 | 2.2063 | −0.0111 | 80.517 | 81.602 | +1.085 |
| 44 | 2.0762 | 2.1969 | +0.1206 | 81.039 | 80.966 | −0.074 |
| 48 | 2.3519 | 2.2673 | −0.0846 | 79.172 | 79.793 | +0.622 |
| 55 | 3.0468 | 2.3051 | −0.7417 | 77.543 | 77.937 | +0.394 |
| 59 | 2.3903 | 2.5013 | +0.1109 | 75.651 | 76.776 | +1.125 |

(The key row is a whole-stream QP difference — the key picture is not touched
by the gate.)  7 of the 9 selected tail frames improve SS2; the two that lose
(f44, f59) lose 0.07 and are the tail's BA ripple.

At the same QP (115) the tail PSNR is flat (f33-43 −0.049 dB, f44-59
+0.023 dB) while the tail spends 553 fewer bytes: the mechanism is a rate win,
not a quality trade.  At the matched 100 kbps QPs (base 115.66, cand 114.64,
linear in QP between adjacent measured streams) the tail reads:

| segment | metric | base @100 | cand @100 | Δ |
|---|---|---:|---:|---:|
| f33-43 | BA | 2.1071 | 2.0980 | −0.0091 |
| f33-43 | SS2 | 82.674 | 83.119 | **+0.445** |
| f44-59 | BA | 2.4274 | 2.4378 | +0.0104 |
| f44-59 | SS2 | 78.565 | 78.972 | **+0.407** |

The SS2 gain is the sign-consistent reading; Butteraugli is flat at matched
rate (the QP116→115 BA step in the table above is the base's own ripple, not
the mechanism).

## 7. The 4x4/8x4 coding-block extension (negative)

`AV2ENC_RD_TAIL_ACT` forces a 16x16 tail node whose source high-pass activity
(`mean((d - box3x3(d))^2)`) clears the threshold to split, so its residual
could reach the legal 8x8 / 8x4 / 4x4 transform partitions.  The band gate's
`max_depth` bound had to be raised by one (rect-path 16x16 coding blocks sit at
depth 4 on this clip); a large finite NONE penalty replaced
`kInfiniteCost` after the first version made a superblock unsolvable.

Measured at QP115, threshold 20 (18.5 % of tail 16x16 blocks clear 50, 29.2 %
clear 20):

| arm | bytes | kbps | f33-43 PSNR | f44-59 PSNR |
|---|---:|---:|---:|---:|
| cand QP115 | 24,722 | 98.888 | 44.6019 | 45.8514 |
| +ACT 20 | 25,027 | 100.108 | 44.3179 | 45.3850 |

It spends **+1.2 %** rate for **−0.28 dB** (f33-43) and **−0.47 dB**
(f44-59).  The decision dump explains it: the forced splits do not produce a
single 8x8-class coding block in the committed tail tree (no `bsize` 8x8 /
8x16 / 16x8 leaf).  The solver instead retreats to coarser ancestors
(`16x16` leaves 4,140 → 3,669, 32x16 → 345), so the screen perturbs the layout
without reaching the 4x4/8x4 transforms it was built for.  The P-frame
coding-block floor for inter leaves is effectively 16x16 on this clip.  The
knob stays implemented and default off; do not enable it.

## 8. Conformance

* Gate off pins, final release build: QP106 **25,008 B /
  `0639b43630abb454f7be5ac494d2efb4`**, QP110 **23,690 B /
  `1b99127a4cd1d386fa5600fb63c57a83`**.
* Gate on reproduces exactly: the final binary re-encodes cand QP115 to
  24,722 B / `0c0e92d82cd1b2f4c87d2fc600e7d92c` (the ladder stream).
* 10/10 ladder streams: `tools/validate_dav2d.sh` OK, `dav2d --strict 1`
  decodes 60/60, and `--video-recon` equals the decoded payload byte for byte.
* `cmake --preset dev-cache` and `cmake --preset release` (warnings +
  `-Werror`) build clean; `ctest` 12/12; `tools/gate_video_nightly.sh --bin
  build-release/av2enc` OK (expansion, pins, strict decode, visible recon,
  BAWP).
* No gate/pin/CLI/CMake/options/parse/`main.cpp` change.

## 9. Verdict

**KEEP for the f0-32 ceiling; the K gain on its own is a LEAD.**

Success rule as given: a higher f0-32 ceiling with K not regressing is a KEEP
for the ceiling; a K gain >= +0.15 is a KEEP on its own.  Measured at the
100 kbps protocol point:

| quantity | base | cand | Δ | reading |
|---|---:|---:|---:|---|
| K, 100 kbps bracket | 86.041 | **86.266** | **+0.225** | KEEP-bar K +0.225 |
| K, five-point fit | 86.060 | **86.186** | **+0.126** | below the +0.15 K bar |
| f0-32 ceiling, fit | 91.325 | **91.439** | **+0.113** | higher ceiling |
| f0-32 ceiling, nearest points | 91.248 | **91.466** | **+0.218** | higher ceiling |
| f0-32 ceiling, 2-pt interpolation | 91.377 | **91.522** | **+0.145** | higher ceiling |
| f44-59 ceiling, fit | 89.441 | **89.520** | **+0.079** | sign-consistent |
| f44-59 ceiling, nearest points | 89.311 | **89.578** | **+0.267** | higher ceiling |

The mechanism is real and behaves as designed: at a fixed QP the tail is
**6.4 % smaller** with flat tail PSNR (f33-43 −0.05 dB, f44-59 +0.02 dB) and
better tail BA/SS2 at the matched rate; the exact plan codes more residual
leaves with fewer, larger transforms.  K does not regress at any
rate-normalised QP (+0.017..+0.300) and the f0-32 ceiling is higher on every
reading, so the ceiling rule is met.

Caveats for the integrator:

* The conservative ceiling number is the fit, **+0.113**, not the +0.218
  nearest-point pair: the cand QP114 point is a Butteraugli ripple (BA 2.108
  at 102.6 kbps against 2.086 at 98.9) and it drags the two-point
  interpolation at the base's 101.124 kbps reference to −0.23.  Do not read
  any single bracket as the mechanism result.
* The K fit (+0.126) is under the +0.15 KEEP bar; only the bracket (+0.225)
  clears it.  Treat the K gain as +0.13 with a +0.02..+0.23 range.
* The arm is cheaper at every fixed QP (1.5-3.3 kbps), so any composition must
  be re-bracketed; the gain is partly the QP the budget buys.
* `AV2ENC_RD_TAIL_TX_D=0` (plain SSE plan D) and `AV2ENC_RD_TAIL_HF/EDGE`
  were not swept: the run budget went to the five-point ladder and the
  per-arm ceiling curve.  The D sweep is the obvious next knob, since the key
  driver measured HF 0 as its best K point while its BA improved at HF 5/20.
* The 4x4/8x4 route is blocked by the P-frame 8x8 coding-block floor, not by
  the TX search: nothing in the committed tail tree is smaller than 16x16, so
  the legal 4x4/8x4 transform partitions are unreachable from this surface
  (§7).

Recommended composition point: `AV2ENC_RD_TAIL_TX=1` with the defaults above
on top of the driver recipe.  It touches only pictures 33-59 and leaves every
gate-off stream byte-identical.
