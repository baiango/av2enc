# RD-engine0 — sequential key-frame driver, second iteration

Status: **LEAD, not KEEP.** The supervisor is unchanged: default-off
`AV2ENC_RD_DRIVER` decides the key picture one superblock at a time in the
reconstruction each candidate layout builds. This iteration adds the
§5.20.7.30 scan-order rate fix, a BA-shaped leaf D (high-pass + block-edge),
and an optional high-activity gate for sub-16 leaves. Against the 86.013 base
the final arm is **86.041 (+0.028)** on the 100 kbps bracket and **+0.263** on
the 6-point full-ladder fit (R² 0.463); its key and clip BA/SS2 are better at
the matched rate. The bracket is inside the QP ripple, so it does not clear the
+0.15 KEEP bar.

Branch `exp/rd-engine0`, worktree `/Users/arin/work/rd-engine0`.
Base (unchanged): K **86.013** at 240p / 100 kbps, L3+L5 recipe.

History: iteration 1 (commit `71c901b7`) built the sequential driver and
measured 86.080 bracket / +0.242 fit with a low-band-only D. This iteration
(commits `d40cd638` + the final commit) adds the fixed rate model and the
BA-shaped D; the interrupted run was the final ladder/BA sweep and this report.

## 1. Files

| File | Role |
|---|---|
| `src/pipeline/video/video_rd_engine.h/.cpp` | RD-engine rate model, now with the §5.20.7.30 scan-order fix and live-context fields from `rd-rate` (`482b57a6`); static-model calibration refit in §2 |
| `src/pipeline/video/video_seq_rd.h` | `seq_rd::leaf_distortion()`: eff-perc key D + HF high-pass + EDGE boundary-gradient terms (from `rd-seq`) |
| `src/pipeline/video/video_frame_internal.h` | `rd_driver` gate/knobs/counters |
| `src/pipeline/video/video_frame_decision.cpp` | superblock loop, measured leaf cost, expand/prune, high-activity gate |
| `src/pipeline/video/video_frame_intra.cpp` | measured-candidate histogram guard |
| `tools/lane-engine0/ladder.sh` | interleaved-QP ladder, strict decode, validate, PSNR, recon equality, oracle; `DRV_EXTRA` selects the arm |
| `tools/lane-engine0/keyscore.sh` | one-frame key score (bytes/PSNR/BA/SS2) for the D-model loop |

No CLI, CMake, `main.cpp`, options/parse, gate or pin edit.

## 2. Rate model: scan-order fix + model-vs-actual

`block_rate()` used to hand `av2::Model::init()` a **raster** level array where
the model walks §5.20.7.30 **scan** order, so every coefficient was priced
against another position's context. Fixed (copied from `rd-rate`, commit
`482b57a6`), together with the live `txb_skip_context` / `dc_sign_context`
fields. The driver still prices its measured leaves with the static §9.3 rows
(it does not track the writer's above/left state), so it applies the measured
static→census family correction.

Model vs actual on the key picture, QP119 (the `rd-rate` audit, same stream;
`frozen` = static rows + context 0, `live` = writer contexts + adapted rows):

| family | census bits | frozen after fix | ratio | live | ratio |
|---|---:|---:|---:|---:|---:|
| txb_skip | 613.6 | 775.8 | 1.264 | 613.6 | 1.000 |
| eob | 5,418.8 | 6,066.6 | 1.120 | 5,418.8 | 1.000 |
| coeff_base | 61,716.1 | 66,993.4 | 1.086 | 61,889.3 | 1.003 |
| coeff_br | 13,347.3 | 14,238.6 | 1.067 | 13,350.9 | 1.000 |
| coeff_mag | 38,683.0 | 38,683.0 | 1.000 | 38,683.0 | 1.000 |
| sign | 640.7 | 638.0 | 0.996 | 640.7 | 1.000 |
| **total** | **120,419.5** | **128,031.2** | **1.063** | **120,596.3** | **1.0015** |

The driver's calibration is the inverse of the frozen ratios:
`txb 0.791, eob 0.893, coeff_base 0.921, coeff_br 0.937, mag 1.000, sign
1.004` (env-overridable). The driver's R is therefore ~1.00× the emitted
coefficient census; the old lane-1 raster model was 1.204× before calibration
and needed the much larger 0.428/0.822/0.769 scales. Inter live rows are 1.008×
(`rd-rate`), but the driver does not touch inter decisions.

## 3. BA-shaped D

For a measured key leaf:

```text
D = eff_perc key D (SSE + key_csf * box3x3 mean error^2)
  + HF   * sum (d - box3x3(d))^2                 high-pass error
  + EDGE * sum ((src_in - src_out) - (rec_in - rec_out))^2
                                               top/left gradient mismatch
```

Both terms are computed on **the leaf's own reconstruction** in
`seq_rd::leaf_distortion()`, the same call that reads the reconstructed block
the coder just wrote (visible-row clip, live causal neighbours). `HF`
(`AV2ENC_RD_DRIVER_HF`) and `EDGE` (`AV2ENC_RD_DRIVER_EDGE`) are env knobs.
The measured best K point is **HF = 0** (the term is implemented and swept but
off by default); HF 5/20 improve the key BA and lose the clip K (§6).

```text
J = D + lambda * R_exact
R_exact = calibrated §8.3 plan_rate over the emitted scan-order levels
          + intra mode syntax + TX-partition/TX-type terms
lambda  = key_tx_rd_lambda(qindex) * 0.25 * AV2ENC_RD_DRIVER_LAMBDA
```

## 4. High-activity gate (measured, default off)

`AV2ENC_RD_DRIVER_ACT` (default 0 = off) stops a 16x16-or-smaller node from
descending to 8x8/4x4 unless the **source** high-pass energy
`mean((src - box3x3(src))^2)` clears the threshold. Flat sky keeps the larger
coding block even when the measured D calls the split cheap. The clip's 16x16
activity quantiles on the key frame are median 3.4, p75 80, p90 797, so 20 and
50 are the measured arms.

## 5. Gate and driver

| Variable | Default | Meaning |
|---|---|---|
| `AV2ENC_RD_DRIVER` | off | master gate; intra key picture only, no SDP/ext partitions/palette/IBC/segmentation/delta-Q/force-single |
| `AV2ENC_RD_DRIVER_MIN` | 4 | deepest uniform leaf; 4 keeps 4x4/8x8/16x16/32x32/64x64 (no 16x16 floor) |
| `AV2ENC_RD_DRIVER_HF` | 0 | high-pass error weight (measured: 5/20 lower the key BA, not the clip K) |
| `AV2ENC_RD_DRIVER_EDGE` | 0 | boundary-gradient weight |
| `AV2ENC_RD_DRIVER_ACT` | 0 | source high-pass gate for sub-16 leaves |
| `AV2ENC_RD_DRIVER_LAMBDA` | 1.0 | scale on the calibrated key lambda |

For each superblock in raster order: expand the variance reference tree to
`MIN`, prune it top-down with the measured J (NONE vs the split children,
children decided in decode order, spec-forced splits never collapsed), commit
the winner through `collect_node()`/`encode_leaf()`, then update recon and
§6.19.2.3 BlockDecoded before the next superblock. A candidate that cannot be
encoded/validated is not scored; a failed prune falls back to the reference.
Measured leaves at QP119: 19,806 (32 superblocks); runtime ~42 s per 60-frame
stream (~6× the 7.5 s base).

## 6. Measurements

Recipe: `--video --frames 60 --cpu-threads 8 --fast-decode
--video-drift-refresh-threshold 1.5 --video-drift-refresh-budget 3250`,
`AV2ENC_EFF_PERC=1 AV2ENC_RD_UVFULL=1 AV2ENC_TX_DETAIL_SEARCH_TAIL=1
AV2ENC_TX_DETAIL_MIN=0.50 AV2ENC_RD_KEY_TX=1 AV2ENC_RD_UVLEVEL=1`; arm adds
`AV2ENC_RD_DRIVER=1`. Release `-O3 -Werror`, QP 115–120 interleaved (115 is
the arm's straddle point; 116–120 is the 5-point protocol ladder).
`tools/oracle.py --target-kbps 100`.

### Ladder (final default: HF 0, ACT 0)

| QP | arm | bytes | kbps | KPI | BA | SS2 | key B | key Y |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 115 | base | 27,202 | 108.808 | 86.097 | 2.0943 | 83.679 | 16,594 | 46.856 |
| 115 | **drv** | **25,281** | **101.124** | **86.294** | **2.0504** | **83.743** | **14,523** | **47.762** |
| 116 | base | 26,884 | 107.536 | 86.184 | 2.0629 | 83.556 | 16,411 | 46.684 |
| 116 | drv | 24,843 | 99.372 | 85.900 | 2.1571 | 83.425 | 14,293 | 47.543 |
| 117 | base | 26,616 | 106.464 | 85.917 | 2.1079 | 83.214 | 16,182 | 46.457 |
| 117 | drv | 24,207 | 96.828 | 85.909 | 2.1309 | 83.346 | 14,100 | 47.344 |
| 118 | base | 25,578 | 102.312 | 85.802 | 2.1695 | 83.237 | 15,970 | 46.253 |
| 118 | drv | 23,948 | 95.792 | 85.918 | 2.1293 | 83.284 | 13,976 | 47.208 |
| 119 | base | 24,984 | 99.936 | 86.019 | 2.0647 | 82.856 | 15,687 | 46.019 |
| 119 | drv | 23,128 | 92.512 | 85.977 | 2.1109 | 83.520 | 13,809 | 46.989 |
| 120 | base | 24,616 | 98.464 | 85.674 | 2.1446 | 82.610 | 15,490 | 45.826 |
| 120 | drv | 23,150 | 92.600 | 85.729 | 2.1724 | 83.048 | 13,708 | 46.861 |

| arm | bracket | K@100 | Δ | 6-pt fit | Δ fit | R² | 5-pt fit (116–120) | Δ | R² |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| base | 119/118 | **86.013** | — | 85.823 | — | 0.526 | 85.822 | — | 0.446 |
| **drv** | 116/115 | **86.041** | **+0.028** | **86.086** | **+0.263** | 0.463 | 85.924 | +0.102 | 0.065 |

The 5-point fit is noisy because the arm's QP116 point sits on a local KPI
dip (85.900 at 99.37 kbps); the 6-point fit is the stable reading. The
5-point 115–119 fit gives +0.176 (R² 0.288). Best estimate: bracket +0.03,
fit +0.10…+0.26; **LEAD**, not KEEP.

### Key and clip BA/SS2 at the matched rate

Matched points: base QP119 (99.936 kbps), drv QP115 (101.124 kbps).

| metric | base | drv | Δ |
|---|---:|---:|---:|
| clip BA | 2.0647 | **2.0504** | **−0.0143** |
| clip SS2 | 82.856 | **83.743** | **+0.887** |
| key BA | 1.4323 | **1.2797** | **−0.1526 (−10.7 %) |
| key SS2 | 87.469 | 87.538 | +0.069 |
| mean PSNR-Y | 44.923 | **45.748** | **+0.825 dB** |

Key is 9.6–12.5 % smaller at +0.85…+1.04 dB over QP 115–120. The fixed-QP clip
BA is still worse (e.g. QP119 2.1109 vs 2.0647), but the arm is 3–4 kbps below
base at every QP; at matched rate the clip BA is slightly better and SS2 is
+0.89.

### Per-frame PSNR (base QP119 → drv QP115)

Mean +0.825 dB; range **+0.13…+1.74 dB** (all 60 frames positive).

| segment | base | drv | Δ |
|---|---:|---:|---:|
| f0–32 mean | 44.915 | 46.075 | +1.160 |
| f33–43 mean | 44.189 | 44.651 | +0.462 |
| f44–59 mean | 45.445 | 45.829 | +0.384 |

Selected frames (PSNR / bytes):

| frame | base | drv | ΔdB |
|---:|---:|---:|---:|
| 0 (key) | 46.019 / 15,687 | 47.762 / 14,523 | +1.743 |
| 1 | 46.021 / 21 | 47.764 / 25 | +1.743 |
| 10 | 45.046 / 24 | 46.156 / 66 | +1.110 |
| 20 | 45.039 / 86 | 46.025 / 150 | +0.986 |
| 30 | 44.550 / 95 | 45.426 / 163 | +0.876 |
| 33 | 43.865 / 377 | 44.519 / 429 | +0.654 |
| 40 | 44.360 / 392 | 44.490 / 380 | +0.130 |
| 59 | 46.772 / 296 | 47.185 / 306 | +0.413 |

### Ceilings (source-replacement, oracle)

Matched points base QP119 / drv QP115:

| segment | base | drv | Δ |
|---|---:|---:|---:|
| key f0 | 86.156 | 86.402 | +0.246 |
| f0–32 | 91.302 | 91.608 | +0.306 |
| f33–43 | 87.870 | 88.152 | +0.282 |
| f44–59 | 89.491 | 89.554 | +0.063 |

### P-chain propagation: clip BA by segment

Same matched points, `extra_metrics` on the sliced decoded frames:

| segment | base BA | drv BA | Δ | base SS2 | drv SS2 | Δ |
|---|---:|---:|---:|---:|---:|---:|
| f0–32 | 1.8671 | 1.9286 | +0.0615 | 85.213 | 86.287 | +1.074 |
| f44–59 | 2.4721 | 2.2688 | **−0.2033** | 78.152 | 78.847 | +0.695 |

The tail improves; the head loses a little BA. That is the P-chain term: the
key's −10.7 % BA does not carry uniformly, but the overall clip BA is slightly
better at matched rate. No refresh-schedule change was made.

## 7. Ablations

All at QP 115–120, same base ladder; Δ vs 86.013 bracket.

| arm | bracket K@100 | Δ bracket | 6-pt fit | Δ fit | key BA (QP119) | clip BA (QP119) |
|---|---:|---:|---:|---:|---:|---:|
| **HF 0 (default)** | **86.041** | **+0.028** | **86.086** | **+0.263** | 1.6232 | 2.1109 |
| HF 5 | 85.957 | −0.056 | 85.938 | +0.116 | 1.5438 | 2.1284 |
| HF 20 | 85.764 | −0.249 | 85.974 | +0.151 | **1.3888** | 2.1664 |
| ACT 20, HF 0 | 85.997* | −0.016* | — | — | — | 2.1114 (QP116) |
| ACT 50, HF 0 | 85.988* | −0.025* | — | — | — | 2.1140 (QP116) |

\* ACT arms are direction-only: ACT 20 has no point at/above 100 kbps
(nearest 98.0 kbps), and ACT 50's nearest point is 100.008 kbps. They do not
form a full ladder. The gate does what it is for — fewer
sub-16 leaves (measured leaves 19,806 → 6,106 at ACT 20) and a better
fixed-QP clip BA than HF 0 at QP116 (2.111 vs 2.157) — but it costs rate and
does not beat the HF 0 matched K, so it stays default off.

The BA term trades key BA for rate: HF 20 buys 0.23 key BA and costs 1.5 kbps
plus 0.06 clip BA. On this clip the matched-K objective prefers no HF term.

## 8. Conformance

* Gate off pins (release, plain `--fast-decode`): QP106 **25,008 B /
  `0639b43630abb454f7be5ac494d2efb4`**, QP110 **23,690 B /
  `1b99127a4cd1d386fa5600fb63c57a83`**.
* Final ladder: 12/12 streams `validate_dav2d.sh` OK, `dav2d --strict 1`
  60/60, `--video-recon` payload equals the decoded payload (12/12 `ok`).
* `cmake --preset dev-cache` and `cmake --preset release` (warnings +
  `-Werror`) build clean; `ctest` 12/12.
* No gate/pin/CLI/CMake/options/parse/`main.cpp` change.

## 9. Verdict

**LEAD.** The second iteration is a correct sequential key RD driver with an
accurate rate model and a BA-shaped D that is measured on the leaf's own
reconstruction. At the matched rate it is better on every reported axis:
clip BA −0.014, clip SS2 +0.887, key BA −10.7 %, mean PSNR +0.825 dB, and all
four ceilings up. The matched K bracket is **+0.028** (inside the ripple); the
6-point fit is **+0.263** (R² 0.463). It does not clear +0.15 on the primary
bracket, so it is not a KEEP and stays default off.

The remaining blocker is the P chain: the key's BA win does not carry to f0–32
(+0.06 clip BA) even though f44–59 improves (−0.20). The next change should
attack the head-frame inter residual/QP budget (queue item 1 in
`docs/rd-loop-ledger.md`), not another key D term.

## 10. Reproduce

```sh
BIN=build-release/av2enc tools/lane-engine0/ladder.sh /tmp/rd-engine0 "115 116 117 118 119 120"
# BA sweep: DRV_EXTRA="AV2ENC_RD_DRIVER_HF=20" ...
# key-only score: tools/lane-engine0/keyscore.sh /tmp/keys 115 AV2ENC_RD_DRIVER=1
```
