# RD P-driver — head-only measured RD on the P chain

Status: **LEAD, not KEEP.** The head-only exact-rate / BA-shaped TX and level RD
(`AV2ENC_RD_HEAD_TX`, the extended `AV2ENC_RD_TAIL_TX` mechanism) makes the head
**1.5x cheaper at better PSNR and better BA/SS2 at the same QP**, and lifts every
source-replacement ceiling. At the matched 100 kbps point it moves K by **+0.160
on the bracket / +0.075 on the six-point fit**, which does not clear the +0.15
KEEP bar on the reading that survives the QP ripple. The requested head
drift-aware skip/reference term (`AV2ENC_P_DRIFT`) was implemented, measured,
and is a **no-op on this clip** with a precise reason (§4); the head's
under-spend is a TX/level defect, not a drift defect.

Branch `exp/rd-pdriver`, worktree `../rd-pdriver`, from `c6116f04`.
Base K **86.013** bracket (reproduced as **86.041** on this six-point ladder;
`docs/rd-engine0.md` §6 gives 86.041 for the same five base points), 240p /
100 kbps, the merged L3+L5+driver recipe plus `--fast-decode
--video-drift-refresh-threshold 1.5 --video-drift-refresh-budget 3250`.

## 1. Scope

The composition measurement (`docs/rd-compose.md`) shows the whole remaining K
gap is Butteraugli (−2.82 K) and SSIMULACRA2 (−1.06 K); it splits ~48 % head
f1-32 / ~48 % tail f33-59 / 1 % key, and the constant head penalties (QP
offset, non-zero-MV skip penalty, low-band refresh trigger) are all closed
negative. Two mechanisms were built, in the priority order this lane was given:

1. **`AV2ENC_P_DRIFT`** — head-only drift-aware skip/reference cost, priced from
   the real future source frames the flat P chain already holds. §4.
2. **`AV2ENC_RD_HEAD_TX`** — head-only exact-rate / BA-shaped TX and level RD
   (the prescribed fallback), arming the existing tail mechanism on pictures
   1-32. §5.

Two further probes are reported as negatives: a head partition-descent gate
(§7) and an all-zero-residual collapse (§7).

## 2. Files

| File | Role |
|---|---|
| `src/pipeline/video/video_p_drift.h` | new: `AV2ENC_P_DRIFT` gate, eff-perc-units future-frame distance, counters |
| `src/pipeline/video/video_tail_tx.h` | `AV2ENC_RD_HEAD_TX` / `AV2ENC_RD_HEAD_LAST`: the exact-rate stage armed on pictures 1-32; `tail_armed()` is a **union** of the head and tail windows |
| `src/pipeline/video/video_frame_internal.h` | `tx_detail::band_descent_armed()` (tail window or head window), `AV2ENC_RD_HEAD_PART` |
| `src/pipeline/video/video_frame_decision.cpp` | the drift surcharge in the inter-leaf skip/residual comparison |
| `tools/lane-pdriver/ladder.sh` | interleaved multi-arm ladder; strict decode, validate, per-frame PSNR, recon equality, oracle |
| `tools/lane-pdriver/perframe_metrics.py` | per-segment BA / SS2 / PSNR-Y (key, head f1-32, f33-43, f44-59, all) |
| `tools/lane-pdriver/head_stats.py` | head bytes/picture and the residual-arm composition from the census/decision dumps |
| `tools/lane-pdriver/kfit.py` | bracket K, OLS fit K, R², per-QP rate-adjusted deltas |

No CLI, CMake, `main.cpp`, options/parse, gate or pin edit. 4:2:0 only by
construction (the drift term reads `VideoInputFrame::y/u/v`). Every gate is
default off.

## 3. Gates

| Variable | Default | Meaning |
|---|---|---|
| `AV2ENC_RD_HEAD_TX` | off | arm the exact-rate / BA-shaped TX + level stage on pictures 1-32. Composes with the tail window (union) |
| `AV2ENC_RD_HEAD_LAST` | 32 | last head picture |
| `AV2ENC_P_DRIFT` | off | head-only drift-aware skip/reference cost |
| `AV2ENC_P_DRIFT_K` | 4 | horizon depth (1..8) |
| `AV2ENC_P_DRIFT_ALPHA` | 0.5 | weight on the summed future term |
| `AV2ENC_P_DRIFT_FIRST/LAST` | 1 / 32 | armed picture window |
| `AV2ENC_P_DRIFT_STATS` | off | print counters |
| `AV2ENC_RD_HEAD_PART` | off | band-gated partition descent on the head (§7, measured negative) |

## 4. Arm 1 — head drift-aware skip/reference cost (measured no-op)

### Design

A skipped leaf installs its prediction error in the reconstructed reference,
where the following pictures copy it through fractional-pel MC. The leaf
decision prices the skip arm with this picture's error only, so a stream that
never repairs a frozen error keeps skipping it. The gate adds the error each arm
installs in the next K **real source pictures** (`frames.subspan(i + 1)`, already
carried as `InterFrameEncoder::horizon_future_`):

```text
J_skip = D_perc(prediction) + alpha * sum_{k=1..K} 0.5^k * D_perc(src[i+k], skip_recon) + lambda * R_skip
J_res  = D_perc(recon)      + alpha * sum_{k=1..K} 0.5^k * D_perc(src[i+k], res_recon)  + lambda * R_res
```

Two properties separate it from the merged `AV2ENC_PORT_HORIZON`, which lost
0.26-0.50 K on this exact recipe (`docs/port-lane2.md`):

* **Units.** The port added a *plain* future SSE to a *perceptual* current D, so
  on high-activity leaves the unweighted future term could not outbid a
  low/mid-band-weighted current one and skip share rose while every segment lost
  PSNR. `p_drift::error_distortion()` reproduces the eff-perc form exactly —
  `sum(w * (e^2 + csf * mean3x3(e)^2)) / 256` with the leaf's own Q8 weight map
  and the configured CSF/CSF2 gains — so both arms compare one quantity.
* **Scope.** Head pictures 1-32 only. The drift-refresh schedule code and its
  thresholds are untouched (its refresh count still moves, because it is
  decided from the reconstruction).

(The first cut of the header dropped the Q8 neutral 256 and was 256x too small;
instrumenting the term against the current-frame D found it, and the numbers
below are the corrected implementation.)

### Measurement (QP116, 60 frames, base recipe)

| arm | α | head skip→res flips | head B/picture | head BA | total B |
|---|---:|---:|---:|---:|---:|
| base | — | — | 517.2 | 2.0174 | 24,843 |
| drift | 0.5 | 5 | 517.2 | 2.0174 | 24,843 |
| drift | 1 | 25 | 521.5 | 1.9836 | 25,149 |
| drift | 2 | 32 | 529.8 | — | 24,997 |
| drift | 4 | 28 | 539.2 | 2.0178 | 25,111 |
| drift | 8 | 27 | 558.0 | — | 25,397 |
| drift | 50 | 44 | 566.2 | — | 25,391 |
| drift | 1000 (forcing) | 0 | 562.8 | 1.9396 | 25,511 |
| `RD_HEAD_TX`, for comparison | — | — | **349.0** | **1.9380** | **23,798** |

**The mechanism cannot fire on this clip.** The decision dump
(`tools/lane-pdriver/head_stats.py --segment head`) shows the head is not a skip
chain at all. At QP116, of 1,372 head inter leaves:

* pictures 1-4 are 100 % skip (166 leaves),
* pictures 5-32 select the residual arm on **every** leaf (1,160 leaves),
* but **916 of those 1,160 code `eob_y = eob_u = eob_v = 0`**.

So 79 % of the head's residual-arm leaves already cost nothing but the TX
symbols and a skip/residual term has nothing to trade there; and on the 244
leaves it *could* move, the two arms' future error differ by only ~2.3 %
(`fut[skip]` 30.5 M vs `fut[res]` 29.9 M at α = 1), which the coefficient rate
outbids. Forcing the term to α = 1000 changes nothing structural: the head goes
517 → 563 B/picture rather than toward the 349 B/picture the TX mechanism
reaches.

The head's real defect is that the residual it does code is priced with the
RDOQ-calibrated static model — the exact defect `AV2ENC_RD_TAIL_TX` fixed on the
tail. That is arm 2.

The gate stays implemented and default off: it is a correct, unit-consistent
implementation of the requested mechanism, and the negative is the evidence
that the head under-spend is a TX/level problem, not a drift problem. Do not
re-open it with another constant.

## 5. Arm 2 — head-only exact-rate / BA-shaped TX and level RD (the arm)

### Design

`AV2ENC_RD_TAIL_TX` (`docs/rd-tail.md`) already applies the L3 mechanism to
pictures 33-59: on the inter luma residual the plan is chosen with

```text
D = seq_rd::leaf_distortion_gains() on the assembled-block reconstruction
    (eff-perc inter CSF; HF / EDGE as configured)
R = plan_rate_bits()  -- §8.3 through §8.2.6 interval, live carry image
J = D + pipe_lambda * AV2ENC_RD_TAIL_TX_LAMBDA * R
```

by a two-stage search (the historical screen, then one exact re-search at
`model_rate_scale = 1.0` with the coefficient trellis). `AV2ENC_RD_HEAD_TX`
arms exactly that stage on pictures 1-32. Nothing is duplicated: the window is
the only change, and `tail_armed()` is a **union**, so `AV2ENC_RD_HEAD_TX=1`
and `AV2ENC_RD_TAIL_TX=1` compose. The drift-refresh schedule code and thresholds
are untouched.

### Fixed-QP effect (QP116)

| segment | base B | head-tx B | Δ |
|---|---:|---:|---:|
| key f0 | 114,072 | 114,072 | 0.0 % |
| head f1-32 | 16,552 | 11,168 | **−32.5 %** |
| f33-43 | 25,336 | 25,944 | +2.4 % |
| f44-59 | 28,976 | 29,344 | +1.3 % |
| all | 184,936 | 180,428 | −2.4 % |

| metric (QP116) | base | head-tx |
|---|---:|---:|
| head B/picture | 517.2 | **349.0** |
| head BA | 2.0174 | **1.9380** |
| head SS2 | 86.152 | **86.212** |
| mean PSNR-Y | 45.6545 | **45.6622** |
| drift-refresh pictures | 32 | 28 |

The refresh *schedule and thresholds* are unchanged; the refresh *count* moves
(32 → 28 at QP116, 30 → 34 at QP114) because the schedule is decided from the
reconstructed reference the head has just changed. Nothing in the schedule code
was touched.

Head bytes per picture at every rung of the ladder:

| QP | 113 | 114 | 115 | 116 | 117 | 118 |
|---|---:|---:|---:|---:|---:|---:|
| base | 645.0 | 616.2 | 517.2 | 517.2 | 415.0 | 394.0 |
| head-tx | 497.0 | 458.0 | 404.0 | 349.0 | 363.0 | 312.2 |
| Δ | −23 % | −26 % | −22 % | −33 % | −13 % | −21 % |

The head becomes **1.5x cheaper at flat-to-better PSNR**; the tail's +2 % is
footprint drift (the head's cheaper reconstruction changes the reference the
tail predicts from, and the refresh schedule re-plans against it), not the
tail's own objective.

## 6. Measurements

Recipe: `--video --frames 60 --cpu-threads 8 --fast-decode
--video-drift-refresh-threshold 1.5 --video-drift-refresh-budget 3250`,
`AV2ENC_EFF_PERC=1 AV2ENC_RD_UVFULL=1 AV2ENC_TX_DETAIL_SEARCH_TAIL=1
AV2ENC_TX_DETAIL_MIN=0.50 AV2ENC_RD_KEY_TX=1 AV2ENC_RD_UVLEVEL=1
AV2ENC_RD_DRIVER=1`; arm adds `AV2ENC_RD_HEAD_TX=1`. Release `-O3 -Werror`, six
interleaved QPs 113-118, arm order rotated per QP, driven under `bash` by
`tools/lane-pdriver/ladder.sh`. Matched K is `tools/oracle.py --target-kbps 100`.

### Ladder

| QP | arm | bytes | kbps | K | BA | SS2 | JOD | PSNR-Y |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 113 | base | 26,372 | 105.488 | 86.310 | 2.0675 | 83.812 | 9.8415 | 45.859 |
| 113 | **head-tx** | **25,376** | **101.504** | 86.162 | 2.0974 | 83.748 | 9.8314 | 45.775 |
| 114 | base | 26,160 | 104.640 | 86.254 | 2.0677 | 83.781 | 9.8315 | 45.661 |
| 114 | **head-tx** | **25,073** | **100.292** | **86.236** | 2.0858 | **83.974** | 9.8320 | 45.769 |
| 115 | base | 25,281 | 101.124 | 86.294 | 2.0504 | 83.743 | 9.8306 | 45.674 |
| 115 | head-tx | 24,436 | 97.744 | 85.930 | 2.1692 | 83.648 | 9.8292 | 45.696 |
| 116 | base | 24,843 | 99.372 | 85.900 | 2.1571 | 83.425 | 9.8239 | 45.580 |
| 116 | **head-tx** | **23,798** | **95.192** | **86.161** | **2.0786** | 83.647 | 9.8237 | 45.576 |
| 117 | base | 24,207 | 96.828 | 85.909 | 2.1309 | 83.346 | 9.8135 | 45.478 |
| 117 | head-tx | 23,469 | 93.876 | 85.906 | 2.1543 | 83.507 | 9.8207 | 45.497 |
| 118 | base | 23,948 | 95.792 | 85.918 | 2.1293 | 83.284 | 9.8165 | 45.370 |
| 118 | head-tx | 23,192 | 92.768 | 85.801 | 2.1556 | 83.153 | 9.8123 | 45.349 |

The union arm's head row is byte-for-byte the head-only arm's head row
(QP116: head B/pic 349.0, head BA 1.9380, `MD5` of the 23,798-byte union stream
equals the earlier all-P experiment's), so the two gates are orthogonal: the
head gate owns pictures 1-32, the tail gate owns 33-59, and
`tail_armed()` being a union is the only interaction.

The `head-tx` column is the **union arm** (`AV2ENC_RD_HEAD_TX=1
AV2ENC_RD_TAIL_TX=1`): the head-only arm alone is reported in §7 and is K −0.008
bracket / +0.058 fit, i.e. the tail mechanism carries the rest.

| arm | bracket | K@100 | Δ bracket | 6-pt fit | Δ fit | slope | R² |
|---|---|---:|---:|---:|---:|---:|---:|
| base | 116/115 | **86.041** | — | **86.073** | — | +0.0452 | 0.755 |
| head-tx (union) | 115/114 | **86.201** | **+0.160** | **86.148** | **+0.075** | +0.0371 | 0.556 |

Per-QP rate-adjusted deltas (base slope 0.0452 K/kbps):

| QP | Δkbps | ΔK | rate-adjusted ΔK | ΔBA | ΔSS2 |
|---:|---:|---:|---:|---:|---:|
| 113 | −3.98 | −0.148 | +0.032 | +0.0299 | −0.065 |
| 114 | −4.35 | −0.018 | +0.179 | +0.0180 | +0.193 |
| 115 | −3.38 | −0.364 | −0.211 | +0.1188 | −0.095 |
| 116 | −4.18 | +0.261 | +0.450 | −0.0785 | +0.222 |
| 117 | −2.95 | −0.003 | +0.130 | +0.0234 | +0.162 |
| 118 | −3.02 | −0.117 | +0.019 | +0.0264 | −0.130 |

The arm is 3.0-4.3 kbps cheaper at every fixed QP; the rate-adjusted sign is
positive at five of six QPs (mean **+0.100**) and matches the fit. QP115 is the
one negative: it is the base's best BA point of the ladder (base 2.0504 vs 2.0675
and 2.1571 on its neighbours), so it flatters the base rather than indicting the
arm.

The bracket (+0.160) clears +0.15; the fit (+0.075) does not. The bracket's own
bracket pair (`head-tx` 115/114) has 115 as the arm's ripple point, so the
honest reading is **+0.075…+0.160**, and the lane does not claim KEEP.

### Matched-rate segments

Match points: base QP116 (99.372 kbps) vs head-tx QP114 (100.292 kbps).

| segment | metric | base | head-tx | Δ |
|---|---|---:|---:|---:|
| key f0 | BA | 1.4993 | **1.4151** | −0.0842 |
| key f0 | SS2 | 87.637 | 88.006 | +0.369 |
| head f1-32 | BA | 2.0174 | **1.9295** | **−0.0879** |
| head f1-32 | SS2 | 86.152 | **86.589** | **+0.437** |
| head f1-32 | PSNR-Y | 45.8882 | 46.1316 | +0.243 |
| f33-43 | BA | 2.1117 | **2.0537** | −0.0580 |
| f33-43 | SS2 | 82.386 | **83.275** | **+0.889** |
| f44-59 | BA | 2.5090 | 2.4622 | −0.0468 |
| f44-59 | SS2 | 78.420 | **78.975** | +0.555 |
| clip | BA | 2.1571 | **2.0858** | −0.0713 |
| clip | SS2 | 83.425 | **83.974** | +0.549 |
| clip | JOD | 9.8239 | 9.8320 | +0.0081 |
| clip | mean PSNR-Y | 45.580 | **45.849** | +0.269 |

Every segment improves on both perceptual metrics at the matched rate. At the
*base's own* QP115 compare (101.124 kbps) the arm at QP113 reads head BA 1.9899
/ SS2 86.498 vs the base's 1.9489 / 86.248 — head BA slightly worse, SS2 +0.25 —
so the head gain is sign-consistent on SS2 and mixed on BA below 100 kbps. The
100 kbps reading above is the protocol point.

### Skip share

Head inter-leaf skip share is 15.5 % at QP116 base (212 of 1,372) and the arm
changes the *residual*, not the skip/residual split: the head's rate falls
32.5 % with the residual-arm count unchanged. The clip-wide inter skip share is
47.9 % (3,304 of 6,898 leaves) and is not the mechanism's target.

### Ceilings (source replacement, oracle)

Matched points base QP116 (99.372 kbps) vs head-tx QP114 (100.292 kbps):

| segment | base | head-tx | Δ |
|---|---:|---:|---:|
| key f0 | 86.020 | **86.346** | **+0.326** |
| f0-32 | 91.248 | **91.396** | **+0.148** |
| f33-43 | 87.747 | **88.041** | **+0.294** |
| f44-59 | 89.311 | **89.696** | **+0.385** |

All four ceilings rise. The f0-32 row is the head mechanism's own ceiling and
the f44-59 row (which pools the head plus f33-43) rises the most.

## 7. Ablations

All at QP 116 fixed, same base, head bytes/picture and head BA
(`tools/lane-pdriver/head_stats.py`, `perframe_metrics.py`):

| arm | head B/pic | head BA | bytes | verdict |
|---|---:|---:|---:|---|
| base | 517.2 | 2.0174 | 24,843 | — |
| **`RD_HEAD_TX` (head only)** | **349.0** | **1.9380** | **24,327** | the mechanism alone; K −0.008 bracket / +0.058 fit |
| **`RD_HEAD_TX` + `RD_TAIL_TX` (union)** | **349.0** | **1.9380** | **23,798** | the §6 ladder arm; the head value is identical, the tail is exact-rate too |
| `P_DRIFT` α = 1 | 521.5 | 1.9836 | 25,149 | no-op (916/1,160 residual-arm leaves are all-zero EOB) |
| `RD_HEAD_PART` (band descent) | **804.2** | — | 26,203 | **negative**: +56 % head rate |
| `RD_HEAD_PART` + `RD_HEAD_TX` | 804.2 | — | 26,203 | identical — the TX stage is behind the partition gate |
| `RD_HEAD_PART` + `DETAIL_MIN=0.75` | 804.2 | — | 26,203 | identical — the threshold never binds |

`AV2ENC_RD_HEAD_PART` opens the measured probe's first-level clamp
(`partition_probe_depth` = 1, which is why the head codes 597 of 1,372 leaves at
64x64 and 358 at 64x32) and runs the tail's band-gated descent on pictures 1-32.
It spends +56 % head rate: the same retreat-to-coarser-ancestors behaviour
`docs/rd-tail.md` §7 measured for the tail sub-16 gate, one level up. It stays
implemented and default **off**. Its being identical with the TX stage and with
`DETAIL_MIN` 0.75 confirms the gate, not the threshold, is what fired.

An all-zero-residual collapse (choose skip when the residual arm emits
`eob = 0` on all planes) was implemented and is **bit-identical** to the base:
the pipeline's exact-rate comparison already prefers the skip symbol, so there
is no leak to fix. It was removed rather than left as a no-op knob.

## 8. Conformance

* Gate off pins, final release build: QP106 **25,008 B /
  `0639b43630abb454f7be5ac494d2efb4`**, QP110 **23,690 B /
  `1b99127a4cd1d386fa5600fb63c57a83`** — reproduced after every code change,
  including with the probe knob removed.
* Final ladders: 24/24 streams `tools/validate_dav2d.sh` OK, `dav2d --strict 1`
  60/60, `--video-recon` payload equals the decoded payload (24/24 `ok`).
* `cmake --preset release` (warnings + `-Werror`) and `cmake --preset
  dev-cache` build clean; `ctest --test-dir build-release` 12/12;
  `tools/gate_video_nightly.sh` and `src/pipeline/video/gate_video_eff_perc.sh`
  OK.
* The final binary reproduces the ladder stream: `AV2ENC_RD_HEAD_TX=1
  AV2ENC_RD_TAIL_TX=1` at QP116 is 23,798 B / `775d76acd6ebdf52c9baa9928807cd26`.
* No gate/pin/CLI/CMake/options/parse/`main.cpp` change.

## 9. Verdict

**LEAD.** Judged on the rule as given (KEEP at ≥ +0.15 with a consistent
ladder), the union arm reads **+0.160 bracket / +0.075 fit**, and the
rate-adjusted per-QP mean is **+0.100** with five of six QPs positive. It is not
a KEEP.

What is unambiguous:

* the head mechanism works as designed — **−32.5 % head bytes/picture at
  QP116** with head BA −0.079 and head SS2 +0.060 at the same QP, and −13…−33 %
  at every rung of the ladder;
* at the matched 100 kbps point **every segment** improves on BA and SS2, and
  **all four source-replacement ceilings rise** (+0.148 f0-32 … +0.385 f44-59);
* the exact-rate stage composes with the tail (union = −4.2 % total bytes).

Why the K gain is smaller than the head byte win: the head is only 9 % of the
stream at this QP (16.5 KB of 185 KB), and K is convex in BA — the arm moves the
large head improvements into a stream whose K is set by the f44-59 segment,
which moves least.

The requested head drift-aware skip/reference term is implemented, correct in
the eff-perc units the merged port got wrong, and a **no-op on this clip**: 79 %
of the head's residual-arm leaves emit all-zero EOB, so there is no
skip/residual trade to make. That is a result, not a failure — it identifies the
head's defect as TX/level pricing.

Blockers for the next iteration:

1. The residual head under-spend is in the **partition**, not the TX: the
   measured probe clamps the P tree to `partition_probe_depth` = 1 and 70 % of
   head leaves are 64x64/64x32. `RD_HEAD_PART` attempted it and is negative as a
   *forced* descent; the next step is a measured per-superblock P partition RD
   (the key's `seq_driver` mechanism, `docs/rd-engine0.md` §5) rather than a
   band gate.
2. K is still set by f44-59, and the head gain cannot reach it there.

## 10. Reproduce

```sh
BIN=build-release/av2enc tools/lane-pdriver/ladder.sh /tmp/rd-pdriver "113 114 115 116 117 118"
ARMS="base headtail" BIN=build-release/av2enc \
  tools/lane-pdriver/ladder.sh /tmp/rd-pdriver-union "113 114 115 116 117 118"
# fit + rate-adjusted deltas
python3 tools/lane-pdriver/kfit.py --oracle /tmp/rd-pdriver/union.json
# segment BA / SS2 / PSNR-Y at the matched pair
python3 tools/lane-pdriver/perframe_metrics.py --ref testdata/nasa_480x240_30fps_60f.y4m \
  --arm base_q116=/tmp/rd-pdriver/base_q116.dec.y4m \
  --arm htail_q114=/tmp/rd-pdriver-union/headtail_q114.dec.y4m
# head bytes/picture and the residual-arm composition
python3 tools/lane-pdriver/head_stats.py --segment head \
  --pair base=/tmp/rd-pdriver/base_q116.log \
  --decisions base=/tmp/rd-pdriver/base_q116.dec
```
