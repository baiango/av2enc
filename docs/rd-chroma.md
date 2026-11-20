# Chroma tool coverage: full UV mode search on the video key picture

Status: **KEEP** (matched-rate K 85.297 → 85.634, **+0.337** at 240p30 /
100 kbps by the specified oracle; median ≈ +0.25 over the estimator and
refresh-configuration range in §6.1). Gate `AV2ENC_RD_UVFULL`, env only,
default off; every pin is byte-identical with it unset.

Experiment worktree `exp/rd-chroma`, based on main `7f331e22`. Reference
recipe is the measured best aggregate:

```sh
REF=testdata/nasa_480x240_30fps_60f.y4m
AV2ENC_EFF_PERC=1 AV2ENC_TX_DETAIL_SEARCH_TAIL=1 AV2ENC_TX_DETAIL_MIN=0.50 \
  build-dev-cache/av2enc "$REF" -o /tmp/x.ivf --video --frames 60 \
  --cpu-threads 8 --qp <Q> --fast-decode \
  --video-drift-refresh-threshold 1.5 --video-drift-refresh-budget 3250
```

The base arm is that command; the candidate arm adds `AV2ENC_RD_UVFULL=1`.

## 1. The gap

On the 240p30 / 100 kbps point the key picture is ~76% of the bits and the
video key path coded **every** chroma leaf as DC_PRED (`entry.uv_mode = 0` in
`video_frame_decision.cpp`) plus the opt-in `--video-cfl` hypothesis, which the
nightly aggregate did not enable. `--fast-decode` closes with Chroma-from-Luma
off, so 100% of key chroma leaves were DC and no chroma directional / smooth /
Paeth predictor was ever coded.

The M1 still product already has the full search (`--still-uv-full`,
`Encoder::choose_uv_mode()`), and `docs/negative-results.md` records that
enabling it by default was rate-neutral **because its score was U-only SSE with
no rate term**. This port changes the score, not just the candidate set.

## 2. What changed

| file | change |
|---|---|
| `src/pipeline/video/video_uv_search.h` | **new**, header-only chroma search: 13 §5.20.5.6 base modes + the CfL candidate, eff-perc perceptual D, §8.2 symbol information-bit rate, top-N transform re-score |
| `src/pipeline/video/video_frame_decision.cpp` | gate + decision wiring in the intra chroma block; `entry.uv_mode` now carries the winner; `AV2ENC_RD_UVFULL_STATS` diagnostic |
| `src/pipeline/video/video_frame_internal.h` | `uv_smooth_` map (§7.13.2.15/.16 chroma neighbour smooth flags) + accessors |
| `src/pipeline/video/video_frame_intra.cpp` | `chroma_neighbor_smooth()` / `set_uv_smooth()` |
| `src/pipeline/video/video_frame.cpp` | size `uv_smooth_` |

`video_frame_decision.cpp`'s per-pass reset also clears both grids.

### Candidate set

* The 13 base modes DC, SMOOTH, SMOOTH_V, SMOOTH_H, PAETH, V, H, D45, D135,
  D67, D113, D157, D203 through `intra::predict_intra` with `color_plane = 1/2`
  (bilinear directional interpolation, no IDIF).
* `AngleDeltaUV` is **not** a signalled symbol: §6.19.5.6 derives it as
  `AngleDeltaY` only when `UVMode == YMode` and `YMode` is directional, so each
  candidate is evaluated with that derived delta and there is no UV
  angle-delta loop. This matches the still encoder.
* The CfL hypothesis (`CFL_DERIVED_ALPHA`) comes from
  `chroma_search::choose()` — the shared M1/video helper — so the derived-alpha
  derivation, MHCCP (when `--video-mhccp` is on) and the bottom/right-edge
  padded-plane handling are the existing verified code, not a copy. With the
  gate on, the helper is called with an unreachable DC score so it always
  returns the CfL prediction even when its own U-only SSE test would drop it;
  the full search then ranks CfL against the 13 modes on one scale.
* `CFL_EXPLICIT_ALPHA` is **not** reachable in the video writer: `code_uv_mode`
  only codes the explicit-alpha syntax under `enable_mhccp`, otherwise it
  writes the two-symbol `cfl_index`, i.e. derived alpha only. Unchanged here.

### Decision

```
J = D + lambda * R
D = eff_perc::weighted_sse(neutral Q8 weights, CSF gains, chroma weight)
    over both planes        (same D the eff-perc leaf charges chroma)
R = uv_mode / is_cfl / cfl_index §8.2 information bits from the static §9.3
    CDF rows the writer starts from (-log2 p)
lambda = leaf_rd_lambda(qindex) * (eff_perc on ? AV2ENC_EFF_PERC_SCALE : 1)
```

Two stages, no SSE/perceptual mixing:

1. every candidate is ranked by the **prediction** perceptual D + `lambda*R`;
2. the cheapest `AV2ENC_RD_UVFULL_TOPN` (default 3) are re-scored on the **real
   chroma transform**: D from the quantised reconstruction
   (`compute_tx` + `eff_perc::weighted_sse`), R from
   `pipe_plan_coefficient_bits()` — the static §8.3 coefficient syntax
   information bits of the emitted plan — plus the mode bits.

Only the winner is returned; the leaf then recomputes its chroma TX on the
winning prediction exactly as before, so `--video-cctx`, QM, the edge mask and
TCQ plumbing are untouched. A stage-1 proxy cost may rank a candidate but can
never win: only re-scored candidates are compared.

The chroma directional modes need the §7.13.2.15/.16 neighbour smooth flags,
which the video encoder tracked only for luma (`intra_joint_`). `uv_smooth_` is
the chroma mirror; the still encoder's `uv_smooth_` semantics are reproduced
exactly (`UVMode is SMOOTH/SMOOTH_V/SMOOTH_H and not CfL`).

### Env surface (no CLI, no CMake, no options/parse/main.cpp)

| var | default | meaning |
|---|---|---|
| `AV2ENC_RD_UVFULL` | unset (off) | master gate; `0`/`false`/`off`/`no` keep it off |
| `AV2ENC_RD_UVFULL_TOPN` | 3 | candidates re-scored on the real transform (0..8) |
| `AV2ENC_RD_UVFULL_STATS` | unset | print the key-picture U/V coefficient bits and the UVMode histogram at exit |

## 3. Correctness

* Gate off (unset, `0`, `off`): the pinned streams are byte-identical —
  QP106 25,008 B `0639b43630abb454f7be5ac494d2efb4`, QP110 23,690 B
  `1b99127a4cd1d386fa5600fb63c57a83` (plain `--fast-decode`, both QPs).
* Gate on: **encoder recon == decoder output** byte for byte on
  240p 480x240x60 (measured QP120 stream `49228567…`, recon `c3de26a8…`) and on
  720p `nasa_1440x720` 12 frames (`2ce0ea7b…`), i.e. the chroma
  edge-filter/smooth-flag state and the coded `uv_mode` are decoder-exact.
* Every stream below decodes 60/60 with `tools/validate_dav2d.sh` (`--strict 1`).
* Gates: `tools/gate_video_nightly.sh` OK, `tools/test.sh` 10/10,
  `gate_video_eff_perc.sh` OK, `gate_video_qp_lambda.sh` OK (pins
  byte-identical), `gate_video_cdf_carry.sh` PASS, `gate_video_pred_surface.sh`
  PASS, `gate_video_sub8.sh` 23/23, `gate_video_any_dims.sh` PASS.
* Edge geometry smoke tests with the gate on: 100x70 (non-aligned), 256x256,
  640x360x4 — all decode.

## 4. UV mode distribution, key picture (QP120 key, 255 leaves)

`AV2ENC_DECISION_DUMP` for the UVMode histogram (`order_hint == 0`); CfL is
counted from `AV2ENC_RD_UVFULL_STATS` because a CfL leaf also codes
`uv_mode = DC_PRED` and `AV2ENC_DECISION_DUMP` has no `is_cfl` field:

| UVMode | base | candidate |
|---|---:|---:|
| DC_PRED (0) | 255 (all) | 22 |
| V (1) | 0 | 7 |
| H (2) | 0 | 31 |
| D45 (3) | 0 | 10 |
| D135 (4) | 0 | 3 |
| D113 (5) | 0 | 6 |
| D157 (6) | 0 | 9 |
| D203 (7) | 0 | 11 |
| D67 (8) | 0 | 5 |
| SMOOTH (9) | 0 | 64 |
| SMOOTH_V (10) | 0 | 12 |
| SMOOTH_H (11) | 0 | 14 |
| PAETH (12) | 0 | 8 |
| **CfL** (`is_cfl`, DC base) | 0 | 53 |

So ~91% of key chroma leaves leave DC, and SMOOTH alone takes 25%.

## 5. Chroma bytes and quality

Key-picture chroma coefficient **information bits** (static §8.3 model,
`AV2ENC_RD_UVFULL_STATS`, key picture only):

| QP | base U / V | candidate U / V | Δ bits |
|---:|---:|---:|---:|
| 117 | 6,827.2 / 5,944.3 | 5,737.2 / 4,877.2 | −2,157 |
| 118 | 6,685.3 / 5,768.4 | 5,743.4 / 4,659.4 | −2,051 |
| 119 | 6,493.4 / 5,619.2 | 5,598.9 / 4,453.6 | −2,060 |
| 120 | 6,348.1 / 5,510.1 | 5,460.0 / 4,356.9 | −2,041 |
| 121 | 6,187.8 / 5,327.4 | 5,286.0 / 4,232.1 | −1,997 |
| 122 | 5,986.2 / 5,121.7 | 5,124.3 / 4,116.9 | −1,867 |
| 123 | 5,752.0 / 4,965.9 | 4,945.2 / 3,941.7 | −1,831 |

That is a stable **−15..−17%** of key chroma coefficient bits. Key picture
total: QP119 16,098 → 15,944 B, QP120 15,868 → 15,709 B, QP121 15,599 →
15,442 B (−1.0%), i.e. roughly 60% of the chroma saving survives after the
extra `uv_mode` symbols and CDF adaptation.

Key-frame PSNR at QP120 (per-frame `psnr` stats): base Y 45.860 / U 48.420 /
V 48.950, candidate Y 45.860 / U 48.420 / V 49.070.

Per-frame (60 frames, QP120 pair): mean PSNR-Y 44.781 → 44.832,
mean U 48.125 → 48.058, mean V 48.745 → 48.749; worst frame 33 unchanged
(43.490 dB both). Chroma V gains, luma is flat-to-better, and the frame's rate
is 559 B lower.

## 6. Matched-rate K

5 QPs (118..122) per arm, interleaved, `tools/oracle.py --target-kbps 100`:

| QP | base B | kbps | K | cand B | kbps | K |
|---:|---:|---:|---:|---:|---:|---:|
| 118 | 26,217 | 104.868 | 85.671 | 26,165 | 104.660 | 85.933 |
| 119 | 25,569 | 102.276 | 85.809 | 25,330 | 101.320 | 85.837 |
| 120 | 25,319 | 101.276 | 85.438 | 24,760 | 99.040 | 85.487 |
| 121 | 24,709 | 98.836 | 85.168 | 24,471 | 97.884 | 85.115 |
| 122 | 24,114 | 96.456 | 85.099 | 23,997 | 95.988 | 85.024 |

| candidate | matched kbps | K @ target | method |
|---|---:|---:|---|
| base | 100.000 | 85.297 | linear (base@121..base@120) |
| candidate | 100.000 | **85.634** | linear (cand@120..cand@119) |

**ΔK = +0.337** at iso-rate 100.0 kbps. The base number reproduces the parent
measurement exactly (85.297), so the harness matches.

Adding QP117 and QP123 to both ladders (7 points) leaves every point and the
matched K unchanged, so the result is not a bracket artifact.

### 6.1 Estimator and configuration sensitivity

The K-vs-rate curve is not smooth: the drift-refresh decision flips one frame on
or off when the chroma reference changes, which moves K at a point by up to
±0.2. The two-bracket interpolation the oracle reports is therefore the most
optimistic of several reasonable estimators. Fitting the same 7-point ladders
instead:

| estimator | ΔK @ 100 kbps |
|---|---:|
| oracle bracket (121..120 / 120..119) | +0.337 |
| linear fit, full 94..107 kbps ladder | +0.110 |
| quadratic fit, full ladder | +0.213 |
| linear fit, 98..105 kbps | +0.268 |
| linear fit, 97..103 kbps | +0.263 |

An independent 5-point ladder with the refresh switch **off**
(`--video-drift-refresh-off`, same recipe, QP116..120, so both arms have the
same refresh schedule and the curve is monotone) gives a cleaner read of the
same decision change:

| QP | base B | kbps | K | cand B | kbps | K |
|---:|---:|---:|---:|---:|---:|---:|
| 116 | 25,855 | 103.420 | 85.716 | 25,440 | 101.760 | 85.816 |
| 117 | 25,490 | 101.960 | 85.679 | 25,063 | 100.252 | 85.729 |
| 118 | 25,047 | 100.188 | 85.630 | 24,877 | 99.508 | 85.673 |
| 119 | 24,466 | 97.864 | 85.437 | 24,475 | 97.900 | 85.479 |
| 120 | 24,188 | 96.752 | 85.094 | 24,135 | 96.540 | 85.257 |

Matched K@100: base 85.614, candidate 85.710, **ΔK +0.096**; the candidate has
higher K at every QP *and* a lower rate at four of five. The refresh-off
configuration simply converts rate into K more slowly (≈0.03 K/kbps vs
≈0.11 K/kbps with refresh on), so the same ~1% rate saving is worth less there.

Read together, the effect is **+0.10 .. +0.34 K, median ≈ +0.25**, at
iso-rate 100 kbps: clearly positive and rate-real (every QP codes fewer bytes
and better chroma), but only the specified 5-QP oracle bracket clears +0.15
with margin. The verdict below uses the specified harness and states the range.

Iso-rate components (linear in rate, both arms at 100.0 kbps):

| | base | candidate | Δ |
|---|---:|---:|---:|
| PSNR-Y | 44.663 | 44.828 | **+0.165 dB** |
| PSNR-U | 47.981 | 48.133 | +0.152 dB |
| PSNR-V | 48.637 | 48.802 | +0.165 dB |
| CVVDP JOD | 9.7877 | 9.7959 | +0.008 |
| Butteraugli | 2.2418 | 2.1589 | **−0.083** |
| SSIMULACRA2 | 82.297 | 82.699 | **+0.40** |
| K | 85.297 | 85.634 | **+0.337** |

The gain is on Butteraugli and SSIMULACRA2, the two chroma-sensitive metrics —
the expected signature of a chroma-prediction change, and the reason CVVDP
alone would have called it ripple.

## 7. Attribution (QP120, same recipe)

| arm | bytes | key U/V coef bits | K @ 100 kbps |
|---|---:|---:|---:|
| base (DC only) | 25,319 | 6,348 / 5,510 | 85.297 (3-pt ladder 85.297) |
| `--video-cfl` only | 25,099 | 6,051 / 5,304 | 85.316 (+0.019) |
| `AV2ENC_RD_UVFULL=1` | 24,760 | 5,460 / 4,357 | 85.634 (+0.337) |
| gate on, `TOPN=0` (prediction only) | 25,142 | — | not scored |
| gate on, `TOPN=8` | 24,968 | — | not scored |

CfL alone is rate-neutral in K — the same conclusion
`docs/negative-results.md` records for the still path. The mode search is what
pays, and the stage-2 transform re-score is worth 382 B over the
prediction-only ranking at QP120.

## 8. Verdict

**KEEP**: +0.337 K at matched rate by the specified harness (5 QPs around
100 kbps, `tools/oracle.py --target-kbps 100`), median ≈ +0.25 over the
estimator/configuration range in §6.1, with key chroma coefficient bits down
17%, decoder-exact reconstruction, and byte-identical pins. The effect is
unambiguously rate-real (fewer bytes *and* better chroma at every QP); only its
K conversion is estimator-sensitive. The tool stays behind
`AV2ENC_RD_UVFULL` (default off) until the CUDA chroma decision is ported.

## 9. Blockers and follow-ups

* **CUDA backend not ported.** The GPU product path runs
  `still::chroma_decide()` (DC + derived CfL) from
  `src/cuda/pipeline/still/chroma/`, so a GPU encode still codes DC-only
  chroma. The CPU decision here is not mirrored on device; the sizes and the
  §7.13.2.15/.16 chroma smooth map would have to move into the device job.
* **IBP**: the gate is forced off when `--video-ibp` is on (verified
  byte-identical to gate-off). IBP changes the DC candidate (§7.13.2.12
  post-filter) and the CLI already rejects `--video-ibp` with `--video-cfl`;
  supporting both would need an `ibp::predict` candidate loop plus a parity
  gate that does not exist today.
* **`CFL_EXPLICIT_ALPHA`** is unreachable without MHCCP in `code_uv_mode`, so
  only the derived-alpha hypothesis is searched. The explicit-alpha search
  (`cfl::search_alpha`) exists in the still path.
* **Rate model**: stage 1 prices mode symbols only; the residual is priced at
  stage 2 for the top-N. A per-candidate residual model (or a live-CDF slice)
  is the obvious next step, together with a sweep of `TOPN` on more content.
* Only 240p30 / 100 kbps and the two clips used for smoke tests were measured;
  720p was checked for decode/recon parity, not for K.
* The K delta is estimator-sensitive because a chroma change can flip one
  drift-refresh frame (§6.1). A refresh-schedule-insensitive measurement (for
  example a fixed refresh map for both arms) would settle the point estimate;
  none exists today.
