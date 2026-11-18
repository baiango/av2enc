> **ARCHIVED.** The implementation was removed because it used a port of
> libjxl's Butteraugli; third-party code is not allowed in this tree. The
> measurement is kept as a record. The shipped opt-in is the perceptual RD core
> (`docs/eff-perc.md`).

# Perceptual reference refresh (`AV2ENC_PIPE_BA_REFRESH`)

Status: **opt-in, env-gated, CPU-only; measured at 240p/100 kbps as K 84.81
(QP111) / 84.63 (QP110), iso-rate 84.80 against the 83.80 wall — +1.00 K.**
Off by default; every pinned stream is byte-identical. No CLI flags and no new
syntax.

> **Deep-anchor follow-up (negative).** `docs/pipe-deep.md` runs the same
> planner with the anchors at the key's QP depth (`AV2ENC_PIPE_DEEP=1`). Depth
> alone loses at this budget: 84.02 K interpolated at 100 kbps (depth −14)
> against 84.81 here, and the rate-free all-anchor ceiling is 87.69, below the
> 90 wall. The CDF composition fix (seed the §9.3 image from the coded
> `base_q_idx`) is recorded there and is independent of the allocation.

The drift-refresh lever (+1.65 K) triggers on the primary reference's luma PSNR
and spends its budget chronologically. This pipeline replaces the trigger and
the selection rule with the measured Butteraugli error and a global allocation
under the same payload allowance. It uses the existing drift-refresh plumbing
(frame QP drop + `force_inter_residual`), so the bitstream is still a normal
key+P chain.

## Recipe

```sh
AV2ENC_PIPE_BA_REFRESH=1 build-dev/av2enc \
  testdata/nasa_480x240_30fps_60f.y4m -o /tmp/ba.ivf \
  --video --frames 60 --cpu-threads 8 --qp 111 --nightly \
  --video-drift-refresh-threshold 1.5 --video-drift-refresh-budget 3250
```

`--nightly` supplies the enabled drift-refresh plumbing; the env gate replaces
its PSNR trigger. `--video-drift-refresh-threshold 1.5` is part of the tuned
recipe and is ignored by the BA planner; the budget is not.

## K oracle (required by `docs/k-oracle.md`)

**Announced before tuning: the achievable envelope is 84.80 K at 240p/100 kbps
and the optimistic allocation ceiling is 84.92. Both exceed the wall (83.80).**

| Bound | Value | Method |
|---|---:|---|
| Achievable envelope (primary, lower bound on `K*`) | **84.81 at 99.900 kbps** (84.80 interpolated at 100.0) | QP111 + budget 3250 measured directly; QP110 at 101.388 kbps is 84.63. The coarse allocation sweep (QP109 + budget 1600) gives 83.81 at 98.948 kbps, so halving the refresh budget does not raise the envelope. |
| Allocation ceiling (optimistic) | **84.92** | Give every frame the measured all-refresh BA floor (QP110 probe, mean 2.432) but keep the QP111 JOD 9.7927 / SS2 82.903. This over-states the floor because QP111 refreshes are noisier, so the true ceiling is ≤ 84.92. |
| Decision bound | not applicable | The lever is frame-level; there is no per-block candidate set and no new syntax. |

Headroom: **+1.00 K over the 83.80 wall** and **+0.12 K under the allocation
ceiling**. The independent-frame knapsack (refresh a frame only for itself)
gives BA 3.056 / K 83.50, *below* the measured chain, because a refresh also
repairs every following frame; it is not a valid upper bound and was discarded.

The envelope clears the wall, so the result below is reported rather than a
ceiling. The measured point is within 0.12 K of the per-frame BA floor, so
further allocation tuning has at most ~0.1 K of headroom.

## What it does

1. **Pass 1 (measure the chain).** Encode with the refresh trigger disabled.
   For every coded frame, measure `BA(recon, source)` with a self-contained
   scalar port of libjxl Butteraugli (`src/pipeline/video/video_butteraugli.h`)
   and record the tile payload.
2. **Pass 2 (probe the refresh).** Encode every inter frame as a refresh and
   record the same per-frame BA and payload. This gives each candidate's
   measured refresh floor `BA_refresh[i]` and its measured extra payload
   `payload_refresh[i] - payload_no_refresh[i]`.
3. **Plan.** A dynamic program chooses the refresh set that minimises the
   predicted sum of per-frame BA under `--video-drift-refresh-budget`
   (3250 B here). Between refreshes the prediction starts at the measured
   refresh floor and the no-refresh gap re-accumulates with 0.8 survival per
   frame (the measured refresh benefit is mostly gone after ~8 frames;
   `docs/video-drift-refresh.md`). This is the "marginal effect per byte"
   ranking: the DP is a global knapsack over the measured BA drops and priced
   payloads, not a threshold.
4. **Emit + refine.** Apply the schedule through the drift-refresh plumbing.
   If the probe under-priced the isolated refreshes (it measures against an
   always-fresh reference) and the payload cap dropped scheduled frames, the
   measured actual/estimated ratio re-plans once and emits a refinement pass.

No refresh decision uses PSNR. The only PSNR value left in the code is the
untouched default drift trigger, which is bypassed when the env gate is on.

## Metric validation

The internal score is the max of the Butteraugli diffmap (the number
`butteraugli_main` prints), from XYB opsin conversion, three-scale frequency
separation, Malta pooling and masking. On the 60-frame no-refresh QP110 stream
against `butteraugli_main` frame by frame:

| | internal | `butteraugli_main` |
|---|---:|---:|
| mean BA | 5.142 | 4.986 |
| f0 | 1.397 | 1.365 |
| f4 | 1.619 | 1.660 |
| f9 | 2.130 | 2.129 |
| f52 | 9.939 | 9.939 |

The absolute bias (+3.1% mean) comes from the 8-bit limited-range 4:2:0
BT.601/sRGB front end versus ffmpeg's PNG path. The planner only uses
differences against the same metric's own key/probe values, so the bias
cancels; every reported K number is measured by the external
`tools/extra_metrics.py` on the decoded stream, not by the internal value.

## Results at 240p

`testdata/nasa_480x240_30fps_60f.y4m`, 60 frames, `--nightly`, QP / budget
bracketing 100 kbps. Quality: `tools/quality.py` (CVVDP 0.5.7, `standard_fhd`,
MPS), `tools/extra_metrics.py`, `tools/kpi.py`.

| Stream | QP | Bytes | kbps | JOD | BA | SS2 | PSNR-Y | K |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| nightly pin | 106 | 25,008 | 100.032 | 9.7334 | 3.420 | 80.837 | 44.67 | **82.15** |
| tuned PSNR drift (wall) | 110/111 | 25,000 | 100.0 | 9.7536 | 2.712 | 81.38 | 44.73 | **83.80** |
| PSNR trigger, same QP | 111 | 24,950 | 99.800 | 9.7513 | 2.731 | 81.422 | 44.703 | **83.75** |
| **BA refresh** | **111** | **24,975** | **99.900** | **9.7927** | **2.4768** | **82.903** | **44.922** | **84.81** |
| BA refresh | 110 | 25,347 | 101.388 | 9.7863 | 2.5271 | 82.796 | 44.981 | 84.63 |
| BA refresh, half budget (allocation sweep) | 109 | 24,737 | 98.948 | 9.7793 | 2.8360 | 82.349 | 44.820 | 83.81 |
| **BA refresh at 100.0** | — | — | 100.0 | — | — | — | — | **84.80** |

At the matched 100.0 kbps point: **+2.65 K versus the 82.15 nightly pin** and
**+1.00 K versus the 83.80 tuned-drift wall**. The same-QP QP111 comparison is
+1.06 K for +0.10 kbps, so the gain is not a rate artefact.

The allocation sweep is the continuous-knob evidence: halving the refresh
budget at QP109 (1600 B) gives 98.948 kbps / K 83.81, i.e. at the wall but not
above it; the QP111 3250 B point is 84.81 at 99.900 kbps. The envelope is not a
single-QP artefact.

Same-QP (QP111) delta against the PSNR trigger, **+0.100 kbps**:

| | PSNR trigger | BA refresh | Δ |
|---|---:|---:|---:|
| JOD | 9.7513 | 9.7927 | **+0.0414** |
| Butteraugli | 2.7310 | 2.4768 | **−0.2542** |
| SSIMULACRA2 | 81.422 | 82.903 | **+1.481** |
| PSNR-Y | 44.703 | 44.922 | **+0.219 dB** |
| K | 83.75 | **84.81** | **+1.06** |

Late-band allocation (mean over frames 44-59 / 50-59):

| | BA 44-59 | BA 50-59 | PSNR-Y 50-59 |
|---|---:|---:|---:|
| no refresh (QP110) | 7.579 | 8.791 | — |
| PSNR trigger (QP111) | 4.358 | 4.487 | 43.547 |
| BA refresh (QP111) | **3.268** | **3.116** | **44.380** |

The gate's late-band check is +0.83 dB mean luma PSNR on frames 50-59 against
the PSNR trigger at the same QP.

## Which frames refresh

QP111, 60 frames, budget 3250 B.

| Rule | Frames | Count |
|---|---|---:|
| PSNR trigger | 6, 7, 8, 9, 16, 18, 20, 21, 23-42 | 28 |
| BA refresh | 3, 11, 13, 14, 15, 16, 18, 21, 23, 25, 26, 27, 28, 29, 31, 33, 36, 37, 39, 41, 43, 46, 49, 51, 52, 54 | 26 |

At QP110 the BA refinement refreshes 10, 12, 14, 16, 18, 19, 21, 22, 23, 25,
26, 27, 29, 31, 33, 36, 38, 41, 44, 46, 48, 51, 53, 55 (24 frames).

**Trigger difference.** The PSNR trigger is a mean-luma-MSE statistic with a
1.5 dB allowance below the key; it concentrates 22 of its 28 refreshes in the
20-42 trough and never fires after frame 42. The BA rule prices the worst local
error: the 44-59 detail/text band has BA 4.4 in the PSNR arm and 7.6 with no
refresh, while its mean PSNR looks acceptable, so the BA planner keeps 6
refreshes in that band (46, 49, 51, 52, 54 and 43 at its edge). That is the
whole difference: same payload (3259 vs 3249 extra B), same number of refreshes
within 2, but the bytes go where the perceptual error is, not where the mean
error is.

The BA stream is 24,975 B / 99.900 kbps; the PSNR stream is 24,950 B /
99.800 kbps. The K gain is not a rate gain.

## Verification

* Gate off and `AV2ENC_PIPE_BA_REFRESH=0`: nightly pin 25,009 B /
  `a1d0dba1259b20a4b628d4cd1e14c33c`; quality pin 25,141 B /
  `3984677b9b665952984b07b7cf75c6ea`; nightly QP110 pin 23,690 B /
  `1b99127a4cd1d386fa5600fb63c57a83`; 720p pin 36,572 B /
  `c4ba28138d8f6569d5e08721a66d755c`.
* BA QP111 stream: 24,975 B, MD5 `4318f39cd0da547baa3446f5aaa41ac4`;
  `dav2d-macos --strict 1` decodes 60/60 and the `--video-recon` payload
  (10,368,000 B) equals the decoded payload byte for byte. A second identical
  encode reproduced the same MD5.
* `tools/test.sh --build build-dev`: 10/10.
* `gate_video_nightly`, `gate_video_qp_lambda`,
  `video_drift_refresh_gate` and the new
  `src/pipeline/video/video_ba_refresh_gate.sh` pass. The new gate checks the
  off pins, the plan/budget/tail allocation, strict decode, recon equality and
  a +0.20 dB late-band floor against the PSNR trigger.

## Caveats

* One clip, 240p, CPU-only, single tile, 8-bit 4:2:0, flat key+P chain. The
  720p check was not run: the host was at load average ~58, the 720p point is
  JOD-dominated (the PSNR refresh is −0.48 K there), and this pipeline's
  measure/probe/emit/refine passes cost more than the flag-off encode. Do not
  quote a 720p result.
* The metric is a scalar port, not a bit-exact libjxl build. The validation
  above is the contract; it is used for decisions only.
* The carry model (0.8/frame) and the probe-cost refinement are model
  assumptions. The allocation ceiling (84.92) bounds what a better allocation
  could buy: about +0.12 K, so there is no point fitting those constants on
  this clip.
* Encode cost is not quoted: the shared host was heavily loaded during the
  measurement. The metric alone is ~0.12 s/frame at 240p in an unloaded
  standalone -O2 build; the pipeline runs it on two passes and encodes the
  sequence two to four times.

## Named scope

The env gate only replaces the trigger of an already-enabled
`--video-drift-refresh` chain (`--nightly` includes it). Without that chain it
is inert. It keeps the drift-refresh restrictions (CPU, single tile, flat
key+P, 8-bit 4:2:0) and adds no syntax, no CLI flag and no default-on path.
