> **ARCHIVED.** The implementation was removed because it used a port of
> libjxl's Butteraugli; third-party code is not allowed in this tree. The
> measurement is kept as a record. The shipped opt-in is the perceptual RD core
> (`docs/eff-perc.md`).

# Deep inter anchors (`AV2ENC_PIPE_DEEP`)

Status: **opt-in, env-gated, CPU-only; negative at the 100 kbps point.** The
anchor-depth sweep peaks at **K 84.02** (matched-rate interpolation, depth
−14), the same-budget depth −27 point is **K 83.06** at 104.7 kbps, and the
rate-free ceiling — every inter frame at the key's QP 84 — is **K 87.69** at
236.6 kbps. All are below the current best 84.81 and the stage-1 wall 90.0, so
the pipeline is **not tuned further**. Off by default; every pinned stream is
byte-identical.

The differential (`docs/svt-diff.md` mechanism #1) predicted +2.0–3.0 K from
giving the BA refresh the key's quantizer depth. This document measures that
lever directly. The depth is real — an anchor at QP 84 codes 9–14k coefficient
levels and chroma EOBs on 32–41 of its 55–58 leaves, against 14/26 shallow
anchors with **zero** chroma EOB — but the anchor does **not** reach the key's
BA floor on the fade content, and at the tuned 3,250 B allowance it buys only
3 anchors instead of 26. Depth is necessary, not sufficient: the remaining gap
is anchor *coverage* and prediction, not the quantizer depth of the refresh.

## Recipe

```sh
AV2ENC_PIPE_DEEP=1 build/av2enc \
  testdata/nasa_480x240_30fps_60f.y4m -o /tmp/deep.ivf \
  --video --frames 60 --cpu-threads 8 --qp 111 --fast-decode \
  --video-drift-refresh-threshold 1.5 --video-drift-refresh-budget 3250
```

`--fast-decode` supplies the enabled drift-refresh chain and the CDF chain
(`--avg-cdf --adaptive-cdf --cdf-continuity`). The env gate replaces the PSNR
trigger with the BA planner (same as `AV2ENC_PIPE_BA_REFRESH=1`) and deepens
every scheduled anchor. The default depth is the measured key offset (27 QP at
the 84/111 recipe); `--video-drift-refresh-qp <N>` overrides it for the oracle.
The budget is unchanged at 3,250 B.

## What it does

1. **Placement is unchanged.** Pass 1 measures the no-refresh chain, pass 2
   probes every inter frame as a refresh, and the same BA dynamic program
   picks the schedule under `--video-drift-refresh-budget`. With the gate on,
   pass 2 probes the **deep** floor, so the planner prices the deeper frames.
2. **Anchor depth.** The existing drift-refresh plumbing drops the frame
   `base_q_idx` by `drift_refresh_qp` and sets `force_inter_residual`. At QP 84
   the forced transform now keeps chroma coefficients; no separate chroma
   switch is needed because the flag is all-plane.
3. **CDF seed.** The carried §9.3 image needs no new syntax:
   `--cdf-continuity` already signals the per-frame carry, and only a frame
   the decoder resets (§5.18.2, the first P frame of the flat chain) derives
   its defaults from its own `base_q_idx`. The encoder's `seed_frame_cdfs()`
   now re-derives that image from the **coded** QP instead of the static plan's
   QP. That is the whole composition fix; the mixed form (carry without
   continuity and without `--video-cdf-per-frame-qp`) stays rejected.
4. **No new syntax.** The anchor is a normal P frame with a lower
   `base_q_idx` in the existing frame header. Nothing else changes.

The CDF fix is load-bearing, not defensive. A 2-frame 640x360 stream whose
first P frame is an anchor at QP 71 while the static plan says 111 (coefficient
qctx slice 0 vs slice 1):

| build | decoded pictures | `--video-recon` |
|---|---:|---|
| seed from the static plan QP | **1/2** (`Error decoding frame: Invalid argument`) | n/a |
| seed from the coded QP | **2/2** | byte-equal |

Existing shallow streams do not move: at `drift_refresh_qp == 0` the coded QP
equals the plan QP, so `AV2ENC_PIPE_BA_REFRESH=1` still produces the pinned
24,975 B / MD5 `4318f39cd0da547baa3446f5aaa41ac4` stream.

## K oracle (required by `docs/k-oracle.md`)

**Announced before tuning: at the tuned budget the depth curve peaks at
K 84.02 at 100 kbps (depth −14); the rate-free ceiling of the mechanism at the
key's depth is K 87.69. Both are below the stage-1 wall 90.0 and below the
current best 84.81. This pipeline is not a path to 90.**

### Depth sweep at the tuned budget (QP 111, budget 3,250 B)

| Depth | Anchor QP | Anchors | Bytes | kbps | JOD | BA | SS2 | PSNR-Y | K |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 (shallow control) | 111 | 26 | 24,975 | 99.900 | 9.7927 | 2.4768 | 82.903 | 44.922 | **84.81** |
| −14 | 97 | 13 | 25,751 | 103.004 | 9.7861 | 2.7255 | 82.654 | 45.104 | **84.14** |
| −27 | 84 | 3 | 26,180 | 104.720 | 9.7834 | 3.2131 | 82.438 | 44.990 | **83.06** |
| −40 | 71 | 1 | 26,510 | 106.040 | 9.7628 | 4.3956 | 81.834 | 44.727 | **80.93** |

### Matched-rate points

| Depth | QP | Anchors | Bytes | kbps | JOD | BA | SS2 | K |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| −14 | 113 | 14 | 24,894 | **99.576** | 9.7774 | 2.7069 | 81.826 | **84.00** |
| −27 | 114 | 4 | 25,326 | **101.304** | 9.7752 | 3.0394 | 81.722 | **83.25** |
| −14 | 111 | 13 | 25,751 | 103.004 | 9.7861 | 2.7255 | 82.654 | 84.14 |
| −27 | 111 | 3 | 26,180 | 104.720 | 9.7834 | 3.2131 | 82.438 | 83.06 |
| −40 | 111 | 1 | 26,510 | 106.040 | 9.7628 | 4.3956 | 81.834 | 80.93 |

Linear interpolation between the two depth −14 points gives **K 84.02 at
100.0 kbps**; the two depth −27 points bracket the same region above 101 kbps
and extrapolate to ≈83.3. The envelope of the depth sweep is therefore
**84.02**, **−0.79 K against the shallow BA pipeline** (84.81) and **−5.98 K
against the 90 wall**.

### Rate-free ceiling (all 59 inter frames coded as deep anchors)

| Depth | Anchors | Bytes | kbps | JOD | BA | SS2 | PSNR-Y | K |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| −27 (key depth) | 59 | 59,151 | 236.604 | 9.8742 | 1.8352 | 86.814 | 46.123 | **87.69** |
| −40 | 59 | 61,612 | 246.448 | 9.8781 | 1.9007 | 86.812 | 46.420 | **87.50** |

Even with every inter frame at the key's QP (or 13 QP finer) and **2.4x the
rate**, the mechanism tops out at **K 87.69**, 2.31 below the wall and 1.14
below SVT-AV1 p0 (88.83). Deeper does not help: at −40 the all-anchor stream
pays 2.5 kB more and the BA mean is 1.90 vs 1.84 because the finer quantizer
changes the loop-filter/residual balance without moving SS2. The differential's
upper bound ("replace all inter frames with the key's measured BA 1.43 / SS2
87.90") is not reachable by the inter path: on the fade content the inter
anchor at QP 84 measures BA 1.8–2.5, and the all-anchor mean is 1.835 / 86.814.

## Per-anchor bytes and chroma coverage

Shallow vs deep at the same 3,250 B allowance. Coverage is from
`AV2ENC_BLOCKDUMP` on the emitted pass; bytes are the IVF frame sizes.

| Arm | Anchors | Anchor bytes | Leaves | Leaves with EOB | Chroma EOB | Zero-chroma anchors |
|---|---:|---:|---:|---:|---:|---:|
| shallow (depth 0) | 26 | 4,154 | 1,212 | 583 (48%) | 183 | **14/26** |
| deep (depth −27) | 3 | 4,028 | 171 | 121 (71%) | 106 | **0/3** |

Deep anchor detail (depth −27, QP 111 stream):

| Anchor | Bytes | Leaves | EOB | Luma EOB | Chroma EOB | Levels | BA | SS2 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| f17 | 767 | 55 | 37 | 31 | 32 | 9,007 | 1.747 | 87.69 |
| f33 | 1,111 | 58 | 41 | 39 | 33 | 12,150 | 2.083 | 86.99 |
| f49 | 2,150 | 58 | 43 | 39 | 41 | 14,439 | 2.523 | 84.55 |

The shallow arm spends 4,154 B on 26 frames (mean 160 B, largest 399 B); the
deep arm spends almost the same 4,028 B on 3 frames. The deep anchors code
chroma on every frame; the shallow detail-band anchors already code some
(f46 294 B / 22 chroma EOB, f49 377 B / 24, f51 167 B / 13, f52 292 B / 20,
f54 399 B / 23), while the 14 zero-chroma anchors are all in the static first
half and the f32–f43 trough. Deepening the first-half anchors buys almost
nothing because their BA is already near the key (f3/f11/f16 sit at 1.63–1.77).

The probe under-prices an isolated deep anchor by ~3x because pass 2 refreshes
*every* frame, so each anchor predicts from a fresh reference. Measured probe
extra vs actual extra on the depth −27 chain: f17 214 → 767 B, f33 428 →
1,111 B, f49 712 → 2,150 B. The planner therefore schedules 7 anchors, the
emit path's allowance keeps 3, and the one refinement pass cannot recover a
per-anchor 3x bias with a single global scale. This is a *second* reason the
same-budget deep point is poor; the all-anchor ceiling shows it is not the
binding one.

## Detail band (frames 44–59), per-frame BA / SS2

`shallow` is the pinned BA-refresh stream (26 anchors, QP 111), `deep` is the
depth −27 stream at the tuned budget (3 anchors), `all` is every inter frame
at QP 84, `SVT` is the AV1 p0 row from `docs/svt-diff.md`.

| f | shallow BA | shallow SS2 | deep BA | deep SS2 | all BA | all SS2 | SVT BA | SVT SS2 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 44 | 3.55 | 78.2 | 6.57 | 72.1 | 2.09 | 84.8 | 1.62 | 86.7 |
| 45 | 3.96 | 76.5 | 6.93 | 71.2 | 1.94 | 84.4 | 1.57 | 85.8 |
| 46 | 2.68 | 79.7 | 6.35 | 73.4 | 1.94 | 85.0 | 1.57 | 86.0 |
| 47 | 3.01 | 76.7 | 6.66 | 71.7 | 1.86 | 84.8 | 1.66 | 85.2 |
| 48 | 3.30 | 75.2 | 8.43 | 68.4 | 2.22 | 83.8 | 2.29 | 84.2 |
| 49 | 2.75 | 79.0 | 2.52 | 84.5 | 1.95 | 83.9 | 1.87 | 84.4 |
| 50 | 3.11 | 76.8 | 2.82 | 82.2 | 1.96 | 84.2 | 1.73 | 84.8 |
| 51 | 2.96 | 77.8 | 2.91 | 82.7 | 2.10 | 85.1 | 1.65 | 84.8 |
| 52 | 3.12 | 77.2 | 3.53 | 79.6 | 2.35 | 84.1 | 1.58 | 86.0 |
| 53 | 3.31 | 73.2 | 3.35 | 75.8 | 2.21 | 83.9 | 1.80 | 85.3 |
| 54 | 2.97 | 75.9 | 3.87 | 73.1 | 2.96 | 83.8 | 1.96 | 84.6 |
| 55 | 2.96 | 73.0 | 4.30 | 71.4 | 2.89 | 83.5 | 2.34 | 83.5 |
| 56 | 2.96 | 73.4 | 4.22 | 72.8 | 2.11 | 83.9 | 2.34 | 83.6 |
| 57 | 3.21 | 71.5 | 5.12 | 71.3 | 2.20 | 83.4 | 1.53 | 86.2 |
| 58 | 3.81 | 66.0 | 4.79 | 65.4 | 2.23 | 81.5 | 1.92 | 82.6 |
| 59 | 4.87 | 65.4 | 4.39 | 65.2 | 2.11 | 82.3 | 3.50 | 76.2 |

Band means:

| Arm | 44–59 BA | 44–59 SS2 | 48–59 BA | 48–59 SS2 |
|---|---:|---:|---:|---:|
| shallow (depth 0) | 3.284 | 74.715 | 3.278 | 73.702 |
| deep −27 at budget | 4.798 | 73.798 | 4.188 | 74.368 |
| all-deep −27 | **2.194** | **83.903** | **2.272** | **83.626** |
| SVT p0 | 1.934 | 84.369 | 2.043 | 83.847 |

The deep anchor at f49 repairs its own frame and the next two (f49–f51 BA
2.5–2.9 / SS2 82–85) but f44–f48 has no anchor for 11–15 frames and stays at
BA 6.4–8.4. The shallow arm's 26 anchors keep the trough at 3.0–3.3. The
all-anchor arm shows the depth can reach SVT's anchor quality (2.19 vs 1.93
mean BA in the detail band) but only by paying 2.4x the rate; the remaining
0.26 BA is the prediction/leaf gap, not depth.

## Encode cost

Same host (M4, 8 CPU threads, single run), 60 frames:

| Arm | User CPU | Wall | Passes |
|---|---:|---:|---:|
| shallow BA refresh | 57.0 s | 41.2 s | 3–4 |
| deep −27 at budget | 57.0 s | 51.9 s | 4 |

The user CPU is flat because the measured-cost passes dominate; the wall time
rose ~26% on this host, which was not otherwise controlled. The pipeline
encodes the sequence 3–4 times plus the BA metric on two passes (≈0.12 s/frame
at 240p in an unloaded standalone build).

## Why the lever fails here

* **The key's floor is not an inter floor on this content.** The key is
  intra-coded at QP 84 with full intra search; a P anchor at the same QP
  predicts from a drifting reference and reaches BA 1.75–2.52 on the fade,
  not 1.43. SVT's anchors are at 1.87–2.29 in the same band; the all-deep arm
  matches them, so the encoder can reach anchor quality at key depth — it
  cannot reach *key quality* at anchor rate.
* **3,250 B buys 3 deep anchors or 26 shallow ones.** The measured isolated
  deep anchors cost 0.8–2.2 kB each (the all-refresh probe under-prices them
  3x). Three anchors leave 11–15-frame holes in the fade; the 0.8/frame carry
  model already predicts that this loses to the shallow schedule (predicted
  mean BA 3.31 vs 2.70 for depth −14, both worse than the measured shallow
  2.48).
* **Infinite anchor rate still does not reach 90.** The all-anchor stream at
  the key's depth is K 87.69 at 236.6 kbps, 2.31 below the wall. The
  differential's +4.07 K all-key-floor bound uses BA 1.43 / SS2 87.90; the
  inter path delivers 1.84 / 86.81 on the same frames even with no rate
  constraint.
* **What would have to change.** The gap is anchor coverage at a rate the
  100 kbps budget can afford (SVT uses 9 anchors / 4.9 kB, three of them
  1.1–1.6 kB) plus the two-sided/distance-weighted prediction that keeps its
  leaves at 0.84–0.89 MSE between anchors. Depth alone at this budget moves
  **−0.8 K**; depth with unlimited coverage recovers at most +2.9 K and still
  stops 2.3 K short of the wall.

## Verification

* Gate off and `AV2ENC_PIPE_DEEP=0`: nightly QP110 pin 25,009 B /
  `a1d0dba1259b20a4b628d4cd1e14c33c`; quality pin 25,141 B /
  `3984677b9b665952984b07b7cf75c6ea`; QP110 60-frame pin 23,690 B /
  `1b99127a4cd1d386fa5600fb63c57a83`; 720p pin 36,572 B /
  `c4ba28138d8f6569d5e08721a66d755c`.
* Shallow BA pin with the gate off: 24,975 B / MD5
  `4318f39cd0da547baa3446f5aaa41ac4`.
* Depth −27 at 3,250 B: 26,180 B / MD5
  `5d6b8802334d69e0f3f63f35bc2de72e`; a second identical encode reproduced it.
* All reported streams decode 60/60 with `dav2d-macos --strict 1` and the
  `--video-recon` payload equals the decoded payload byte for byte (10,368,000
  B). The all-deep depth −27 and −40 ceiling streams are recon-exact too.
* `tools/test.sh --build build-dev`: 10/10.
* `gate_video_nightly`, `gate_video_qp_lambda`, `video_drift_refresh_gate`,
  `video_ba_refresh_gate` and the new `video_pipe_deep_gate.sh` pass. The new
  gate checks the off pins, the default −27 depth, the key-QP anchor, strict
  decode + recon, the CDF slice-cross A/B and the mixed-form rejection.

## Caveats

* One clip, 240p, CPU-only, single tile, 8-bit 4:2:0, flat key+P chain. The
  source's second half is a global fade, which is the hardest case for a
  single-reference chain; the negative result is still about rate at 100 kbps,
  not about a property of the fade.
* The matched-rate depth −27 point is an extrapolation from 101.3 and
  104.7 kbps; depth −14 is interpolated between 99.576 and 103.004 kbps.
* The all-anchor streams exceed the 3,250 B allowance by design; they are
  quality ceilings, not recipes.
* The probe under-pricing is measured but not fixed: a per-anchor re-probe or
  a chain-context probe would change the same-budget allocation. It cannot
  change the conclusion, because the all-anchor ceiling is below 90.

## Named scope

The env gate only changes the refresh QP of an already-enabled
`--video-drift-refresh` chain (`--fast-decode` includes it). Without that chain it
is inert. It keeps the drift-refresh restrictions (CPU, single tile, flat
key+P, 8-bit 4:2:0), adds no CLI flag of its own, and emits no syntax the
existing writer cannot already signal. `--video-drift-refresh-qp` remains the
explicit depth knob for the oracle.
