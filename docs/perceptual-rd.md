# Perceptual TX-RD experiment (`AV2ENC_PERCEPTUAL_RD`)

Status: **opt-in, default off.**  The best measured strength is `0.75`, kept
behind the environment gate; no preset sets it.  The gain is modest and the
strength response is narrow (the neighbouring 0.5 / 1.0 / 2.0 points are
negative), so this is an experiment, not a default.

## What it does

The encoder prices distortion with SSE (or SATD), so every decoded sample
error costs the same.  Butteraugli does not: on this clip the measured error
mass is on the fine-detail / text band of the P frames (frames 44-59), while
the flat background is already coded at about 54 dB PSNR.  Plain SSE
under-prices the detail-rich blocks that produce that error, and the TX search
prunes their high-frequency coefficients first.

The proxy re-weights the TX-search distortion by local source activity
(`src/pipeline/video/video_perceptual.h`):

```
activity = mean absolute deviation (MAD) of an 8x8 source luma cell
raw      = (kReferenceActivity + kActivitySoftness) / (activity + kSoftness)
weight   = 1 + strength * (1 - raw / frame_mean_raw)
```

`kReferenceActivity = kActivitySoftness = 12` (8-bit luma), the weight is
clamped to `[0.2, 5.0]`, and `frame_mean_raw` is a deterministic 4-cell-stride
subsample of the whole luma frame, so the mean weight over the frame is 1 and
the knob is close to a redistribution.  Detail-rich cells get weight > 1,
smooth cells < 1.

The TX search prices candidates as `sse + lambda * rate`, so weighting its SSE
by the leaf factor is exactly dividing lambda by that factor.  That is how it
is applied (both the intra and inter luma TX search sites in
`video_frame_decision.cpp`): the shared TX-search module, and therefore the
CUDA reference contract, is untouched.  The leaf-level candidate / skip-
residual / intra-inter SSE is deliberately **not** re-weighted; that variant
was measured too (below) and is a wash or worse.

The environment variable is read once.  Strength 0 builds no weight map, keeps
the historical SSE and passes both TX lambdas through unchanged; the QP 106
nightly pin is byte-identical (25,008 B, MD5
`0639b43630abb454f7be5ac494d2efb4`).

## Measurement

`testdata/nasa_480x240_30fps_60f.y4m`, 60 frames, `--nightly --cpu-threads 8`.
Quality: `tools/quality.py` (CVVDP JOD), `tools/extra_metrics.py` (Butteraugli,
SSIMULACRA2), `tools/kpi.py`.  One metric job at a time, `nice -n 10`.  The
baseline QP106 point is the pin: 100.032 kbps, JOD 9.7334, BA 3.4202,
SS2 80.837, K 82.15.

Iso-rate 100.0 kbps is a linear interpolation over rate between the two
bracketing measured QPs.  Note that K vs QP is a sawtooth of about ±1.5 K on
this clip (the drift-refresh schedule flips), so single-bracket numbers are
noisy; the six-QP window and the no-refresh controls below are the robust
checks.

### Final strength sweep (TX lambda weighting)

QP 104/106/108, strength `0` / `0.5` / `0.75` / `1.0` / `2.0`:

| strength | QP | kbps | JOD | BA | SS2 | K |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 104 | 104.632 | 9.7185 | 4.1720 | 80.108 | 80.73 |
| 0 | **106** | **100.032** | **9.7334** | **3.4202** | **80.837** | **82.15** |
| 0 | 108 | 97.164 | 9.7216 | 3.6824 | 80.633 | 81.60 |
| 0.5 | 104 | 104.024 | 9.6907 | 3.9596 | 79.678 | 80.84 |
| 0.5 | 106 | 100.484 | 9.7279 | 3.4900 | 80.144 | 81.89 |
| 0.5 | 108 | 96.932 | 9.7319 | 3.7807 | 80.353 | 81.45 |
| **0.75** | 104 | 104.876 | 9.7077 | 4.1769 | 80.178 | 80.68 |
| **0.75** | **106** | **99.468** | **9.7243** | **3.0557** | **80.659** | **82.78** |
| **0.75** | 108 | 95.904 | 9.7244 | 3.5908 | 80.706 | 81.79 |
| 1.0 | 104 | 104.440 | 9.7141 | 4.0590 | 80.177 | 80.89 |
| 1.0 | 106 | 100.200 | 9.7164 | 3.8627 | 80.509 | 81.26 |
| 1.0 | 108 | 97.492 | 9.7128 | 4.5674 | 80.503 | 80.23 |
| 2.0 | 104 | 103.688 | 9.7157 | 3.5125 | 80.232 | 81.80 |
| 2.0 | 106 | 100.272 | 9.7017 | 4.0577 | 80.257 | 80.84 |
| 2.0 | 108 | 97.196 | 9.7084 | 4.4092 | 80.296 | 80.38 |

Iso-rate 100.0 kbps:

| strength | JOD | BA | SS2 | K | ΔJOD | ΔBA | ΔSS2 | ΔK |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 9.7333 | 3.4231 | 80.835 | 82.15 | — | — | — | — |
| 0.5 | 9.7284 | 3.5296 | 80.172 | 81.83 | −0.005 | +0.107 | −0.663 | −0.32 |
| **0.75** | **9.7227** | **3.1660** | **80.612** | **82.54** | **−0.011** | **−0.257** | **−0.223** | **+0.39** |
| 1.0 | 9.7161 | 3.9147 | 80.509 | 81.18 | −0.017 | +0.492 | −0.326 | −0.97 |
| 2.0 | 9.7023 | 4.0888 | 80.260 | 80.80 | −0.031 | +0.666 | −0.575 | −1.35 |

Only the 0.75 point is positive; it is a narrow optimum between two worse
neighbours, which is why the gate is off and `docs`/the commit call it out.

### Robustness of the 0.75 point

Six QPs (paired against the baseline at the same QP):

| QP | base kbps | base K | 0.75 kbps | 0.75 K | Δrate | ΔK |
|---:|---:|---:|---:|---:|---:|---:|
| 104 | 104.632 | 80.73 | 104.876 | 80.68 | +0.23% | −0.05 |
| 105 | 101.368 | 81.81 | 101.224 | 81.91 | −0.14% | +0.10 |
| 106 | 100.032 | 82.15 | 99.468 | 82.78 | −0.56% | +0.63 |
| 107 | 99.540 | 80.51 | 98.364 | 81.42 | −1.18% | +0.91 |
| 108 | 97.164 | 81.60 | 95.904 | 81.79 | −1.30% | +0.19 |
| 110 | 94.760 | 81.41 | 94.988 | 81.77 | +0.24% | +0.36 |
| mean | 99.58 | 81.37 | 99.14 | 81.73 | −0.45% | **+0.36** |

Five of six points are positive; the mean gain is +0.36 K at a 0.45% lower
mean rate.  Two no-refresh controls (`--video-drift-refresh-threshold 29`, so
no refresh frame can flip and hide the RD effect):

| QP | base bytes | base K | 0.75 bytes | 0.75 K | Δrate | ΔK |
|---:|---:|---:|---:|---:|---:|---:|
| 106 | 23,822 | 79.78 | 23,887 | 80.17 | +0.27% | +0.39 |
| 108 | 23,132 | 79.35 | 23,344 | 79.61 | +0.92% | +0.26 |

The QP106 control is the cleanest evidence: +0.39 K for +0.27% rate with no
refresh flips.  QP108 is +0.26 K for +0.92% (about +0.09 K after a rough
0.19 K/% rate adjustment).

### Per-frame Butteraugli on the flagged band

At QP 106, mean Butteraugli over frames 44-59 (the P-frame detail/text band)
versus the baseline:

| stream | frames 1-43 | frames 44-59 | all |
|---|---:|---:|---:|
| baseline QP106 | 2.462 | 5.844 | 3.420 |
| 0.75 QP106 | 2.197 | **5.228** | **3.056** |
| 0.75, no refresh QP106 | 3.843 | **6.260** | **4.528** |
| baseline, no refresh QP106 | 3.795 | 7.268 | 4.779 |

The proxy moves the band by about −0.6 BA at the pinned operating point and
−1.0 without refresh, which is the intended effect; the early frames improve
too at QP106.

## Ablations and the failed inverse direction

**Smooth-weighted masking (the textbook direction) is negative.**  The same
activity map with the sign reversed (smooth cells heavier) was measured first
at QP 104/106/108:

| strength | ΔJOD | ΔBA | ΔSS2 | ΔK at iso-rate |
|---:|---:|---:|---:|---:|
| 0.25 | −0.036 | +0.476 | −0.655 | −1.10 |
| 0.5 | −0.030 | +0.408 | −0.850 | −0.99 |
| 1.0 | −0.089 | +0.496 | −1.872 | −1.59 |
| 2.0 | −0.101 | +0.713 | −2.754 | −2.13 |

Flat 8x8 cells are already at 53.9 dB PSNR (MSE 0.27) versus 40.5 dB in the
textured cells, so there is nothing to buy in the smooth half: the bits only
come out of the detail band, where the Butteraugli error is.

**Weighting the leaf SSE on top of the TX lambda is a wash.**  Applying the
same weight to `sse_vis` (inter candidates, skip/residual, intra/inter) and to
chroma at strength 0.75 gives 100.844 kbps, JOD 9.7415, BA 3.1855, SS2 80.924,
K 82.65 at QP 106 (baseline 82.15 at 100.03); at iso-rate 100.0 over QP
104/106/108 it interpolates to K 82.15, i.e. exactly a tie.  The per-band gain
is already obtained from the TX precision, and the extra leaf terms only move
bits between candidates and planes.  The final code therefore applies the
proxy only where the TX search combines distortion and rate.

**TX lambda weighting at 1.0 / 2.0 is negative** (see the table); only 0.75 is
positive.  The strength response is discontinuous because the drift-refresh
and partition decisions flip, which is why the off/default is not changed.

## Verification with the gate off

With `AV2ENC_PERCEPTUAL_RD` unset:

* `tools/refactor/check_wavefront_pins.sh`: 25,009 / 25,141 / 23,690 / 24,113 B
  and the 720p pin 36,572 B MD5 `c4ba28138d8f6569d5e08721a66d755c`.
* `tools/gate_video_nightly.sh` and
  `src/pipeline/video/gate_video_qp_lambda.sh`: OK.
* `tools/test.sh --build build-dev`: 10/10.

Not merged and not pushed.
