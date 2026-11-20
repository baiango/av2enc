# Lane 3 — key-intra TX plan + coefficient level RD (`AV2ENC_RD_KEY_TX`)

Status: **built, measured, env-gated, default off. LEAD, not a KEEP.** The
key picture picks the same J as the rest of the RD engine
(`J = D_eff_perc + lambda * R_exact`), the plan rate is the §8.3 coefficient
model, and the winning leaves can be re-quantised by the coefficient trellis.
Two independent 7-point ladders (QP 114-120 and 118-124) both give matched K
**85.942 vs 85.634 (+0.308)**, but the sign is not consistent away from the
100 kbps bracket: the independent QP deltas span -0.32..+0.17 and full-ladder
line fits give +0.056 / +0.043. Best estimate: **+0.19 K**
(midpoint of the +0.308 bracket and the +0.04..+0.06 full-ladder fits; range
+0.05..+0.31). The
mechanism is real at fixed QP: the key is **0.75-1.57 % smaller at
-0.05..+0.04 dB over QP 114-121** (122-124 are flat), and the TX weight sweep
peaks at the calibrated 0.25.

Branch `exp/rd-lane3`, worktree `../rd-lane3`.

## Files

| File | Change |
|---|---|
| `src/pipeline/video/video_frame_intra.cpp` | new `AV2ENC_RD_KEY_TX` branch in `encode_intra_luma_tx_partition`: two-stage plan search, exact §5.20.6.3 partition cost, eff-perc plan D, plan/level counters |
| `src/pipeline/video/video_tx_search.h` | `Options::intra_rdoq`, `Options::model_rate_scale` (both default-off-equivalent: 0.4 is the historical RDOQ scale) |
| `src/pipeline/video/video_tx_search.cpp` | the RDOQ/model-rate branches accept intra leaves when `intra_rdoq` is set; `info.is_inter` follows the block; `kModelRateScale` becomes the option |
| `src/pipeline/video/video_frame_internal.h` | `search_luma_tx` forwards `intra_rdoq` / `model_rate_scale` |
| `tools/lane-3/key_tx_rd_ladder.sh` | reproducible interleaved-QP ladder, decode + metrics + K interpolation |

No CLI, `options.h`, `parse.cpp`, `main.cpp`, gate, or pin edit. Gate unset is
byte-identical to the pins (verified below).

## Gate

Environment only. The master gate plus the calibrated defaults reproduce the
measured point:

| Variable | Default | Meaning |
|---|---|---|
| `AV2ENC_RD_KEY_TX` | off | master gate |
| `AV2ENC_RD_KEY_TX_TOP` | 4 | screen candidates re-scored exactly (1..8) |
| `AV2ENC_RD_KEY_TX_SCALE` | 0.25 | multiplies `key_tx_rd_lambda` (the historical key weight) |
| `AV2ENC_RD_KEY_LEVEL` | 1 | run the coefficient trellis on the winning leaves |
| `AV2ENC_RD_KEY_LEVEL_SCALE` | auto | trellis `sse_scale`; auto = `0.25 * rdoq_sse_scale_at(qindex)` |
| `AV2ENC_RD_KEY_TX_D` | 1 | exact-stage plan D: 1 = eff-perc key CSF, 0 = plain coded SSE |
| `AV2ENC_RD_KEY_STATS` | off | print the plan histogram on the historical path too |

The measurement recipe is the standing one:

```sh
AV2ENC_EFF_PERC=1 AV2ENC_RD_UVFULL=1 \
AV2ENC_TX_DETAIL_SEARCH_TAIL=1 AV2ENC_TX_DETAIL_MIN=0.50 \
AV2ENC_RD_KEY_TX=1 \
build-release/av2enc testdata/nasa_480x240_30fps_60f.y4m -o out.ivf \
    --video --frames 60 --qp 119 --cpu-threads 8 --fast-decode \
    --video-drift-refresh-threshold 1.5 --video-drift-refresh-budget 3250
```

## Design

### One J

```text
D  = eff_perc::weighted_sse over the coding block's assembled luma recon
     (key CSF: AV2ENC_EFF_PERC_KEY_CSF, default = the inter gain; 7x7 band 0)
R  = sum(per-leaf exact coefficient + TX-type bits)
   + exact §5.20.6.3 partition symbols
J  = D + lambda * R,  lambda = key_tx_rd_lambda(qindex) * AV2ENC_RD_KEY_TX_SCALE
```

`D` is computed once over the block, not summed from the per-part candidate
costs, so the plan comparison and the committed recon read the same pixels.
With `AV2ENC_RD_KEY_TX_D=0` the same code uses plain coded SSE; that ablation
was worse (see below), so the key keeps the eff-perc CSF.

### Exact plan rate, no SSE proxy

The per-part search is the existing CPU reference (`search_luma_tx`) with
`use_context_rate = true`. That branch prices every candidate with
`block_syntax_bits` (`txb_skip`, EOB prefix/extra, `coeff_base`,
`coeff_base_eob`, `coeff_br`, the `read_quant` escape, the sign) from the §9.3
initial CDF slice for the **key picture's** QIndex. The TX-type cost is the
§5.20.8.3 static CDF cost the writer uses. The partition symbols are priced by
walking the real writer (`txpartition::write_tx_partition`) with a counting
sink, so `do_partition`, `tx_partition_type` and `tx_2or3_partition_type` are
counted with their real contexts and values. Nothing in the exact stage is
the old `coefficient_bits * nonzero` heuristic; that estimate survives only in
the cheap screen.

The historical path's total rate estimate on this key is **25,533 bits**; the
exact §8.3 model prices the gated plan at **121,408 bits** (4.76x). That
ratio is why the calibrated TX lambda is ~0.25x: `0.25 * 8 = 2.0` is the
exact-rate equivalent of the historical `8 * 1-bit-per-nonzero`.

### Two-stage search

1. **Screen** (255 leaves x the legal set): the historical cost
   `plain SSE + key_tx_lambda * estimate`; every partition evaluates one cheap
   leaf search.
2. **Exact top-K** (`AV2ENC_RD_KEY_TX_TOP`, default 4): re-run the same
   partition with the exact rate, the eff-perc D, the calibrated lambda and
   optionally the trellis; the winner is chosen by exact J. Ties keep the
   screen order.

The screen is not the same total order as the exact cost: **57 of 255 leaves
(22.4 %)** changed their winning partition at QP119.

### Candidate set

For each key coding block the plan enumerates `TX_PARTITION_NONE` plus every
legal `SPLIT`/`HORZ`/`VERT` from `txpartition::legal_splits()`. The transform
sizes follow the coding-block size: 64x64 gives 64x64 / four 32x32; 32x32 gives
32x32 / four 16x16 (and 32x16 / 16x32); 16x16 gives 16x16 / four 8x8 (and
16x8 / 8x16). Subsizes are decoded in the real per-transform prediction chain,
so each part predicts from the previous part's reconstruction.

`txpartition` signals one partition level per coding block. A 4x4 transform
therefore needs an 8x8 coding block; the key partition tree's floor is 16x16
(`--video-min-block` default). **4x4 selected 0 times is correct** for this
tree, not a search failure. Under the gate the selected transform sizes at
QP119 are 8x8 632, 16x16 24, 32x32 3, 64x64 8; the historical path selects
8x8 432, 16x16 32, 32x32 3, 64x64 8. The exact coefficient rate moves work
from 16x16 into 8x8.

IDTX is not evaluated: `--fast-decode` leaves `enable_idtx_intra = 0`, and the
intra TX-set query forces the DCT_IDTX set to DCT_DCT (`tx_type.cpp`
`transform_type()` override). Turning the sequence bit on changes the key
header and is a separate measurement; it is not part of this gate.

### Coefficient level trellis

`AV2ENC_RD_KEY_LEVEL=1` turns on the RDOQ DP for intra leaves:
`codec/rate/rdoq` over the §5.20.7.30 scan, rate from
`rdoq::av2::Model` (exact base/br/mag/sign/EOB costs, `is_inter = false`),
distortion `(original - level*step)^2`, lambda `sse_scale * step^2`. It runs on
the winning leaf only (the scalar levels seed the contexts), and a refined leaf
is kept only if its exact `J` improves and the §5.20.7.29 intra codability rule
now holds (`eob >= 2` or DCT_DCT).

Calibration (`AV2ENC_RD_KEY_LEVEL_SCALE`, QP119, plan arm fixed, key bytes /
key PSNR / stream mean PSNR):

| scale | key B | key dB | mean PSNR | note |
|---:|---:|---:|---:|---|
| off | 15,970 | 46.087 | 44.973 | plan only |
| 0.0035 | 15,973 | 46.080 | 44.922 | |
| **0.007** | **15,976** | **46.032** | **44.953** | closest to auto (0.25x inter) |
| 0.014 | 15,906 | 46.055 | 44.947 | best key bytes |
| 0.030 | 15,900 | 46.036 | 44.892 | over-zeroes |
| 0.100 | 15,909 | 45.974 | 44.904 | as hard as the inter RDOQ scale loses key PSNR |

The trellis is not a large independent lever on this key: its best point moves
key bytes by <=0.4 % at <=0.05 dB. At QP119 the level arm's KPI is 86.077 vs
85.864 for plan-only, but the gain is mostly Butteraugli ripple (2.038 vs
2.102) and the mean PSNR is 0.02 dB lower; the plan search is the mechanism and
the trellis is kept because the task requires the level RD under the same J and
because the auto scale is the only setting that does not lose mean PSNR by
more than 0.05 dB.

## Results

Host Apple M4, `build-release` (`-O3`, warnings + `-Werror`),
`--cpu-threads 8`, 60 frames, 8-bit 4:2:0, one tile. The ladders are driven by
`tools/lane-3/key_tx_rd_ladder.sh` in bash, base and arm interleaved per QP.
Quality: CVVDP 0.5.7 (standard_fhd, MPS), Butteraugli and SSIMULACRA2 from
`tools/extra_metrics.py`, `tools/kpi.py`. K interpolated linearly in rate to 100.0 kbps.

### Matched K: two 7-point ladders

Two separate sessions, base and arm interleaved per QP, `build-release`, driven
by `tools/lane-3/key_tx_rd_ladder.sh` (bash). Ladder A is QP 114-120, ladder B
is QP 118-124. The overlapping QPs were encoded independently in each session;
the streams are byte-identical (see Determinism).

**Ladder A (114-120)**

| QP | base kbps | arm kbps | dKPI | dBA | dSS2 |
|---:|---:|---:|---:|---:|---:|
| 114 | 113.50 | 112.10 | +0.090 | -0.037 | -0.15 |
| 115 | 111.17 | 109.32 | -0.323 | +0.102 | +0.04 |
| 116 | 109.02 | 108.42 | +0.170 | -0.047 | +0.24 |
| 117 | 106.05 | 106.07 | -0.113 | +0.038 | +0.15 |
| 118 | 104.66 | 103.22 | -0.360 | +0.125 | +0.06 |
| 119 | 101.32 | 100.59 | +0.241 | -0.087 | -0.15 |
| 120 | 99.04 | 98.45 | +0.102 | -0.021 | +0.06 |
| **K@100** | **85.634** | **85.942** | **+0.308** | | |

**Ladder B (118-124)**

| QP | base kbps | arm kbps | dKPI | dBA | dSS2 |
|---:|---:|---:|---:|---:|---:|
| 118 | 104.66 | 103.22 | -0.360 | +0.125 | +0.06 |
| 119 | 101.32 | 100.59 | +0.241 | -0.087 | -0.15 |
| 120 | 99.04 | 98.45 | +0.102 | -0.021 | +0.06 |
| 121 | 97.88 | 97.07 | +0.005 | +0.007 | +0.09 |
| 122 | 95.99 | 96.16 | +0.050 | -0.001 | +0.01 |
| 123 | 93.76 | 94.53 | -0.004 | -0.027 | -0.40 |
| 124 | 92.75 | 92.04 | -0.038 | +0.007 | -0.20 |
| **K@100** | **85.634** | **85.942** | **+0.308** | | |

**Stream sizes (IVF, bytes)**

| QP | base | arm | delta |
|---:|---:|---:|---:|
| 114 | 28,376 | 28,025 | -351 |
| 115 | 27,793 | 27,331 | -462 |
| 116 | 27,254 | 27,104 | -150 |
| 117 | 26,513 | 26,518 | +5 |
| 118 | 26,165 | 25,805 | -360 |
| 119 | 25,330 | 25,148 | -182 |
| 120 | 24,760 | 24,612 | -148 |
| 121 | 24,471 | 24,268 | -203 |
| 122 | 23,997 | 24,041 | +44 |
| 123 | 23,439 | 23,633 | +194 |
| 124 | 23,187 | 23,010 | -177 |

The base interpolation reproduces the programme base K = 85.634 on both
ladders. The arm reproduces **+0.308** on both because 100 kbps is bracketed by
the same two points (QP 119 and 120) in both ladders: +0.241 at 100.59 kbps and
+0.102 at 98.45 kbps. Those two points are the sign-consistent bracket.

The rest of each ladder is not sign-consistent. Both ladders score 4 positive
and 3 negative same-QP deltas. The independent QPs are 114-117 (A) and 121-124
(B), with deltas `+0.090, -0.323, +0.170, -0.113` and `+0.005, +0.050, -0.004,
-0.038`. A line fit over the whole 7-point ladder is a diagnostic only, but it
is much less flattering than the bracket:

| ladder | base fit K@100 | arm fit K@100 | delta |
|---|---:|---:|---:|
| A (114-120) | 85.650 | 85.706 | +0.056 |
| B (118-124) | 85.519 | 85.562 | +0.043 |

The arm wins the operating bracket and loses above ~102 kbps; at 92-97 kbps the
delta is zero to within 0.05 K.

**Equal-rate pairs** (closest rate in the same QP range, both ladders pooled):

| base QP | base kbps | arm QP | arm kbps | rate delta | dKPI |
|---:|---:|---:|---:|---:|---:|
| 115 | 111.17 | 114 | 112.10 | +0.93 | +0.070 |
| 116 | 109.02 | 115 | 109.32 | +0.31 | -0.014 |
| 116 | 109.02 | 116 | 108.42 | -0.60 | +0.170 |
| 117 | 106.05 | 117 | 106.07 | +0.02 | -0.113 |
| 118 | 104.66 | 118 | 103.22 | -1.44 | -0.360 |
| 119 | 101.32 | 119 | 100.59 | -0.73 | +0.241 |
| 120 | 99.04 | 120 | 98.45 | -0.59 | +0.102 |
| 121 | 97.88 | 120 | 98.45 | +0.56 | +0.473 |
| 122 | 95.99 | 122 | 96.16 | +0.18 | +0.050 |
| 123 | 93.76 | 123 | 94.53 | +0.78 | -0.004 |
| 124 | 92.75 | 124 | 92.04 | -0.71 | -0.038 |

The only positive run is 98-101 kbps. Below and above it the pair deltas are
mixed around zero. At the exact same rate pair (106.05 / 106.07) the arm is
-0.113.

### TX weight sweep

One 5-point ladder (117-121), one base run, four arm runs at
`AV2ENC_RD_KEY_TX_SCALE` 0.15 / 0.25 / 0.35 / 0.50:

| scale | K@100 | delta | QP119 arm kbps | QP119 key B / dB |
|---:|---:|---:|---:|---:|
| 0.15 | 85.341 | -0.294 | 102.40 | 16,043 / 46.085 |
| **0.25** | **85.942** | **+0.308** | **100.59** | **15,805 / 46.019** |
| 0.35 | 85.678 | +0.044 | 100.62 | 15,730 / 45.990 |
| 0.50 | 85.615 | -0.019 | 99.66 | 15,570 / 45.870 |
| base | 85.634 | 0 | 101.32 | 15,984 / 46.048 |

0.25 is the peak of the calibration curve and remains the default. 0.15 spends
too many bits (1-2 kbps above the 0.25 arm at the same QP) and its matched K is
0.60 lower; 0.35 and 0.50 are flat to slightly negative on the matched metric,
though their BA is smoother. The
exact-rate ratio argument (0.25 x 8 = 2.0, the exact-rate equivalent of the
historical 8, because the exact model is 4.76x the old estimate) is consistent
with the measured peak.

### Key RD at fixed QP (the mechanism)

Key bytes and key PSNR at every unique ladder QP (0.25 arm):

| QP | base key B | arm key B | delta | base key dB | arm key dB | delta |
|---:|---:|---:|---:|---:|---:|---:|
| 114 | 17,418 | 17,155 | -263 (-1.51%) | 47.071 | 47.115 | +0.044 |
| 115 | 17,033 | 16,765 | -268 (-1.57%) | 46.822 | 46.856 | +0.034 |
| 116 | 16,799 | 16,594 | -205 (-1.22%) | 46.673 | 46.684 | +0.011 |
| 117 | 16,528 | 16,322 | -206 (-1.25%) | 46.474 | 46.457 | -0.017 |
| 118 | 16,236 | 16,115 | -121 (-0.75%) | 46.301 | 46.253 | -0.048 |
| 119 | 15,984 | 15,805 | -179 (-1.12%) | 46.048 | 46.019 | -0.029 |
| 120 | 15,743 | 15,621 | -122 (-0.78%) | 45.862 | 45.826 | -0.036 |
| 121 | 15,476 | 15,308 | -168 (-1.09%) | 45.643 | 45.651 | +0.008 |
| 122 | 15,265 | 15,291 | +26 (+0.17%) | 45.460 | 45.505 | +0.045 |
| 123 | 15,084 | 15,074 | -10 (-0.07%) | 45.338 | 45.328 | -0.010 |
| 124 | 14,870 | 14,797 | -73 (-0.49%) | 45.163 | 45.151 | -0.012 |

At 114-121 the key is 0.75-1.57 % smaller at -0.05..+0.04 dB; at 122-124 the
key difference is +0.17..-0.49 % and the PSNR is flat. That is the
"key TX/level RD improves at fixed QP" half of the LEAD criterion, and it is the
part of the result that does not depend on the operating rate.

### Determinism

- A clean out-of-tree build (`build-verify`, fresh configure, no ccache, full
  compile, release flags) reproduces the arm QP119 stream exactly:
  25,148 B, MD5 `1354bc5929f1c8552a112977c03c09a6`, and the same plan counters
  (`none=43 split=162 horz=36 vert=14; tx8=632 tx16=24; dct=241 adst=550;
  parts=791; screen_flips=57`). The earlier incremental-build chroma anomaly
  does not reproduce on a clean build.
- The two independent ladder sessions produce byte-identical overlapping
  streams: arm QP120 MD5 `42b4811fda3f0921cccdb22e64c0ec1a` and base QP120
  MD5 `49228567054c354f2ebd573f9845198f` in both sessions.
- The scale sweep's base 117-121 streams match ladder A's base streams
  byte-for-byte (same size and metrics).

### Key bytes by family (QP119, `AV2ENC_BIT_CENSUS=<path>`, first picture)

Coarse (bits):

| family | base | arm | delta |
|---|---:|---:|---:|
| tile payload (B) | 15,944 | 15,765 | -179 |
| info bits | 127,456.5 | 126,055.4 | -1,401.2 |
| partition | 247.4 | 247.4 | 0 |
| intra_mode | 3,248.2 | 3,622.6 | +374.4 |
| residual | 81,515.9 | 79,724.4 | -1,791.5 |
| ccso_blk | 28 | 32 | +4 |
| cdef_index | 0 | 0 | 0 |

The residual split (`fine`) shows where the -1,791.5 bits come from:

| fine family | base bits | arm bits | delta |
|---|---:|---:|---:|
| `txb_skip` | 577.1 | 598.0 | +20.9 |
| `eob` | 5,098.7 | 5,561.1 | +462.3 |
| `coeff_base` | 63,912.4 | 62,401.2 | -1,511.2 |
| `coeff_br` | 13,854.8 | 13,339.0 | -515.8 |
| `sign` | 537.4 | 640.7 | +103.3 |
| `tx_partition` | 453.5 | 350.5 | -103.0 |
| `coeff_mag` (escape) | 39,031 | 38,827 | -204.0 |

The exact plan trades +374 bits of intra-mode syntax and +462 bits of EOB for
1,511 fewer `coeff_base` and 516 fewer `coeff_br` bits: fewer, better-placed
coefficients rather than a partition change (the coding-block partition row is
flat).

### Plan counts (QP119, 255 key luma leaves)

| | base | arm |
|---|---:|---:|
| NONE | 59 | 43 |
| SPLIT | 113 | 162 |
| HORZ | 62 | 36 |
| VERT | 21 | 14 |
| transform leaves | 677 | 791 |
| 4x4 | 0 | 0 |
| 8x8 | 432 | 632 |
| 16x16 | 32 | 24 |
| 32x32 | 3 | 3 |
| 64x64 | 8 | 8 |
| DCT | 269 | 241 |
| ADST | 408 | 550 |
| IDTX | 0 | 0 |
| screen->exact flips | - | 57 (22.4 %) |

### Per-frame PSNR-Y (QP119, selected frames)

| frame | base | arm | delta | base B | arm B |
|---:|---:|---:|---:|---:|---:|
| 0 (key) | 46.048 | 46.019 | -0.029 | 15,984 | 15,805 |
| 1 | 46.049 | 46.021 | -0.028 | 20 | 21 |
| 10 | 45.081 | 45.046 | -0.035 | 23 | 24 |
| 20 | 45.138 | 45.038 | -0.100 | 96 | 87 |
| 30 | 44.536 | 44.555 | +0.019 | 50 | 99 |
| 33 | 43.872 | 43.868 | -0.004 | 398 | 360 |
| 36 | 43.963 | 44.021 | +0.058 | 145 | 139 |
| 40 | 44.376 | 44.234 | -0.142 | 444 | 385 |
| 48 | 44.813 | 44.810 | -0.003 | 309 | 298 |
| 59 | 46.718 | 46.826 | +0.108 | 215 | 251 |
| mean | 44.934 | 44.893 | -0.041 | | |

The old-build anomaly (a P-tail collapse from a key-only change, seen during
development) is not present in the final binary: the P chain stays within
0.15 dB of the base on every frame and the mean moves by -0.04 dB. The final
binary is deterministic (two identical encodes produce the same MD5).

## Conformance

- Gate off pins, release `-Werror`: QP106 **25,008 B / `0639b436`**, QP110
  **23,690 B / `1b99127a`**. Gate off every changed branch is not entered.
- Gate on QP119: `dav2d --strict 1` decodes 60/60;
  `tools/validate_dav2d.sh` OK; `--video-recon` payload equals the decoded
  payload byte for byte (`cmp` clean, 10,368,000 B).
- `tools/gate_video_nightly.sh --bin build-release/av2enc`: OK (expansion,
  pins, strict decode, visible recon, BAWP).
- `ctest` 12/12.
- Dev and release builds: `cmake --preset dev-cache` and
  `cmake --preset release` (warnings + `-Werror`) both clean.
- Clean rebuild determinism: a fresh `build-verify` configure/compile (no
  ccache) reproduces the arm QP119 stream exactly (25,148 B, MD5
  `1354bc5929f1c8552a112977c03c09a6`), and the two ladder sessions reproduce
  their overlapping streams (arm QP120 `42b4811fda3f0921cccdb22e64c0ec1a`,
  base QP120 `49228567054c354f2ebd573f9845198f`).
- Exact sizes at the bracket: QP119 base 25,330 B / arm 25,148 B; QP120 base
  24,760 B / arm 24,612 B; QP118 base 26,165 B / arm 25,805 B.

## Verdict

**LEAD.** Two independent 7-point ladders both give matched K **+0.308**
(85.942 vs 85.634), and the 100 kbps bracket is positive in both (+0.241 at
100.59 kbps, +0.102 at 98.45 kbps). The sign is *not* consistent across either
ladder: each has 4 positive and 3 negative same-QP deltas, the independent
points 114-117 and 121-124 span -0.323..+0.170 and -0.038..+0.050, and a line
fit over the full ladder gives only +0.056 (A) / +0.043 (B). The +0.308 is the
favorable operating bracket plus a Butteraugli dip at arm QP119 (BA 2.038 vs
2.126), not a resolved iso-rate gain. Best estimate: **+0.19 K**, range
+0.05..+0.31: the protocol gives +0.308, the full-ladder line fits give
+0.043 / +0.056, and the midpoint is the defensible number. Not a claim of
a KEEP; the integrator can compose and measure, but the lane should not
publish +0.31 as a resolved gain.

The mechanism is real in the key: at fixed QP the key is 0.75-1.57 % smaller
at -0.05..+0.04 dB over QP 114-121 (122-124 are flat), and the TX weight sweep
peaks at the calibrated 0.25 (0.15 -0.294, 0.25 +0.308, 0.35 +0.044, 0.50
-0.019). The unresolved part is the conversion of that key RD win into the
5-metric K through the P chain, which is dominated by the QP ripple.

Recommended composition point: `AV2ENC_RD_KEY_TX=1` with the defaults above
(0.25 TX weight, top-4, trellis on, key CSF plan D). `AV2ENC_RD_KEY_TX_D=0`
(plain SSE plan D) is worse on the objective that scores: at QP119 it is
25,316 B (101.26 kbps) with key 15,843 B / 46.042 dB and stream mean PSNR
44.937, but KPI 85.642 against the CSF arm's 86.077 at 100.59 kbps (base
85.837 at 101.32). PSNR prefers the plain-SSE plan; the K metric does not.
Restricting the exact stage to the screen winner (`AV2ENC_RD_KEY_TX_TOP=1`) is
not a better point either: 25,400 B with key 15,979 B / 46.058 dB and mean
44.913, against the default TOP=4's 25,148 B with key 15,805 B / 46.019 dB and
mean 44.893.

## Blockers / open items

- **Lane 1's `video_rd_engine.*` is not in this worktree.** The local
  equivalent is documented above: `exact_partition_bits()`,
  `search_luma_tx(..., context_rate=true)` and `txtsearch::Options` are the
  interface. When lane 1 lands, replace `exact_partition_bits` + the exact-rate
  `block_syntax_bits` calls with its per-family rate and re-calibrate
  `AV2ENC_RD_KEY_TX_SCALE`; do not run both.
- **IDTX is not in the candidate set** under `--fast-decode` (`enable_idtx_intra`
  is 0 and the intra TX-set query forces DCT_IDTX to DCT_DCT). Measuring it
  needs the sequence bit and a header change, which is outside this lane's
  no-syntax-change rule.
- **4x4 is unreachable**, not unchosen: the key coding-block floor is 16x16,
  and §5.20.6.3 signals one partition level per coding block. An 8x8 coding
  block is required for a 4x4 transform. If the key partition tree ever gains
  an 8x8 floor, this search will see 4x4 without a code change.
- **QP ripple.** The two-ladder result is still one bracket (QP 119/120); the
  extra points are not sign-consistent. The safest reading is LEAD with the
  range above. A composition measurement on the merged tree is the next test.
- **The earlier cross-build chroma difference is resolved by a clean build.**
  The earlier anomaly came from an incremental build; a fresh out-of-tree
  release build (`build-verify`, no ccache) reproduces the arm QP119 stream
  byte-for-byte (25,148 B, MD5 `1354bc59...`), and the two ladder sessions
  reproduce their overlapping streams exactly. Use a clean build for a
  composited measurement.
- The key `partition` family is unchanged (247.4 bits) because the number of
  coding blocks and the partition-symbol contexts do not move; only the TX
  partitions inside the leaves change. The exact plan rate's 4.76x gap to the
  historical estimate is the calibration story, not a bitstream cost.
