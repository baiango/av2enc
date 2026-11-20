# lpl36 perceptual refresh (`AV2ENC_PIPE_LPL_REFRESH`)

Status: **opt-in, env-gated, CPU-only; measured at 240p/100 kbps as K 83.93
(interpolated from the tuned QP bracket) and 83.73-84.03 on the direct
allocation sweep, against the 83.80 tuned-drift wall and the 82.15 nightly
pin.** The tuned 3,250 B recipe at QP 111 is K 84.17 at 101.48 kbps. At
1440x720/~292.6 kbps the pipeline is flat against the 45.61 pin (45.69
interpolated). Both points are below the 85.10 perceptual RD core and far below
the 90 wall, so the pipeline is not tuned further.

This revives the removed perceptual refresh architecture with the user-owned
**lpl36** metric instead of the deleted third-party metric port. The planner
still uses only a perceptual error and priced payloads: no PSNR term, no new
syntax, and every pinned stream is byte-identical when the env gate is off.
The C++ metric is first-party av2enc code under the repository's Unlicense; the
header carries a one-line attribution to the metric's author (baiango).

## Metric

`src/pipeline/video/video_lpl36.h` is a self-contained port with no OpenCV or
other external library:

```
error = (mean + std)(|L * G2 (Y_ref - Y_dist)|)
      + 0.5 * (mean + std)(|L * G4 (Y_ref - Y_dist)|)
```

* `G` is the OpenCV `pyrDown` 5x5 Gaussian `[1 4 6 4 1]/16`, reflect-101
  border, applied `log2(scale)` times and sampled at even positions;
* `L` is the 5x5 LoG (centre 16, mid-ring -2, outer-ring -1, reflect-101);
* `mean`/`std` are population statistics of the absolute filtered plane.

Lower is better; identical planes score exactly 0. The planner consumes
`error = -score`, so the DP and the carry model keep their original
minimum-sum direction. `Metric::Init` owns every scratch plane and the
per-frame `Error` call performs no allocation. On the parity gate's lossy
4-frame pair the maximum C++/Python relative error is **2.7e-11** (mean
1.6e-11), well inside the 1e-3 contract.

## Recipe

```sh
AV2ENC_PIPE_LPL_REFRESH=1 build-release/av2enc \
  testdata/nasa_480x240_30fps_60f.y4m -o /tmp/lpl.ivf \
  --video --frames 60 --cpu-threads 8 --qp 111 --fast-decode \
  --video-drift-refresh-threshold 1.5 --video-drift-refresh-budget 3250
```

`--fast-decode` supplies the enabled drift-refresh plumbing; the env gate replaces
its PSNR trigger. `--video-drift-refresh-threshold 1.5` is part of the tuned
recipe and is ignored by the planner; the budget is not.

The 720p pin recipe is unchanged:

```sh
AV2ENC_PIPE_LPL_REFRESH=1 build-release/av2enc \
  testdata/space8k_1440x720_60f.y4m -o /tmp/lpl720.ivf \
  --video --frames 60 --cpu-threads 8 --qp 217 --fast-decode \
  --video-key-qp-offset -57 --video-leaf-rd-lambda 250 \
  --video-ext-partitions
```

## What it does

1. **Pass 1 (measure the chain).** Encode with no refresh and record the
   per-frame lpl36 error of the reconstruction against its source plus the tile
   payload.
2. **Pass 2 (probe the refresh).** Encode every inter frame as a refresh and
   record the same quantities. This gives each candidate's measured refresh
   floor and its measured extra payload.
3. **Plan.** A dynamic program chooses the refresh set that minimises the
   predicted sum of per-frame lpl36 error under the existing
   `--video-drift-refresh-budget`. Between refreshes the prediction starts at
   the measured all-refresh floor and closes the gap to the no-refresh curve
   with 0.8 survival per frame.
4. **Emit + refine.** Apply the schedule through the unchanged drift-refresh
   plumbing (frame QP drop + `force_inter_residual`). If the probe under-priced
   the isolated refreshes and the payload cap dropped scheduled frames, the
   measured actual/estimated ratio re-plans once and emits one refinement pass.

No refresh decision uses PSNR. The untouched default drift trigger is bypassed
when the env gate is on.

## Results at 240p

`testdata/nasa_480x240_30fps_60f.y4m`, 60 frames, `--fast-decode`, QP / budget
bracketing 100 kbps. Quality: `tools/quality.py` (CVVDP 0.5.7,
`standard_fhd`, MPS), `tools/extra_metrics.py`, `tools/kpi.py`.

| Stream | QP | Budget | Bytes | kbps | JOD | BA | SS2 | PSNR-Y | K |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nightly pin | 106 | — | 25,008 | 100.032 | 9.7334 | 3.4202 | 80.837 | 44.668 | **82.15** |
| PSNR trigger (wall) | 111 | 3,250 | 24,950 | 99.800 | 9.7513 | 2.7310 | 81.422 | 44.703 | **83.75** |
| **lpl36 refresh (tuned)** | **111** | **3,250** | **25,370** | **101.480** | **9.7626** | **2.6343** | **82.297** | **44.664** | **84.17** |
| lpl36 refresh, matched allocation | 111 | 2,950 | 25,009 | 100.036 | 9.7660 | 2.8450 | 82.411 | 44.735 | **83.73** |
| lpl36 refresh, allocation sweep | 111 | 3,000 | 25,099 | 100.396 | 9.7789 | 2.7569 | 82.672 | 44.830 | **84.03** |
| lpl36 refresh, allocation sweep | 111 | 2,900 | 24,835 | 99.340 | 9.7593 | 2.9835 | 81.894 | 44.672 | **83.31** |
| lpl36 refresh | 112 | 3,250 | 24,748 | 98.992 | 9.7677 | 2.8002 | 82.002 | 44.682 | **83.77** |
| no refresh | 111 | — | 22,526 | 90.104 | 9.6751 | 5.1666 | 78.928 | 43.841 | **79.05** |

Adjacent-QP interpolation between QP 112 (98.992 kbps) and QP 111
(101.480 kbps) gives **K 83.93 at 100.0 kbps**. The direct allocation points
span 83.31-84.03 between 99.34 and 100.40 kbps, so the iso-rate curve is
rippled by about ±0.4 K and the result is **at the 83.80 wall**, not a new
best. Against the 82.15 nightly pin it is **+1.78 K**; against the tuned PSNR
trigger at the same QP it is **+0.42 K** at 101.48 kbps. The perceptual RD core
remains the internal best at 85.10.

The no-refresh control is the mechanism check: the planner buys **+5.12 K**
over no refresh (79.05 -> 84.17) on the tuned point, and lowers the mean lpl36
error from 13.128 to 12.425 (5.4%).

## Results at 720p

`testdata/space8k_1440x720_60f.y4m`, 60 frames at 60 fps, pin recipe
(QP 217, key -57, leaf lambda 250, ext partitions). The lpl36 gate replaces the
drift trigger; the default 1,200 B allowance is unchanged.

| Stream | QP | Bytes | kbps | JOD | BA | SS2 | PSNR-Y | K |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pin | 217 | 36,572 | 292.576 | 7.3236 | 28.8713 | -10.146 | 31.43 | **45.61** |
| lpl36 refresh | 217 | 36,230 | 289.840 | 7.3243 | 29.3809 | -10.367 | 31.430 | **45.54** |
| lpl36 refresh | 216 | 37,429 | 299.432 | 7.3766 | 30.7635 | -8.304 | 31.533 | **46.05** |

Linear interpolation to the pin's 292.576 kbps gives **K 45.69**, +0.08 over
the 45.61 pin. The 720p K-vs-QP curve is documented as rippled by ±1.9 K, so
this is flat: the refresh allocation is not a 720p lever. The lpl36 arm
refreshes 8 frames (4-11) and charges 1,385 B; the pin's PSNR trigger refreshes
13 frames and charges 1,459 B. At 720p the planner's late-detail ranking is
strong but the chronological emit check spends the small allowance on the
early refreshes first, so the planned late anchors are dropped. That is the
same allocation bias the earlier refresh pipeline had; it is not re-tuned here
because the 720p point is JOD-dominated and the measured result is flat.

## Which frames refresh

QP 111, 60 frames, 3,250 B, applied frames (19):

```
4, 6, 18, 23, 26, 28, 31, 33, 35, 37, 39, 41, 43, 45, 46, 48, 49, 51, 53
```

The tuned PSNR trigger refreshes 28 frames (6-9, 16, 18, 20, 21, 23-42) and
never fires after frame 42. The lpl36 objective is an absolute error, so it
prefers the high-detail late band even more strongly than the previous metric:
the 44-59 mean lpl36 error is 14.906 for the lpl36 arm against 15.691 for the
PSNR trigger (-5.0%), while the PSNR arm keeps more of the 9-17 band. The
external metrics see that trade: JOD +0.011, BA -0.097, SS2 +0.875 at
101.48 kbps, which is the +0.42 K at that rate. At the matched 100 kbps
allocation the gain over the PSNR trigger is within the curve ripple.

## Encode cost

M4, 8 CPU threads, 60 frames, single run, no trace:

| Point | Arm | Wall | User CPU | Passes |
|---|---|---:|---:|---:|
| 240p QP111 | lpl36 refresh | 16.91 s | 29.86 s | 4 |
| 240p QP111 | PSNR trigger | 4.05 s | 7.27 s | 1 |
| 720p QP217 | lpl36 refresh | 113.89 s | 384.64 s | 4 |
| 720p QP217 | pin | 27.68 s | 95.83 s | 1 |

The pipeline costs **4.1x** the one-pass encode at both points, almost entirely
the three extra encodes (measure, probe, refinement). The metric itself is
negligible: `lpl36_dump` scores 60 frames in 0.02 s at 240p (0.33 ms/frame) and
0.13 s at 720p (2.2 ms/frame).

## Verification

* Gate off and `AV2ENC_PIPE_LPL_REFRESH=0` keep the pins: default QP189
  `a1d0dba1259b20a4b628d4cd1e14c33c`, quality QP190
  `3984677b9b665952984b07b7cf75c6ea`, nightly QP110
  `1b99127a4cd1d386fa5600fb63c57a83`, nightly QP106 `0639b436...`, 720p QP217
  `c4ba28138d8f6569d5e08721a66d755c`.
* 240p lpl36 stream: 25,370 B, `dav2d-macos --strict 1` decodes 60/60 and the
  `--video-recon` payload equals the decoded payload byte for byte.
* `tools/test.sh --build build-release`: **10/10**.
* `gate_video_nightly`, `gate_video_qp_lambda`, `gate_video_lpl_refresh`,
  `gate_video_lpl_deep` and `gate_lpl36_parity` pass.
* `gate_lpl36_parity` compares the C++ port and the Python package on the same
  Y4M luma planes: max relative error 2.7e-11; identical planes are 0.0 in
  both. Point `LPL36_SRC` at the Python package's `src` directory; without it
  the gate skips (exit 77).
* The 240p refresh gate also requires the lpl36 mean to beat the no-refresh
  control by >=3% and the 44-59 mean to beat the PSNR trigger by >=2%; the
  measured margins are 5.4% and 5.0%.

## Deep anchors

`AV2ENC_PIPE_LPL_DEEP=1` is retained. It runs the same lpl36 planner with every
scheduled refresh dropped to the key picture's quantizer depth (default the
measured key offset, -27 QP at the 84/111 recipe) using the existing
`--video-drift-refresh-qp` plumbing. `gate_video_lpl_deep` checks the off pins,
that `AV2ENC_PIPE_LPL_DEEP=0` is byte-identical to the shallow lpl36 stream, the
key-depth anchors at QP 84, strict decode/recon, and the CDF slice-cross case
(anchor QP 71 against a static 111 plan). The deep lever was already negative
at 100 kbps in the earlier pipeline's measurement (the same placement and carry
model; only the metric changed), so this document does not quote a deep K point
and the arm is not tuned.

## Named scope and caveats

* One 240p clip and one 720p clip, CPU-only, single tile, 8-bit 4:2:0, flat
  key+P chain. The gate requires the existing `--video-drift-refresh` chain;
  without it the env gate is inert.
* lpl36 is an absolute luma-difference objective. It ranks the late detail band
  above the mid-band trough, which is why the lpl36 mean improves more than the
  external K at the tuned rate. This is a metric choice, not a tuning defect.
* The 240p K result is at the wall, not above it; the perceptual RD core
  (85.10) remains the internal best. The 720p result is flat.
* The metric port is deterministic and allocation-free per frame, but the
  filters are scalar float32 and the pooling float64; the parity gate is the
  contract, not bit-exact OpenCV equivalence.
