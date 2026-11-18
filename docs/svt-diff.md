# SVT-AV1 preset 0 vs the BA-refresh stream: a per-frame differential

Status: **diagnostic only.** No encoder source was changed. The two streams are
the README SVT-AV1 row and the (now removed) `AV2ENC_PIPE_BA_REFRESH=1` recipe, both
at 240p / 100 kbps on `testdata/nasa_480x240_30fps_60f.y4m` (60 frames). The
goal is to locate the 4.02 K deficit, not to tune a parameter.

## Headline

| | ours (BA refresh) | SVT-AV1 p0 CRF 36.5 | gap |
|---|---:|---:|---:|
| IVF bytes / kbps | 24,975 / 99.900 | 25,064 / 100.256 | −89 B |
| CVVDP JOD | 9.7927 | 9.8304 | −0.0377 |
| Butteraugli (mean) | 2.4768 | 1.5117 | +0.9651 |
| SSIMULACRA2 (mean) | 82.903 | 88.301 | −5.398 |
| PSNR-Y (MSE-weighted) | 44.922 | 48.800 | −3.878 dB |
| PSNR-Y (mean of frames) | 45.103 | 48.978 | −3.875 dB |
| **K** | **84.81** | **88.83** | **−4.02** |

K decomposes additively at these values into **JOD +0.19, Butteraugli +2.93,
SSIMULACRA2 +0.90** (the residual 0.02 is the cross-term). Butteraugli owns 73%
of the gap, SS2 22%, CVVDP 5%. Anything that does not move Butteraugli cannot
close more than ~1.1 K.

## What was encoded and decoded

```sh
# SVT-AV1, the README row (100.256 kbps; the app's own "Byte Count" excludes
# the 32-byte IVF header and 12-byte frame headers)
/opt/homebrew/bin/SvtAv1EncApp \
  -i testdata/nasa_480x240_30fps_60f.y4m -b out/svt.ivf \
  --preset 0 --crf 36.5 --fps-num 30 --fps-denom 1 --progress 0
ffmpeg -v error -i out/svt.ivf -pix_fmt yuv420p out/svt.y4m

# ours, the BA-refresh recipe; `build/av2enc` is the current main build
AV2ENC_PIPE_BA_REFRESH=1 build/av2enc \
  testdata/nasa_480x240_30fps_60f.y4m -o out/ba.ivf \
  --video --frames 60 --cpu-threads 8 --qp 111 --nightly \
  --video-drift-refresh-threshold 1.5 --video-drift-refresh-budget 3250
tools/bin/dav2d-macos -i out/ba.ivf -o out/ba.y4m --muxer yuv4mpeg2 -q
```

The AV2 stream cannot be decoded by `ffmpeg`; `dav2d-macos --strict 1` decodes
60/60 and the existing gate already pins the decoded payload to `--video-recon`.
The AV1 stream was decoded by `ffmpeg` (libdav1d 1.5.3), not `dav2d`.

Metrics: `tools/extra_metrics.py` and `tools/quality.py` on the decoded Y4Ms
(CVVDP 0.5.7, `standard_fhd`, MPS). Per-frame bytes, PSNR, Butteraugli and SS2
and the decoded-domain decomposition come from `bench/svt_diff/` scripts added
with this doc. The SVT `--enable-stat-report` file was also parsed: it gives
per-picture PSNR/SSIM only, and its per-frame MSE matches the independent
measurement to the printed 0.01 MSE / 0.01 dB; there is no
mode/partition/skip report in SVT-AV1, so
the block-level comparison for SVT is inferred from payload size and the
decoded domain, as the task allows.

## Per-frame table

`srcΔ` is the source's own frame-to-frame luma MSE (how much new signal the
frame contains). `SVT qidx` is the AV1 `base_q_idx` from the frame header
(trace_headers); `SVT TL` is SVT's temporal layer from its stat report. `ΔdB`
and `ΔSS2` are **SVT − ours** (positive = SVT better); `ΔBA` is **ours − SVT**
(positive = ours worse). `R` marks one of our 26 BA refreshes.

| f | srcΔ | ours B | SVT B | SVT qidx | SVT TL | ours dB | SVT dB | ΔdB | ours BA | SVT BA | ΔBA | ours SS2 | SVT SS2 | ΔSS2 | R |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0 | 0.00 | 18365 | 17490 | 25 | 0 | 47.62 | 50.15 | +2.53 | 1.43 | 1.96 | -0.53 | 87.90 | 90.29 | +2.39 |  |
| 1 | 0.00 | 25 | 23 | 146 | 5 | 47.62 | 50.27 | +2.65 | 1.43 | 1.96 | -0.53 | 87.90 | 90.42 | +2.51 |  |
| 2 | 0.07 | 20 | 32 | 116 | 4 | 47.48 | 50.30 | +2.82 | 1.38 | 1.68 | -0.30 | 88.19 | 90.49 | +2.29 |  |
| 3 | 0.08 | 35 | 25 | 146 | 5 | 47.44 | 50.19 | +2.75 | 1.76 | 1.50 | +0.26 | 88.11 | 90.53 | +2.42 | R |
| 4 | 0.36 | 19 | 28 | 137 | 3 | 46.59 | 49.97 | +3.38 | 1.42 | 1.44 | -0.02 | 87.72 | 90.29 | +2.57 |  |
| 5 | 0.31 | 35 | 24 | 146 | 5 | 45.59 | 50.18 | +4.58 | 2.46 | 1.35 | +1.10 | 86.98 | 90.38 | +3.40 |  |
| 6 | 0.00 | 17 | 35 | 117 | 4 | 45.59 | 50.18 | +4.59 | 2.46 | 1.35 | +1.10 | 86.99 | 90.38 | +3.40 |  |
| 7 | 0.28 | 35 | 24 | 146 | 5 | 46.05 | 48.83 | +2.77 | 1.70 | 1.71 | -0.00 | 87.30 | 89.91 | +2.61 |  |
| 8 | 0.25 | 18 | 37 | 88 | 2 | 45.52 | 50.32 | +4.80 | 1.88 | 1.14 | +0.74 | 87.04 | 90.49 | +3.45 |  |
| 9 | 0.33 | 27 | 24 | 146 | 5 | 46.30 | 49.94 | +3.64 | 1.76 | 1.15 | +0.60 | 87.58 | 90.38 | +2.80 |  |
| 10 | 0.07 | 17 | 26 | 107 | 4 | 46.32 | 50.13 | +3.81 | 2.07 | 1.18 | +0.89 | 87.64 | 90.51 | +2.86 |  |
| 11 | 0.00 | 35 | 21 | 146 | 5 | 46.35 | 50.13 | +3.78 | 1.63 | 1.19 | +0.44 | 87.59 | 90.48 | +2.89 | R |
| 12 | 0.08 | 16 | 35 | 84 | 3 | 46.34 | 50.22 | +3.88 | 2.68 | 1.20 | +1.48 | 87.60 | 90.54 | +2.94 |  |
| 13 | 0.12 | 58 | 21 | 146 | 5 | 46.48 | 50.09 | +3.61 | 1.66 | 1.21 | +0.45 | 87.66 | 90.50 | +2.84 | R |
| 14 | 0.08 | 39 | 22 | 103 | 4 | 46.39 | 50.20 | +3.81 | 2.04 | 1.23 | +0.81 | 87.48 | 90.63 | +3.15 | R |
| 15 | 0.08 | 38 | 21 | 146 | 5 | 46.34 | 50.16 | +3.82 | 1.70 | 1.27 | +0.42 | 87.40 | 90.67 | +3.26 | R |
| 16 | 0.00 | 25 | 168 | 44 | 1 | 46.35 | 50.19 | +3.84 | 1.70 | 1.27 | +0.42 | 87.41 | 90.71 | +3.31 | R |
| 17 | 0.07 | 16 | 21 | 146 | 5 | 46.22 | 50.00 | +3.78 | 1.60 | 1.10 | +0.50 | 87.34 | 90.52 | +3.18 |  |
| 18 | 0.17 | 37 | 26 | 120 | 4 | 46.31 | 49.97 | +3.66 | 2.04 | 1.16 | +0.89 | 87.34 | 90.50 | +3.16 | R |
| 19 | 0.26 | 26 | 22 | 146 | 5 | 45.54 | 49.53 | +3.99 | 1.87 | 1.52 | +0.35 | 86.84 | 90.19 | +3.35 |  |
| 20 | 0.74 | 35 | 55 | 98 | 3 | 46.20 | 50.17 | +3.97 | 2.37 | 1.07 | +1.30 | 87.08 | 90.43 | +3.35 |  |
| 21 | 0.00 | 57 | 22 | 146 | 5 | 46.26 | 50.16 | +3.89 | 1.54 | 1.07 | +0.46 | 87.20 | 90.39 | +3.19 | R |
| 22 | 0.09 | 24 | 24 | 138 | 4 | 46.17 | 50.04 | +3.87 | 1.81 | 1.10 | +0.71 | 87.15 | 90.43 | +3.27 |  |
| 23 | 0.14 | 34 | 21 | 146 | 5 | 46.23 | 50.13 | +3.90 | 1.82 | 1.07 | +0.75 | 87.22 | 90.43 | +3.21 | R |
| 24 | 0.40 | 25 | 63 | 72 | 2 | 45.32 | 49.95 | +4.62 | 2.41 | 1.21 | +1.20 | 86.64 | 90.12 | +3.48 |  |
| 25 | 0.10 | 160 | 21 | 146 | 5 | 45.40 | 49.71 | +4.31 | 2.56 | 1.37 | +1.19 | 86.55 | 90.00 | +3.45 | R |
| 26 | 0.00 | 79 | 22 | 112 | 4 | 45.52 | 49.70 | +4.18 | 2.56 | 1.37 | +1.19 | 86.68 | 89.96 | +3.28 | R |
| 27 | 0.08 | 51 | 20 | 146 | 5 | 45.53 | 49.90 | +4.37 | 1.99 | 1.32 | +0.67 | 86.68 | 89.97 | +3.29 | R |
| 28 | 0.11 | 39 | 30 | 119 | 3 | 45.60 | 49.95 | +4.35 | 1.83 | 1.22 | +0.61 | 86.73 | 90.16 | +3.44 | R |
| 29 | 0.15 | 97 | 20 | 146 | 5 | 45.28 | 49.79 | +4.51 | 2.88 | 1.27 | +1.61 | 86.28 | 90.07 | +3.79 | R |
| 30 | 0.17 | 38 | 26 | 117 | 4 | 45.18 | 49.20 | +4.02 | 1.99 | 1.49 | +0.50 | 86.37 | 89.82 | +3.45 |  |
| 31 | 0.00 | 112 | 23 | 146 | 5 | 45.43 | 49.23 | +3.81 | 1.99 | 1.49 | +0.50 | 86.43 | 89.82 | +3.39 | R |
| 32 | 1.93 | 61 | 1621 | 34 | 0 | 43.45 | 45.35 | +1.90 | 2.49 | 1.49 | +1.01 | 84.36 | 87.78 | +3.42 |  |
| 33 | 3.22 | 254 | 20 | 146 | 4 | 44.15 | 47.07 | +2.92 | 2.36 | 1.37 | +0.99 | 85.46 | 88.21 | +2.74 | R |
| 34 | 2.80 | 68 | 32 | 140 | 3 | 43.31 | 47.29 | +3.98 | 2.81 | 1.45 | +1.36 | 84.18 | 88.63 | +4.46 |  |
| 35 | 2.65 | 88 | 24 | 146 | 4 | 42.79 | 47.96 | +5.18 | 2.88 | 1.32 | +1.57 | 81.39 | 88.98 | +7.59 |  |
| 36 | 0.00 | 219 | 125 | 96 | 2 | 43.80 | 47.98 | +4.17 | 2.38 | 1.32 | +1.06 | 83.57 | 88.99 | +5.43 | R |
| 37 | 3.42 | 283 | 20 | 146 | 4 | 43.80 | 48.17 | +4.37 | 2.46 | 1.23 | +1.23 | 83.03 | 88.53 | +5.50 | R |
| 38 | 3.34 | 48 | 38 | 128 | 3 | 43.31 | 47.88 | +4.57 | 3.48 | 1.31 | +2.17 | 81.42 | 88.23 | +6.81 |  |
| 39 | 2.76 | 369 | 21 | 146 | 4 | 43.85 | 47.84 | +3.99 | 2.71 | 1.55 | +1.16 | 82.15 | 88.13 | +5.98 | R |
| 40 | 3.26 | 57 | 160 | 77 | 1 | 43.29 | 47.46 | +4.18 | 3.12 | 1.54 | +1.58 | 79.90 | 88.12 | +8.22 |  |
| 41 | 0.00 | 209 | 35 | 146 | 4 | 43.96 | 48.06 | +4.11 | 2.67 | 1.59 | +1.08 | 81.90 | 87.40 | +5.50 | R |
| 42 | 3.44 | 125 | 28 | 144 | 3 | 43.59 | 48.34 | +4.75 | 3.27 | 1.49 | +1.79 | 80.12 | 88.21 | +8.10 |  |
| 43 | 3.60 | 395 | 21 | 146 | 4 | 43.97 | 47.95 | +3.99 | 3.01 | 1.45 | +1.55 | 80.21 | 86.52 | +6.30 | R |
| 44 | 2.80 | 40 | 73 | 97 | 2 | 43.22 | 48.07 | +4.85 | 3.55 | 1.62 | +1.93 | 78.18 | 86.73 | +8.55 |  |
| 45 | 2.89 | 65 | 25 | 146 | 4 | 43.29 | 47.43 | +4.14 | 3.96 | 1.57 | +2.39 | 76.47 | 85.77 | +9.30 |  |
| 46 | 0.00 | 294 | 44 | 118 | 3 | 44.40 | 47.44 | +3.03 | 2.68 | 1.57 | +1.11 | 79.66 | 86.02 | +6.36 | R |
| 47 | 3.70 | 76 | 24 | 146 | 4 | 43.67 | 47.16 | +3.50 | 3.01 | 1.66 | +1.35 | 76.70 | 85.22 | +8.52 |  |
| 48 | 3.61 | 92 | 1619 | 77 | 0 | 43.42 | 46.61 | +3.19 | 3.30 | 2.29 | +1.01 | 75.25 | 84.23 | +8.98 |  |
| 49 | 2.98 | 377 | 25 | 146 | 3 | 44.54 | 47.67 | +3.14 | 2.75 | 1.87 | +0.88 | 79.00 | 84.39 | +5.38 | R |
| 50 | 2.95 | 81 | 47 | 112 | 2 | 44.02 | 48.20 | +4.19 | 3.11 | 1.73 | +1.38 | 76.76 | 84.75 | +7.99 |  |
| 51 | 0.00 | 167 | 27 | 146 | 3 | 44.75 | 48.32 | +3.57 | 2.96 | 1.65 | +1.31 | 77.83 | 84.79 | +6.97 | R |
| 52 | 3.75 | 292 | 38 | 100 | 1 | 44.62 | 48.90 | +4.28 | 3.12 | 1.58 | +1.54 | 77.17 | 85.96 | +8.80 | R |
| 53 | 3.97 | 114 | 27 | 146 | 3 | 43.79 | 48.94 | +5.15 | 3.31 | 1.80 | +1.51 | 73.22 | 85.34 | +12.13 |  |
| 54 | 3.14 | 399 | 53 | 126 | 2 | 44.72 | 48.62 | +3.90 | 2.97 | 1.96 | +1.01 | 75.91 | 84.63 | +8.73 | R |
| 55 | 3.13 | 108 | 26 | 146 | 3 | 44.42 | 48.30 | +3.88 | 2.96 | 2.34 | +0.61 | 72.99 | 83.46 | +10.47 |  |
| 56 | 0.00 | 43 | 1065 | 74 | 0 | 44.61 | 48.31 | +3.70 | 2.96 | 2.34 | +0.61 | 73.37 | 83.60 | +10.22 |  |
| 57 | 3.99 | 70 | 25 | 146 | 1 | 44.44 | 49.43 | +4.99 | 3.21 | 1.53 | +1.68 | 71.51 | 86.21 | +14.71 |  |
| 58 | 3.84 | 91 | 273 | 110 | 0 | 44.01 | 48.00 | +3.98 | 3.81 | 1.92 | +1.89 | 66.02 | 82.62 | +16.59 |  |
| 59 | 3.28 | 84 | 84 | 146 | 1 | 44.42 | 47.01 | +2.59 | 4.87 | 3.50 | +1.37 | 65.40 | 76.18 | +10.78 |  |

Three things jump out:

* **The early frames are close; the fade section is not.** Frames 0-4 differ by
  −0.26 to 0.53 BA (ours is better at f3) and 2.3-2.6 SS2. Frames 44-59 differ
  by 0.6-2.4 BA and 5-17 SS2. The gap is not uniform; it opens where the source starts its global
  fade at f32 and never closes.
* **Our refreshes do not reset the error.** Frames at distance 0 from a refresh
  have MSE 1.98 on average and distance 1 has 2.30; SVT has 1.00 and 0.84. A
  refresh is not a sawtooth reset in our stream.
* **In the fade, SVT's leaves are better than its anchors.** SVT's f48 anchor
  is 1.42 MSE but f49-f55 sit at 0.83-1.11; f32 is 1.90 but f33-f35 are
  1.04-1.28. Our frames after a refresh generally get worse, not better.

The close set is f0-f4 and f7: |ΔBA| ≤ 0.53 and ΔSS2 ≤ 2.7. The five worst
Butteraugli frames are f45 (+2.39), f38 (+2.17), f44 (+1.93), f58 (+1.89) and
f42 (+1.79); the five worst SS2 frames are f58 (+16.59), f57 (+14.71), f53
(+12.13), f59 (+10.78) and f55 (+10.47). The BA ranking is concentrated in the
fade's mid-band changes; the SS2 ranking is concentrated in the last 12 frames.

Segment summary (K-gap share is the exact-K segment decomposition with JOD
held at ours; the 12.3 + 20.5 + 38.7 + 28.6% sums to that 3.44 K partial,
leaving the JOD and cross terms unattributed):

| segment | | bytes | BA | SS2 | PSNR-Y | K-gap share |
|---|---|---:|---:|---:|---:|---:|
| 0-15 | ours | 18,799 | 1.842 | 87.568 | 46.50 | 12.3% |
| 0-15 | SVT | 17,888 | 1.409 | 90.429 | 50.08 | |
| 16-31 | ours | 855 | 2.060 | 86.871 | 45.78 | 20.5% |
| 16-31 | SVT | 584 | 1.257 | 90.221 | 49.85 | |
| 32-47 | ours | 2,651 | 2.928 | 81.169 | 43.62 | 38.7% |
| 32-47 | SVT | 2,311 | 1.471 | 87.592 | 47.59 | |
| 48-59 | ours | 1,918 | 3.278 | 73.702 | 44.31 | 28.6% |
| 48-59 | SVT | 3,309 | 2.043 | 83.847 | 48.19 | |

The last two segments carry 67% of the attributable K gap. In 48-59 we
*underspend* by 1.4 KB and lose 10 SS2 points; in 32-47 we overspend slightly
and still lose 1.46 BA.

## Where the bits go

| class | frames | bytes | mean B/frame | share |
|---|---:|---:|---:|---:|
| ours key | 1 | 18,365 | 18,365 | 73.5% |
| ours refresh | 26 | 4,154 | 160 | 16.6% |
| ours copy (non-refresh) | 33 | 1,704 | 52 | 6.8% |
| SVT key | 1 | 17,490 | 17,490 | 69.8% |
| SVT inter anchors (qidx ≤ 96) | 9 | 4,893 | 544 | 19.5% |
| SVT other inter | 50 | 1,709 | 34 | 6.8% |

The anchor budgets are almost identical: **ours key + refreshes = 22,519 B,
SVT key + inter anchors = 22,383 B (0.6% apart).** The difference is
concentration. Our 4.2 KB of inter refresh is spread over 26 frames (160 B
each, largest 399 B); SVT's 4.9 KB is concentrated in 9 anchors (544 B each)
and three of those — f32 1,621 B, f48 1,619 B, f56 1,065 B — carry 4,305 B,
**65% of all SVT inter bytes** against our top three's 1,171 B (20%).

SVT is bimodal: three 1.1-1.6 KB anchors at qidx 34-77, cheaper anchors at
qidx 72-96, then 20-30 B leaves at qidx 137-146. Ours is unimodal: every inter frame is a 16-400 B shallow refresh or
copy at qidx 111. The source change is also bimodal in the same place: frames
0-31 are nearly static (source Δ 0-0.7), the last 28 are a global
fade (source Δ 1.9-4.0 per frame).

Bytes outside the tile payload are 1,858 B of our 24,975 (7.4%, ~31 B/frame:
IVF container/frame headers plus the per-frame tile-group header). SVT's
leaf frames are 34 B on average; our non-refresh copy frames are 52 B. About
600-900 B of our inter budget is per-frame signalling that SVT does not pay.

## Quantizer depth

The relevant comparison is the sample-domain luma step, not the QP number,
because AV2 and AV1 number their quantizers differently.

| stream / frame class | QP | luma step (8-bit) | relative to our inter |
|---|---:|---:|---:|
| ours key (QP 84, -27 offset) | 84 | 55 | 0.46× |
| ours inter (all frames) | 111 | 120 | 1.00× |
| SVT key | 25 | 28 | 0.23× |
| SVT inter anchors f32/f16/f56/f48 | 34/44/74/77 | 36/44/69/71 | 0.30-0.59× |
| SVT leaves | 137-146 | 158-182 | 1.32-1.52× |

SVT codes its leaves **coarser** than our inter frames and its anchors
**1.7-3.3× finer**. That is the whole rate-quality trade: it spends the bits
where they persist (anchors) and lets the leaves be almost pure copy; we spend
a middle QP on every frame.

The key is the in-stream proof that the shallow refresh is the problem: it is
coded at the same encoder's QP 84 and reaches **BA 1.43 / SS2 87.90**, while
the 26 inter refreshes at QP 111 average **BA 2.30 / SS2 84.3**. The key is
coarser than SVT's key and still beats SVT's key on Butteraugli (1.43 vs 1.96)
and loses only 2.4 SS2 at +5% bytes. The perceptual quality exists in this
encoder at roughly key depth; the inter refresh path never asks for it.

## Where the error lives (decoded domain)

### Luma vs chroma

| frames | | MSE-Y | MSE-U | MSE-V | Y gap | U gap | V gap |
|---|---|---:|---:|---:|---:|---:|---:|
| 0-15 | ours | 1.47 | 0.69 | 0.60 | | | |
| 0-15 | SVT | 0.64 | 0.34 | 0.29 | 2.30× | 2.01× | 2.07× |
| 16-31 | ours | 1.72 | 0.71 | 0.61 | | | |
| 16-31 | SVT | 0.67 | 0.35 | 0.29 | 2.56× | 2.01× | 2.08× |
| 32-47 | ours | 2.84 | 0.81 | 0.67 | | | |
| 32-47 | SVT | 1.15 | 0.34 | 0.28 | 2.47× | 2.41× | 2.43× |
| 48-59 | ours | 2.42 | 1.07 | 0.92 | | | |
| 48-59 | SVT | 1.00 | 0.25 | 0.20 | 2.41× | **4.31×** | **4.57×** |

Luma is a flat 2.3-2.6× everywhere. Chroma is 2.0-2.4× early and then
**4.3-4.6×** in the last 12 frames. The SS2 gap across frames correlates
**0.92 with the chroma MSE ratio** and only 0.29 with the luma MSE ratio
(tail: 0.88 vs 0.36). Chroma is not the whole SS2 gap (the regression has a
large frame-independent term), but it is the strongest per-frame correlate of
where SS2 collapses.

### Flat vs detail

8×8 source blocks, fixed variance bins, pooled over the segment. The cell is
"share of SSE / MSE":

| frames | | flat <1 | low 1-10 | mid 10-40 | detail 40-160 |
|---|---|---:|---:|---:|---:|
| 0-15 | ours | 1% / 0.03 | 15% / 2.06 | 57% / 3.48 | 27% / 4.01 |
| 0-15 | SVT | 0% / 0.01 | 20% / 1.18 | 61% / 1.61 | 19% / 1.26 |
| 16-31 | ours | 1% / 0.04 | 14% / 2.23 | 58% / 4.07 | 28% / 4.86 |
| 16-31 | SVT | 1% / 0.01 | 18% / 1.17 | 61% / 1.68 | 20% / 1.41 |
| 32-47 | ours | 2% / 0.11 | 15% / 3.34 | 59% / 6.73 | 24% / 10.10 |
| 32-47 | SVT | 1% / 0.02 | 16% / 1.44 | 52% / 2.40 | 31% / 5.31 |
| 48-59 | ours | 5% / 0.22 | 35% / 3.84 | 57% / 6.85 | 3% / 9.24 |
| 48-59 | SVT | 3% / 0.05 | 32% / 1.47 | 59% / 2.93 | 6% / 7.91 |

Flat pixels are 55-57% of the frame and only 1-5% of the error; both encoders
reproduce them essentially exactly. The error lives in the **mid-activity body**
(20-25% of pixels, 57-61% of SSE) and the detail tail. The relative gap is
*worst in the mid band* (2.3-2.8×) and smallest in the sparse high-detail band
in the fade (1.17× at 48-59), so this is not an "edges/detail preservation"
failure; it is a broad quantization floor over the body of the picture.

### Frequency bands

8×8 DCT error energy relative to the source's own band energy, and the
ours/SVT ratio:

| frames | | low (1-2) | mid (3-5) | high (≥6) | DC |
|---|---|---:|---:|---:|---:|
| 0-15 | ours | 0.00102 | 0.00295 | 0.00949 | 0.00001 |
| 0-15 | SVT | 0.00035 | 0.00116 | 0.00437 | 0.00001 |
| 0-15 | ratio | **2.89×** | 2.55× | 2.17× | 1.28× |
| 16-31 | ours | 0.00122 | 0.00349 | 0.01085 | 0.00001 |
| 16-31 | SVT | 0.00038 | 0.00124 | 0.00456 | 0.00001 |
| 16-31 | ratio | **3.17×** | 2.81× | 2.38× | 1.83× |
| 32-47 | ours | 0.00382 | 0.00716 | 0.01874 | 0.00009 |
| 32-47 | SVT | 0.00116 | 0.00263 | 0.00843 | 0.00004 |
| 32-47 | ratio | **3.29×** | 2.72× | 2.22× | 2.03× |
| 48-59 | ours | 0.01088 | 0.01625 | 0.03501 | 0.00020 |
| 48-59 | SVT | 0.00434 | 0.00708 | 0.01566 | 0.00006 |
| 48-59 | ratio | 2.51× | 2.30× | 2.24× | 3.05× |

The gap is broadband, but it is **largest in the low and mid bands** (2.5-3.3×)
and smallest in high frequencies (2.2×). Butteraugli is most sensitive to that
low/mid range, which is exactly where the BA gap is. This is the signature of a
coarse quantizer zeroing low/mid contrast, not of a missing high-frequency
transform. DC error is negligible in absolute terms (0.05-0.41 MSE), so the
global brightness is tracked correctly.

### Butteraugli diffmap

`butteraugli_main --rawdistmap` over all 60 frames:

| frames | | mean diffmap | max | top-1% share |
|---|---|---:|---:|---:|
| 0-15 | ours / SVT | 0.243 / 0.160 | 1.84 / 1.41 | 4.5% / 5.2% |
| 16-31 | ours / SVT | 0.265 / 0.170 | 2.06 / 1.26 | 4.5% / 4.8% |
| 32-47 | ours / SVT | 0.420 / 0.253 | 2.93 / 1.47 | 4.3% / 4.1% |
| 48-59 | ours / SVT | 0.485 / 0.272 | 3.28 / 2.04 | 4.3% / 4.5% |

The top 1% of pixels carries only ~4-5% of the diffmap mass in both streams,
and the spatial correlation is 0.94-0.96 for most frames (0.84-0.93 in the
fade) — the errors are in the same places
and are broadly distributed, not a few bad blocks. That is why BA tracks the
average error level rather than a localized artifact. The one exception is the
key: our key has a lower BA max (1.43 vs 1.96) but a higher diffmap mean
(0.210 vs 0.158), i.e. SVT's key trades a localized low-frequency artifact for
a lower average error; our key is uniformly noisier but has no peak.

## Why the error does not decay: the source is a global fade

The last 28 source frames are not motion; they are an illumination ramp.
Block matching (16×16, ±16 integer) finds zero motion gain on every pair
(0.0-0.8%). The luma mean falls from 58.2 at f31 to 37.8 at f59. A per-frame
global affine fit `src_t ≈ a·src_{t-1} + b` explains almost all of the change:
a = 0.968/0.964/0.945/0.898 and b = +0.9/+1.1/+1.6/+3.1 at f33/f37/f48/f58,
leaving a residual of 0.08-0.48 MSE. Both decoders track the ramp: our recon's
own affine relation matches the source's gain to ≤0.013 and its offset to
≤0.4 level. The
change itself is therefore *not* the deficit; the residual quantization error
is. A frame that perfects the affine prediction still inherits the previous
frame's error scaled by `a`:

`recon_t − src_t ≈ a·(recon_{t−1} − src_{t−1})`.

That is why the error floor, not the source change, is the thing to attack.

**Repair fraction.** Treat "copy the previous recon" as the zero-bit baseline.
The fraction of that baseline error removed by the coded frame:

| frames | ours | SVT |
|---|---:|---:|
| 1-15 | 5.3% | 8.9% |
| 16-31 | 7.4% | 8.0% |
| 32-47 | 46.4% | 68.0% |
| 48-59 | 53.6% | 75.0% |
| all 1-59 | 38.0% | 60.3% |

In the static first half there is almost nothing to repair (the source change
is below the quantization noise). In the fade, SVT removes ~3/4 of the
would-be drift; we remove ~1/2. But the
absolute error is the product of the baseline and the repair, and our baseline
is higher because the previous recon is noisier. The error-vs-distance table
shows the refresh *frequency* is not the lever:

| frames since our refresh | ours MSE | n | frames since SVT anchor | SVT MSE | n |
|---:|---:|---:|---:|---:|---:|
| 0 | 1.98 | 26 | 0 | 1.00 | 9 |
| 1 | 2.30 | 18 | 1 | 0.84 | 10 |
| 2 | 2.31 | 7 | 2 | 0.85 | 10 |
| 3 | 2.07 | 2 | 3 | 0.87 | 10 |
| 4-5 | 2.09-2.10 | 4 | 4-5 | 0.76-0.79 | 10 |
| 6-7 | 1.52-1.53 | 2 | 6-7 | 0.82-0.89 | 10 |

Our error is flat at 2.0-2.3 whether the frame is a refresh or five frames
past one. SVT's is flat at 0.8-1.0 whether it is an anchor or seven frames past
one. Adding more shallow refreshes cannot lower a flat floor; only a deeper
anchor or a better predictor can.

## Skip/residual balance (our block dump)

`AV2ENC_BLOCKDUMP` on the same encode (last of the pipeline's passes, 12,437
leaf records). The picture is unambiguous:

| class | leaves | skip=1 | leaves with any EOB | nonzero levels |
|---|---:|---:|---:|---:|
| 33 non-refresh frames | 1,631 | 1,560 (95.6%) | 71 (all in f32-f59) | small |
| 26 refresh frames | 1,212 | 0 (forced residual) | 11-37 per frame | 12-2,084 per frame |

In the static first half (f1-f31) every non-refresh leaf is a **pure skip**:
`skip_flag=1`, zero coefficients in all three planes. The entire inter quality
is frozen at the last refresh. In the fade (f32-f59) the encoder's own RD lets
1-9 leaves per non-refresh frame code a residual (71 leaves total); that is the
encoder tracking the fade's DC/low-frequency, and it is the only non-forced
residual in the inter chain. On a refresh frame,
`force_inter_residual` makes every leaf code residual, but at QP 111 the
quantizer still zeroes almost everything: the early refreshes have as few as
12 nonzero luma levels across a whole 480×240 frame (f16), several are under
100, and 14 of the 26 refresh frames code **zero chroma coefficients**. Even
the biggest refresh (f54, 399 B) has 35 of 53 leaves with any EOB.

The key is the contrast: 256 leaves (222 of them 16×16), 549 of 768
plane-leaves with EOB, and BA 1.43. The intra path is willing to code; the
inter refresh path at QP 111 is not, even when forced, because the quantizer
has nothing left to code.

This also explains the "close" frames: f3's 35 B refresh changes BA from 1.38
to 1.76 (worse); f14's 39 B changes 1.66 to 2.04; f29's 97 B changes 1.83 to
2.88. Four of the 26 refreshes increase luma MSE. At QP 111 a shallow refresh
can inject noise as easily as remove it.

## SVT's structure (OBU trace)

`ffmpeg -bsf:v trace_headers` on the SVT IVF, re-ordered by AV1 `order_hint`:

* 1 key (qidx 25), 30 coded inter pictures, 29 `show_existing_frame` units.
* 10 anchors at qidx ≤ 96: f0 25, f8 88, f12 84, f16 44, f24 72, f32 34,
  f36 96, f40 77, f48 77, f56 74. The three big ones are f32 (1,621 B),
  f48 (1,619 B), f56 (1,065 B).
* 34 leaves at qidx 137-146, 20-30 B; 16 mid frames at qidx 97-136, 840 B
  total.
* `reference_select = 1`, `enable_jnt_comp = 1`, `skip_mode_present = 1`;
  the sequence enables two-reference compound prediction and skip mode. The leaves'
  measured error stays at 0.84-0.89 MSE out to seven frames from an anchor,
  consistent with a distance-weighted two-sided prediction of a linear fade;
  a single past reference at the same distance would inherit the fade.
* Filters are not the differentiator: CDEF has a nonzero strength on 6/60
  pictures, loop restoration on 8/60, loop filter on 17/60. SVT switches most filters
  off at this rate. Our CDEF is on for 60/60 but is a no-op on 33 of them.

## Ranked mechanisms by expected K recovery

The estimates below are measured deltas where possible, and ranges where the
mechanism has to be composed with the existing pipeline. They overlap: #1-#3
all attack the same propagated floor, so they do not add to 4.02.

| rank | mechanism | evidence | expected K | blocker / cost |
|---:|---|---|---:|---|
| **1** | **Deep inter anchors**: give the refresh the key's existing quantizer depth (-27 offset, QP 84), and let the BA planner price that floor instead of the QP 111 floor. | The key at QP 84 is BA 1.43/SS2 87.9; the 26 refreshes at QP 111 are BA 2.30/SS2 84.3; SVT's 9 anchors are BA 1.53/SS2 88.3. Replacing all inter frames with the key's measured floor is +4.07 K (upper bound, no rate); replacing only the refresh frames is +1.27 K; the rest arrives through the copy frames. | **2.0-3.0** | The CDF chain rejects `--video-drift-refresh-qp != 0` (`validate.cpp`), because a runtime `base_q_idx` change desyncs the §9.3 carried image. The key's offset already proves the quality point. Fix is a per-anchor §9.3 CDF re-seed (the existing `--video-cdf-per-frame-qp` reset path is the control), not new syntax. |
| **2** | **Two-sided / distance-weighted prediction** between anchors (a future reference, or a past+future pair), so the fade is interpolated instead of propagated. | Source is a global affine fade with no motion; SVT's leaf error is flat 0.84-0.89 at past-anchor distance 1-7, below its hard-section anchors (1.04-1.90) on many frames. On f37/f43/f44/f52 the simple time-weighted interpolation of the bracketing anchors is within 0.1 MSE of SVT's actual frame; f35/f41 still need residual. Our chain is single-reference: all 2,843 inter leaves use `ref=0`. | **0.5-1.0** incremental | The nightly flat key+P chain rejects B/reorder/temporal layers; the current closed-GOP B path is not a fair test (it inserts 10 keys and falls to K 36.9). Needs a hidden-ARF pyramid or a two-reference P path, not the existing `--b-frames` schedule. |
| **3** | **Chroma depth and coverage at the anchors**: the same QP-depth fix applied to chroma, or a chroma-only delta on reference frames. | Tail chroma MSE is 4.3-4.6× SVT while luma is 2.4×; 14/26 refreshes code zero chroma EOB; SVT chroma PSNR is 52-56 dB in the tail vs ours 44-48. SS2 gap correlates 0.92 with the chroma MSE ratio. | **0.3-0.6** (subset of #1) | Same §9.3 qctx/CDF issue for a chroma delta. `--video-uv-delta-q` exists but is unused; it is a mechanism arm, not a sweep. |
| **4** | **Intra/keyframe RD**: close the 2.5 dB / 2.4 SS2 key gap at equal bytes. | Key: 18,365 B / 47.62 dB / BA 1.43 / SS2 87.90 vs SVT 17,490 B / 50.15 dB / BA 1.96 / SS2 90.29. Error is uniformly ~2× in every DCT band, consistent with a ~2× coarser step at equal rate. | **0.1-0.3** | Direct f0 K contribution is ~0 (our BA is better and SS2 only +0.026); it matters only if the key becomes a long-lived anchor. Intra partition/prediction/TX work. |
| **5** | **Per-frame overhead**: compact the 31 B/frame header and the 52 B copy frames. | 1,858 B of 24,975 is header (7.4%); 33 non-refresh frames cost 1,704 B (52 B each) vs SVT's 34 B leaf mean. | **0.1-0.2** | 600-900 B recovered is ~2-4% of the budget; it can fund a deeper anchor but does not change the floor. |
| **6** | **Per-frame residual at the current QP, filters, or re-allocation within the current mechanism.** | All-refresh QP 110 probe floor is BA 2.432 vs the measured 2.477; the allocation ceiling is +0.12 K. SVT uses CDEF on 6/60 and LR on 8/60. | **<0.2 combined** | This is the negative result: "code residual on every P frame" is not the missing lever. Our all-refresh floor is the QP-111 quantization floor, and SVT's leaves are also skip-dominated. |

The mechanism answer to "prediction, TX, quantisation, or refresh" is therefore:
**quantisation depth of the reference frames (refresh) first, prediction
structure second, chroma depth third. TX and per-frame residual coding are not
the binding constraint.** The key is the control that proves it: the same transform and entropy stack
reaches BA 1.43 on the key at QP 84, while the inter refresh at QP 111 sits at
2.30.

## What this does and does not say

* It locates the gap in the **anchor/refresh mechanism**, not in the BA
  allocation. The current BA planner is already within 0.12 K of its own
  ceiling; making the refresh deeper changes the floor it is optimising over.
* It does not propose a QP, CRF or budget sweep. The measured points are the
  two existing streams, the key's existing offset, and SVT's published
  structure. The change is structural: a QP-changing anchor with a coherent
  CDF re-seed, and a two-sided predictor if the fade section is to stay flat
  between anchors.
* Matching SVT reaches ~88.8 K, not 90. The remaining ~1.2 K to the target is
  outside this differential: SVT itself is at 88.83 on this clip, so >90
  requires beating SVT's anchor/leaf trade, not only copying it.

## Reproduce

The analysis helpers are in `bench/svt_diff/` (read-only, no encoder changes):

```sh
python3 bench/svt_diff/stream_bits.py \
  --ivf out/svt.ivf --trace out/svt_trace.txt --json out/svt_bits.json
python3 bench/svt_diff/perframe_metrics.py \
  --ref testdata/nasa_480x240_30fps_60f.y4m --test out/svt.y4m \
  --ivf out/svt.ivf --bits-json out/svt_bits.json --json out/svt_perframe.json
python3 bench/svt_diff/perframe_metrics.py \
  --ref testdata/nasa_480x240_30fps_60f.y4m --test out/ba.y4m \
  --ivf out/ba.ivf --json out/ba_perframe.json
python3 bench/svt_diff/domain_metrics.py \
  --ref testdata/nasa_480x240_30fps_60f.y4m --ours out/ba.y4m \
  --svt out/svt.y4m --json out/domain.json
python3 bench/svt_diff/band_table.py \
  --ref testdata/nasa_480x240_30fps_60f.y4m --ours out/ba.y4m \
  --svt out/svt.y4m --json out/band_table.json
python3 bench/svt_diff/ba_maps.py \
  --ref testdata/nasa_480x240_30fps_60f.y4m --ours out/ba.y4m \
  --svt out/svt.y4m --json out/ba_maps.json
```

For the block dump, re-run the BA encode with
`AV2ENC_BLOCKDUMP=out/ba_blocks.txt`; the last 60 groups of the dump are the
emitted pass. The SVT trace is
`ffmpeg -v verbose -i out/svt.ivf -c copy -bsf:v trace_headers -f null -`.

## Caveats

* One clip, 240p, 8-bit 4:2:0, single tile, 30 fps. The source's second half is
  a global fade, which is unusually friendly to distance-weighted two-sided
  prediction; the anchor-depth result does not depend on that, but the size of
  mechanism #2 does.
* The AV2 stream is decoded with `dav2d-macos`, not `ffmpeg` (ffmpeg has no AV2
  decoder). The AV1 stream is decoded with `ffmpeg`/libdav1d.
* SVT-AV1's stat report has no mode/partition/skip statistics. The leaf/EOB
  balance for SVT is inferred from 20-30 B payloads and the decoded domain; the
  block-level skip counts are exact only for our stream.
* The K recovery ranges are counterfactual and overlap. The only exact
  counterfactual is "all inter frames at the key's measured BA/SS2" (+4.07 K,
  no rate), which is an upper bound, not a schedule.
* The encoder was not modified. The `--video-drift-refresh-qp` path is blocked
  by the CDF composition check, so the deep-anchor arm could not be measured
  directly without source changes; the key and SVT's anchors are the measured
  evidence for its floor.
