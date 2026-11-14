# Rate-aware video partition tree (`--video-partition-rd`)

`--video-partition-rd` replaces the CPU `--video` partition decision cost with a
single-domain RD cost. It is an explicit CPU opt-in on top of
`--video-rd-preset quality`; it is not in `--nightly` because it measured worse
when stacked on the bottom-edge path. The flag-less default keeps the
historical tree and every pinned stream byte-identical.

## The defect

The historical `InterFrameEncoder::decide()` cost mixed two units:

```cpp
// leaf distortion: source variance over the block (SSE units)
costs.leaf = block_variance_cost_at(info);

// signal cost: SAD-domain lambda
lambda      = 0.85 * 2^((base_q_idx - 12) / 6);
signal_cost = 2.0 * lambda;
```

At QP 190 the signal cost is `1.44e9`, while the largest possible variance
reduction of a 64x64 8-bit block is `4096 * 127.5^2 = 6.66e7` (half the samples
0, half 255). Every optional split is therefore unaffordable above
`q ≈ 163`. The P-frame tree collapses to one 64x64 leaf per superblock (only
geometry-forced edge splits survive), so a frame gets ~30 motion vectors, one
per 64x64 area. The measured consequence at 480x240/100 kbps is severe:
PSNR-Y 27.58 dB for the measured preset, with the partition search contributing
almost nothing.

## The change

Both terms now live in the source-variance (SSE) domain:

```text
leaf  = variance(block) + price * leaf_bits
split = price * 2 bits
```

* `price = 700` SSE per estimated syntax bit.
* `leaf_bits = 10` for an inter leaf (skip + `is_inter` + single mode + MVD
  shell) and `6` for an intra leaf (YMode + angle).
* The partition symbol is `2` bits.

A binary split adds one leaf and one partition symbol, so it costs
`700 * (10 + 2) = 8,400` SSE for an inter leaf and `700 * (6 + 2) = 5,600` for
an intra leaf. `price` is calibrated from the measured local slope of this
clip's RD curve at the ~100 kbps operating point:

| Point | local slope |
|---|---:|
| default preset near QP 190 | ≈ 4,000 SSE/bit |
| new tree near QP 168 | ≈ 6,300 SSE/bit |

The leaf distortion deliberately remains the source variance, which is
QP-independent. That is why the price is a calibrated constant rather than the
quantiser Lagrangian: the Lagrangian applies to quantised coefficient
distortion, while the variance proxy measures the prediction energy a larger
block cannot represent. Spreading the measured 4,000–6,300 SSE/bit slope over
the 12 syntax bits a split adds gives the 700 price above; the resulting 8,400
SSE split penalty sits in the measured plateau.

The tree still uses the existing legal-partition rules, `min_block_size`
(default 16 on this surface), `max_partition_depth`, the >64x64
`kInfiniteCost` rule, and the existing deterministic solver. The change is
only the cost of a leaf and of a partition symbol, so it is a decision-layer
model change, not a new tool or a flag combination.

## Measured RD

Clip: `testdata/nasa_480x240_30fps_60f.y4m`, 60 frames, 30 fps, 480x240,
CPU `--video`, `--cpu-threads 8`, `AV2ENC_CPU_FAST=ON`, Apple M4. Quality:
`tools/quality.py` (CVVDP 0.5.7, `standard_fhd`, MPS) against the pinned
`dav2d-macos`.

The new preset at the same QP:

| QP | old preset kbps | old JOD | new preset kbps | new JOD | Δ rate | Δ JOD |
|---:|---:|---:|---:|---:|---:|---:|
| 189 | 103.70 | 6.3657 | 81.376 | 7.3391 | -21.5% | +0.973 |
| 190 | 100.564 | 6.3021 | 81.920 | 7.3181 | -18.5% | +1.016 |
| 191 | 101.40 | 6.2466 | 82.776 | 7.2597 | -18.4% | +1.013 |
| 192 | 98.42 | 6.1832 | 81.980 | 7.2329 | -16.7% | +1.050 |

At the original ~100.56 kbps operating point the new curve reaches it at
QP 164/165:

| QP | kbps | CVVDP JOD | PSNR-Y |
|---:|---:|---:|---:|
| 164 | 102.132 | 8.3782 | 33.802 |
| 165 | 100.252 | 8.3438 | 33.547 |
| 166 | 99.280 | 8.3337 | 33.463 |
| 167 | 98.572 | 8.2810 | 33.363 |
| 168 | 98.324 | 8.2533 | 33.218 |
| 169 | 96.976 | 8.2026 | 33.036 |
| 170 | 96.812 | 8.2058 | 32.956 |

Iso-rate at 100.564 kbps (linear interpolation between QP 164 and QP 165):

| Stream | Rate | CVVDP | PSNR-Y |
|---|---:|---:|---:|
| old preset, QP 190 | 100.564 kbps | 6.3021 | 27.5816 |
| new preset, interpolated | 100.564 kbps | **8.3495** | **33.589** |
| **delta** | matched | **+2.047 JOD** | **+6.008 dB** |

Nearest measured point (QP 165, 100.252 kbps, -0.31% rate): **+2.042 JOD**,
+5.965 dB.

At iso-CVVDP the new curve is far below the old one. The new stream reaches
6.3021 JOD near QP 208:

| QP | kbps | CVVDP JOD |
|---:|---:|---:|
| 205 | 78.316 | 6.4597 |
| 208 | 75.108 | 6.3098 |
| 210 | 75.080 | 6.2725 |

Interpolated at CVVDP 6.3021: **75.1 kbps vs 100.564 kbps (-25.3% rate)**.

## Encode time

Host-relative, a single run, 60 frames:

| Stream | Wall time |
|---|---:|
| old preset, QP 190 | 10.7 s |
| new preset, QP 165 | 11.9 s |
| new preset, QP 190 | 9.2 s |

The finer tree has more leaves but each leaf is smaller, so the ME and TX
search work per leaf shrinks by about the same factor. The change is ~1.1x at
the 100 kbps point, not a multiple.

## Verification

The stream is decoded and compared sample for sample with the encoder
reconstruction:

```sh
DAV2D=/Users/arin/work/av2enc/tools/bin/dav2d-macos
build-algo/av2enc testdata/nasa_480x240_30fps_60f.y4m -o /tmp/p.ivf \
    --video --frames 60 --qp 165 --cpu-threads 8 \
    --video-rd-preset quality --video-recon /tmp/p.y4m
$DAV2D -q --strict 1 -i /tmp/p.ivf -o /tmp/p_dec.y4m --threads 1
```

QP 165 and QP 190 both decode with `--strict 1`, 60 pictures, and the 10,368,000
byte recon payload equals the decoder payload byte for byte.
`tools/validate_dav2d.sh` also passes. `tools/gate_video_rd_preset.sh` checks
the preset expansion (including `--video-partition-rd`), the pinned flag-less
480x240 default (`25,009 B`, `a1d0dba1...`), `--strict 1`, and the recon
comparison.

## Interactions

* CPU `--video` only. `--video-gpu` is rejected by name; `--still` uses its own
  tree.
* `--video-ext-partitions` and `--video-sdp` have separate cost models and are
  rejected by name rather than silently ignoring the flag; SVC layers are
  rejected for the same reason.
* The lossless path uses a fixed block size and is rejected by name.
* `--video-partition-rd` alone is valid; the measured preset adds the
  inter-residual, deblock/CCSO, TX-search and intra-full tools that make the
  tree's extra leaves affordable.
