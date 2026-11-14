# Measured video partition tree (`--video-partition-rd`)

`--video-partition-rd` replaces the CPU `--video` P-frame partition cost with a
measured inter RD probe: each candidate leaf is priced by one NEWMV prediction
plus one quantised luma residual. It is an explicit CPU opt-in and is part of
the `--nightly` aggregate. The flag-less default, the stable `quality` preset
and every GPU path keep the historical tree and byte-identical streams.

```sh
av2enc testdata/nasa_480x240_30fps_60f.y4m -o /tmp/n.ivf \
    --video --frames 60 --qp 136 --cpu-threads 8 --nightly
```

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
0, half 255). Every optional split is therefore unaffordable above `q ≈ 163`,
and the P-frame tree collapses to one 64x64 leaf per superblock (only
geometry-forced edge splits survive). A first-rate-aware fix priced a
variance leaf plus a constant `700` SSE per estimated syntax bit; it recovered
part of the loss but variance still cannot see prediction: a split only
recovers the between-child mean term, not the inter-residual reduction.

## The change

The probe measures the quantity the encoder actually spends:

```text
leaf = D + lambda * R
D    = min(prediction SSE, quantised-residual reconstruction SSE)
R    = 8 mode bits + luma_plan_bits(plan)          (or 8 skip bits)
lambda = scale * step^2,  step = rc::quant_step_from_qindex(base_q_idx)
```

* `D` comes from one NEWMV search and one `compute_tx()` on the winner; the
  plan's `luma_plan_bits()` is the encoder's existing residual-rate estimate.
* `lambda` is quantiser-scaled. The default scale is `1.1e-5`; `--nightly`
  uses the measured `2.75e-6` (see the sweep below). Both are exposed as
  `--video-partition-rd-lambda <S>`.
* The probe is bounded to the first split level: a superblock leaf against its
  HORZ/VERT/SPLIT children. `--video-partition-rd-depth <N>` (1..3) extends it;
  depth 1 is the measured default.
* Candidates are precomputed on the frame worker pool and keyed by
  `(mi_row, mi_col, size)`. Nodes the solver reaches outside that set (forced
  boundary partitions) are probed synchronously, so the tree never consults a
  stale or missing cost.
* The probe is read-only with respect to encoder state. The shared
  `partition::decide_frame` solver still owns legality, forced boundaries,
  chroma ownership and the deterministic tie-break, so the emitted tree is
  exactly `validate_frame`'s tree.
* Leaves that cannot be inter-predicted (sub-8, right-edge, ME failure) and the
  keyframe keep the SSE-domain variance + syntax-bit estimate. The keyframe
  tree is unchanged.
* The bottom row is probed over its full coded 64x64 extent, the same extent
  `encode_leaf()` uses with the `--video-edge-inter` padded reference, so the
  probe and the coder agree on which leaves are inter-capable.

## Measured RD

Clip: `testdata/nasa_480x240_30fps_60f.y4m`, 60 frames, 30 fps, 480x240,
CPU `--video`, `--cpu-threads 8`, `AV2ENC_CPU_FAST=ON`, Apple M4. Quality:
`tools/quality.py` (CVVDP 0.5.7, `standard_fhd`, MPS) against the pinned
`dav2d-macos`.

`--nightly` + probe (`--qp`, probe lambda `2.75e-6`, depth 1):

| QP | kbps | CVVDP JOD | PSNR-Y |
|---:|---:|---:|---:|
| 135 | 103.736 | 9.1651 | 37.101 |
| **136** | **100.116** | **9.1542** | **36.946** |
| 137 | 98.592 | 9.1198 | 36.929 |

At the established ~100.25 kbps point:

| Stream | Rate | CVVDP | PSNR-Y |
|---|---:|---:|---:|
| `--nightly` QP142 (edge + CCSO, historical tree) | 100.248 kbps | 8.7465 | 35.3754 dB |
| `--nightly` QP136 (edge + CCSO + probe) | 100.116 kbps | **9.1542** | **36.946 dB** |
| probe, interpolated to 100.248 kbps (QP 135/136) | 100.248 kbps | **9.1546** | **~36.95 dB** |
| **delta (interpolated)** | matched | **+0.408 JOD** | **+1.57 dB** |

The probe also reaches 100.116 kbps at QP136 versus QP142 for the same
edge+CCSO path: the finer tree removes rate, so the iso-rate point moves to a
lower QP. The gain is content-dependent; this fixture is mostly static text
over a slowly moving model, where cheap skip leaves with per-32-block MVs beat
the legacy 64x64 tree by a wide margin.

### Lambda sweep

Iso-rate CVVDP for the same nightly aggregate and depth 1, interpolating each
lambda's bracketing QPs to 100.248 kbps:

| probe lambda | iso-rate JOD |
|---|---:|
| `1e-9` (rate-free) | ≈9.10 |
| `0.1375e-5` | 9.131 |
| `1.1e-5` (probe default) | 9.134 |
| `2.2e-5` | 9.136 |
| `0.22e-5` | 9.143 |
| `0.55e-5` | 9.145 |
| `0.33e-5` | 9.151 |
| **`0.275e-5`** | **9.155** |

The measured optimum is a quarter of the original probe scale; the nightly
aggregate carries `2.75e-6`. Larger values overprice the extra leaves and the
rate-free limit over-splits.

### Depth

Depth 2 (QP 138/139 bracket, same lambda) measures ≈9.109 JOD at 100.248 kbps:
worse than depth 1 at ≈1.2x the encode work. The first split level already
captures the win on this clip.

### Encode time

60 frames on the M4 (host otherwise idle): edge+CCSO QP142 is 8.2-8.9 s wall
(6.0 s CPU); nightly + probe QP136 is 10.5-11.4 s wall (14.5 s CPU across the
8-thread pool). The wall cost is ~+25% because the probe precompute runs on the
frame pool; the per-leaf ME/TX work is parallel.

## Verification

```sh
BIN=build-a/av2enc
DAV2D=/Users/arin/work/av2enc/tools/bin/dav2d-macos
$BIN testdata/nasa_480x240_30fps_60f.y4m -o /tmp/n.ivf --video --frames 60 \
     --qp 136 --cpu-threads 8 --nightly --video-recon /tmp/n.y4m
tools/validate_dav2d.sh /tmp/n.ivf                 # OK, 60 frames
$DAV2D -q --strict 1 -i /tmp/n.ivf -o /tmp/n_dec.y4m --threads 1
# encoder recon payload == decoder payload, 10,368,000 B, byte for byte
tools/quality.py --ref testdata/nasa_480x240_30fps_60f.y4m --ivf /tmp/n.ivf \
     --json /tmp/n.json                            # 9.1542 JOD, 36.946 dB
```

`tools/gate_video_nightly.sh` checks the `--nightly` expansion (including the
probe flags), the pinned default (`25,009 B`, `a1d0dba1...`) and `quality`
(`25,141 B`, `3984677b...`) streams, `--strict 1` decode and visible recon
equality.

## Interactions

* CPU `--video` only, 8-bit 4:2:0, single tile, flat P chain. `--video-gpu`,
  `--still`, the non-4:2:0 format cores, `--tiles > 1x1`, the B/temporal-layer/
  random-access/TIP schedules and the 128/256 superblock trees are rejected by
  name. `--video-ext-partitions`, `--video-sdp`, SVC layers and the lossless
  path are likewise rejected.
* The probe is not a tool toggle: it replaces the partition decision cost. The
  surrounding tools (`--video-inter-residual`, `--video-deblock`,
  `--video-ccso`, `--video-tx-search`, `--video-intra-full`) make the extra
  leaves affordable; `--nightly` adds rate-aware CCSO and bottom-edge inter.
* `--video-tx-context-rate` still measures negative when stacked on this path
  (9.109 JOD at 101.13 kbps with the probe), so it stays a separate opt-in.
