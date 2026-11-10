# Multi-GPU planner (simulation only)

Analytic planning tool for av2enc tile-parallel and GOP-parallel multi-GPU
work distribution. It needs **no CUDA and no multi-GPU host**; it is pure Python
3.9+ stdlib.

This is not an encoder, not a benchmark, and not validation. Every number is a
model estimate. The encoder CUDA path is not wired into the encode path yet, and
the single-GPU 1 fps denominator is a product target, not a measurement. See
`docs/multi-gpu.md` for the design contract and the pending-hardware
disclaimer.

## Quick start

From the repository root:

```sh
cd bench/multi_gpu_plan

# 1/2/4 GPUs at 1080p and 4K, default layouts (4x2, 8x2, 16x2)
python3 multi_gpu_plan.py --matrix

# Machine-readable result, including exact tile index sets
python3 multi_gpu_plan.py --matrix --json /tmp/multi_gpu_plan.json

# Markdown tables for a design review
python3 multi_gpu_plan.py --matrix --markdown

# Model invariants and determinism checks
python3 multi_gpu_plan.py --self-test
```

Expected first lines:

```text
av2enc multi-GPU planner - SIMULATION ONLY
No CUDA, no multi-GPU hardware, no measured scaling.
PCIe 4.0 x16 (effective 26.8 GB/s), refs=4, target 1 fps, serial fraction 3.5%, content skew +/-15%
scale mode: quality (default quality)
```

## Simulated matrix

Default assumptions: PCIe 4.0 ×16, 85% link efficiency, refs=4, lookahead=4,
content skew ±15%, balanced LPT assignment, 1 Mbps payload at 1 fps. The
`work ms` column is the same-quality parallel work time, not an fps claim.

```text
Scheduling and scaling
res    gpus  layout  tiles/GPU  SB max/mean  load imb  work ms  speedup  eff
-----  ----  ------  ---------  -----------  --------  -------  -------  -----
1080p  1     4x2     8          72/63.8      1.000     997.6    1.000    1.000
1080p  2     8x2     8/8        36/31.9      1.007     520.4    1.917    0.959
1080p  4     16x2    8/8/8/8    18/15.9      1.030     283.9    3.514    0.878
4k     1     4x2     8          255/255.0    1.000     3990.2   1.000    1.000
4k     2     8x2     8/8        136/127.5    1.008     2083.0   1.916    0.958
4k     4     16x2    8/8/8/8    68/63.8      1.025     1130.0   3.531    0.883
```

Memory and transfer:

```text
res    gpus  ref/GPU MB  dup extra MB  xfer MB  xfer ms  xfer %1fps  live MB  cap MB  fits
-----  ----  ----------  ------------  -------  -------  ----------  -------  ------  ----
1080p  1     11.87       0.00          3.09     0.221    0.02        614.9    7500    yes
1080p  2     11.87       11.87         12.35    0.734    0.07        588.3    7500    yes
1080p  4     11.87       35.60         24.21    1.198    0.12        574.9    7500    yes
4k     1     47.46       0.00          11.98    0.569    0.06        851.7    7500    yes
4k     2     47.46       47.46         47.94    2.127    0.21        762.5    7500    yes
4k     4     47.46       142.38        95.41    3.986    0.40        711.5    7500    yes
```

`MB` means MiB. `live MB` is a model estimate including 512 MiB planning
headroom; the arena still enforces the 7,500 MB cap per GPU.

## Inputs

| Flag | Default | Meaning |
|---|---|---|
| `--resolution` | `1080p` | `1080p`, `1440p`, `4k`, `2160p` |
| `--width`, `--height` | — | Explicit frame size; use instead of `--resolution` |
| `--matrix` | off | Run 1080p and 4K |
| `--gpus` | `1,2,4` | GPU counts to simulate (not device IDs) |
| `--tiles` | `auto` | `CxR`, `auto`, or a comma list (`4x2,8x2,16x2`) |
| `--multi-gpu` | `tile` | `tile` or `gop` |
| `--scale-mode` | `quality` | `quality` or `throughput` (label and interpretation) |
| `--pcie-gen` | `4` | 3, 4, or 5 |
| `--pcie-width` | `16` | 1, 2, 4, 8, or 16 lanes |
| `--pcie-efficiency` | `0.85` | Effective fraction of raw line rate |
| `--refs` | `4` | Reference frames per GPU |
| `--lookahead` | `4` | Lookahead frames per GPU |
| `--fps` | `1.0` | Target fps for payload size and the transfer budget |
| `--bitrate-mbps` | `1.0` | Encoded payload bitrate |
| `--ms-per-mp` | `481.0` | Single-GPU ms per luma megapixel (calibrated to ~1 s at 1080p) |
| `--serial-fraction` | `0.035` | Single-GPU work that does not shrink with more GPUs |
| `--content-skew` | `0.15` | Synthetic per-tile cost variation, ± fraction |
| `--assignment` | `balanced` | `balanced` (LPT) or `contiguous` (optimal DP ranges) |
| `--vram-cap-mb` | `7500` | Per-GPU cap |
| `--rdo-multiplier` | `1.0` | Candidate-cache size multiplier |
| `--headroom-mb` | `512` | Planning reserve per GPU |
| `--frames` | `240` | Clip length for GOP mode |
| `--gop-length` | `24` | Segment length for GOP mode |
| `--keyframe-ratio` | `6.0` | Keyframe bits / inter bits, for the GOP penalty |
| `--baseline-keyframe-interval` | `240` | Single-GPU keyframe interval for the GOP penalty |
| `--json` | — | Write the full result as JSON |
| `--markdown` | off | Print Markdown tables |
| `--self-test` | off | Run geometry/assignment/VRAM/determinism invariants |

## Outputs

- `tiles_per_gpu` and `gpu_tile_indices`: exact assignment.
- `gpu_area_sb`, `gpu_cost_sb_equiv`: area and synthetic content-weighted load.
- `area_imbalance`, `load_imbalance`, `load_min_ratio`, `load_cv`.
- `ref_pool_mb_per_gpu`, `ref_dup_total_mb`, `ref_dup_extra_mb`.
- `host_transfer_parts_mb`, `host_transfer_mb`, `host_transfer_ms`,
  `host_transfer_pct_1fps`.
- `compute_ms_1gpu`, `frame_ms`, `throughput_fps`, `speedup_vs_1gpu`,
  `efficiency`.
- `vram_breakdown_mb` (persistent, scratch, candidate, headroom), `vram_live_mb`,
  `vram_cap_mb`, `vram_fits`.
- GOP mode: `segments`, `segments_per_gpu`, `keyframe_bitrate_penalty_pct`.

JSON output is stable and includes `"measured": false` plus a disclaimer.

## Examples

```sh
# One configuration: 2 GPUs, 1080p, 8x2, PCIe 4.0 x16
python3 multi_gpu_plan.py --width 1920 --height 1080 --gpus 2 --tiles 8x2

# 4 GPUs, 4K, 16x2, realistic x4 lane split
python3 multi_gpu_plan.py --width 3840 --height 2160 --gpus 4 --tiles 16x2 \
  --pcie-gen 4 --pcie-width 4

# Compare balanced and contiguous ownership
python3 multi_gpu_plan.py --matrix --assignment balanced
python3 multi_gpu_plan.py --matrix --assignment contiguous

# Every layout at every GPU count
python3 multi_gpu_plan.py --matrix --tiles 4x2,8x2,16x2

# GOP-parallel: 240 frames, 24-frame segments
# (GOP mode assigns whole segments; the per-frame tile layout is the default
# for the GPU count, and --tiles does not change segment assignment)
python3 multi_gpu_plan.py --matrix --multi-gpu gop --frames 240 --gop-length 24

# GOP-parallel with fewer, longer segments (smaller keyframe penalty, worse tail)
python3 multi_gpu_plan.py --matrix --multi-gpu gop --frames 480 --gop-length 48

# 8 references (duplicated per GPU)
python3 multi_gpu_plan.py --matrix --refs 8

# Markdown for a review doc
python3 multi_gpu_plan.py --matrix --markdown
```

## Model

The planner computes tile rectangles from AV2 uniform spacing
(`docs/tiles.md` §5.18.7.5), including the partial superblock row. Assignment is
deterministic LPT on tile pixel area (or an optimal contiguous DP partition).
A stable per-tile hash adds synthetic content skew; it is normalized so total
work is unchanged.

Timing:

```text
compute_1gpu = (luma_pixels / 1e6) * ms_per_mpx
T_G = serial_fraction * compute_1gpu
    + (1 - serial_fraction) * load_imbalance * compute_1gpu / gpus
    + host_transfer_ms
speedup = (compute_1gpu + host_transfer_ms_1gpu) / T_G
```

The default `ms_per_mpx = 481` makes the 1080p single-GPU model ~1.00 s. This is
calibrated to the product floor, **not measured**. The serial fraction is the
non-parallel share (coordinator, launch, merge, CDF, frame header/rate control).
Load imbalance is content-aware `max_gpu_cost / mean_gpu_cost`.

Host traffic (tile mode) includes source upload, tile payload D2H,
reconstruction gather, reference fill, and CDF gather. The CDF term is a
conservative upper bound: with the default `disable_cdf_update=1` it is zero,
and it only applies when the opt-in `--adaptive-cdf` path is used. PCIe
effective bandwidth is `per-lane raw * width * efficiency`; transfer time
assumes a serialized host-mediated sequence plus 0.05 ms per active phase. GOP
mode has no per-frame reference exchange, only source upload and compressed
segment output.

VRAM uses the `docs/vram.md` split: persistent frames + per-16×16 state,
per-tile scratch, RDO candidate cache, and 512 MiB headroom. `local_tiles` is
the tiles on that GPU, not the frame total.

## Interpreting `quality` vs `throughput`

- `quality` (default): hold aggregate fps at the target (1 fps at 1080p) and
  spend the parallel work time on quality. The `speedup` column is the quality
  headroom; actual fps stays at the target.
- `throughput`: pin the single-GPU quality level, tile layout, QP, search, and
  references. The `speedup` column is the fps scaling factor. This is the mode
  for bit-exactness and scaling measurements.

The generator does not change its arithmetic based on `--scale-mode`; the flag
labels how the same-quality work budget is spent. `docs/multi-gpu.md` defines
the encoder behavior.

## Top scaling risk

**Content-dependent tile cost: the per-frame barrier waits for the slowest
GPU.** Area-based assignment can look perfect while a detailed tile or a
thermally throttled GPU gates every frame. At 3.5% serial work, a 10% straggler
drops 2-GPU speedup to ~1.77×, below the 1.8× target. The synthetic skew in this
tool is not a substitute for measured per-tile time. Secondary risks are
coordinator/host serialization, PCIe in throughput mode, and the serial entropy
tail. See `docs/multi-gpu.md` §9.

## Limitations

- No CUDA, no physical multi-GPU, no measured validation.
- Homogeneous 3060s assumed; mixed GPUs change the slowest-GPU calculus.
- Absolute timing is a planning assumption, not a 3060 measurement.
- No tile bitrate/BD-rate model; tile overhead is discussed separately in
  `docs/multi-gpu.md`.
- No thermal/power/clock model; a throttled GPU is a straggler.
- Synthetic content skew is deterministic but arbitrary.
- 16×2 is not yet verified with `dav2d`; test it before relying on 4-GPU tile
  mode.
