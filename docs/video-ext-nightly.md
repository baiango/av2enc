# Extended partitions on the nightly chain

`--video-ext-partitions` now composes with `--fast-decode`, including the carried
CDF chain, the loop filters (CDEF / CCSO / deblock), the BAWP carrier, the
bottom-edge mask and the drift-refresh frames. The measured win is the one the
keyframe partition oracle predicted (`docs/ce-partition-oracle.md`, branch
`exp/ce11-part`): the `HORZ_3` / `VERT_3` family that the rect tree never
searches is a **Pareto** improvement — fewer bytes *and* a higher
reconstruction — at the nightly operating point.

```sh
av2enc testdata/nasa_480x240_30fps_60f.y4m -o /tmp/ext.ivf \
    --video --frames 60 --qp 110 --cpu-threads 8 \
    --fast-decode --video-ext-partitions
```

## What made the two trees incompatible

The oracle ran the extended solver with `AV2ENC_PART_HARNESS=1`, a diagnosis
switch that (a) let `--video-partition-rd` and `--video-ext-partitions` start
together and (b) gave the extended solver the rect solver's leaf rate term.
That bypass hid five real mismatches, all of which are fixed here:

1. **Leaf pricing.** `InterFrameEncoder::decide()` had two cost functions. The
   rect solver priced a leaf as
   `variance + price * leaf_bits` (keyframes: 6 bits) or, on P frames, with the
   measured `--video-partition-rd` probe; the extended solver priced the same
   leaf as the **bare variance**. The extended tree therefore compared a
   differently-scaled child cost with the same signal cost. Both solvers now
   call one `partition_leaf_cost()` closure, so an extended child of a given
   `(mi_row, mi_col, size)` costs exactly what a `NONE/HORZ/VERT/SPLIT` child
   of that geometry costs, including the probe's own `(mi_row, mi_col, size)`
   cache.
2. **Search bound.** The extended solver ignored the probe's depth clamp, so on
   P frames it searched depth 4 against the rect tree's depth 1 and "won" on
   tiny leaves. `pe::DecisionOptions` is now copied from the rect solver's
   `partition::DecisionOptions` (min block, max depth, SPLIT legality), so the
   candidate *set* is the only difference.
3. **Probe candidate set.** `build_partition_probe_costs()` enumerated only
   `NONE/HORZ/VERT/SPLIT` children. It now also walks the §5.20.3.1 child list
   of `HORZ_3`, `VERT_3` and the four uneven 4-way rows, so the frame-level
   probe covers the extended geometry (64x16 / 16x64 / ...) instead of falling
   back to a synchronous probe per node.
4. **CDF image.** The extended writer adapts its own copy of the five §9.3
   partition rows (`still::partition_stream::TilePartitionCdfs`), while the
   §8.2 image packed only `CdfState::do_split / do_square_split / rect_type`.
   `do_ext_partition` and `do_uneven_4way_partition` were absent, so with
   `--cdf-continuity` the encoder adapted rows the decoder reset every frame:
   the first extended frame desynchronised. Both rows are now part of
   `CdfState` (and the average-CDF bank), and the writer bridges its private
   copy in/out of the image around the tile. With the tool off the rows never
   code a symbol, so the image is value-identical and the pins do not move.
5. **Loop filters / drift refresh / BAWP / edge mask.** These were rejected by
   name. They consume the coded leaf list (`blocks_`): the §7.18.1 `Skips[]`
   grid, the `LrTxSkip` grid and the deblock leaf grid are all derived per leaf
   from `(mi_row, mi_col, size, skip, eob)`, which the extended collector fills
   exactly like the rect one. BAWP's carrier and the bottom-edge mask are
   per-leaf decisions inside `encode_leaf()`. All were verified by decoding the
   composed 60-frame stream, not just by removing the rejection.

## What is still rejected, and why

| Rejection | Reason |
|---|---|
| `--video-gpu` | CUDA keeps the rect candidate set; no GPU extended solver |
| `--video-sb256` / SVC / `--tiles` | the extended solver is wired for one tile and 64/128 roots |
| B / temporal-layer / TIP / random-access schedules | the extended writer rides the flat key+P chain |
| segmentation / delta-Q | both own the per-frame effective qindex |
| `--video-ref-mvs` | the temporal field is not wired through the extended collector |

The **4-wide vertical uneven child** stays rejected by the cost model: the
vertical uneven family is only offered at `w >= 64` (an 8-wide 1/8 subsize).
The 4-wide `VERT_4A/4B@32x32` row is the documented `dav2d` bottom-edge desync
(`docs/negative-results.md`). With the default 64x64 superblock and
`--video-min-block 16` no 4-way row is reachable (its 1/8 subsize is 8 pixels,
below the floor); at `--video-min-block 8` the horizontal rows and the 64-wide
vertical rows become reachable and are gated (the emitted `VERT_4A/4B` nodes
are all 64-wide; `HORZ_4` appears at 16/32/64-wide parents).

## Bottom edge

The bottom/right edge is where the two solvers disagreed structurally. A
partial superblock has no legal `PARTITION_NONE`, so the decoder derives a
forced partition (HORZ or VERT) without syntax and may need the tree to
subdivide past the encoder's `max_depth`/min-block bound. The rect solver
relaxes its policy at a boundary (`partition::solve_node`'s two-pass loop); the
extended solver did not, and a 720p frame (720 = 11x64 + 16, a 16-row bottom
superblock) failed with "no partition candidate has a finite cost". The
extended solver now mirrors the same relax, and the relaxed pass **skips the
extended rows**, so the relax can never code an 8-pixel strip below the
configured floor. This is the trap the earlier component gate documented; the
`w >= 64` rule for the vertical uneven family is unchanged and the 4-wide
`VERT_4A/4B@32x32` row remains unreachable.

## Measured: 240p front-end, QP 108..112

`testdata/nasa_480x240_30fps_60f.y4m`, 60 frames, `--fast-decode`, both arms with
`--fast-decode`'s own flags (key offset -27, drift refresh on). Quality:
`tools/quality.py` (CVVDP 0.5.7, `standard_fhd`, MPS) with the pinned `dav2d`.

| QP | nightly B | kbps | CVVDP | PSNR-Y | +ext B | kbps | CVVDP | PSNR-Y | bytes | PSNR |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 108 | 24,567 | 98.268 | 9.7256 | 43.8994 | 24,176 | 96.704 | 9.6870 | 44.3043 | -1.59% | +0.405 |
| 109 | 24,313 | 97.252 | 9.6996 | 43.8561 | 23,938 | 95.752 | 9.7170 | 44.3438 | -1.54% | +0.488 |
| **110** | **24,113** | **96.452** | **9.7130** | **43.8901** | **23,587** | **94.348** | **9.6991** | **44.2385** | **-2.18%** | **+0.348** |
| 111 | 23,705 | 94.820 | 9.7048 | 43.8523 | 23,305 | 93.220 | 9.7108 | 44.2409 | -1.69% | +0.389 |
| 112 | 23,421 | 93.684 | 9.7166 | 43.8942 | 22,914 | 91.656 | 9.7195 | 44.2586 | -2.16% | +0.364 |

Iso-rate against the pin, 96.452 kbps (interpolating the extended arm between
QP108 and QP109): **9.6949 JOD / 44.315 dB** versus **9.7130 / 43.8901**,
i.e. **-0.018 JOD and +0.425 dB at the same rate**. Across the measured range
the iso-rate deltas are -0.022 .. +0.014 JOD and +0.37 .. +0.59 dB. This is a
rate/PSNR Pareto win with a CVVDP-neutral result (the ±0.03 single-clip
ripple is twice the residual JOD delta). The oracle's expected
"+0.19 .. +0.30 dB" was a keyframe-only, pre-filter PSNR delta; at stream level
the delta is larger because the extended family is also selected on P frames.

The tree is the oracle's: at QP120 the extended keyframe has **378 nodes / 232
leaves** (rect: 363 / 189) with **36 `HORZ_3`/`VERT_3` nodes** (32 `VERT_3`,
4 `HORZ_3`), i.e. the 5.2% keyframe residual-shape win reproduces with the
carried CDFs, CDEF/CCSO and the edge mask in place. The keyframe partition tree
is QP-independent under this cost model (source variance + a constant syntax
price), so the same 36 nodes appear at QP90/105/120.

## Measured: keyframe curve

One picture, `--fast-decode` (which applies the -27 key offset), `--frames 1`:

| QP | nightly B | CVVDP | PSNR-Y | +ext B | CVVDP | PSNR-Y | bytes | PSNR |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 90 | 25,387 | 9.9203 | 47.7419 | 24,764 | 9.9457 | 50.7073 | -2.45% | +2.965 |
| 105 | 20,266 | 9.9083 | 46.2163 | 19,688 | 9.9265 | 47.9646 | -2.85% | +1.748 |
| 120 | 16,156 | 9.8837 | 44.8985 | 15,444 | 9.8892 | 45.2235 | -4.41% | +0.325 |

All three are strict Pareto (fewer bytes, higher PSNR, higher CVVDP). The gain
grows as the keyframe gets more bits — the extended leaves are residual *shape*,
so they matter most where coefficients are dense. At the QP120 point the shipped
nightly keyframe is 16,156 B (not the 16,322 B oracle figure: that was the
reduced no-filter configuration).

## Measured: 720p at ~300 kbps

`testdata/space8k_1440x720_60f.y4m`, 60 frames, `--qp 217
--video-key-qp-offset -57` (the 720p offset from
`bench/space8k_720p/README.md`). 60 frames at 60 fps is one second, so the IVF
size is the kbps x 1000 / 8:

| arm | bytes | kbps | CVVDP | PSNR-Y |
|---|---:|---:|---:|---:|
| nightly QP217 | 37,003 | 296.024 | 7.2726 | 31.3342 |
| +ext QP217 | 39,090 | 312.720 | 7.4039 | 31.7681 |
| **+ext QP219 (iso-rate)** | **37,157** | **297.256** | **7.3231** | **31.5768** |

At iso-rate (+0.4% bytes for the ext arm) the extended stream is **+0.051 JOD
and +0.243 dB**. At the same QP it buys +0.131 JOD / +0.434 dB for +5.6% rate.
This is the first point where the extended family is CVVDP-positive outside the
ripple (the 240p CVVDP delta was neutral), which fits the oracle's keyframe
finding: at 720p one improved key picture is worth +0.356 JOD because the
skip-dominated P chain carries its error forward.

Both 720p streams decode 60/60 under `dav2d --strict 1` with the decoded
payload equal to `--video-recon`; the 16-row bottom superblock row is the case
that exposed the missing boundary relax (see "Bottom edge").

## Verification

```sh
src/pipeline/video/gate_video_ext_nightly.sh --bin build-t/av2enc
# PASS default/quality/nightly pins (25,009 / 25,141 / 24,113, byte-identical)
# PASS ext nightly 23,587 B, 60/60 pictures, recon == decoded
# PASS ext keyframe 15,444 B (rect 16,156 B), 1 picture, HORZ_3=4 VERT_3=32
# PASS min-block 8 bottom edge, 16-row bottom SB and rejections
```

The gate also encodes a generated 128x80 clip (a 16-row partial bottom
superblock, the 720p failure) and requires 2/2 pictures with an exact recon.

The gate checks the decoded picture count itself (`dav2d` exits 0 even when a
picture fails), compares the decoded payload with `--video-recon` byte for byte,
and parses the `AV2ENC_PART_TREE` dump for the extended nodes and the 4-wide
geometry rule.

## Reproduction

```sh
# flag off: the three pins must stay byte-identical
build-t/av2enc testdata/nasa_480x240_30fps_60f.y4m -o /tmp/p.ivf \
    --video --frames 60 --qp 189
build-t/av2enc testdata/nasa_480x240_30fps_60f.y4m -o /tmp/q.ivf \
    --video --frames 60 --qp 190 --video-rd-preset quality
build-t/av2enc testdata/nasa_480x240_30fps_60f.y4m -o /tmp/n.ivf \
    --video --frames 60 --qp 110 --cpu-threads 8 --fast-decode

# the composed stream
build-t/av2enc testdata/nasa_480x240_30fps_60f.y4m -o /tmp/e.ivf \
    --video --frames 60 --qp 110 --cpu-threads 8 \
    --fast-decode --video-ext-partitions

# the emitted tree
AV2ENC_PART_TREE=/tmp/tree.txt build-t/av2enc \
    testdata/nasa_480x240_30fps_60f.y4m -o /tmp/k.ivf \
    --video --frames 1 --qp 120 --cpu-threads 8 \
    --fast-decode --video-ext-partitions
awk '{print $9}' /tmp/tree.txt | sort | uniq -c
```
