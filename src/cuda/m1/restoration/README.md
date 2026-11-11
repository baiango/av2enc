# M1 post-CDEF restoration stage — CCSO + Wiener NS

Workstream: in-loop restoration for the M1 GPU path, owned by
`src/cuda/m1/restoration/`.  It consumes the deblocked reconstruction, applies
CCSO (§7.19) and loop restoration / non-separable Wiener (§7.20–§7.20.3), and
produces the final reference/output planes.

Normative order (do not reorder):

```
reconstruct -> deblock -> CDEF -> CCSO -> loop restoration -> output
```

`CCSO` is part of the restoration stage and runs before loop restoration.
GDF is *after* Wiener inside the same §7.20 block loop, but GDF is blocked (see
below) and must be rejected.

## API

```cpp
#include "cuda/m1/restoration/m1_restoration.h"

using namespace av2enc::cuda::m1;

RestorationStageParams params;
params.ccso.ccso_luma_size_log2 = 6;         // §5.18.7.12 derivation
params.ccso.plane[p] = ...;                  // parsed §5.18.7.12 scalars
params.ccso.offsets_dev[p] = d_offsets;      // 576 int16, device
params.ccso.blks[p] = d_ccso_blks;           // one byte per CCSO unit
params.ccso.blk_cols[p] = cols;
params.ccso.blk_rows[p] = rows;
params.lr = ...;                             // §5.18.7.11 + §5.20.10.6

RestorationFrame frame;                      // all pointers device-resident
frame.curr[p] = pre_cdef_plane;              // deblocked, pre-CDEF
frame.cdef[p] = post_cdef_plane;             // CCSO modifies in place
frame.lr[p]   = lr_plane;                    // distinct output buffer

cudaError_t st = restoration_stage_apply(frame, params, stream);
```

Two entry points:

| Function | Purpose |
|---|---|
| `restoration_stage_validate(frame, params)` | Host-side validation; never dereferences device grids and never launches. |
| `restoration_stage_apply(frame, params, stream)` | CCSO (optional), then the §7.20 copy + Wiener filter. |
| `restoration_stage_ccso_core(frame, ccso)` | Builds the §7.19 core for callers that want to issue CCSO separately. |

Rules enforced before any launch:

- 8-bit 4:2:0, one tile, `mi_cols > 0`, `mi_rows > 0`.
- `lr` must be a distinct buffer from `cdef` (LR reads `CdefFrame` while writing
  `LrFrame`).
- If a plane's CCSO is enabled, `curr[plane]` must be a distinct pre-CCSO
  snapshot; CCSO reads luma from `curr[0]` while writing `cdef`.
- `ccso.offsets_dev[plane]` must be present for every enabled plane.
- `LoopRestorationSize` conformance (§6.17.7.7): power of two, luma 64–512,
  chroma 32–256 for 4:2:0, `[2] == [1]`.
- `frame_filters_on[0]` with `num_filter_classes > 1` requires a class map
  (`filter_class_grid` + `subclass_lookup`, or the legacy pre-looked-up
  `subclass_grid`).
- GDF and PC-Wiener are rejected with `cudaErrorNotSupported`.
- CCSO scalar fields must be representable by §5.18.7.12
  (`ccso_header_conforms()`): `max_band_log2 <= 3` unless `bo_only`, and
  `edge_clf` 0/1.

No allocation, no `cudaDeviceSynchronize()`, no global state.  The stage issues
at most one CCSO kernel plus the loop-restoration copy and filter.

### Data residency

Every pointer in `RestorationFrame` and `RestorationStageParams.lr` is a device
pointer: planes, `unit_type`, `unit_coeffs`, `frame_coeffs`,
`filter_class_grid`, `subclass_lookup`, `subclass_grid`, segment ids, lossless
array, BruModes and the four tile grids.  The CCSO offset table is passed
separately as `offsets_dev` because `CcsoParams` stores it inline on the host.
Upload once per parameter change, never per frame.

## CCSO details (§5.18.7.12, §6.17.7.8, §5.20.10.2, §7.19)

- **Unit size / block grid.** `CcsoLumaSizeLog2` is 6, 7 or 8.  For M1's 64×64
  superblock with `ccso_unit_matches_sb_size = 1` it is 6.
  `ccso_select_luma_size_log2(sb_size, tile_start_or, matches_sb)` implements
  the header rule; `ccso_blk_grid_size(...)` computes the `CcsoBlks` grid for
  each plane.  A CCSO unit smaller than the superblock is legal; the §7.19.1
  inner loops step by `sbBlkW/sbBlkH`.
- **Per-unit enable.** `CcsoBlks[plane][unitRow][unitCol]` is signaled by
  `read_ccso()` only at unit origins (`MiRow/MiCol` multiples of
  `1 << (CcsoLumaSizeLog2 - MI_SIZE_LOG2)`).  A zero entry skips the unit.
- **Offsets.** `ccso_offset_idx` is 7 bits; the value is
  `Ccso_Offset[idx] * (ccso_scale_idx + 1)`, stored per
  `[band][cls0][cls1]`.  `ccso_set_offsets()` fills that table in syntax order
  (`d0, d1, band`).
- **Bands.** `band = CurrFrame[0][yLuma][xLuma] >> (BitDepth - max_band_log2)`;
  the classifier always reads *luma* even for chroma (`xLuma = x3 << subX`).
- **Edge classes.** `ccso_score(diff, quantStep, edgeClf)` with
  `quantStep = CCSO_Quant_Sz[scale_idx][quant_idx]`; `edge_clf` 0/1/2 changes
  the class count (`3 - edge_clf`).  All 7 `Ccso_Pos` extension filters are
  supported.  A bitstream can only code `edge_clf` 0/1; the core keeps 2 for
  diagnostics.
- **Planes.** Per-plane `ccso_planes[p]`, scalars, offsets and block grid.  The
  filter modifies `CdefFrame` in place and clips to `[0, 255]`.
- **Lossless / BRU.** A sample is skipped when its segment's `LosslessArray`
  entry is 1, and when `BruModes[mi][mi] != BRU_ACTIVE` for the
  superblock-aligned sub-block.  Both are implemented in the shared core and
  covered by the CPU tests.
- **Tile clipping.** With `disable_loopfilters_across_tiles`, the luma fetch
  clamps to `MiColStart/End`, `MiRowStart/End`.

## Loop restoration / Wiener NS details

- **Units and stripes.** 4×4 luma block walk for all planes; stripes are 64
  luma lines offset 8 upward (`stripeNum = (lumaY + 8) / 64`).  Samples inside
  the stripe come from `CdefFrame`; outside from `CurrFrame`, with the
  two-line clamp.  Unit index clamps to the last unit.
- **Per-unit filters.** `LrType[plane][unitRow][unitCol]` selects
  `RESTORE_NONE` / `RESTORE_WIENER_NONSEP` /
  `RESTORE_PC_WIENER` (rejected) / `RESTORE_SWITCHABLE` grid.
- **Luma filter.** 32 taps mapped to 16 coefficient indices.
- **Chroma filter.** 12 chroma taps + 12 chroma-from-luma taps mapped to
  coefficient indices 0–5 and 6–17.  `get_luma_sample()` applies
  `Wiener_Filters_420` with `cfl_ds_filter_index` (3 aliases to 0).
- **Frame-level filters.** `frame_filters_on[plane]` uses
  `FrameLrWienerNs[plane][subcls][coeff]` and ignores per-unit coefficients.
  For luma, `subcls = SubclassLookup[FilterClass[y>>2][x>>2]]`; chroma always
  uses class 0.
- **Class map requirement.** §7.20.4's `Pc_Wiener_Lut_To_Class` and
  `Pc_Wiener_Sub_Classify2` are missing from the spec text.  The API therefore
  requires the caller to supply `filter_class_grid` + `subclass_lookup` (or a
  pre-looked-up `subclass_grid`).  Without it, validation fails loudly; there is
  no silent identity fallback for a real class map.
- **Coefficient ranges.** `Wiener_Ns_Taps_Min/K` are normative and implemented
  (`wiener_ns_clip_coeff()`); an encoder-side fit must clip to
  `[min, min + 2^k - 1]`.
- **`LrFrame := CdefFrame`** is unconditional, so `RESTORE_NONE` units and
  planes pass through byte-exactly.

## Header / syntax checklist

Sequence header (`sequence_filter_config()`; M1 currently clears all filter
flags):

| Element | Spec | M1 value for this stage |
|---|---|---|
| `disable_loopfilters_across_tiles` | §5.4.10 | 0 (one tile) |
| `enable_cdef` | §5.4.10 | 0 until CDEF lands |
| `enable_gdf` | §5.4.10 | 0 |
| `gdf_unit_matches_sb_size` | §5.4.10 | absent (GDF off) |
| `enable_restoration` | §5.4.10 | 1 |
| `lr_tools_disable[0][RESTORE_PC_WIENER]` | §5.4.10 | 1 (blocked) |
| `lr_tools_disable[0][RESTORE_WIENER_NONSEP]` | §5.4.10 | 0 |
| `lr_tools_disable[1][RESTORE_PC_WIENER]` | §5.4.10 | 1 (inferred) |
| `lr_tools_uv_present` | §5.4.10 | 1 if any chroma LR |
| `lr_tools_disable[1][RESTORE_WIENER_NONSEP]` | §5.4.10 | 0 |
| `enable_ccso` | §5.4.10 | 1 |
| `ccso_unit_matches_sb_size` | §5.4.10 | 1 (64×64 units) |

Frame header:

| Syntax | Spec | Notes |
|---|---|---|
| `lr_params()` | §5.18.7.11 | `toolsCount/indexToTool`, `FrameRestorationType`, `frame_filters_on`, `num_filter_classes_idx`, `LoopRestorationSize` flags, frame coefficients |
| `ccso_params()` | §5.18.7.12 | `ccso_frame_flag`, `ccso_planes`, `bo_only`, scale/quant/ext/edge, `max_band_log2`, offset list |
| `gdf_params()` | §5.18.7.9 | not used; `gdf_frame_enable = 0` |
| `CodedLossless` | §5.18.7.1 | forces CCSO and LR off |

Tile syntax:

| Syntax | Spec | Where |
|---|---|---|
| `read_ccso()` | §5.20.10.2 | at each CCSO unit origin; writes `CcsoBlks` |
| `read_lr(row, col, bSize)` | §5.20.10.4 | from the block/partition syntax walk |
| `read_lr_unit()` | §5.20.10.5 | per restoration unit; writes `LrType` and per-unit coefficients |
| `read_wienerns_filter()` | §5.20.10.6 | frame-level filter (one class set) and per-unit filter |

Semantics: `FrameRestorationType` §6.17.7.7, CCSO fields §6.17.7.8, filter
processes §7.19/§7.20.  Requirements: `LoopRestorationSize[2] == [1]`,
`64 >> SubsamplingY <= LoopRestorationSize[1]`, `check_ru_size()`,
`ccso_ext_filter != 7`, `1 << ccso_max_band_log2 <= CCSO_BAND_NUM`.

## M1 integration notes (files outside this directory)

These are **not** edited by this agent.  They are the exact steps to wire the
stage into the M1 GPU path.

1. **Sequence header** — `src/encoder/m1/m1_headers.cpp` (currently
   `enable_restoration = 0`, `enable_ccso = 0` around lines 148–149).  Emit the
   full `sequence_filter_config()` row above.  M1 has no GDF/CDEF yet, so those
   stay 0.
2. **Uncompressed frame header** — same file: add `ccso_params()` and
   `lr_params()` with `single_picture_header_flag` semantics.  For an intra
   still, `reuse_ccso`/`sb_reuse_ccso` are 0 and `temporal_pred_flag` is 0.
3. **Tile syntax / entropy** — `src/encoder/m1/m1_syntax.cpp` and
   `src/encoder/m1/stream/`: write `ccso_blk S()`, `use_wiener_ns S()`, the
   `read_wienerns_filter()` symbols and the LR/CCSO offset symbols with their
   §8.3 CDFs.  `read_lr()` is invoked from the block walk, not once per frame.
   Entropy stays CPU-resident, matching the current M1 design.
4. **Orchestrator** — `src/cuda/m1/m1_gpu.h`: add `Stage::kRestoration`
   after `kDeblock`; `src/cuda/m1/m1_gpu.cu`: allocate (from the arena) a
   `curr` copy of the deblocked planes, a `cdef` set (a copy of deblocked when
   CDEF is off), and an `lr` set; upload the CCSO offset table and all grids;
   call `restoration_stage_apply`; use `lr` as the final recon/reference
   output.  Do not alias `curr`/`cdef` while CCSO is on.
5. **Build** — root `CMakeLists.txt` GLOBs `src/cuda/m1/**/*.cu`, so
   `m1_restoration.cu` is picked up automatically, but three dependencies are
   not:
   - `src/cuda/restoration/ccso.cu` and `src/cuda/restoration/wiener.cu` are
     not in `AV2ENC_M1_GPU_SUPPORT_SOURCES`;
   - `src/encoder/filters/restoration/restoration.cpp` is not in
     `av2enc_core`, yet the stage calls its validators and
     `ccso_header_conforms()` (the CUDA component tests link it explicitly).

   Add the two `.cu` files to the support list and `restoration.cpp` to the
   core sources (owner: agent 9), or link an equivalent object library.
   Without that the M1 GPU target will fail to link.
6. **Gates** — enabling filters changes the pinned M1 MD5s.  Add one stream per
   tool: CCSO-only, then +Wiener.  Each must pass
   `tools/validate_dav2d.sh`, and `tools/m1_verify` must show decoded ==
   encoder recon for Y/U/V.  Keep the old no-filter MD5 as the regression pin.

### Standalone minimal stream gate

Not feasible from this directory alone.  A gate needs (a) sequence/frame header
flags, (b) tile symbols and CDFs, and (c) writer plumbing.  Those live in
`src/encoder/m1/m1_headers.cpp`, `src/encoder/m1/m1_syntax.cpp`,
`src/encoder/m1/stream/` and `src/main.cpp`, none of which this agent owns.
Creating a second mini-encoder that writes §8.2 syntax would duplicate the
entropy writer and violate the single-codepath rule.  The parity and quality
evidence here is therefore component-level; the stream gate is the integration
step above.

## CPU-vs-GPU parity

The CPU reference and the CUDA kernels call the *same* `__host__ __device__`
inline functions (`ccso_core.h`, `wiener_core.h`).  Every operation is integer
(clip, shift, add, multiply by an `int16` coefficient), so parity is structural,
not tolerance-based.  The stage test additionally checks:

- whole-frame `CdefFrame` (CCSO output) and `LrFrame`;
- guard columns/rows on both buffers;
- per-unit and frame-level filters, luma + chroma-from-luma;
- lossless/BRU metadata and switchable units;
- real decoder recon for 64×64 and 640×360.

Run on a CUDA host:

```sh
# 1. produce real QP90 reconstructions
build-cuda/av2enc tests/data/m0_64x64_1f.y4m   -o /tmp/s64.ivf  --m1 --no-cfl
build-cuda/av2enc tests/data/m0_640x360_1f.y4m -o /tmp/s360.ivf --m1 --no-cfl

# 2. CPU-vs-GPU parity (synthetic + real)
cd src/cuda/m1/restoration
RUN_ARGS="--ivf /tmp/s64.ivf --ivf /tmp/s360.ivf" ./build_and_test.sh
```

The test decodes the IVF with `tools/bin/dav2d` (override with
`--dav2d PATH`) and uses the decoded first frame as `curr = cdef` (CDEF off).

**Status on this host:** the development Mac has no CUDA toolchain (`nvcc` is
absent), so the GPU binaries were **not executed here**.  Both CUDA translation
units pass a host syntax check against a stubbed `cuda_runtime.h`; the shared
CPU core passes its full suite.  Do not report the parity test as run until the
command above has executed on the 3060 box.

CPU suite status (this host):

```
CCSO:   PASS: 5518 checks, 0 failures   (ASan+UBSan clean)
Wiener: PASS:  114 checks, 0 failures   (ASan+UBSan clean)
real recon (64x64, 640x360, 256x256, per-unit + frame filters):
        PASS: 36 checks, 0 failures
```

The real-recon CPU/oracle check is `real_recon_oracle_test.cpp`; it compares the
production CPU path against the independent spec-order oracle in
`restoration_test_common.h` on decoded QP90 frames.  Build/run:

```sh
c++ -std=c++20 -O2 -I src \
  src/cuda/m1/restoration/real_recon_oracle_test.cpp \
  src/encoder/filters/restoration/restoration.cpp \
  -o /tmp/real_recon_oracle_test
/tmp/real_recon_oracle_test /tmp/recon64.y4m /tmp/recon360.y4m
```

## Quality delta at QP 90

`restoration_quality_bench.cpp` is an encoder-side oracle experiment.  It reads
a source Y4M plus the decoder's reconstruction of the same QP90 frame, trains
CCSO offsets and frame-level Wiener coefficients on that pair, applies the
filters in normative order, and measures PSNR/CVVDP.  **No stream is produced
and no rate cost is included**, so this is filter potential, not BD-rate.

Build:

```sh
c++ -std=c++20 -O2 -I src \
  src/cuda/m1/restoration/restoration_quality_bench.cpp \
  src/encoder/filters/restoration/restoration.cpp \
  -o /tmp/restoration_quality
```

Inputs (all QP90):

| Name | Source | Recon |
|---|---|---|
| 64×64 | `testdata/m0_64x64_1f.y4m` | `dav2d` of `s64_dc.ivf` |
| 640×360 | `build-m1/baseline/src_640x360.y4m` | `dav2d` of `s360_dc.ivf` |
| 256×256 | `testdata/shine_256x256_1f.y4m` | `dav2d` of a `--m1 --no-cfl` encode |

`--max-band-log2 3` (8 bands × 3×3 classes = 72 offsets/plane; the cheapest
configuration that captured nearly all the gain).  Plane-sample-weighted PSNR:

| Case | none Y/U/V (avg) | CCSO Y/U/V (avg) | Wiener Y/U/V (avg) | CCSO+Wiener Y/U/V (avg) |
|---|---|---|---|---|
| 64×64 | 42.50 / 45.80 / 45.12 (43.49) | **42.84 / 46.26 / 45.37 (43.83)** | 42.80 / 45.93 / 46.59 (43.95) | **42.95 / 47.15 / 46.96 (44.32)** |
| 640×360 | 43.15 / 48.59 / 47.38 (44.76) | **44.23 / 48.62 / 47.43 (45.50)** | 43.13 / 48.60 / 46.81 (44.65) | 44.23 / 48.60 / 47.14 (45.44) |
| 256×256 | 46.93 / 47.92 / 49.29 (47.49) | **47.42 / 47.99 / 49.37 (47.84)** | 46.93 / 48.71 / 49.89 (47.72) | **47.22 / 48.77 / 50.08 (47.96)** |

With `--max-band-log2 2` (36 offsets/plane) the CCSO luma gains are still
+0.31 dB (64×64), +0.97 dB (640×360), +0.47 dB (256×256).

CVVDP 0.5.7, `standard_fhd`, 1 frame (64×64 and 256×256 need
`--temp-resample` because the probe writes F1; the 640×360 source is F1):

| Case | none | CCSO | CCSO+Wiener |
|---|---:|---:|---:|
| 64×64 | 9.9009 | 9.9001 | 9.9075 |
| 640×360 | 9.9018 | 9.9107 | 9.9110 |
| 256×256 | 9.9507 | 9.9515 | 9.9481 |

Report string: `ColorVideoVDP v0.5.7, 37.84 [pix/deg], Lpeak=200,
Lblack=0.2, Lrefl=0.3979 [cd/m^2], (standard_fhd)`.

**Verdict.**

- **CCSO is a PSNR win** at QP90 on all three inputs: +0.34/+0.74/+0.35 dB
  average at band 3, and it is the only component that wins on every input.
  The luma gain is largest on 640×360 (+1.08 dB).
- **Frame-level Wiener NS is not a consistent win.**  It helps 64×64 and chroma
  at 256×256, but is slightly negative on 640×360 luma and average.  A single
  least-squares frame filter is too coarse; use per-unit coefficients (or
  per-unit RDO) before claiming it.  The component is ready; the *selection*
  is not.
- **CCSO+Wiener** beats no-filter everywhere, but at 640×360 it is marginally
  *below* CCSO-only.  Do not enable Wiener by default until per-unit selection
  is measured.
- **CVVDP is effectively flat** (±0.01 JOD) because QP90 on these sources is
  already near the top of the scale.  Do not claim a perceptual win from these
  numbers; a lower-QP/content matrix is needed.
- **Rate cost matters.**  Band-3 CCSO signals ≤72 offsets per plane (nonzero
  counts in this run: 64×64 35/24/20, 640×360 16/14/17, 256×256 41/22/15).
  Even with entropy coding, the 64×64 stream is only 936 bytes, so the header is
  a large fraction of it; 640×360 (17.4 KB) and 256×256 (6.0 KB) are the
  plausible targets.  A real BD-rate number requires the stream gate.

## Blockers: PC-Wiener, GDF, film grain

These remain blocked because the normative tables are referenced but not
printed in the spec text (PDF or HTML).  Hand-typing them is forbidden by
`docs/negative-results.md`.

| Tool | Missing data (name in §9.8 / §7.21.7) |
|---|---|
| PC-Wiener | `pc_wiener_sub_classify2.h`, `pc_wiener_lut_to_class.h`, `pc_wiener_sub_classify.h`, `pc_wiener_filters.h` |
| GDF | `gdf_alpha.h`, `gdf_weight.h`, `gdf_bias.h`, `gdf_intra_error.h`, `gdf_inter_error.h` |
| Film grain | `gaussian_sequence.h` (`Gaussian_Sequence`, zero-mean σ≈512, clipped ±2048, multiples of 4) |

What *is* in the text and therefore implemented or available:
`Pc_Wiener_Config`, `Pc_Wiener_Normalizer`, `Mode_Weights`, `Mode_Offsets`,
`PC_WIENER_*` constants, the `get_box_features`/`get_qval_given_tskip`
algorithm, `Gdf_Coords`, and the full film-grain AR/blending process.  The
missing piece for each is the lookup/filter table, not the algorithm.  No table
values are guessed or copied from another encoder.

## Files

```text
src/cuda/m1/restoration/
  m1_restoration.h             stage PODs + API
  m1_restoration.cu            CCSO -> LR stage implementation
  m1_restoration_test.cu       CPU-vs-GPU parity (synthetic + real recon)
  real_recon_oracle_test.cpp   CPU reference vs independent oracle on real recon
  restoration_quality_bench.cpp  offline QP90 CCSO/Wiener quality probe
  build_and_test.sh            standalone nvcc build/run
  README.md                    this file
```

The stage intentionally does not include `src/cuda/m1/m1_gpu.h`, so it can be
built and tested before the orchestrator hook lands.
