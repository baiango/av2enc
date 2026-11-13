# M1 post-CDEF restoration stage — CCSO + Wiener NS

Workstream: in-loop restoration for the M1 GPU path, owned by
`src/cuda/pipeline/still/restoration/`.  It consumes the deblocked reconstruction, applies
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
#include "cuda/pipeline/still/restoration/still_restoration.h"

using namespace av2enc::cuda::still;

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

1. **Sequence header** — `src/encoder/m1/still_headers.cpp` (currently
   `enable_restoration = 0`, `enable_ccso = 0` around lines 148–149).  Emit the
   full `sequence_filter_config()` row above.  M1 has no GDF/CDEF yet, so those
   stay 0.
2. **Uncompressed frame header** — same file: add `ccso_params()` and
   `lr_params()` with `single_picture_header_flag` semantics.  For an intra
   still, `reuse_ccso`/`sb_reuse_ccso` are 0 and `temporal_pred_flag` is 0.
3. **Tile syntax / entropy** — `src/encoder/m1/still_syntax.cpp` and
   `src/encoder/m1/stream/`: write `ccso_blk S()`, `use_wiener_ns S()`, the
   `read_wienerns_filter()` symbols and the LR/CCSO offset symbols with their
   §8.3 CDFs.  `read_lr()` is invoked from the block walk, not once per frame.
   Entropy stays CPU-resident, matching the current M1 design.
4. **Orchestrator** — `src/cuda/pipeline/still/still_gpu.h`: add `Stage::kRestoration`
   after `kDeblock`; `src/cuda/pipeline/still/still_gpu.cu`: allocate (from the arena) a
   `curr` copy of the deblocked planes, a `cdef` set (a copy of deblocked when
   CDEF is off), and an `lr` set; upload the CCSO offset table and all grids;
   call `restoration_stage_apply`; use `lr` as the final recon/reference
   output.  Do not alias `curr`/`cdef` while CCSO is on.
5. **Build** — root `CMakeLists.txt` GLOBs `src/cuda/pipeline/still/**/*.cu`, so
   `still_restoration.cu` is picked up automatically, but three dependencies are
   not:
   - `src/cuda/restoration/ccso.cu` and `src/cuda/restoration/wiener.cu` are
     not in `AV2ENC_STILL_GPU_SUPPORT_SOURCES`;
   - `src/encoder/filters/restoration/restoration.cpp` is not in
     `av2enc_core`, yet the stage calls its validators and
     `ccso_header_conforms()` (the CUDA component tests link it explicitly).

   Add the two `.cu` files to the support list and `restoration.cpp` to the
   core sources (owner: agent 9), or link an equivalent object library.
   Without that the M1 GPU target will fail to link.
6. **Gates** — enabling filters changes the pinned M1 MD5s.  Add one stream per
   tool: CCSO-only, then +Wiener.  Each must pass
   `tools/validate_dav2d.sh`, and `tools/still_verify` must show decoded ==
   encoder recon for Y/U/V.  Keep the old no-filter MD5 as the regression pin.

### Standalone minimal stream gate

Not feasible from this directory alone.  A gate needs (a) sequence/frame header
flags, (b) tile symbols and CDFs, and (c) writer plumbing.  Those live in
`src/encoder/m1/still_headers.cpp`, `src/encoder/m1/still_syntax.cpp`,
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
# 1. produce real QP90 reconstructions.  The pair (no CDEF, CDEF) is the real
#    deblock -> CDEF -> CCSO -> Wiener chain: `pre` is post-deblock, `post` is
#    post-CDEF.
build-cuda/av2enc tests/data/basic_64x64_1f.y4m   -o /tmp/pre64.ivf  --still
build-cuda/av2enc tests/data/basic_64x64_1f.y4m   -o /tmp/post64.ivf --still --still-cdef
build-cuda/av2enc tests/data/basic_640x360_1f.y4m -o /tmp/pre360.ivf --still
build-cuda/av2enc tests/data/basic_640x360_1f.y4m -o /tmp/post360.ivf --still --still-cdef

# 2. CPU-vs-GPU parity (synthetic + single-frame + real CDEF chain)
cd src/cuda/pipeline/still/restoration
RUN_ARGS="--ivf /tmp/pre64.ivf --ivf-pair /tmp/pre360.ivf /tmp/post360.ivf" \
  ./build_and_test.sh

# Or run steps 1+2 together:
BIN=build-cuda/av2enc DAV2D=tools/bin/dav2d ./run_real_chain_gate.sh
```

The test decodes the IVF with `tools/bin/dav2d` (override with
`--dav2d PATH`).  `--ivf` uses the decoded frame as `curr = cdef` (CDEF off).
`--ivf-pair PRE POST` fills `curr = PRE` (post-deblock) and `cdef = POST`
(post-CDEF), so CCSO reads the real pre-CDEF luma and the whole normative chain
is exercised.

**Status (RTX 3060, `sm_86`, CUDA 13.0, 2026-11-13):** executed and passing.

```
synthetic (validation + randomized CCSO/LR):  PASS: 278 checks, 0 failures
+ real deblock->CDEF pairs (64x64, 640x360):   PASS: 448 checks, 0 failures
+ QP1 pairs (64x64, 640x360):                  PASS: 448 checks, 0 failures
```

`run_real_chain_gate.sh` runs the QP90 pair set by default; `QP=1` selects the
near-lossless fixtures.  Both were run through `tools/bin/dav2d`.

CPU suite status:

```
CCSO:   PASS: 5518 checks, 0 failures   (ASan+UBSan clean)
Wiener: PASS:  114 checks, 0 failures   (ASan+UBSan clean)
real recon (64x64, 640x360, 256x256, per-unit + frame filters):
        PASS: 36 checks, 0 failures
real deblock+CDEF pair (64x64 + 640x360, per-unit + frame filters):
        PASS: 24 checks, 0 failures
```

The real-recon CPU/oracle check is `real_recon_oracle_test.cpp`; it compares the
production CPU path against the independent spec-order oracle in
`restoration_test_common.h` on decoded QP90 frames.  Build/run:

```sh
c++ -std=c++20 -O2 -I src \
  src/cuda/pipeline/still/restoration/real_recon_oracle_test.cpp \
  src/encoder/filters/restoration/restoration.cpp \
  -o /tmp/real_recon_oracle_test
# single recon, or the real deblock+CDEF pair:
/tmp/real_recon_oracle_test /tmp/post64.y4m /tmp/post360.y4m
/tmp/real_recon_oracle_test --pair /tmp/pre64.y4m /tmp/post64.y4m \
                            --pair /tmp/pre360.y4m /tmp/post360.y4m
```

The pair mode was run on this host against 64×64 and 640×360 QP90:
`PASS: 24 checks, 0 failures` (the production CPU path matches the independent
naive oracle for both per-unit and two-class frame filters on the real
post-CDEF frame).

## Quality delta at QP 90

`restoration_quality_bench.cpp` is an encoder-side oracle experiment.  It reads
a source Y4M plus the decoder's reconstruction of the same QP90 frame, trains
CCSO offsets and frame-level Wiener coefficients on that pair, applies the
filters in normative order, and measures PSNR.  With `--recon-pre` it uses the
real pre-CDEF frame for CCSO/§7.20 sourcing, so the measured chain is
deblock→CDEF→CCSO→Wiener.  The optional `--cost` report links the M2 adapters
and prints exact header + symbol bits.

Build (the `--cost` report additionally links the M2 restoration adapters and
the M1 syntax module):

```sh
c++ -std=c++20 -O2 -I src \
  src/cuda/pipeline/still/restoration/restoration_quality_bench.cpp \
  src/encoder/filters/restoration/restoration.cpp \
  src/encoder/m2/restoration/video_restoration.cpp \
  src/encoder/m1/filters_stream/filters_stream.cpp \
  src/entropy/av2/av2_entropy.cpp \
  src/bitstream/bitwriter.cpp \
  -o /tmp/restoration_quality
```

Inputs (all QP90, current build):

| Name | Source | Recon |
|---|---|---|
| 64×64 | `tests/data/basic_64x64_1f.y4m` | `dav2d` of `--still` and `--still --still-cdef` |
| 640×360 | `tests/data/basic_640x360_1f.y4m` | `dav2d` of `--still` and `--still --still-cdef` |
| 256×256 | `testdata/shine_256x256_1f.y4m` | `dav2d` of `--still` and `--still --still-cdef` |

`--max-band-log2 3` = 8 bands × 3×3 classes = 72 offsets/plane.

Measured on the real deblock→CDEF chain (`--recon-pre` supplies the pre-CDEF
frame for CCSO/§7.20 sourcing, `--recon` is the CDEF decode).  avg is
sample-weighted over the 4:2:0 planes.

| Case | none Y/U/V (avg) | CCSO Y/U/V (avg) | Wiener Y/U/V (avg) | CCSO+Wiener Y/U/V (avg) |
|---|---|---|---|---|
| 64×64 | 42.58 / 45.80 / 45.12 (43.54) | **42.78 / 46.26 / 45.37 (43.80)** | 42.89 / 46.09 / 46.52 (44.02) | **42.75 / 47.39 / 46.92 (44.22)** |
| 640×360 | 45.24 / 52.47 / 50.68 (47.35) | **45.41 / 52.52 / 50.70 (47.47)** | 45.27 / 52.85 / 51.31 (47.54) | **45.41 / 52.81 / 51.32 (47.63)** |
| 256×256 | 47.40 / 51.46 / 50.01 (48.51) | **47.58 / 51.59 / 50.08 (48.67)** | 47.32 / 51.83 / 50.90 (48.67) | **47.48 / 51.92 / 50.92 (48.79)** |

Bit cost, two independent measurements:

1. Real integrated CCSO stream (M1 QP90).  Without CDEF: `--still` → `--still-ccso`
   is 936 → 970 B (+34 B), 13,355 → 13,397 B (+42 B), 5,336 → 5,373 B
   (+37 B).  Inside the CDEF chain: `--still --still-cdef` → `+ --still-ccso` is
   938 → 972 B (+34 B), 13,375 → 13,415 B (+40 B), 5,349 → 5,384 B (+35 B).
   This is the real cost of the sequence flag, `ccso_params()` and the per-unit
   `read_ccso()` symbols.  M1's default is band-offset-only with 64 bands; the
   fitted M2 row below is edge-classified with `max_band_log2 = 3`.
2. M2 syntax cost from the fitted parameters (exact header bits + standalone
   `read_ccso()` / `read_lr()` symbol bytes up to coder termination, via
   `av2enc::video::restoration::measure_restoration_cost`):

| Case | CCSO | Wiener NS | both |
|---|---:|---:|---:|
| 64×64 | 51 B | 28 B | 79 B |
| 640×360 | 55 B | 33 B | 88 B |
| 256×256 | 53 B | 25 B | 78 B |

The header dominates; a 936-byte 64×64 stream cannot absorb a ~79-byte filter
header.  This is still not a BD-rate number: the M2 core tile writer has no
restoration hook (see `src/encoder/m2/restoration/README.md`), so CCSO's real
cost above comes from the M1 stream gate and the Wiener cost is syntax-only.

**Verdict.**

- **CCSO is a consistent PSNR win** at QP90 on the real post-CDEF frame:
  +0.21 / +0.17 / +0.18 dB luma and +0.26 / +0.13 / +0.15 dB average at band 3.
- **Frame-level Wiener NS is not a consistent luma win.**  It adds chroma
  (+0.46 dB U at 64×64, +0.38 dB at 640×360) and is −0.08 dB luma at 256×256.
  A single least-squares frame filter is too coarse; use per-unit coefficients
  (or per-unit RDO) before enabling it by default.  The component is ready; the
  *selection* is not.
- **CCSO+Wiener** beats no-filter on the average metric, but its luma is flat
  to slightly negative versus CCSO-only (−0.03 / +0.00 / −0.10 dB); the extra
  average gain is chroma.  Do not enable Wiener for luma by default until
  per-unit selection is measured.
- **Rate cost matters.**  Band-3 CCSO signals ≤72 offsets per plane (nonzero
  counts: 64×64 35/21/18, 640×360 42/8/3, 256×256 41/18/13).  The 640×360 and
  256×256 streams are the plausible targets; 64×64 is header-dominated.

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
src/cuda/pipeline/still/restoration/
  still_restoration.h             stage PODs + API
  still_restoration.cu            CCSO -> LR stage implementation
  still_restoration_test.cu       CPU-vs-GPU parity (synthetic + real recon)
  real_recon_oracle_test.cpp   CPU reference vs independent oracle on real recon
  restoration_quality_bench.cpp  offline QP90 CCSO/Wiener quality + cost probe
  build_and_test.sh            standalone nvcc build/run
  run_real_chain_gate.sh       encode/decode real CDEF pair + run parity
  README.md                    this file
```

The stage intentionally does not include `src/cuda/pipeline/still/still_gpu.h`, so it can be
built and tested before the orchestrator hook lands.
