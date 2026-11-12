# In-loop restoration — CCSO and Wiener loop restoration (CPU reference)

AV2 v1.0.0 (commit `38f28e7`) §7.19 CCSO and §7.20 loop restoration, written
from the spec text.  Scalar CPU reference plus a CUDA port that shares the
same arithmetic.

**Integration status (2026-11-12).**  Per-unit non-separable Wiener NS is now
wired into the opt-in CPU still path: `--still-wiener` (luma) and
`--still-chroma-lr` (U/V), applied after CDEF/CCSO.  Both are gated by
`tools/still_verify/loop_restoration_gate.sh` with `dav2d --strict 1` and a
sample-for-sample encoder-recon compare at 64x64 and 640x360.  The CUDA
restoration kernel is still unwired, and PC-Wiener/GDF/film grain remain
unintegrated (see blockers below).

## Status

| Tool | Spec | CPU reference | CUDA kernel | Bit-exact test | Status |
|---|---|---|---|---|---|
| CCSO (cross-component sample offset) | §5.18.7.12, §6.17.7.8, §5.20.10.2, §7.19 | yes | yes | yes | **implemented** |
| Loop restoration framework (units, stripes, source sampling) | §5.18.7.11, §6.17.7.7, §5.20.10.4–§5.20.10.5, §7.20–§7.20.2 | yes | yes | yes | **implemented** |
| Non-separable Wiener (luma 32-tap, chroma 12-tap + 6 luma taps) | §5.20.10.6, §7.20.3 | yes | yes | yes | **implemented** |
| Frame-level Wiener with pixel classes | §5.18.7.11 `frame_filters_on`, §7.20.3 | partial | partial | yes | arithmetic done; the class map must be supplied (see blockers) |
| PC-Wiener (classification + filtering) | §7.20.4 | no | no | — | reference in `../spec_tables/`; stream integration pending |
| GDF (guided detail filter) | §5.18.7.9, §7.20.5 | no | no | — | reference in `../spec_tables/`; stream integration pending |
| Film grain synthesis | §5.14, §7.21.7 | no | no | — | **TODO — Gaussian table missing** |

Nothing here has been decoded with `dav2d`; the filters have no syntax path yet.
Per `docs/tool-roadmap.md` these are M3/M4 tools and each needs its own minimal
stream before any quality claim.

## Scope and data model

- 8-bit 4:2:0 planar.  Other bit depths and subsampling return an error instead
  of being approximated.
- `RestorationFrame` carries three plane views each for `curr` (deblocked,
  pre-CDEF), `cdef` (post-CDEF/CCSO; CCSO modifies it in place) and `lr` (the
  loop-restoration output, never aliasing `cdef`).
- The MI grid (`mi_cols`, `mi_rows`) defines plane extents exactly as the spec
  does: `MiCols * 4 >> subX` by `MiRows * 4 >> subY`.
- Optional metadata, all luma MI grids: `segment_ids`, `chroma_segment_ids`,
  `lossless[8]`, `bru_modes`, and the `MiColStart/End`, `MiRowStart/End` tile
  grids.  `nullptr` means "absent / single tile / not lossless / BRU active".
- `disable_loopfilters_across_tiles` switches the allowed filter extent from the
  frame to the tile, as §7.19.1 and §7.20.1 require.  When set, the four tile
  grids must be supplied.
- `sb_width`/`sb_height` are only used to re-derive the BruModes lookup
  position (`sbBlkW`/`sbBlkH`).

## CCSO

```cpp
#include "encoder/filters/restoration/restoration.h"
using namespace av2enc::restoration;

CcsoParams& p = params.plane[0];
p.enabled = true;
p.bo_only = false;
p.scale_idx = 2; p.quant_idx = 1; p.ext_filter = 4; p.edge_clf = 0;
p.max_band_log2 = 6;
// Fill the offset table from the ccso_offset_idx values in syntax order.
std::uint8_t indices[ccso_offset_count(p)];
ccso_set_offsets(p, indices, sizeof indices);      // offset = Ccso_Offset[i] * (scale+1)

params.blks[0] = ccso_blk_grid;                    // one byte per CCSO unit
params.blk_cols[0] = ...; params.blk_rows[0] = ...;

RestorationStatus st = ccso_frame(frame, params);
```

Implemented details:

- band-offset-only (`ccso_bo_only`) and edge-classified modes;
- `ccso_score()` with `ccso_edge_clf` 0/1/2 and every `CCSO_Quant_Sz` entry
  (including `quantStep == 0`);
- all 7 `Ccso_Pos` extension filters;
- luma taps read from `CurrFrame[0]` (pre-CDEF), clamped to the allowed extent,
  for every plane (`xLuma = x3 << subX`);
- `CcsoBlks[plane]` unit grids, per-plane enable, lossless segments,
  BRU-inactive superblock regions;
- clipped writes to `[0, 2^BitDepth - 1]`;
- partial units at the right/bottom plane edge.

**CCSO unit size.** §5.18.7.12 derives `CcsoLumaSizeLog2` from
`ccso_unit_matches_sb_size` or the tile alignment (`8` for 64-aligned tile
starts, `7` for 32-aligned, else `6`).  Use
`ccso_select_luma_size_log2(sb_size, tile_start_or, matches_sb)` and
`ccso_blk_grid_size(...)` instead of hard-coding 64.  A unit *smaller* than the
superblock is legal: §7.19.1's sample loops step by `sbBlkW/sbBlkH`, so when
`blkW < sbBlkW` the inner loop runs once per unit and every sample is still
visited exactly once.  This was previously rejected; it is now supported and
tested.

**Header-representable values.** `ccso_max_band_log2` is `f(2 + bo_only)`:
0–3 for edge-classified CCSO and 0–6 for `bo_only`.  `ccso_edge_clf` is `f(1)`,
i.e. 0 or 1.  The filter core accepts a wider range for diagnostics; a real
syntax writer must call `ccso_header_conforms()`.

## Loop restoration / Wiener

```cpp
params.frame_type[0] = kRestoreWienerNs;         // kRestoreNone / kRestorePcWiener / kRestoreSwitchable
params.unit_size[0]  = 64;                       // LoopRestorationSize[plane]
params.unit_rows[0]  = restoration_count_units(64, frame.mi_rows * 4);
params.unit_cols[0]  = restoration_count_units(64, frame.mi_cols * 4);
params.unit_type[0]  = lr_type_grid;             // LrType[plane][unitRow][unitCol]
params.unit_coeffs[0] = lr_coeffs;               // 18 int16 per unit
loop_restoration_frame(frame, params);           // writes frame.lr
```

Implemented details:

- the normative block walk over the *luma* MI grid for every plane, with
  chroma blocks of 2×2 samples for 4:2:0;
- stripes offset 8 luma lines upwards (`stripeNum = (lumaY + 8) / 64`), sample
  sourcing from `CdefFrame` inside the stripe and `CurrFrame` outside, with the
  two-line clamp;
- luma `Wiener_Ns_Config_Y` (32 taps) and chroma `Wiener_Ns_Config_Uv`
  (12 taps) plus the six chroma-from-luma coefficient groups;
- `cfl_ds_filter_index` including the `Wiener_Filters_420` 2×2 downsampling
  (index 3 aliases 0);
- per-unit coefficients (`frame_filters_on = 0`) and per-class frame
  coefficients (`frame_filters_on = 1`, class map supplied);
- `LrFrame := CdefFrame` copy, so `RESTORE_NONE` units and planes pass through
  byte-exactly;
- lossless segments, BRU-inactive regions, tile extents, partial units;
- §5.18.7.11 unit-size conformance: power of two, at most
  `RESTORATION_TILESIZE_MAX` (512 luma / 256 chroma for 4:2:0), at least one
  stripe (`64 >> SubsamplingY` for chroma), `LoopRestorationSize[2] == [1]`,
  and §6.17.7.7 `check_ru_size()` against supplied tile grids.

`frame_filters_on` needs `FilterClass` from §7.20.4's pixel classifier.  The
classifier's LUT (`Pc_Wiener_Lut_To_Class`) is not in the spec text, so the
caller supplies the class map explicitly:

- preferred: `filter_class_grid` (luma MI grid) plus `subclass_lookup`;
  §7.20.3 computes `subcls = SubclassLookup[FilterClass[y>>2][x>>2]]`.
  `FilterClass` is `Pc_Wiener_Lut_To_Class` output, whose range is
  `NUM_PC_WIENER_LUT_CLASSES == 256` (not 16), so set
  `filter_class_count = 256` and supply 256 lookup entries.  The historical
  16-entry grid/lookup remains the default and is what existing callers use;
- legacy: `subclass_grid` already holds `subcls` per 4×4 cell.

Without either grid the CPU path returns `kMissingSubclassGrid` and the CUDA
launcher returns `cudaErrorInvalidValue`; nothing is silently approximated.

## Encoder-side search (`classes/`)

The encoder needs decisions, coefficient fits and exact syntax, not just the
decoder filter core.  `classes/` adds that layer without changing the tested
CPU reference:

| File | Purpose |
|---|---|
| `wiener_tables.h` | spec constants, `Wiener_Ns_Taps_Present`, `Shuffled_Index`, `Mode_Weights/Offsets`, `get_q` (§7.14.2), `Encode/Decode_Num_Filter_Classes` |
| `wiener_cdfs.h`, `wiener_qlookup.h` | generated from `spec/tables/default_cdf.json` / `quant.json` (`generate_classes_tables.py`); never hand-edited |
| `wiener_class.*` | §7.20.4 `FilterClass` derivation and `SubclassLookup` mapping.  The algorithm is implemented; `Pc_Wiener_Lut_To_Class` / `Pc_Wiener_Sub_Classify2` are now generated in `../spec_tables/` (recovered from the pinned PDF and hash-pinned attachments) and are caller-supplied |
| `wiener_fit.*` | stripe-aware least-squares fit of the 16 luma / 18 chroma coefficients, per-subset and `uv_sym` variants, deterministic quantisation and clipping |
| `wiener_syntax.*` | exact §5.18.7.11 header order and §5.20.10.4–.6 tile order, including `search_frame_filters` group coding, `wiener_ns_length`, bank merges, CDF adaptation, temporal frame-filter prediction, and `wiener_write_tile` in `read_lr()` order |
| `wiener_decision.*` | per-unit vs frame-level selection, `use_wiener_ns`/`LrType`, unit-size search, class merging, RD cost from exact syntax bits |

### What is available without the missing tables

- `NumFilterClasses == 1`: fully available, including frame-level luma/chroma
  filters and per-unit filters.
- `NumFilterClasses > 1`: the search refuses unless the caller supplies a
  256-entry `Pc_Wiener_Lut_To_Class` and the
  `Pc_Wiener_Sub_Classify2[4][8][256]` table.  `wiener_derive_filter_class_grid`
  and `wiener_build_subclass_grid` return `kMissingNormativeTable` otherwise,
  and `wiener_decide` records a note and falls back to one class.
- PC-Wiener itself remains unimplemented (also needs `Pc_Wiener_Sub_Classify`
  and `Pc_Wiener_Filters`).  The syntax writers reject a PC-Wiener selection
  rather than emit a stream that cannot be reproduced.

### Decisions and integration

```cpp
#include "encoder/filters/restoration/classes/wiener_decision.h"

WienerDecisionInput in;
in.core = lr_core;                       // curr/cdef/dst + MI geometry
in.target[0] = source_y;                 // encoder source picture per plane
in.target_stride[0] = source_y_stride;   // (same for U/V)
in.base_q_idx = base_q_idx;
in.tables = class_tables;                // nullptr -> one class only
in.lr_tx_skip = lr_tx_skip_grid;         // only needed for multi-class
in.refs = &reference_set;                // nullptr for intra

WienerDecisionOptions options;
options.lambda_bits = rd_lambda;         // 0 = pure distortion
WienerDecision decision;
wiener_decide(in, options, &decision);

// Feed the restoration core:
//   params.frame_type / unit_size / unit_rows/cols / unit_type / unit_coeffs
//   params.frame_filters_on / frame_coeffs / class maps
// then call loop_restoration_frame().
```

Exact call sites in the existing tree:

1. **Sequence header** — `src/encoder/m2/restoration/video_restoration.cpp`
   `make_sequence_filter_config()` already disables luma PC-Wiener
   (`lr_disable_luma_pc_wiener = true`), which is the precondition for
   `wiener_write_lr_params`.
2. **Frame header** — `src/encoder/m2/video_headers.cpp`
   `write_frame_header_syntax()`: after the deblocking/delta-q fields and
   before `tx_mode_select` (the same slot as
   `av2enc::video::restoration::write_lr_params`).  M1's
   `src/encoder/m1/filters_stream/filters_stream.cpp`
   `write_lr_params()` is the equivalent call site.  Use
   `wiener_write_lr_params()` when temporal prediction or
   `search_frame_filters` ref groups are needed; the M1 adapter only supports
   intra.
3. **Tile** — per superblock, before `read_partition()`:
   `wiener_write_tile()/wiener_read_tile()` (or `wiener_write_unit()` per unit
   in `read_lr()` order).  The M2 adapter's `write_read_lr()` /
   `write_read_lr_unit()` are the existing wrappers; map `WienerUnitPlan` to
   `filters_stream::LrUnitDecision` field-for-field (`type`, `merged`,
   `use_bank_k`, `subset`, `uv_sym`, `coeffs`).
4. **Apply** — after CCSO:
   `loop_restoration_frame(frame, params_from_decision)` in
   `src/encoder/m2/restoration/video_restoration.cpp` `apply_lr_frame()`.

The decision's `sse` is the exact §7.20 core SSE for the chosen plan; the tests
assert that `loop_restoration_frame()` reproduces it and that a header+tile
write→read round-trip produces the same recon.

## Blockers (why PC-Wiener / GDF / film grain are TODO)

§9.8 of the specification does not print these tables; it says they are
"defined in the … header file", and those headers are not part of the text
(PDF/HTML) or of `spec/tables/` in this repo.  Hand-typing them is exactly what
`docs/negative-results.md` forbids, so the tools are left unimplemented:

| Tool | Missing normative data |
|---|---|
| PC-Wiener | `pc_wiener_filters.h` (`Pc_Wiener_Filters[64][?][13]`), `pc_wiener_sub_classify.h`, `pc_wiener_sub_classify2.h`, `pc_wiener_lut_to_class.h` (4096-entry LUT) |
| GDF | `gdf_alpha.h` (`Gdf_Alpha[6][3][22][4]`), `gdf_weight.h`, `gdf_bias.h`, `gdf_intra_error.h`, `gdf_inter_error.h` |
| Film grain | `gaussian_sequence.h` (`Gaussian_Sequence[2048]`) |

Everything else that film grain needs (`ScalingLut` interpolation, AR
coefficient filter, noise-stripe blending) is plain text in §7.21.7 and can be
implemented once the Gaussian table is available.  Film grain is output-only
and must be measured perceptually, never with PSNR alone.

`gdf_frame_enable` is rejected by both the CPU (`kUnsupportedGdf`) and the CUDA
launcher (`cudaErrorNotSupported`), so an integration mistake fails loudly.

## Files

```text
src/encoder/filters/restoration/
  restoration_common.h       constants, Round2/Clip3, Wiener and CCSO tables
  ccso_core.h                §7.19.1 shared host/device core
  wiener_core.h              §7.20–§7.20.3 shared host/device core
  restoration.h              public CPU API (frames, params, status)
  restoration.cpp            CPU reference implementation
  restoration_test_common.h  test frames + independent spec-transcription oracles
  ccso_test.cpp              CCSO CPU tests (5,518 checks)
  wiener_test.cpp            Wiener CPU tests (114 checks)
  build_and_test.sh          standalone CPU build/test
  classes/                   encoder-side search, decisions and syntax
    wiener_tables.h          spec constants + generated helper declarations
    wiener_cdfs.h            generated §9.3 CDFs
    wiener_qlookup.h         generated §7.14.2 qlookup
    wiener_class.h/.cpp      §7.20.4 classification (missing LUT gated)
    wiener_fit.h/.cpp        least-squares Wiener-NS coefficient search
    wiener_syntax.h/.cpp     lr_params/read_lr_unit/read_wienerns_filter
    wiener_decision.h/.cpp   per-unit vs frame-level selection
    wiener_classes_test.cpp  fit/class/decision/real-recon tests
    wiener_syntax_test.cpp   pinned order + CDF round-trip tests
    generate_classes_tables.py
    build_and_test.sh        standalone CPU build/test
  README.md                  this file
```

## Tests

```sh
src/encoder/filters/restoration/build_and_test.sh
src/encoder/filters/restoration/classes/build_and_test.sh
```

Both suites cross-check the reference against a second, independent
transcription of the spec pseudocode (`restoration_test_common.h`), on random
and hand-built frames, and verify that guard columns/rows are never touched.
ASan/UBSan-clean:

```sh
c++ -std=c++20 -O1 -g -fsanitize=address,undefined -I src \
  src/encoder/filters/restoration/restoration.cpp \
  src/encoder/filters/restoration/ccso_test.cpp -o /tmp/ccso_asan && /tmp/ccso_asan
```

Coverage highlights:

- CCSO: `bo_only` × 6 band counts, edge classes × 7 extension filters, all 16
  `(scale_idx, quant_idx)` pairs, per-plane unit grids, lossless, BRU, tile
  clamping, 0/255 clipping, partial units, unit < superblock, header
  representability, 40 randomized frames (108,192 samples).
- Wiener: zero-coefficient identity, per-unit filters, chroma-from-luma with
  all four `cfl_ds_filter_index` values, explicit stripe-boundary sourcing
  checks, two-class frame filters, mixed unit types, partial/odd frames, 2×2
  tile clipping, lossless/BRU, and 30 randomized frames (81,336 samples).
- Encoder search (`classes/`): synthetic and real (Y4M) recon, luma/chroma
  fits, `uv_sym` pairs, deterministic repeats, missing-table refusal,
  256-class lookup extension, per-unit and frame-level decisions, mixed unit
  sizes, decision-SSE vs `loop_restoration_frame()` equality, and header+tile
  write→read→filter round-trips.  `wiener_syntax_test` pins the `ns(n)` and
  `lr_params()` field order, checks `merged`-before-coefficients ordering, the
  `search_frame_filters` ref groups, temporal frame-filter prediction, and
  byte-level CDF-state round-trips.

CPU mode is a correctness oracle; per `docs/architecture.md` it is not
benchmarked or tuned.  GPU timings are printed by
`src/cuda/restoration/build_and_test.sh` on the target box.

## Integration contract (for the M3/M4 pass)

1. Parse §5.18.7.12 / §5.18.7.11 / §5.20.10.2–§5.20.10.6 and fill
   `CcsoFrameParams` / `LoopRestorationParams`.
2. Run filters in the normative order: reconstruct → deblock → CDEF → **CCSO**
   → loop restoration (Wiener/PC-Wiener/GDF) → output.
3. Feed the *pre-CDEF* deblocked planes as `curr` and the post-CDEF/CCSO planes
   as `cdef`; keep `lr` separate.
4. Update the reference frame from `lr` (luma), and remember film grain is
   output-only.
5. Gate with `tools/validate_dav2d.sh`; compare encoder recon vs decoder Y4M
   with grain disabled.  No BD-rate number before that passes.
