# In-loop restoration — CCSO and Wiener loop restoration (CPU reference)

AV2 v1.0.0 (commit `38f28e7`) §7.19 CCSO and §7.20 loop restoration, written
from the spec text.  Scalar CPU reference plus a CUDA port that shares the
same arithmetic.  **No bitstream integration and no `dav2d` claim** — this
component is not wired into the M0/M1 encode path.

## Status

| Tool | Spec | CPU reference | CUDA kernel | Bit-exact test | Status |
|---|---|---|---|---|---|
| CCSO (cross-component sample offset) | §5.18.7.12, §6.17.7.8, §5.20.10.2, §7.19 | yes | yes | yes | **implemented** |
| Loop restoration framework (units, stripes, source sampling) | §5.18.7.11, §6.17.7.7, §5.20.10.4–§5.20.10.5, §7.20–§7.20.2 | yes | yes | yes | **implemented** |
| Non-separable Wiener (luma 32-tap, chroma 12-tap + 6 luma taps) | §5.20.10.6, §7.20.3 | yes | yes | yes | **implemented** |
| Frame-level Wiener with pixel classes | §5.18.7.11 `frame_filters_on`, §7.20.3 | partial | partial | yes | arithmetic done; the class map must be supplied (see blockers) |
| PC-Wiener (classification + filtering) | §7.20.4 | no | no | — | **TODO — normative tables missing** |
| GDF (guided detail filter) | §5.18.7.9, §7.20.5 | no | no | — | **TODO — normative tables missing** |
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

**Spec ambiguity (documented, not worked around silently).** §7.19.1 steps the
CCSO unit loop by `blkW/blkH` but the sample loops by `sbBlkW/sbBlkH`.  When the
CCSO unit is *smaller* than the superblock those sample loops overlap and a
sample is visited twice; the text does not say what that means.  Every
conforming configuration on 64×64 superblocks has unit ≥ superblock, so this
implementation accepts only that range and returns
`RestorationStatus::kUnsupportedCcsoUnitSize` otherwise (e.g. a 64×64 CCSO unit
with 128×128 superblocks).  If 128/256 superblocks are enabled later, resolve
the ambiguity against the reference decoder before lifting this check.

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
- lossless segments, BRU-inactive regions, tile extents, partial units.

`frame_filters_on` needs `FilterClass` from §7.20.4's pixel classifier.  The
classifier's LUT (`Pc_Wiener_Lut_To_Class`) is not in the spec text, so the
caller supplies the resulting subclass grid explicitly.  Without it the CPU
path returns `kMissingSubclassGrid` and the CUDA launcher returns
`cudaErrorInvalidValue`; nothing is silently approximated.

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
  ccso_test.cpp              CCSO CPU tests (2426 checks)
  wiener_test.cpp            Wiener CPU tests (107 checks)
  build_and_test.sh          standalone CPU build/test
  README.md                  this file
```

## Tests

```sh
src/encoder/filters/restoration/build_and_test.sh
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
  clamping, 0/255 clipping, partial units, 40 randomized frames (108,192
  samples).
- Wiener: zero-coefficient identity, per-unit filters, chroma-from-luma with
  all four `cfl_ds_filter_index` values, explicit stripe-boundary sourcing
  checks, two-class frame filters, mixed unit types, partial/odd frames, 2×2
  tile clipping, lossless/BRU, and 30 randomized frames (81,336 samples).

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
