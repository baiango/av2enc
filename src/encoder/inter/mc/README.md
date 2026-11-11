# Inter motion compensation — CPU reference (AV2 §7.13.3)

M2 component 12 (see `docs/agent-waves.md`). Implements single-reference
translational motion compensation for P-frames, the spatial subset of AV2 motion
vector prediction, and skip/merge + NEWMV/NEARMV decision hooks.

**No bitstream integration, no ME, no BD-rate claim.** This directory only
produces prediction blocks and resolves mode decisions; the integration pass
owns syntax and the ME component (`src/encoder/inter/me/`) owns the search.

Normative reference: AV2 Bitstream & Decoding Process Specification v1.0.0
(2026-05-28, commit `38f28e7`):

| Clause | Used for |
|---|---|
| §7.13.3.18 | Block inter prediction: sampling locations, subpel filters, clipping |
| §7.13.3.17 | MV-to-sample-location derivation (unscaled refs) |
| §7.12.2 | Find MV stack (spatial subset) |
| §5.20.7.13 | `assign_mv`: PredMvs + diffMvs, MV integer clamp |
| §5.20.9.4/5 | `clamp_mv_row` / `clamp_mv_col` picture bounds |
| §4.7 | `Round2`, `Round2Signed`, `Clip3` |
| §9.2 | `SUBPEL_BITS`, `SCALE_SUBPEL_BITS`, `REF_SCALE_SHIFT`, `MV_*`, stack sizes |

## Scope

Implemented:

- 8-bit 4:2:0 planar references, single reference per block.
- 8-tap EIGHTTAP / EIGHTTAP_SMOOTH / EIGHTTAP_SHARP / BILINEAR, plus the
  normative four-tap downgrade for a pass whose extent is <= 4 samples
  (4x4 luma, 2x2 chroma).
- All 16 subpel phases; 1/8-luma-sample input MVs (M2 baseline precision).
- Partial (edge) blocks: any rectangle inside the MI-aligned reference plane.
- Spatial MV stack: global MV, scan points, weight sorting, dedup, MVP
  combinations, picture-bound clamping, DRL indices.
- Skip/merge, NEARMV and NEWMV resolution, including an absolute-MV hook for the
  ME component.
- Encoder-side merge candidate scoring by SAD.

Not implemented (deliberately):

- Compound prediction, B frames, inter-intra, BAWP, TIP, OBMC.
- Global/local/extend warp, refine-MV, optical-flow refinement.
- Temporal MV projection, the derived-SMVP stack, the ref-MV bank, warp
  candidates. These slots are simply absent, equivalent to a frame with
  `use_ref_frame_mvs=0`, `enable_refmvbank=0`, `enable_warped_motion=0`.
- Reference scaling (`is_scaled() == 1`). References must match current-frame
  dimensions. Supporting it needs the current frame dimensions in the API and
  multiple intermediate-height budgets; out of M2 scope.
- 10-bit, 4:2:2/4:4:4, scaled/superres references, tiles.
- Any bitstream syntax and any rate/RDO decision.

## Files

```text
src/encoder/inter/mc/
  mc_types.h            POD types, constants, reference-frame view
  mc_subpel_tables.h    generated Subpel_Filters[6][16][8]
  gen_subpel_tables.py  extracts the table from the pinned spec PDF
  mc_core.h             shared host/device normative arithmetic
  mc.h                  CPU API
  mc.cpp                validation, two-pass prediction, MV stack, modes
  mc_test_common.h      synthetic/Y4M frame builders, guard bands
  mc_test.cpp           CPU tests (3,536 checks)
  build_and_test.sh     standalone build/run (no root CMake changes)
  README.md             this file
```

The CUDA kernel is in `src/cuda/inter/mc/`. It calls the same `mc_core.h`
arithmetic, so CPU and GPU output is bit-exact by construction.

## API

```cpp
namespace av2enc::inter::mc {

struct Mv { int32_t y, x; };                 // 1/8 luma sample
struct PlaneView { const uint8_t* data; int stride; };
struct RefFrame {
  PlaneView plane[3];
  int mi_cols, mi_rows;                      // luma MI grid; planes are MI-aligned
};
struct BlockRect { int x, y, w, h; };        // plane samples

// Normative prediction.
McStatus predict_plane(const RefFrame&, int plane, const BlockRect&,
                       Mv, InterpFilter, uint8_t* dst, int dst_stride);
McStatus predict_block_420(const RefFrame&, const BlockRect& luma_rect,
                           Mv, InterpFilter, MutablePlane dst[3]);

// MV prediction / mode hooks.
McStatus build_mv_stack(const MvStackParams&, MvStack*);
McStatus resolve_mode(const MvStack&, const MvStackParams&,
                      const ModeDecision&, ResolvedMode*);
Mv clamp_mv_to_picture(Mv, const MvStackParams&);

// Encoder-side merge choice.
McStatus select_merge_candidate(...);        // SAD over stack candidates
int64_t sad_block(...);
}
```

### Reference frame

`RefFrame` is a read-only 8-bit 4:2:0 view. Planes cover the MI-aligned extent
`mi_cols*4 x mi_rows*4` (chroma half). `stride` must be >= the plane width.
The spec clips taps to `(RefMiCols*4 >> sub_x) - 1`, so the MI-aligned padding
row/column must exist; decoders already pad reference frames this way.

### Prediction

`predict_plane` implements the two loops of §7.13.3.18 exactly:

1. horizontal 8-tap convolution per intermediate row, `Round2(s, 3)`, stored as
   `int16_t`;
2. vertical 8-tap convolution, `Round2(s, 11)`, then `Clip1` on the write.

For a pass whose extent is <= 4, the 8-tap regular/sharp filter is replaced by
`FOURTAP` and smooth by `FOURTAP_SMOOTH` (zeros in the other taps). This is the
normative §7.13.3.18 downgrade, not an approximation.

`predict_block_420` derives the chroma rectangle as `(x>>1, y>>1, w>>1, h>>1)`
and requires even luma geometry. The same MV is passed to all three planes; the
kernel applies `(2*mv) >> sub_x/sub_y` internally, which is the correct
1/16-chroma-sample mapping for 4:2:0.

### MV stack (spatial subset of §7.12.2)

`MvStackParams` supplies the current block (`mi_row`, `mi_col`, `bw4`, `bh4`),
the superblock height in MI units (`sb_h4`, 16 for 64x64), the MI grid of
neighbour mode info, and the reference slot. `MiInfo` stores both reference
lists so a future compound/B encoder can reuse the grid.

Implemented steps:

1. setup global MV (`global_mv` input; identity for M2);
2. scan points 7-13 with the superblock-border column alignment;
3. `num_nearest` snapshot;
4. col scan at `deltaCol = -3`;
5. weight sort of the nearest candidates (`DRL_REORDER_CONSTRAINT` default,
   sort when `num_nearest >= 4`);
6. pre-clamp, global-MV insertion with dedup, MVP combinations, final
   picture-bound clamp.

A block with no coded neighbours still yields one candidate: the global MV
(zero for M2). That makes skip/merge always signalable.

`MiInfo` contract: fill one entry per 4x4 luma MI cell of the already-coded
area. Set `ref[0] = kRefIntra` (0) for written intra blocks, `kRefNone` (-1) for
unwritten cells, and `is_inter = 1` only for inter blocks. `w4/h4`,
`col_base/row_base` are the block size/origin in MI units. Unwritten cells are
never scanned, so the encoder can fill the grid incrementally in raster order.

### Skip / merge, NEARMV, NEWMV

```cpp
ModeDecision d;
d.mode = InterMode::kSkipMerge;  // or kNearmv, kNewmv
d.drl_index = 0;                 // signaled DRL index
d.mvd = {3, -4};                 // NEWMV only
d.ref_frame = kRefLast;
d.filter = InterpFilter::kEighttap;
ResolvedMode r;
resolve_mode(stack, params, d, &r);
```

- `kSkipMerge`: MV = `RefStackMv[drl_index][0]`; the encoder additionally sets
  `skip_flag`/no residual. Index 0 is the normative skip/merge default; a
  non-zero index must be signaled as NEARMV/NEAR_NEARMV.
- `kNearmv`: same MV, no MVD.
- `kNewmv`: MV = predictor + MVD, then `mv_clamp_to_integer` (§5.20.7.13).
- `d.me_absolute = true`: the input `me_mv` is the desired absolute MV from the
  ME component; the hook derives `mvd = me_mv - pred`. This is the only place
  this module consumes ME output; it does not search.

`src/encoder/inter/me/` reports MVs in **1/16 luma samples**. AV2's signaled
MvPrecision is at most 1/8 (the M2 baseline is `MV_PRECISION_EIGHTH_PEL`), so
quantise first: `Mv mv = mv_from_me_1_16(me.mv_y, me.mv_x);`. The conversion is
encoder-side and rounds with `Round2Signed(.., 1)`; it is not a normative
process. If a later profile signals quarter- or half-pel MVs, quantise to that
precision instead and keep the MC input in the signaled units.

`select_merge_candidate` predicts luma for each stack entry and returns the
lowest-SAD DRL index. It is an encoder decision, not normative: the chosen index
must still be signaled, and index 0 is the default skip/merge candidate.

## Geometry and guard-band contract

- All rectangles are in the selected plane's samples and must lie inside the
  MI-aligned plane. Edge blocks pass their clipped width/height.
- The implementation never reads outside `[0, plane_w) x [0, plane_h)`; all
  taps are clipped to that region. It never writes outside `dst[0..h) x
  [0..w)`.
- `dst_stride >= w`; the destination may have guard columns/rows.
- Luma block size is limited to 64x64 (`kMaxBlockDim`); this covers the M2
  partition grid and bounds the intermediate buffer. 128/256 superblocks split
  into <=64 blocks before MC.

## Tests

```sh
./build_and_test.sh                 # build + run
./build_and_test.sh --no-run
./build_and_test.sh --check-tables  # verify generated table against the PDF
CXX=clang++ ./build_and_test.sh
```

3,536 checks, 0 failures. ASan + UBSan clean. Coverage:

1. **Table sanity**: 96 phases sum to 128, all coefficients even, phase 0 is a
   delta, and spot-checked phase 1/8/15 rows for all six filters match the PDF.
2. **Integer MV copy**: every filter and seven patterns reproduces the shifted
   reference exactly.
3. **CPU reference vs per-sample core**: the spec's two-pass intermediate form
   equals the per-sample form used by CUDA, for 11 block shapes (4x4 through
   64x64, rectangular), 9 subpel MVs and 4 filters.
4. **Bilinear hand check**: phase 8 is the 64/64 average with the exact
   `Round2(3)` then `Round2(11)` rounding.
5. **Partial blocks**: right/bottom edges, out-of-picture MVs, guard bands.
6. **4:2:0 chroma**: per-plane equivalence, 2x2 chroma (four-tap path), odd
   geometry rejected.
7. **MV stack**: scan order `A,B,D,E,C` then the col-scan `F`, weights
   `2,2,1,1,0,0`, `num_nearest=5`, offsets, dedup, global-MV fallback,
   other-reference candidates ignored, picture-bound clamp `{512,-384}`.
8. **Modes**: skip=stack[0], NEARMV=stack[drl], NEWMV=pred+mvd, ME-absolute mvd
   derivation, DRL/ref mismatch rejection.
9. **Merge selection**: a stack containing the true shifted MV and a wrong MV
   picks the true one with SAD 0.
10. **Validation**: null/out-of-plane/short-stride/128x128 rejection, invalid
    filter, full 17-bit MV range, and 8K sample-location derivation (64-bit).
11. **Real frame**: `tests/data/m0_640x360_1f.y4m` shifted by a known
    `(+2,-4)` integer displacement; all 240 luma and 240 chroma blocks match
    exactly; a fractional MV matches the per-sample core; true-MV SAD is 0.

Measured CPU reference time (oracle only, not a product number): 640x360
luma+chroma, 32x32 blocks, ~1.6–2.0 ms/frame on the dev host and the shared
Vast.ai Xeon (≈170–210 MP/s). Per `docs/architecture.md` this path is not
optimized.

## Integration notes

1. **Where it runs.** After ME/RDO chooses the mode and MV, before residual
   transform; after reconstruction the frame becomes the next reference.
2. **What the caller provides.** A `RefFrame` for the signaled reference slot,
   the block rectangle from the partition tree, the final MV (1/8 luma), the
   signaled interpolation filter, and the MI grid for MV prediction. Map the ME
   module's 1/16 result and its `ref_index` to the AV2 reference slot before
   calling.
3. **Mode decision order.** Build the stack once per block, resolve the mode,
   then predict with `r.mv`. For skip/merge, `drl_index == 0` needs no DRL
   syntax; other indices need the NEARMV path.
4. **Filter signaling.** `interp_filter` is a per-block syntax element in the
   switchable mode; the component consumes the chosen `InterpFilter` and applies
   the four-tap downgrade internally. Do not pre-downgrade.
5. **Reference update.** MC reads only; it never mutates references. Run loop
   filters and store the reconstructed frame before the next P frame.
6. **Not wired in.** Root `CMakeLists.txt`, `src/main.cpp`, entropy, and
   bitstream paths are untouched. No stream is produced by this component, so
   there is no `dav2d` gate yet. The integration pass owns that.
7. **Multi-reference.** The core is per-reference; a multi-reference encoder
   builds the stack with the chosen `ref_frame` and calls `predict_plane` with
   the matching `RefFrame`.

## Blockers / open items

- **No bitstream/decode gate yet.** Per the roadmap, M2 tools are `U1` until a
  minimal key+P stream decodes with `dav2d`; this component cannot claim that.
- **No ME.** Search quality is the dominant M2 gain, and it is component 11.
  This module only resolves what ME/RDO provides. ME emits 1/16-pel MVs; MC
  takes 1/8-pel, so use `mv_from_me_1_16()` (or the signaled precision) and do
  not pass 1/16 values directly.
- **AV2 reference-frame decoders use scaled references for superres.** This
  module rejects/un-supports scaling by design; if superres is enabled later,
  extend the API with current-frame dimensions and re-derive `startX/startY`.
- **Stack subset.** Temporal projection, ref-MV bank, and warp candidates are
  absent. Enabling any of them changes the signaled DRL list, so integrate
  before relying on those tools.
- **No GLOBALMV neighbour substitution.** §7.12.2 `get_mv` returns `GlobalMvs`
  for neighbours coded GLOBALMV; M2 excludes global motion, and `MiInfo` stores
  the plain sub-block MV. Adding global motion requires a mode field in
  `MiInfo` and a per-reference global-MV table.
- **Per-block CPU allocation-free but scalar.** The CPU path is intentionally
  single-threaded and has no SIMD; it is the oracle, not the encoder.
- **Edge conformance detail.** `MiInfo` requires the encoder to write
  `ref[0]=kRefIntra` for intra cells. If the integration forgets this, unwritten
  cells are treated as unavailable, which silently drops valid neighbours.
