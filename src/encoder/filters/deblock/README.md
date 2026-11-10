# Deblocking filter — CPU reference (AV2 §7.17)

M1 component 6 (see `docs/agent-waves.md`).  Implements the normative AV2
deblocking filter for 8-bit 4:2:0 reconstructed frames.  **No bitstream
integration**: this directory only filters planes and reports decisions.

Normative reference: AV2 Bitstream & Decoding Process Specification v1.0.0
(2026-05-28, commit `38f28e7`), §7.17 (general), §7.17.2 (edge process),
§7.17.3 (maximum width), §7.17.4 (filter size), §7.17.5/§7.17.6 (adaptive
strength), §7.17.7 (filter choice + sample filtering).

## Files

```text
src/encoder/filters/deblock/
  deblock_core.h         shared host/device arithmetic + normative tables
  deblock.h              CPU API (planes, metadata, options)
  deblock.cpp            validation + traversal + inspection entry points
  deblock_test_common.h  synthetic frames, metadata builders, Y4M loader
  deblock_test.cpp       CPU tests (562 checks)
  build_and_test.sh      standalone build/run (no root CMake changes)
  README.md              this file
```

The CUDA kernel is in `src/cuda/deblock/` and calls the same
`deblock_core.h` functions, so CPU and GPU output are bit-exact.

## Filter lengths

AV2 §7.17.3 does not use AV1's "4/8/14-tap" names; it derives
`maxWidthPos`/`maxWidthNeg` from the transform size:

| `filterSize` | class | luma `maxWidthPos` | chroma `maxWidthPos` |
|---:|---|---:|---:|
| `<= 4` | 4-tap | 1 | 1 |
| `== 8` | 8-tap | 3 | 3 |
| `== 16` | 14-tap | 6 | 4 |
| `> 16` | wide | 8 | 4 |

`sbEdge` (vertical tile edge, or horizontal edge on the 64×64 grid) clamps
`maxWidthNeg` to `min(maxWidthPos, 6 luma / 2 chroma)`.  Widths 5 and 7 are
unreachable, matching the spec note.

## API

```cpp
namespace av2enc::deblock {

struct DeblockPlane { uint8_t* data; int stride; };            // mutable plane

struct DeblockFrame {
  DeblockPlane planes[3];
  const DeblockMi* mi[3];       // one grid per plane, mi_cols*mi_rows
  int mi_cols, mi_rows;
  int subsampling_x, subsampling_y;   // must be 1,1
  const uint16_t* tile_col_starts; int num_tile_cols;  // MiColStarts
  const uint16_t* tile_row_starts; int num_tile_rows;  // MiRowStarts
};

DeblockStatus deblock_frame(DeblockFrame&, const DeblockOptions&);
DeblockEdgeInfo deblock_inspect_edge(const DeblockFrame&, const DeblockOptions&,
                                     int plane, int pass, int row, int col);
int deblock_plane_width(const DeblockFrame&, int plane);
int deblock_plane_height(const DeblockFrame&, int plane);
bool deblock_supported(const DeblockFrame&, const DeblockOptions&,
                       DeblockStatus*);
}
```

`DeblockOptions` carries `apply_deblocking_filter[4]` (Y-vertical,
Y-horizontal, U, V; §5.18.5.2), `disable_loopfilters_across_tiles` (§5.4.10),
`allow_df_sub_pu`, `frame_is_intra`/`enable_sdp` (skip reset), the chroma AC
quantizer deltas (`DeltaQUAc`/`DeltaQVAc`/`BaseUVAcDeltaQ`), `DfDeltaQ[4]`,
and `bit_depth` (8 only).

`deblock_inspect_edge()` runs the full decision path without writing pixels and
returns `DeblockEdgeInfo`: on-screen/tile/sb flags, block/TX/sub-PU edge flags,
skip, lossless sides, per-side Q/side thresholds, filter size,
`maxWidthPos`/`maxWidthNeg`, chosen `width`, and `apply_filter`.

## Metadata contract

One `DeblockMi` per 4×4-luma MI cell, for **every plane**, row-major.  Chroma
grids are indexed on the full luma MI grid: a chroma block's entry is
replicated over the luma extent it covers, matching the spec's "chroma
information is held in the bottom-right mode info" rule.  Fields:

| Field | Decoder array | Notes |
|---|---|---|
| `block_col_base`, `block_row_base` | `MiColBase`, `MiRowBase` | coding-block origin, luma MI |
| `block_w4`, `block_h4` | `Num_4x4_Blocks_Wide/High[MiSize]` | used for the chroma bottom-right adjustment |
| `sub_pu_size` | `SubPuSize` | `Max_Tx_Size_Rect[MiSize]` |
| `tx_size` | `DeblockingTxSizes[plane]` | chroma uses the chroma TX size |
| `tx_col_base`, `tx_row_base` | `TxColBase`, `TxRowBase` | luma MI |
| `sub_pu_col_base`, `sub_pu_row_base` | `SubPuColBase`, `SubPuRowBase` | luma MI |
| `skip` | `Skips` | forced to 0 for intra regions / SDP |
| `region_intra` | `RegionTypes == INTRA_REGION` | |
| `lossless` | `LosslessArray[segment]` | |
| `qindex` | effective `LumaQIndex` / `ChromaQIndex` | `get_qindex(0, segment)`, i.e. include segment delta when `delta_q_present == 0` |

Geometry:

- `mi_cols = ceil(frame_width / 4)`, `mi_rows = ceil(frame_height / 4)`, and
  both must be **even** (frame padded to a multiple of 8).  This is required
  for an integral 4:2:0 chroma grid and to keep chroma filter reads inside the
  MI-aligned plane.
- Luma plane is `mi_cols*4` × `mi_rows*4`; chroma is `mi_cols*2` × `mi_rows*2`.
  `stride` must be at least the plane width.  Buffers may be padded further
  (the tests use 8 guard columns and 2 guard rows).
- Tile starts are in luma MI units; `num_* == 0` means one implicit tile.

The caller owns all memory; the CPU path allocates nothing and does not
synchronise.

## Traversal / edge decision

`deblock_frame()` follows §7.17.1 exactly:

```text
for plane in {Y,U,V}
  for pass in {vertical, horizontal}
    if apply_deblocking_filter[plane == 0 ? pass : plane + 1]
      for row/col over the plane's 4x4 grid:
        deblocking_filter_edge(plane, pass, row, col)
```

Per edge it computes: on-screen/tile/64-grid flags, previous MI, block base,
chroma bottom-right adjustment, sub-PU edge, transform edge, block edge, skip,
`filt_max_size`, adaptive Q/side strength, `applyFilter` (TX/sub-PU edge plus
boundary strength), screen clipping of `filterSize`, `maxWidthPos/Neg`,
`filter_choice` (side masks, second derivatives, `Q_First` transition), and
finally `sample_filtering` for the four samples along the edge.

All arithmetic is integer and mirrors the spec `Round2`, `Clip3`, `Clip1`.
`Round2` relies on C++20 arithmetic right shift for negative values, which is
exactly the spec's floor definition and matches CUDA.

## Tests

`./build_and_test.sh [--no-run]` builds with
`-Wall -Wextra -Wpedantic -Wconversion -Wshadow` and runs 562 checks:

1. Flat and linear-ramp content: byte-identical (the filter must not invent
   detail).
2. Vertical/horizontal steps: filtered, edge amplitude reduced.
3. Width classes: `maxWidthPos` = 1/3/6/8 luma, 1/3/4/4 chroma; `sbEdge`
   clamps `maxWidthNeg` to 6 luma.
4. Alternating high-frequency texture: rejected by the side mask, unchanged.
5. Skip / boundary strength: a skipped non-block TX edge is not filtered.
6. Lossless sides: the lossless side is never written; the other side filters.
7. Tile edges: `disable_loopfilters_across_tiles` disables exactly the tile
   edge; enabling filters it.
8. Partial blocks / MI padding: frame-edge (x=0, y=0) edges are off-screen;
   guard samples outside the MI-aligned plane stay untouched.
9. 160 randomized frames with random coding/TX blocks, tiles, lossless/skip
   cells, Q indices, apply flags, `DfDeltaQ` and chroma deltas; deterministic
   and guard-clean.
10. A real reconstructed frame (`decoded.y4m`, 640×360): deterministic,
    2,495 luma+chroma samples changed, mean |delta| 0.0075, max 5.

Local results: 562 checks, 0 failures.  ASan + UBSan: clean.

## Measured CPU reference time (not a performance target)

640×360 `decoded.y4m`, 32×32 blocks: ~0.5–2.3 ms/frame depending on host load
(the shared Vast.ai Xeon varies).  1080p synthetic: ~11–15 ms/frame.  The CPU
path is the correctness oracle, per `docs/architecture.md`.

## Integration notes

1. **Where it runs.** After reconstruction of the frame, before CDEF/CCSO/
   restoration and before storing the reference.  Normative filter order is
   deblock → CDEF → CCSO → LR.
2. **What to produce.** The partition/TX/quant stages must fill one `DeblockMi`
   grid per plane.  For M1 intra with 4:2:0: `block_w4/h4` from the partition
   size, `SubPuSize = Max_Tx_Size_Rect[MiSize]`, `tx_size` per transform block
   (chroma TX sizes are half the luma ones), skip = 0 (intra region),
   `lossless` from the segment, `qindex = base_q_idx` (no delta-Q in M1).
   Chroma entries must be replicated over the luma block extent.
3. **Frame-header flags.** `apply_deblocking_filter[0..3]` come from
   §5.18.5.2.  M1 should set all four to 1 for a normal filtered frame; the
   M0 regression stream has them 0 and must not call this filter.
4. **Tile boundaries.** Pass `MiColStarts`/`MiRowStarts`.  If the encoder
   chooses `disable_loopfilters_across_tiles = 1`, tile edges are skipped;
   otherwise they are filtered with the `sbEdge` width clamp.
5. **Lossless.** A lossless block has `lossless=1`; filtering still runs on the
   other side of an edge but never writes the lossless side.  If both sides are
   lossless the edge is skipped.
6. **No bitstream syntax here.** The deblocking filter has no per-block syntax;
   the only frame-level syntax is the four apply flags and `df_delta_q`.  This
   component does not read or write OBUs.
7. **Reference update.** The filtered planes are what later frames predict
   from.  Run all four planes/edges before copying the frame into the reference
   pool.
8. **Not wired in.** Root `CMakeLists.txt`, `src/main.cpp`,
   `src/encoder/m0_still.cpp` and the bitstream paths are untouched; the
   integration pass owns that wiring.

## Limitations / deliberately out of scope

- 8-bit only.  10-bit/4:2:2/4:4:4 are later workstreams.
- 4:2:0 only, even MI grid (frame padded to a multiple of 8).
- BRU (`use_bru` / `BruModes`) is inter-only and not implemented; M1 is intra.
- `DeblockMi.qindex` is the effective per-cell quantizer index.  Segmentation
  feature reconstruction (`base_q_idx + segment delta`) is the caller's job.
- No SIMD, no threading: this is the correctness oracle.
