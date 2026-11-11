# Temporal CCSO reuse (AV2 v1.0.0 §5.18.7.12, §5.20.10.2, §7.23)

Spec-only component for the CCSO temporal mechanisms that were absent from the
tree:

| Mechanism | Spec | Status here |
|---|---|---|
| `reuse_ccso[plane]` / `sb_reuse_ccso[plane]` | §5.18.7.12 | writer + independent reader |
| `ccso_ref_idx[plane]` (index into `ref_frame_idx[]`) | §5.18.7.12 | writer + reader, `f(CeilLog2(NumTotalRefs))` |
| `load_ccso_params()` and the per-slot parameter store | §7.23 | `SlotStore` / `save_ccso_params()` / `resolve_ccso_params()` |
| Reference choice for later frames | §6.17.7.8 | `use_ccso()` encoder decision |
| `ccso_blk S()` and `sb_reuse` suppression | §5.20.10.2 | `TileState::write_block()` / `read_block()` |

Scope is 8-bit 4:2:0, matching `src/encoder/filters/{deblock,cdef,restoration}`.
Nothing in the existing M1/M2 core is edited.

## Why this is a separate module

`src/encoder/m1/filters_stream/filters_stream.cpp` implements the intra/static
form and **deliberately rejects** every reuse bit ("CCSO reference reuse is not
supported").  The module therefore owns a complete §5.18.7.12 row (including the
non-reuse branches) so the reuse decisions are not bolted onto a writer that
refuses them.  For a row with no reuse bits it is byte-identical to the existing
writer; the unit test and the decoder gate both check that.

## API (`ccso_reuse.h`)

### Frame syntax

```cpp
cr::FrameParams params;          // ccso_frame_flag, per-plane params, reuse bits
cr::HeaderContext ctx;           // sequence/frame facts needed by §6.17.7.8
std::string error;

cr::write_ccso_params(writer, params, ctx, &error);   // §5.18.7.12
cr::read_ccso_params(reader, ctx, &params, &error);   // independent mirror
```

`HeaderContext` carries `enable_ccso`, `coded_lossless`,
`single_picture_header_flag`, `frame_is_intra`, `num_planes`, `sb_size`,
`ccso_unit_matches_sb_size`, `tile_start_or`, `mi_cols/mi_rows`, `sub_x/sub_y`,
`num_total_refs`, `ref_frame_idx[]` and a `SlotStore*`.

`ccso_ref_idx` is an index into `ref_frame_idx[]`, **not** a reference slot.
With the M2 core's `num_total_refs = 1` it is zero-width (`f(0)`), so the gate
also pins a 4-reference row with `ccso_ref_idx = 3` (`f(2)`).

`ccso_luma_size_log2` is derived per frame from the same rule as the spec:
`Mi_Width_Log2[SbSize] + MI_SIZE_LOG2` when `ccso_unit_matches_sb_size`, else
8/7/6 from `OR(MiColStarts[]) | OR(MiRowStarts[])`.  **§6.17.7.8 requires
`CcsoLumaSizeLog2 == 8` when `sb_reuse_ccso` is set**, so block-flag reuse is
only legal with 256×256 CCSO units (for example `ccso_unit_matches_sb_size = 0`
with a single tile).  The gate uses exactly that configuration.

### §7.23 slot save/restore

```cpp
cr::SlotStore store;             // NUM_REF_FRAMES (16) slots
cr::SavedFrame saved;            // effective params + final CcsoBlks + geometry
cr::save_ccso_params(store, slot, saved, &error);

cr::EffectiveFrame effective;    // params + inherited/coded CcsoBlks per plane
cr::resolve_ccso_params(ctx, params, store, &effective, &error);
```

`save_ccso_params()` is the §7.23 `save_ccso_params(i, plane)` call: it stores
`SavedCcsoPlanes`, `SavedCcsoLumaSizeLog2`, the scalar parameters, the offset
table and `CcsoBlks`.  It must be invoked for **every refreshed slot** at the
end of the frame, including planes where CCSO is off
(`SavedCcsoPlanes[slot][plane] = 0`).  `resolve_ccso_params()` is the matching
`load_ccso_params()` and enforces the full §6.17.7.8 requirement set:

* `ccso_ref_idx < NumTotalRefs`;
* the reference slot is valid (`RefValid`);
* `SavedCcsoPlanes[idx][plane] == 1`;
* `RefOrderHint[idx] != RESTRICTED_OH`;
* for `sb_reuse_ccso`: `RefMiRows/Cols == MiRows/Cols`,
  `SavedCcsoLumaSizeLog2 == CcsoLumaSizeLog2`, and
  `CcsoLumaSizeLog2 == CCSO_LUMA_SIZE_LOG2 (8)`.

### Encoder decision

```cpp
cr::ReuseChoice choice;
cr::use_ccso(desired_params, desired_blks, blk_cols, blk_rows, plane,
             ctx, store, &choice, &error);
// choice.reuse_ccso, choice.sb_reuse_ccso, choice.ccso_ref_idx, choice.ref_slot
```

It walks the current frame's reference list in order and selects the first slot
whose saved parameters and/or saved block flags match the fresh decision.  It
reports parameter-only reuse, block-flag-only reuse, or both.  This is the
"reference-index selection" half of the mechanism; the spec has no syntax
element named `use_ccso` — its per-block `S()` symbol is `ccso_blk S()`
(`§5.20.10.2`), implemented below.

### Tile symbols

```cpp
cr::TileState tile;
tile.init(effective, mi_cols, mi_rows, sub_x, sub_y, &error);
tile.set_block_flags(plane, blks, cols, rows, &error);  // encoder decisions

tile.write_block(encoder, mi_row, mi_col, &error);      // ccso_blk S()
tile.read_block(decoder, mi_row, mi_col, blk, &error);
```

`write_block`/`read_block` are no-ops unless `(mi_row, mi_col)` is a CCSO unit
origin, exactly like `read_ccso()`.  When `sb_reuse_ccso[plane]` is set the
symbols are suppressed and the saved `CcsoBlks` grid is inherited.  The CDF is
the normative `kCcsoBlkCdf`; the left-neighbour context matches
`filters_stream::ccso_block_context()`.

`make_restoration_params(tile, &restoration_params, &error)` converts the
effective state to the existing §7.19 core input (`restoration::ccso_frame`).

## Where to call it

### 1. `src/encoder/m2/post/loop_filters.cpp` — filter and reference

`run_m2_post_filters()` already runs deblock → CDEF → CCSO and publishes
`output_plane()` (the post-CCSO `LrFrame`).  Two integration points:

* **Before** the call: fill `M2PostFilterConfig::ccso` from the effective frame.
  Pass the effective grids explicitly, because
  `M2PostFilterFrame::init_from_recon()` allocates its default `CcsoBlks`
  grids for `CcsoLumaSizeLog2 = 6` only:

  ```cpp
  for (int p = 0; p < 3; ++p) {
    config.ccso.blks[p] = effective.plane[p].blks.data();
    config.ccso.blk_cols[p] = effective.plane[p].blk_cols;
    config.ccso.blk_rows[p] = effective.plane[p].blk_rows;
  }
  cr::make_restoration_params(tile, &config.ccso, &error);
  // make_restoration_params() overwrites the three pointers/cols/rows with
  // the same effective grid, so it can also be called last.
  ```

  This is the point where `sb_reuse_ccso` matters: the grid is already the
  inherited one, so `run_m2_post_filters()` filters the current frame with the
  reference's block flags.

* **After** the tile payload is coded/reconstructed (the `CcsoBlks` grid is
  final): for every refreshed reference slot call `save_ccso_params()`.  This is
  §7.23's `save_ccso_params(i, plane)` and must happen before the next frame's
  header is written.

The frame that is stored as the reference must be `output_plane()`, not
`m2_frame.cpp`'s raw recon (`m2_frame.cpp:1027,1052`).  CCSO reuse is only
correct if later frames predict from the filtered `LrFrame`.

### 2. M2 frame header

`src/encoder/m2/post/tiled_frame.cpp` already builds a `CcsoHeaderContext` from
the tile layout.  Replace its `filters_stream::write_ccso_params()` call with
this module's writer, adding the reuse fields to `M2TiledFrameHeader` (or
passing the `FrameParams` directly) and filling `HeaderContext::num_total_refs`,
`ref_frame_idx` and `slots`.  Order is unchanged: `ccso_params()` stays where it
is today, between `lr_params()` and `tx_mode_select()`.

### 3. M2 tile writer

`src/encoder/m2/m2_syntax.cpp` has no `read_ccso()` call.  Insert it at the
**start of `mode_info()`**, after the skip/segmentation fields — i.e. at the
start of `write_inter_mode_info()` for `is_inter == 1` and at the start of
`write_intra_leaf()` for the keyframe and `is_inter == 0` cases.  It must not go
before `read_partition()`: §5.20.5.1/§5.20.7.1 place `read_ccso()` inside mode
info.  The gate's `vendor_m2_syntax.py` demonstrates the exact two insertion
points without editing the core.

## Default streams are byte-identical

Three properties keep existing output untouched:

1. Every current sequence writes `enable_ccso = 0`.  `write_ccso_params()` then
   writes **zero bits** and `read_ccso_params()` consumes zero bits; the call
   site can be unconditional.
2. For CCSO-enabled streams without reuse, the writer produces exactly the same
   row as the existing `filters_stream` writer.  `ccso_reuse_test` pins this,
   and the decoder gate compares every fresh frame header against
   `m2::post::write_m2_frame_header_syntax()` bit-for-bit before encoding.
3. `save_ccso_params()` must still be called with CCSO off (it records
   `SavedCcsoPlanes = 0`), but it does not emit bits or touch pixel data.

No default path enables a reuse bit, so the only bitstream change from
integration is in streams that explicitly request temporal reuse.

## Tests

```sh
cd src/encoder/filters/ccso_reuse
./build_and_test.sh
# ccso_reuse_test: 163 checks, 0 failures
```

Coverage:

* six pinned §5.18.7.12 bit strings (intra fresh, inter fresh, reuse-both,
  four-reference `ccso_ref_idx = 3`, `sb_reuse` only, mixed planes), each read
  back by the independent reader;
* byte equality with `filters_stream::write_ccso_params()` for non-reuse rows;
* zero-bit behavior with `enable_ccso = 0`;
* §7.23 save/load, slot overwrite lifetime, `SavedCcsoPlanes = 0`;
* `use_ccso()` selection of both/params-only/sb-only/none;
* every §6.17.7.8 rejection (bad index, invalid/restricted reference, plane not
  saved, luma size for `sb_reuse`, geometry mismatch, mismatched reused
  parameters, reader without a slot store);
* tile symbols round-tripped through the normative §8.2 decoder, plus pinned
  payloads `fc0000` (three fresh flags) and `8000` (`sb_reuse`, no symbols).

## Decoder gate

```sh
cd src/encoder/filters/ccso_reuse/gate
./build_and_test.sh
# PASS: frame 2 reused frame 1's CCSO parameters and CcsoBlks; decoded
# reconstruction bit-exact
```

The gate builds a three-frame M2 stream at 256×192:

| Frame | Type | CCSO |
|---|---|---|
| 0 | CLK | fresh params + fresh `CcsoBlks`, 1 unit/plane |
| 1 | P (skip copy) | fresh params + fresh `CcsoBlks`; saved to slot 0 |
| 2 | P (skip copy) | `reuse_ccso = 1`, `sb_reuse_ccso = 1`, `ccso_ref_idx = 0`, no params or `ccso_blk` symbols |

It decodes both the reuse stream and a no-reuse control stream with
`dav2d --strict 1 --threads 1` and compares every plane of every frame against
the encoder's own `restoration::ccso_frame()` output: **0 diffs** on all three
frames, and 0 diffs for the control frame 2.

Observed bit cost for this fixture:

* frame 2 `ccso_params()` row: **10 bits with reuse vs 133 bits fresh**
  (−123 bits);
* whole streams: baseline (no CCSO) 253 B, no-reuse control 303 B, reuse 288 B
  — the reuse stream is **15 bytes smaller** than the control and still
  bit-exact.

The control proves the saving comes from the reuse syntax, not from a different
picture: its reconstruction is identical.

## Limitations

* 8-bit 4:2:0 only, like the existing filter cores.
* `sb_reuse_ccso` is implemented and gated only with `CcsoLumaSizeLog2 = 8`,
  because §6.17.7.8 forbids it otherwise.  Parameter-only reuse works for any
  luma size.
* The module writes/reads the CCSO rows and carries the slot state; it does not
  choose filter offsets (the caller's fit/search does).
