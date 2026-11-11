# Backwards reference update (BRU)

AV2 v1.0.0 backwards reference update syntax and validation, isolated under
`src/encoder/inter/bru/`.  This directory is the only producer of the fields
that the rest of the tree previously only *consumed* in loop filters.

Normative sections:

| Syntax / process | Section | Page |
|---|---|---|
| sequence `enable_bru` | §5.4.6 | 63 |
| frame `use_bru`, `bru_ref`, `bru_inactive` | §5.18.2 | 102 |
| tile group `bru_tile_active` | §5.19 | 144–145 |
| tile-level `bru_mode S()` / `bru_mode_info()` | §5.20.2.1, §5.20.5.2 | 147, 169–170 |
| `bru_region_valid()` / R10 | §7.2 | 377–378 |
| deblock gating on `BruModes != BRU_ACTIVE` | §7.17.2 | 531 |
| IBC buffer invalidation on `BRU_INACTIVE` | §5.20.2.1 `update_ibc_buffers` | 169 |

## API

`bru.h` exports:

* `make_sequence_header_obu(SequenceConfig)` — non-single-picture sequence
  header with `enable_bru` at the §5.4.6 position. With `enable_bru = false`
  the bytes are identical to `m2::make_sequence_header_obu()` for the same
  width/height/order-hint configuration.
* `write_frame_header()`, `make_tile_group_obu()` — frame-level `use_bru`,
  `bru_ref`, `bru_inactive`, uniform `tile_info()`, `tg_start/tg_end`,
  `bru_tile_active`, `tile_size_minus_1` and the §5.19 early-return form for
  `bru_inactive = 1`.
* `encode_tile_payload()` — the gate subset of §5.20 syntax:
  * active superblocks: `bru_mode = BRU_ACTIVE`, then a `PARTITION_NONE` root,
    `is_inter = 1`, `skip_flag = 1`, `single_ref` selecting reference index 1,
    `single_mode = GLOBALMV`;
  * support/inactive superblocks: just the `bru_mode` symbol (when the tile is
    active), or no symbols at all in an inactive tile.
* `bru_region_valid(cols, rows, modes)` — R10 transcription.  `modes` is one
  entry per 64x64 superblock (the spec samples the top-left MI unit of each
  superblock, so this reduction is exact).
* `deblock_edge_allowed()` — §7.17.2's first check: an edge may be filtered
  only when `use_bru == 0` or the containing superblock is `BRU_ACTIVE`.
* `bru_ibc_buffer_valid()` — §5.20.2.1: a cached IBC buffer is invalidated by
  a `BRU_INACTIVE` superblock; `BRU_SUPPORT` keeps it.
* `parse_stream()` — an independent reader.  It re-parses OBU headers,
  sequence/frame/tile-group headers, and re-decodes the tile CDFs from the
  bytes.  It does not share writer state.

The writer's gate subset deliberately codes only skips, so a mixed frame
reconstructs the reference pictures exactly:

* active blocks reference reference index 1 (zero MV, no residual);
* `BRU_SUPPORT` / `BRU_INACTIVE` blocks use `bru_mode_info()`, whose
  `RefFrame[0]` is `bru_ref` and whose MV is zero.

## M2 call sites

The module is not wired into the M2 core yet; these are the exact integration
points a follow-up change must touch.

1. **Sequence header** — `src/encoder/m2/m2_headers.cpp:177` writes
   `writer.put_bit(false);  // enable_bru`.  Replace with
   `writer.put_bit(config.enable_bru)` and add `enable_bru` to `M2Config`.
2. **Frame header** — after the `ref_frame_idx[]` loop in
   `write_frame_header_syntax()` (`m2_headers.cpp:261-263`) insert the §5.18.2
   block:

   ```cpp
   if (c.enable_bru && !key && !is_tip_frame) {
     writer.put_bit(h.use_bru);
     if (h.use_bru) {
       writer.put_bits(h.bru_ref, ceil_log2(c.num_total_refs));
       writer.put_bit(h.bru_inactive);
     }
   }
   ```

   It must sit before the `allow_intrabc`/`use_ref_frame_mvs` block.  A
   `bru_inactive = 1` frame then returns after `film_grain_config()`; it does
   **not** code `tile_info`, quantization or filter syntax.
3. **Tile group header** — `m2_headers.cpp:318-326` currently emits
   `is_first_tile_group` + frame header + `byte_alignment()` and assumes one
   tile.  Add `tile_start_and_end_present_flag`, `tg_start/tg_end`, and the
   per-tile `bru_tile_active` loop for `NumTiles > 1`; keep
   `BruTileActives[0][0] = 1` for `NumTiles == 1`.  For `bru_inactive = 1`,
   emit `trailing_bits(remainingBits)` and no tile payload.
4. **Tile syntax** — `src/encoder/m2/m2_syntax.cpp`'s `Writer::run()` SB loop
   (`~447-487`) and the per-block `mode_info()` path must code `bru_mode S()`
   once at each `is_sb_start_mi()` when `BruTileActive`, then use
   `bru_mode_info()` (`skip = 1`, `ref = bru_ref`, zero MV) whenever
   `bru_mode != BRU_ACTIVE`.  The active path must add the §5.18.2 `is_inter`
   bit because `MiSize == ChromaMiSize` for a `PARTITION_NONE` superblock.
5. **Post/loop filters** — `src/encoder/m2/post/tiled_frame.cpp` and
   `src/encoder/filters/deblock/` need the per-SB `BruModes` map; deblock,
   CDEF, CCSO and LR must skip non-`BRU_ACTIVE` SBs (§7.17.2, §7.18, §7.19,
   §7.20).  The existing consumers already accept `bru_modes`
   (`filters/restoration/restoration.h:74-78`,
   `filters/restoration/ccso_core.h:80`, `ccso_core.h:157-158`).
6. **Wrap-up** — `bru_inactive` needs §7.2/R5 handling: copy
   `FrameStore[ref_frame_idx[bru_ref]]`, fill `MfRefFrames[y8][x8][list] =
   bru_ref` and zero `MfMvs`, and force all frame filter enables to 0.

## Buffer-mode interaction

* `bru_inactive = 1` is a *whole-frame buffer copy*, not a tile decode.  The
  decoder skips `tile_info`, quantization, deblock, CDEF, CCSO and LR, copies
  the `bru_ref` frame buffer into `LrFrame`, and fills the motion-field
  reference/zero-MV arrays with `bru_ref` (§7.2, p376-377).
* `BRU_INACTIVE` superblocks invalidate the cached IBC buffers at that
  superblock (`update_ibc_buffers`, p169); `BRU_SUPPORT` does not.  The
  writer's helper `bru_ibc_buffer_valid()` mirrors this.
* `bru_mode_info()` blocks are inter skips with `RefFrame[0] = bru_ref` and
  MV `(0,0)`; active blocks must not reference `bru_ref`, `ClosestPast` or
  `ClosestFuture` (§6.19.7.9, p362).  The gate therefore uses two references
  whose content differs: active SBs use reference index 1, `bru_ref` is
  index/slot 0.
* BRU frames must be `immediate_output_frame = 1`, must refresh the `bru_ref`
  DPB slot, and may update only one reference.  The gate and reader keep
  `refresh_frame_flags & (1 << ref_frame_idx[bru_ref])` non-zero.

## Tests and gate

`build_and_test.sh` builds and runs:

* `bru_test` — pinned golden hashes for:
  * `enable_bru = 0/1` sequence-header OBUs,
  * the mixed BRU tile-group OBU,
  * the active and inactive tile payloads;
  plus writer/reader round-trip, whole-frame-inactive reader, exhaustive
  `bru_region_valid()` over all 19,683 3x3 ternary maps and all 65,536 4x4
  binary maps versus an independent oracle, deblock/IBC gating, and the
  `enable_bru = 0` byte-identity control against the untouched M2 writer.
* `bru_gate` — a three-frame gate stream built from the untouched M2
  encoder:
  * CLK A refreshes all slots;
  * P B refreshes slot 1;
  * mixed BRU frame: reference index 0 = slot 0, `bru_ref = 0`, active
    superblock at (0,0), `BRU_SUPPORT` on (0,1)/(1,0)/(1,1), second tile
    inactive;
  * the active region reconstructs A, the support/inactive region reconstructs
    B; the gate compares every Y/U/V sample and the independently parsed
    `BruModes` map.
  The stream is first offered to `dav2d --strict 1 --threads 1`.  The pinned
  decoder `f9dafbc` (and the newer `cb234c3` nightly) do not yet parse
  `enable_bru/use_bru`; when it rejects the stream the gate falls back to the
  AVM decoder in `tools/bin/avm-macos/avmdec`, which implements BRU.  The
  `enable_bru = 0` control always runs on `dav2d --strict 1 --threads 1` and
  is byte-identical to the M2 baseline.

`bru_inactive = 1` is covered by the reader/unit tests.  AVM's decoder also
expects a dummy entropy payload for an inactive tile group (a spec-correct
inactive group has none), so the decoder gate does not include that frame.
