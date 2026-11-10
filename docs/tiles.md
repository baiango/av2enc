# Multi-tile coding (next stage; M0 stays 1x1)

Status: **design notes for the stage after M0**. M0 ships a single tile
(`TileCols = TileRows = 1`). Nothing in this document changes `docs/minimal-bitstream.md`.
Everything below is read off the AV2 v1.0.0 specification (`spec/AV2_Spec_v1.0.0.pdf`,
SHA-256 `e9916f09…b533f`); section numbers are given for every claim.

Target layouts: **1x1, 2x1, 2x2, 4x2, 8x2**. dav2d at `f9dafbc` is reported to decode
all five; 8x2 costs ≈5% bytes on one 720p frame (project measurement, not spec).

---

## 1. Why tiles are cheap for us

Tiles are *independently decodable segments of the same frame*. For an encoder:

- Each tile has its own range-coder instance, its own CDF state, its own trailing bits.
- Everything else (sequence header, frame header, prediction, transforms, quant,
  reconstruction) is unchanged and frame-global.
- The only new syntax is a handful of bits in `tile_info()` plus per-tile size fields
  in `tile_group_payload()`.

For a GPU entropy coder this is the natural parallelism unit: **one CUDA
workgroup per tile**, each with a private CDF copy and a private output buffer.
Tiles are also the only *specified* entropy-independent unit in AV2
(tile groups are just transport grouping; see §5.19).

---

## 2. Frame-header tile info

`tile_info()` — §5.18.7.2; semantics §6.17.7.2. Called from `frame_header_info()`
(§5.18.2) right after `delta_q_params()` / before `quantization_params()`.

Order inside `tile_info()` (§5.18.7.2) for our configuration
(`seq_tile_info_present_flag = 0`, so no sequence-level tile params):

```
# --- geometry, no bits ---
sb4x4  = Num_4x4_Blocks_Wide[SbSize]          # 16 for BLOCK_64X64
sbShift = Mi_Width_Log2[SbSize]               # 4
sbCols = (MiCols + sb4x4 - 1) >> sbShift
sbRows = (MiRows + sb4x4 - 1) >> sbShift

haveTileParams = seq_tile_info_present_flag   # we set 0
reuse_tile_info = 0                            # because haveTileParams == 0

# --- tile_params() --- §5.18.7.3
uniform_tile_spacing_flag            f(1)     # 1 = uniform grid
  # uniform branch:
  tileColsLog2 = minLog2TileCols
  while (tileColsLog2 < maxLog2TileCols)
      increment_tile_cols_log2         f(1)     # 1 = keep going, 0 = stop
  tileRowsLog2 = minLog2TileRows
  while (tileRowsLog2 < maxLog2TileRows)
      increment_tile_rows_log2         f(1)     # 1 = keep going, 0 = stop

# --- back in tile_info() ---
if (TileCols > 1 || TileRows > 1) {
    if (!enable_avg_cdf || !avg_cdf_type)
        context_update_tile_id         f(TileRowsLog2 + TileColsLog2)
    tile_size_bytes_minus_1            f(2)
}
```

Notes that matter:

- `minLog2TileCols` comes from the **level constraint** on tile width:
  `maxTileWidthSb = (Tile_Width_Scaling_Factor[tier][level] * MAX_TILE_WIDTH) >> (sbShift + 4)`
  and `minLog2TileCols = tile_log2(maxTileWidthSb, sbCols)` (§5.18.7.3).
  `MAX_TILE_WIDTH = 4096`, `Tile_Width_Scaling_Factor = 4` for levels 2.0–6.3,
  `8` for 7.0–7.3, `16` for 8.0–8.3 (§9.2; level index mapping in Annex A.4).
  With 64x64 SBs this is 64 SBs = 4096 luma samples at level ≤ 6.3. A 1280-wide
  frame has `sbCols = 20`, so `minLog2TileCols = 0` — no forced columns.
- `minLog2Tiles` also forces enough tiles to respect `MAX_TILE_AREA = 4096*2304`
  (§5.18.7.3). Irrelevant below ~4K.
- `maxLog2TileCols = tile_log2(1, Min(sbCols, 64))`, `maxLog2TileRows` likewise,
  so 256x256 with 64x64 SBs gives `sbCols = sbRows = 4` → at most 4x4 tiles.
- **`context_update_tile_id` is absent for us.** `single_picture_header_flag = 1`
  forces `enable_avg_cdf = 1, avg_cdf_type = 1` (§5.4.8), and §5.18.7.2 only
  codes the field when `!enable_avg_cdf || !avg_cdf_type`. If a later profile
  drops `single_picture_header_flag`, the field reappears.
- `tile_size_bytes_minus_1` is 2 bits, always present when `NumTiles > 1`.
  `TileSizeBytes = tile_size_bytes_minus_1 + 1` (1..4).
- `uniform_tile_spacing_flag = 0` (explicit `width_in_sbs_minus_1` /
  `height_in_sbs_minus_1`, `ns()` descriptors) is legal but we do not need it.

### `tile_log2` (§5.18.7.7)

```
tile_log2(blkSize, target) { for (k = 0; (blkSize << k) < target; k++) {} return k; }
```

### Uniform spacing (§5.18.7.5)

```
uniform_spacing(tileLog2, mis, sbSize) {
    sb4x4 = Num_4x4_Blocks_Wide[sbSize]; sbShift = Mi_Width_Log2[sbSize]
    sbs = (mis + sb4x4 - 1) >> sbShift
    fullSbs = mis >> sbShift
    tileSb = fullSbs >> tileLog2
    extraSbs = (tileSb == 0) ? sbs : fullSbs - (tileSb << tileLog2)
    startSb = 0
    for (i = 0; i < (1 << tileLog2) && startSb < sbs; i++) {
        sbStarts[i] = startSb
        startSb += tileSb
        if (i < extraSbs) startSb += 1
    }
    return (sbStarts, i)
}
```

For 1280x720 (`sbCols = 20`): 4 columns → tile widths 5,5,5,5 SBs; 8 columns →
3,3,3,3,3,3,1,1? No: `fullSbs=20, tileSb=20>>3=2, extraSbs=20-16=4` →
starts 0,3,6,9,12,14,16,18 → widths 3,3,3,3,2,2,2,2. **The last column is not the
narrow one**; the extra SBs are distributed to the *first* `extraSbs` tiles.
(For 4 columns: `tileSb=5, extra=0` → 5,5,5,5.)

At 256x256 (4 SBs each way): 2x2 → 2,2 in each direction; 4x2 → 1,1,1,1 x 2,2.

---

## 3. Tile group OBU and tile payload framing

`tile_group_obu(sz)` — §5.19; semantics §6.18.

```
tile_group_obu(sz) {
    is_first_tile_group                    f(1)   # 1 for the first group of a frame
    if (is_first_tile_group) frame_header_present_flag = 1
    else                     frame_header_present_flag f(1)
    if (frame_header_present_flag) frame_header(is_first_tile_group)
    # (bru_inactive path not used by us)
    NumTiles = TileCols * TileRows
    tile_start_and_end_present_flag = 0
    if (NumTiles > 1) tile_start_and_end_present_flag f(1)
    if (NumTiles == 1 || !tile_start_and_end_present_flag) { tg_start = 0; tg_end = NumTiles-1 }
    else {
        tileBits = TileColsLog2 + TileRowsLog2
        tg_start                               f(tileBits)
        tg_end                                 f(tileBits)
    }
    byte_alignment()                            # zero bits only (§5.2.4)
    endBitPos = get_position(); headerBytes = (endBitPos - startBitPos)/8
    sz -= headerBytes
    tile_group_payload(sz)
}
```

`tile_group_payload(sz)` — §5.20.1:

```
for (TileNum = tg_start; TileNum <= tg_end; TileNum++) {
    tileRow = TileNum / TileCols; tileCol = TileNum % TileCols
    lastTile = (TileNum == tg_end)
    if (lastTile) tileSize = sz
    else { le(TileSizeBytes) tile_size_minus_1; tileSize = tile_size_minus_1 + 1; sz -= tileSize + TileSizeBytes }
    MiRowStart = MiRowStarts[tileRow];  MiRowEnd = MiRowStarts[tileRow+1]
    MiColStart = MiColStarts[tileCol];  MiColEnd = MiColStarts[tileCol+1]
    CurrentQIndex = base_q_idx
    init_symbol(tileSize)
    decode_tile()
    exit_symbol()
}
if (tg_end == NumTiles - 1) { frame_end_update_cdf(); decode_frame_wrapup(); SeenFrameHeader = 0 }
```

Key points:

- **Tile order is raster order** over the tile grid (`TileNum / TileCols`,
  `TileNum % TileCols`). Payloads appear in that order.
- `le(TileSizeBytes)` is a little-endian n-byte integer (§4.11.5), **not** leb128.
- Every tile payload is byte-aligned both ends: `byte_alignment()` before the first
  tile, `init_symbol()` reads whole bytes, `exit_symbol()` consumes the tile's
  trailing bits and leaves the position byte-aligned (§8.2.4: "paddingEndPosition
  will always be a multiple of 8").
- The **last** tile's size is implied by the OBU length (it also contains the OBU
  trailing bits). The per-tile size field for earlier tiles covers exactly the
  tile's coded bytes including the symbol decoder's padding bits (§6.19.1).
- `init_symbol(sz)` reads `Min(sz*8, 15)` bits (§8.2.2), so a tile must be ≥2 bytes.
- The OBU payload size (`num_bytes_in_obu` leb128 in Annex B) must account for the
  frame header + tile-group header + all size fields + all tile payloads.

### One OBU or several?

Two legal packings:

1. **One tile-group OBU with all tiles** (`tile_start_and_end_present_flag = 0`).
   Simplest, and what M0-next should do.
2. **One OBU per tile**: first OBU has `is_first_tile_group = 1` (and carries the
   frame header), later ones must have `is_first_tile_group = 0` +
   `frame_header_present_flag = 0`, and must set `tile_start_and_end_present_flag = 1`
   with the matching `tg_start`/`tg_end`. §6.18 requires
   `SeenFrameHeader != is_first_tile_group`, `tg_start == TileNum` at payload entry,
   and the final group to end at `NumTiles - 1`.

Multiple groups buy nothing for us (the OBUs are adjacent anyway) and add
state-machine risk. Use packing 1.

---

## 4. Independent CDF initialization / reset per tile

- At the start of each tile, `init_symbol(tileSize)` makes a **fresh copy** of the
  frame-level CDF arrays into the `Tile*` working arrays (§8.2.2: "A copy is made of
  each of the CDF arrays mentioned in the semantics for init_coeff_cdfs and
  init_non_coeff_cdfs"). Frame-level arrays are `YModeSetCdf`, `TxbSkipCdf`, … ;
  the working copies are `TileYModeSetCdf`, `TileTxbSkipCdf`, ….
- `read_symbol()` mutates only the `Tile*` copy (§8.2.6). Therefore **each tile starts
  from the same CDF state**, namely the state produced by `init_coeff_cdfs()` +
  `init_non_coeff_cdfs()` at frame-header time (`set_primary_ref_frame_and_ctx(1)`
  with `PRIMARY_REF_NONE` → §5.18.2, §6.17.2).
- Encoder consequence: encode each tile with a private CDF context initialized to the
  *same* default tables, even if another tile is encoded first. Do not chain CDFs
  from tile to tile.
- At `exit_symbol()` (§8.2.4), with `enable_avg_cdf && avg_cdf_type` (forced by
  `single_picture_header_flag`), `avgCdf = TileNum < (1 << numLog2)` where
  `numLog2 = Min(3, FloorLog2(TileCols*TileRows))`. So:
  - 2 tiles → `numLog2 = 1`, tiles 0..1 averaged;
  - 4 tiles → `numLog2 = 2`, tiles 0..3 averaged;
  - 8 tiles → `numLog2 = 3`, all 8 averaged;
  - ≥9 tiles → only the first 8 are averaged.
  The averaging writes `Saved*` arrays; `frame_end_update_cdf()` (§7.5) then installs
  them as the frame CDF state (with the last entry scaled by 3/4). This matters only
  for the *next* frame. A single-frame still can ignore it.
- `context_update_tile_id` is the alternative mechanism (`copyCdf`) and is **not
  signalled** in our `avg_cdf_type = 1` configuration (§5.18.7.2 / §8.2.4).
  Do not write the field.

---

## 5. Tile-boundary constraints and loop-filter signalling

### Prediction and context

- `is_inside(r,c)` is **tile-relative**: it requires
  `MiRowStart <= r < MiRowEnd` and `MiColStart <= c < MiColEnd` (§5.20.9.1).
  `AvailU`/`AvailL` in `decode_partition()` (§5.20.3.1) use `is_inside`, so intra
  prediction never reads samples from another tile, and neighbour-derived contexts
  (partition, intra mode, coefficient level context, MRL, …) reset at tile edges.
- `clear_above_context()` runs once per tile and `clear_left_context()` once per
  superblock row *inside* the tile (§5.20.2.1). `LeftLevelContext`,
  `AboveLevelContext`, `LeftDcContext`, `AboveDcContext`, `LeftMiSizes`,
  `AboveMiSizes`, segment-prediction contexts are therefore zero at tile start.
- Superblock raster order is per tile, not per frame: the outer loops run
  `r = MiRowStart .. MiRowEnd`, `c = MiColStart .. MiColEnd` (§5.20.2.1).
- Encoder consequence: the mode/coefficient context engines must be reset at tile
  boundaries exactly as the decoder does. If the GPU encodes a tile per workgroup,
  each workgroup needs its own above-row context (the left context is internal).

### Loop filters

- `disable_loopfilters_across_tiles` is a **sequence-header** bit (§5.4.10,
  semantics §6.4.10): 1 = loop filters do not access samples from a different tile.
- Where it is consulted:
  - deblocking: `vertTileEdge` / `horzTileEdge` skip edge filtering (§7.17.2,
    §7.17.4);
  - loop restoration: luma/chroma access windows clamp to
    `MiColStart/MiColEnd/MiRowStart/MiRowEnd` (§7.20.1);
  - CDEF/CCSO/GDF: filter block grids are aligned to tile boundaries
    (`MiColStartGrid` etc., §7.18/§7.19; `gdf_per_block` derived from tile
    alignment, §5.18.7.9).
- For M0 (all filters disabled) the bit is inert; write 0 for simplicity.
  If tiles are combined with filters later, write 1 so tile payloads stay
  independently decodable and GPU workgroups stay independent.
- `is_inside_filter_region()` (§5.20.9.3) captures the same rule for CDEF/restoration.

### Geometry constraints

- Minimum tile = one superblock (64x64 luma for us); `tileColsLog2 <= log2(sbCols)`
  and `tileRowsLog2 <= log2(sbRows)` (§5.18.7.3, §6.17.7.2). A 64x64 frame cannot
  have >1x1 tiles; **use 256x256 for tile tests**.
- Max 64x64 tiles (§9.2 `MAX_TILE_COLS/ROWS`).
- Tile width is bounded above by the level-derived `maxTileWidthSb`
  and area by `MAX_TILE_AREA` (§5.18.7.3); there is no "≥256 samples" rule in AV2
  v1.0.0 (that was AV1 folklore) — the bound comes from `maxLog2TileCols`.

---

## 6. Exact syntax element order

### 6.1 2x2 tiles (e.g. 256x256, 4x4 SBs)

Frame header (`frame_header_info()`), only the tile-related tail:

| # | element | value | bits |
|---|---------|-------|------|
| 1 | `uniform_tile_spacing_flag` | 1 | 1 |
| 2 | `increment_tile_cols_log2` | 1 | 1 |
| 3 | `increment_tile_cols_log2` | 0 | 1 |
| 4 | `increment_tile_rows_log2` | 1 | 1 |
| 5 | `increment_tile_rows_log2` | 0 | 1 |
| 6 | `tile_size_bytes_minus_1` | e.g. 1 | 2 |

(`context_update_tile_id` absent: `avg_cdf_type = 1`.)

Then the rest of the frame header (quantisation, segmentation, filters, tx mode,
…), then the tile group:

| # | element | value | bits |
|---|---------|-------|------|
| 7 | `is_first_tile_group` | 1 | 1 |
| — | `frame_header()` | (already counted above) | — |
| 8 | `tile_start_and_end_present_flag` | 0 | 1 |
| 9 | `byte_alignment()` | zeros | 0–7 |
| 10 | `tile_size_minus_1` tile 0 | n0 | `8*TileSizeBytes` |
| 11 | tile 0 payload | | n0+1 |
| 12 | `tile_size_minus_1` tile 1 | n1 | `8*TileSizeBytes` |
| 13 | tile 1 payload | | n1+1 |
| 14 | `tile_size_minus_1` tile 2 | n2 | `8*TileSizeBytes` |
| 15 | tile 2 payload | | n2+1 |
| 16 | tile 3 payload (last) | | rest of OBU |

Tile numbering: 0 = (row 0, col 0), 1 = (0,1), 2 = (1,0), 3 = (1,1).

### 6.2 4x2 tiles (e.g. 256x256)

Same list, but columns reach log2 = 2:

| # | element | value |
|---|---------|-------|
| 1 | `uniform_tile_spacing_flag` | 1 |
| 2 | `increment_tile_cols_log2` | 1 |
| 3 | `increment_tile_cols_log2` | 1 |
| 4 | `increment_tile_cols_log2` | 0 |
| 5 | `increment_tile_rows_log2` | 1 |
| 6 | `increment_tile_rows_log2` | 0 |
| 7 | `tile_size_bytes_minus_1` | `f(2)` |

Then `is_first_tile_group = 1`, `tile_start_and_end_present_flag = 0`,
`byte_alignment()`, then **seven** `tile_size_minus_1` fields
(tiles 0..6, raster order 0..3 = row 0, 4..7 = row 1) and eight payloads;
tile 7's size is implied.

If instead `tile_start_and_end_present_flag = 1` (only needed for partial groups),
`tileBits = TileColsLog2 + TileRowsLog2 = 3`, so `tg_start` and `tg_end` are
3 bits each.

### 6.3 What a tile payload contains

Identical to the M0 single-tile payload (§5.20.2.1 `decode_tile()`):
SB raster over the tile rectangle, `decode_partition()`, `decode_block()`,
`mode_info()`, prediction, `residual()` → `transform_block()` → `coeffs()`.
No syntax element is added or removed inside a tile. Only the *CDF state at entry*
and the *candidate neighbour set* differ.

---

## 7. What stays identical to the 1x1 M0 stream

| Item | 1x1 M0 | Multi-tile |
|---|---|---|
| OBU sequence (TD, SH, frame+tile group) | same | same |
| Sequence header | same | same (only `disable_loopfilters_across_tiles` may change) |
| Frame header fields outside `tile_info` | same | same |
| Partition/mode/coeff syntax | same | same |
| Transform, quant, reconstruction | same | same |
| `base_q_idx` | one value | one value for the whole frame (delta-q off) |
| CDF defaults | same | same, copied per tile |
| Output IVF framing | same | same |

Differences, exhaustively:

1. `tile_info`: `uniform_tile_spacing_flag`, `increment_tile_*_log2` bits,
   `tile_size_bytes_minus_1`.
2. Tile group: `tile_start_and_end_present_flag` (1 bit) and per-tile
   `tile_size_minus_1` fields.
3. Per-tile CDF reset (decoder-visible only through the bitstream, not new syntax).
4. Per-tile neighbour availability at tile edges (changes coded symbols, not syntax).
5. If loop filters are on: `disable_loopfilters_across_tiles` and filter-unit
   alignment.

---

## 8. Encoder implementation notes for the GPU stage

- Buffer each tile's range-coded bytes independently; the size is only known after
  `exit_symbol()`, but `tile_size_bytes_minus_1` must be written *before* the tile
  data. Two-pass or fixed-size reservation:
  - pass 1 encodes tiles into growable buffers, records sizes;
  - choose `TileSizeBytes = max(1, ceil((max_tile_size+1)/256))`, clamp to 4;
  - pass 2 emits header + size fields + buffers.
  Alternative: reserve 2 bytes per tile and fail the frame if any tile exceeds 65536
  bytes (a 720p 8x2 tile at QP 30 is far below that).
- `init_symbol` reads from the *start* of the tile's byte range; the range coder does
  not know the tile's absolute position. So a tile can be encoded in isolation.
- `exit_symbol` writes a trailing 1 bit plus zero padding to the byte boundary
  (§8.2.4 conformance: `SymbolMaxBits >= -14`, the trailing bit at
  `trailingBitPosition` is 1, bits after it are 0). The tile size includes that
  padding.
- Tile payload size must be ≥2 bytes.
- Do not split a tile across OBUs; a tile is one contiguous byte range.

---

## 9. Uncertainties / TODO

- **dav2d CLI and multi-tile behaviour**: the project's pinned build (`f9dafbc`) is
  reported to decode 1x1, 2x1, 2x2, 4x2 and 8x2 layouts, but this has not been
  verified in this repo. TODO: run `tools/validate_dav2d.sh` on a 2x2 stream and
  confirm the decoded YUV matches the 1x1 encode of the same frame.
- **8-row/8-column layouts at 720p**: `sbRows = 12` → `maxLog2TileRows = 3` (8 rows
  possible), columns `sbCols = 20` → `maxLog2TileCols = 5`. 8x2 is within limits at
  level 5.0 only if `maxTileWidthSb` ≥ 20 SB; at level 5.0 it is 64 SB. OK.
- **Tile size field width**: the `f(2)` range only allows 1..4 bytes. For very high
  bitrate/large tiles a 4-byte limit may force smaller tiles. Not yet modelled.
- **`context_update_tile_id` in non-still profiles**: exact encoder policy is a
  rate/robustness trade-off we have not studied (spec only defines the mechanism).
- **`numLog2` averaging with >8 tiles**: only the first 8 tiles contribute to
  `Saved*`. If we ever add a second frame, its CDFs depend on that subset. Unknown
  whether dav2d implements the `TileNum < 1<<numLog2` guard correctly — TODO test.
