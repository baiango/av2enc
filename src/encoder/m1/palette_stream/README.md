# M1 palette syntax adapter — AV2 §5.20.8 / §8.3.2

Integration layer between the gated palette core (`src/encoder/palette/`) and an
M1 block stream. It owns the screen-content flags, the palette syntax position,
the per-MI `PaletteState` (cache/colour source), identity rows, and exact
writer/reader wrappers. It does **not** own partition, intra mode selection,
transform/quant, or the OBU writer.

Status: **4001 checks / 0 failures** and a four-mode `dav2d` gate
(`single`, `split`, `vert`, `wide`) that decodes bit-exactly. No files outside
this directory were modified.

```sh
./src/encoder/m1/palette_stream/build_and_test.sh
WERROR=1 SKIP_GATE=1 ./src/encoder/m1/palette_stream/build_and_test.sh
SANITIZE=1 WERROR=1 SKIP_GATE=1 ./src/encoder/m1/palette_stream/build_and_test.sh
```

## Spec map

| Clause | Used for |
|---|---|
| §5.4.7 | `sequence_scc_config()`: single-picture infers both force values (0 bits) |
| §5.18.3.3 | `screen_content_params()`: frame `allow_screen_content_tools` + `force_integer_mv` |
| §5.20.8.1 | `palette_mode_info()`: `has_palette_y`, size, cache/literal/delta |
| §5.20.8.2 | `get_palette_cache()` (above/left MI palettes, interleaved) |
| §5.20.8.4 | `palette_tokens()`: direction, identity rows, indices, edge replication |
| §5.20.8.5 | ColorOrder + `Palette_Color_Context` |
| §6.19.8.1/.3 | identity rows 0/1/2, sorted colours, cache reuse |
| §7.13.4 | palette prediction (`ColorMapY[y*4+i][x*4+j]`) |
| §8.3.2 | palette CDF selection (`Palette_Color_Context`, per-size tables) |

## Syntax position (the rule callers must get right)

```text
read_intra_y_mode()              // must be DC_PRED for a palette block
read_intra_uv_mode()             // normal chroma mode; CfL allowed
read_cfl_alphas()
palette_mode_info()              // only if YMode == DC_PRED and MiSize >= BLOCK_8X8
palette_tokens()                 // immediately after mode info
read_block_tx_size()/residual    // normal 3 planes; eob=0 = palette-only
```

`palette_mode_info()` is conditioned on `PlaneStart == 0 && YMode == DC_PRED`
(and `MiSize >= BLOCK_8X8`, `Block_Width/Height <= 64`,
`allow_screen_content_tools`). If the luma mode is not DC, **no symbol is
written at all** — even when the tool is enabled. `write()`/`read()` reproduce
that condition from the `y_mode` argument.

The adapter writes mode info and tokens in one call because the core's
`encode_palette_y()`/`decode_palette_y()` do both with the correct CDF
adaptation. The call site is what enforces the position: after UV mode, before
the residual.

## Luma-only rule

AV2 v1.0.0 has no chroma palette. `palette_mode_info()` has no U/V size, colour
or token syntax. Chroma is reconstructed with the normal intra path (DC/CfL)
after the palette-predicted luma. The adapter exposes no U/V palette API;
`test_reachability_and_luma_only` asserts `chroma_rule() == kIntraAfterLuma`.

## Transform-skip (`eob = 0`)

Intra `skip_flag` is only coded for IBC. A palette-only block therefore codes a
normal three-plane residual with `eob = 0` on Y/U/V. The reconstruction equals
the palette prediction. The minimal stream wrapper asserts this mode and the
gate checks the decoded frame against the writer reconstruction.

## API

```cpp
using namespace av2enc::m1::palette_stream;

// Frame/sequence flags.
ScreenContentFlags flags = palette_screen_content_flags();
validate_screen_content_flags(flags, error);
palette_allowed_for_block(flags, bsize, block_w, block_h);
write_sequence_screen_content_params(writer, flags);  // 0 bits for M1
write_frame_screen_content_params(writer, flags);     // 2 bits

// Per-frame writer/reader (own the per-MI PaletteState).
PaletteSyntaxWriter writer(flags);
writer.reset(mi_cols, mi_rows, error);
writer.write(enc, cdfs, mi_row, mi_col, bsize, block_w, block_h,
             visible_w, visible_h, palette_result_or_nullptr, y_mode,
             &mode_info, error);

PaletteSyntaxReader reader(flags);
reader.reset(mi_cols, mi_rows, error);
reader.read(dec, cdfs, mi_row, mi_col, bsize, block_w, block_h,
            visible_w, visible_h, y_mode, decoded_result, &mode_info, error);

// §8.3.2 CDF view + colour-order state.
y_mode_cdf(cdfs); y_size_cdf(cdfs);
color_cdf(cdfs, size, context_index);
identity_cdf(cdfs, prev_identity);
color_order_at(result, block_w, r, c, sample);  // hash/context/symbol/order
```

`PaletteModeInfo` records `has_palette_y`, `size`, whether direction is
signalled, direction, identity rows, cache length/use, and onscreen dimensions.
`PaletteState` stores sizes/colours for every MI the block covers, exactly like
the decoder, so the cache is correct even after partial/edge blocks.

`color_order_at()` exposes the per-sample decoder ColorOrder used by
§5.20.8.5/§7.13.4: `{hash, context_index, colour order, coded symbol}`. It is
derived, not stored, so encoder and decoder cannot disagree.

## Tests

`palette_stream_test.cpp` covers:

* screen-content flag validation and exact §5.4.7/§5.18.3.3 bits;
* palette gating: 4x4/4x8/8x4 rejected, 8x8..64x64 accepted, 64x128/128x64
  rejected, `MiSize >= BLOCK_8X8` cases such as 4x16/16x4 accepted;
* all §8.3.2 default palette CDFs pinned against `scc_cdf.h`;
* round-trips for 4x4..64x64 geometries and 1..8 distinct colours (1 colour
  becomes a legal 2-entry palette), with adaptive and disabled CDF updates;
* partial blocks (10x6 in 16x16, 17x9 in 32x32, 40x20 in 64x64, 33x17 in
  64x32) including edge replication;
* vertical stripes choosing column direction, flat lines using identity rows
  1/2, and 64x64 pinning direction to rows;
* cache hits across MI rows and writer/reader `PaletteState` equality;
* non-DC luma mode consuming zero palette symbols, while a DC non-palette
  block still codes `has_palette_y=0`;
* pinned golden bytes for 8x8/2 colours, 16x16/3 colours (columns), and
  32x32/3 colours with cache reuse.
* minimal stream encoding (NONE/HORZ/VERT/128x64) and 4x4/`allow_palette=0`
  rejection.

Pinned goldens (disabled CDF update):

| Case | Bytes |
|---|---|
| 8x8, 2 colours | `ea856975c1bd54b790deea0000` |
| 16x16, 3 colours, columns | `f066f4b4dd468c0000` |
| 32x32, 3 colours, above cache | `f3cf0f5e8009700000` |

## Gate

`build_and_test.sh` writes one IVF + reference Y4M per mode and requires
`dav2d` to reproduce the writer reconstruction byte-for-byte:

| Mode | Frame | Tile |
|---|---|---:|
| `single` | 64x64, one PARTITION_NONE palette block, 4 colours | 12 B |
| `split` | 64x64, HORZ two 64x32 leaves, 3 colours, above cache | 28 B |
| `vert` | 64x64, VERT two 32x64 leaves, 3 colours, left cache | 24 B |
| `wide` | 128x64, two 64x64 superblocks, 3 colours, left cache | 29 B |

All four decode bit-exactly with `dav2d` f9dafbc.

## Minimal M1 stream

`encode_minimal_palette_frame()` emits a legal still using the existing M1
sequence header and the gated SCC tile writer: 64x64 superblocks with
PARTITION_NONE/HORZ/VERT, palette luma, DC chroma, EOB=0 residual. This is the
narrowest end-to-end path that needs **no core edits**. It is deliberately not
a full M1 partition tree or full M1 mode decision.

## Full M1 integration call sites (not applied)

The following are the exact places to wire the adapter into the production M1
writer. Current line numbers are from the untouched tree.

1. **`src/encoder/m1/m1_headers.h`** `M1Config` (line 58): add
   `bool allow_screen_content_tools = false;`, `bool allow_palette = false;`,
   `bool force_integer_mv = true;` (and IBC flags if needed).
2. **`src/encoder/m1/m1_headers.cpp`** `write_frame_header_syntax()` lines
   170–171 currently write `allow_screen_content_tools=0`,
   `allow_intrabc=0`. Replace with `write_frame_screen_content_params()` (or the
   same two bits). No sequence-header change is needed: M1 already writes
   `single_picture_header_flag=1`, so §5.4.7 infers the force values and codes
   zero sequence bits.
3. **`src/encoder/m1/m1_syntax.cpp`**
   * `CdfState` (line 169): add `palette::PaletteCdfState palette;` and
     `init_palette_cdfs()` in `init_cdfs()` (line 216).
   * `Writer` constructor (line 304): reset a
     `palette::PaletteState palette_state_` with `mi_cols_/mi_rows_`.
   * `write_block()` (line 1372): after `code_uv_mode()` (line 1432) and
     **before** `code_tx(node, 0, ...)` (line 1438), call
     `PaletteSyntaxWriter::write(enc_, cdfs_.palette, r, c, node.bsize, w, h,
     onscreen_w, onscreen_h, node.palette, node.y_mode, ...)`. `code_tx()`
     already handles `eob=0` on all three planes, so `eob=0` is transform-skip
     with no further changes.
   * `m1_syntax.h` `BlockSyntax` (line 73): carry the palette result (a side
     array indexed by leaf, or a pointer) so `write_block()` can reach it.
     `PaletteResult` is a deep struct; a `levels`-style side array avoids
     changing the GPU POD bridge.
4. **`src/encoder/m1/m1_still.cpp`** `encode_block()` (line 1033):
   * after the block geometry and before `choose_luma_mode()` (line 1044),
     query `get_palette_cache()` + `analyze_palette()` (only when
     `palette_allowed_for_block`);
   * if palette wins, force `y_mode = DC_PRED`, predict luma with
     `predict_palette_block()`, set the luma `TxPlan` to `eob=0`, and keep the
     normal chroma mode/CfL path (luma-only rule);
   * fill the new palette field on `syntax::BlockSyntax` (line 1144) next to
     `uv_mode`/`eob` (lines 1147–1155);
   * `assemble_m1_frame()` (line 1851) then writes it with no other change.

Until those edits land, use the minimal stream above for gate coverage and the
call-site map for the production path.

## Limits

* 8-bit 4:2:0, luma-only.
* Palette size 2..8 (1 distinct colour is padded to a legal 2-entry palette).
* Minimal stream: 64x64 superblocks, NONE/HORZ/VERT, palette leaves only.
* CUDA mirror not included here; the shared core is `src/cuda/palette/`.
