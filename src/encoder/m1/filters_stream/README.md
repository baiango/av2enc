# M1 filter/restoration syntax adapters

Host-only, integration-ready bitstream syntax for the three M1 in-loop filter
tools:

| Tool | Header | Tile syntax |
|---|---|---|
| CDEF | §5.18.7.10 `cdef_params()` | §5.20.10.1 `read_cdef()` |
| CCSO | §5.18.7.12 `ccso_params()` | §5.20.10.2 `read_ccso()` |
| Wiener NS / LR | §5.18.7.11 `lr_params()` | §5.20.10.4–§5.20.10.6 |

Plus §5.4.10 `sequence_filter_config()` flags for all three.

This directory owns **only** syntax. It does not decide filter parameters, does
not touch pixels, and does not own the frame OBU. The reference implementations
it mirrors are:

* `src/cuda/m1/cdef/m1_cdef_index_syntax.h` (CDEF `cdef_idx` + §8.3.1 context);
* `src/cuda/m1/cdef/m1_cdef_cdf.h` (CDEF default CDFs);
* `src/cuda/m1/restoration/` / `src/encoder/filters/restoration/restoration.h`
  (CCSO/LR parameter structures and the §7.20 coefficient ranges);
* `src/entropy/av2` (normative §8.2 symbol coder).

## Files

```text
src/encoder/m1/filters_stream/
  filters_stream.h            public API
  filters_stream.cpp          writers + mirrors
  filter_cdfs.h               §9.3 default CDFs (generated)
  bit_io.h                    BitReader + f/ns/tu helpers
  filters_stream_test.cpp     pinned goldens + round-trips
  build_and_test.sh           standalone build/run (no CUDA, no CMake)
  README.md                   this file
```

Run:

```sh
cd src/encoder/m1/filters_stream
./build_and_test.sh
ASAN=1 ./build_and_test.sh
```

Build integration (owner: integrator, `CMakeLists.txt` is outside this
directory): add `src/encoder/m1/filters_stream/filters_stream.cpp` to
`av2enc_core` (it is host-only and has no CUDA link dependency). The public
headers are already reachable through the existing `src` include directory.
The test binary is standalone and deliberately not added to `tests/`.

## Exact call sites and order

The AV2 syntax order is not negotiable. `frame_header_info()` reads:

```text
deblocking_filter_params()   // includes df_delta_q_present bits
gdf_params()                 // no bits for M1 (enable_gdf == 0)
cdef_params()
lr_params()                  // includes frame-level Wiener filters
ccso_params()
read_tx_mode()               // tx_mode_select
...
f(2) reduced_tx_set
```

### 1. Sequence header — `src/encoder/m1/m1_headers.cpp`

Replace the current §5.4.10 block with a single call:

```cpp
#include "encoder/m1/filters_stream/filters_stream.h"

filters_stream::SequenceFilterConfig sf;
sf.disable_loopfilters_across_tiles = false;
sf.enable_cdef = cdef.enable;
sf.enable_gdf = false;
sf.enable_restoration = <true when any FrameRestorationType != NONE>;
sf.enable_ccso = <true when any ccso_planes[p]>;
sf.ccso_unit_matches_sb_size = true;   // 64x64 CCSO units
// lr_disable_luma_pc_wiener = true (PC-Wiener is blocked)
// lr_disable_luma_wiener_ns = false
// lr_uv_present = true if any chroma LR
filters_stream::write_sequence_filter_config(
    writer, sf, /*single_picture_header_flag=*/true, &error);
```

The writer emits the extra restoration/CCSO rows (`lr_tools_disable`,
`lr_tools_uv_present`, `ccso_unit_matches_sb_size`) that the current M1 code
does not. GDF is rejected: M1 has no GDF tables.

### 2. Frame header — `src/encoder/m1/m1_headers.cpp`

Keep the existing deblocking block **including the one `df_delta_q_present`
f(1) bit per enabled plane**. That order is mandatory: the four bits sit after
`apply_deblocking_filter[0..3]` and before `tx_mode_select`. The current M1
code already does this (commit `927f385`); do not move them.

Then write, in this order:

```cpp
// §5.18.7.10
filters_stream::CdefHeaderContext cdef_ctx;
cdef_ctx.enable_cdef = cfg.cdef.enable;
cdef_ctx.single_picture_header_flag = true;
filters_stream::CdefParams cdef_params;   // fill from M1CdefConfig
filters_stream::write_cdef_params(writer, cdef_params, cdef_ctx, &error);

// §5.18.7.11
filters_stream::LrHeaderContext lr_ctx;
lr_ctx.enable_restoration = <any LR>;
lr_ctx.sb_size = 64;
lr_ctx.base_q_idx = cfg.base_q_idx;
filters_stream::LrParams lr_params;       // fill from the LR plan
filters_stream::write_lr_params(writer, lr_params, lr_ctx, &error);

// §5.18.7.12
filters_stream::CcsoHeaderContext ccso_ctx;
ccso_ctx.enable_ccso = <any CCSO>;
ccso_ctx.sb_size = 64;
ccso_ctx.ccso_unit_matches_sb_size = true;
filters_stream::CcsoParams ccso_params;   // fill from the CCSO plan
filters_stream::write_ccso_params(writer, ccso_params, ccso_ctx, &error);

// then the existing tx_mode_select / reduced_tx_set bits.
```

`write_lr_params()` also emits `read_wienerns_filter(plane, 0, 0, 1)` for
frame-level filters. In frame-filter mode every descriptor is raw (`f`, `L`,
`ns`) — no arithmetic-coded `S()` symbols — so it is safe to call from the
header writer.

`LrHeaderContext::lr_tools_disable` must mirror the sequence-header
`lr_tools_disable` values exactly (same for
`CcsoHeaderContext::ccso_unit_matches_sb_size`): the frame-header reader derives
`FrameRestorationType` tool lists and `CcsoLumaSizeLog2` from what the sequence
header signalled.

`cdef_params()` maps the decoded-domain strength fields (`sec` 4 <-> syntax 3)
and emits the `pri_zero` shortcuts. `CcsoParams::offset_idx` is the list of
`tu(7)` values in `d0, d1, band` order; `ccso_fill_offsets()` builds the
`restoration::CcsoParams::offsets` table from it.

### 3. Tile start / per-SB

At the start of every superblock, before `read_partition` (i.e. before the
first `do_split` symbol):

```cpp
filters_stream::cdef_clear_superblock(&cdef_state, mi_row, mi_col);  // §5.20.9.6
filters_stream::lr_write_block(encoder, lr_state, mi_row, mi_col,
                               /*w4=*/16, /*h4=*/16,
                               /*tile_mi_row_start=*/0,
                               /*tile_mi_col_start=*/0,
                               lr_decisions, &error);              // §5.20.10.4
```

`read_lr()` is called **once per superblock**, not per coding block. It runs
before the partition symbol because `decode_partition()` calls
`read_lr(r, c, SbSize)` when `bSize == SbSize`. `lr_block_unit_count()` returns
the exact number of `LrUnitDecision` entries the call consumes (plane-major,
then unit row, then unit column). A `RESTORE_NONE` plane contributes no units.

In `m1_syntax.cpp`, this is the `kFlagSbRoot` branch at the top of
`write_node()`, before `code_do_split()`.

### 4. Tile start / per coding block

In `intra_frame_mode_info`, `read_gdf()`, `read_cdef()` and `read_ccso()` run
after `intra_segment_id()` and before `read_delta_qindex()`/`read_intra_y_mode()`.
M1 codes no segment/skip syntax, so in `write_block()`:

```cpp
// first symbols of the leaf
if (cdef.enable) {
  filters_stream::cdef_write_block(encoder, cdef_state,
                                   /*tile_mi_col_start=*/0,
                                   /*tile_mi_row_start=*/0,
                                   mi_row, mi_col, node.cdef_idx, &error);
}
filters_stream::ccso_write_block(encoder, ccso_state, mi_row, mi_col,
                                 ccso_blk, &error);
// then code_y_mode(), code_uv_mode(), residual...
```

`ccso_blk[3]` is the `CcsoBlks[plane]` value for this block; the call is a no-op
unless the block origin is a CCSO unit origin, so it can be issued from every
leaf. `read_ccso()` derives its context from the left neighbour's `CcsoBlks`
entry, exactly as §8.3 specifies. `ccso_state` must be seeded from the plan's
block grid or updated as symbols are written; the encoder's `blk` value and the
grid entry must agree.

CDEF `cdef_idx` is a no-op when `CdefStrengths == 1` (the decoder reads no
symbol and every block is forced to set 0) and when the 64x64 origin was
already coded.

### 5. State lifetime

* One `CdefTileState`, one `CcsoTileState`, one `LrState` per tile. M1 has one
  tile.
* Initialise them after the header is written: `cdef_tile_init()`,
  `ccso_tile_init()`, `lr_state_init()`.
* `LrState` owns the CDF copies and the Wiener bank state. Do not copy or share
  it between tiles/frames: the §8.3 adaptation is stateful. With M1's
  `disable_cdf_update = 1` the defaults are frozen, but the bank state and the
  per-unit grids are still per frame.
* `lr_state_init()` seeds `RefLrWienerNs[plane][class][bank][coeff]` exactly as
  `decode_tile()` does (`min + ((1 << k) >> 1)`). `lr_write_unit()` never selects
  an unwritten bank slot.

## API summary

```cpp
namespace av2enc::m1::filters_stream {

// §5.4.10
std::size_t write_sequence_filter_config(BitWriter&, const SequenceFilterConfig&,
                                         bool single_picture_header_flag,
                                         std::string* error);
bool read_sequence_filter_config(BitReader&, bool single_picture_header_flag,
                                 bool sb_64x64, SequenceFilterConfig*,
                                 std::string* error);

// §5.18.7.10 / §5.20.10.1
std::size_t write_cdef_params(BitWriter&, const CdefParams&,
                              const CdefHeaderContext&, std::string*);
bool read_cdef_params(BitReader&, const CdefHeaderContext&, CdefParams*,
                      std::string*);
bool cdef_tile_init(CdefTileState*, int mi_cols, int mi_rows, int strengths,
                    int sb_width_log2, std::string*);
void cdef_clear_superblock(CdefTileState*, int mi_row, int mi_col);
bool cdef_write_block(av2::Encoder&, CdefTileState&, int tile_mi_col_start,
                      int tile_mi_row_start, int mi_row, int mi_col, int idx,
                      std::string*);
bool cdef_read_block(av2::Decoder&, CdefTileState&, int tile_mi_col_start,
                     int tile_mi_row_start, int mi_row, int mi_col, int* idx,
                     std::string*);

// §5.18.7.12 / §5.20.10.2
std::size_t write_ccso_params(BitWriter&, CcsoParams&,
                              const CcsoHeaderContext&, std::string*);
bool read_ccso_params(BitReader&, const CcsoHeaderContext&, CcsoParams*,
                      std::string*);
bool ccso_fill_offsets(const CcsoPlaneParams&, restoration::CcsoParams*,
                       std::string*);
bool ccso_tile_init(CcsoTileState*, const CcsoParams&, int mi_cols,
                    int mi_rows, int sub_x, int sub_y, std::string*);
bool ccso_write_block(av2::Encoder&, CcsoTileState&, int mi_row, int mi_col,
                      const int blk[3], std::string*);
bool ccso_read_block(av2::Decoder&, CcsoTileState&, int mi_row, int mi_col,
                     int blk[3], std::string*);

// §5.18.7.11 / §5.20.10.4–§5.20.10.6
std::size_t write_lr_params(BitWriter&, LrParams&, const LrHeaderContext&,
                            std::string*);
bool read_lr_params(BitReader&, const LrHeaderContext&, LrParams*,
                    std::string*);
bool lr_state_init(LrState*, const LrParams&, int mi_cols, int mi_rows,
                   int sub_x, int sub_y, int sb_size, std::string*);
bool lr_unit_range(const LrState&, int plane, int mi_row, int mi_col, int w4,
                   int h4, int tile_mi_row_start, int tile_mi_col_start,
                   LrUnitRange*, std::string*);
int lr_block_unit_count(const LrState&, int mi_row, int mi_col, int w4, int h4,
                        int tile_mi_row_start, int tile_mi_col_start);
bool lr_build_decisions(const LrState&, int mi_row, int mi_col, int w4, int h4,
                        int tile_mi_row_start, int tile_mi_col_start,
                        std::vector<LrUnitDecision>*, std::string*);
bool lr_write_block(av2::Encoder&, LrState&, int mi_row, int mi_col, int w4,
                    int h4, int tile_mi_row_start, int tile_mi_col_start,
                    const std::vector<LrUnitDecision>&, std::string*);
bool lr_read_block(av2::Decoder&, LrState&, int mi_row, int mi_col, int w4,
                   int h4, int tile_mi_row_start, int tile_mi_col_start,
                   std::vector<LrUnitDecision>*, std::string*);
bool lr_write_unit(av2::Encoder&, LrState&, int plane, int unit_row,
                   int unit_col, const LrUnitDecision&, std::string*);
bool lr_read_unit(av2::Decoder&, LrState&, int plane, int unit_row,
                  int unit_col, LrUnitDecision*, std::string*);
}
```

## Field mapping from the existing plan structures

### CDEF

`M1CdefConfig` → `CdefParams`:

| `M1CdefConfig` | `CdefParams` |
|---|---|
| `damping` (3..6) | `damping_minus_3 = damping - 3` |
| `num_strengths` (1..8) | `strengths_minus_1 = num_strengths - 1`, `num_strengths` |
| `on_skip_txfm` | `on_skip_txfm_frame_enable` |
| `strengths[i].y_pri/y_sec/uv_pri/uv_sec` | `set[i]` (same decoded domain) |

Tile: `CdefStagePlan::cdef_idx` is one entry per 64x64 luma block, row-major.
Copy the value into `BlockSyntax::cdef_idx` (already present) or look it up by
`(mi_row/16, mi_col/16)` at the call site.

### CCSO

`restoration::CcsoParams` stores the **computed** offsets, not the syntax
indices. To write the header you must keep the `tu(7)` `ccso_offset_idx` list
alongside the plan (or re-derive it from the offsets when the scale factors are
known). `CcsoPlaneParams::offset_idx` is that list; `ccso_fill_offsets()` fills
`restoration::CcsoParams::offsets` for the §7.19 filter. `ccso_offset_count()`
is the exact number of indices (`bands * interval^2`, or `bands` for
`bo_only`).

`CcsoBlks[plane]` comes from the CCSO plan/search. `CcsoTileState::blks` is the
encoder-side grid; `ccso_write_block()` updates it as it codes and uses it for
the left-neighbour context. The per-block value passed to the writer must be
the plan's value.

### Wiener NS / LR

`restoration::LoopRestorationParams` → `LrParams`:

| `LoopRestorationParams` | `LrParams` |
|---|---|
| `frame_type[p]` | `frame_type[p]` |
| `frame_filters_on[p]` | `frame_filters_on[p]` |
| `num_filter_classes` (plan) | `num_filter_classes` (luma) |
| `unit_size[p]` | `luma_shift` / `chroma_shift` (see below) |
| frame coefficients | `frame_filter[p][c].coeffs` |
| per-unit `unit_type` / `unit_coeffs` | `LrUnitDecision` list |

`luma_shift` is 0..3 with `LoopRestorationSize[0] = 512 >> luma_shift`;
`chroma_shift` is 0..3 with `LoopRestorationSize[1] = (512 >> max(subX,subY)) >>
chroma_shift`. For M1 (4:2:0, 64x64 SB) the legal luma sizes are 64/128/256/512
and chroma sizes 32/64/128/256. `lr_state_init()` enforces the §6.17.7.7 bounds
and power-of-two rule.

For per-unit filters, fill `LrUnitDecision` from `LrType` and the unit
coefficients:

* `type` = `RESTORE_NONE` or `RESTORE_WIENER_NONSEP` (or `PC_WIENER` only for a
  switchable frame type, which M1 does not signal);
* `merged = false` unless the encoder deliberately reuses a bank;
* `subset` = the smallest `Wiener_Ns_Taps_Present` subset that covers every
  non-zero coefficient (0..3 luma, 0..2 chroma);
* `uv_sym = true` when the chroma pairs `(6,7), (8,9), ...` are equal; the
  writer validates this and codes only the first of each pair;
* `coeffs[j]` = the desired coefficient in `[min, min + 2^k)`; absent taps must
  be zero (the writer sets them to zero).

Before calling it, copy the plan's `LrType[plane]` and `LrWienerNs[plane]`
grids into `LrState::unit_type[plane]` and `LrState::unit_coeffs[plane]`
(row-major, 18 coefficients per unit).  `lr_build_decisions()` is the
convenience path: it reads those grids and fills the list with `merged = false`,
the smallest covering subset, and `uv_sym` when the chroma pairs are equal.  Use
the explicit `LrUnitDecision` path when you want bank merges or a specific
subset.

`LrState::unit_coeffs` is only meaningful for per-unit filters.  When
`frame_filters_on[plane]` is set, the §7.20 stage must select coefficients from
`frame_coeffs[plane][class]` using the class map (class 0 for chroma), not from
`unit_coeffs`.

Frame filters: `lr_write_params()` codes the class/match/merge/coefficient
syntax and rewrites `frame_filter[p][c].coeffs` to the effective decoded values.
Call it **before** `lr_state_init()` so the state sees the final coefficients.
`lr_state_init()` then copies them into `LrState::frame_coeffs`.

## Conflicts / blockers with the current tree

1. **`M1Config` has no CCSO/LR fields.** `src/encoder/m1/m1_headers.h` currently
   only has `M1CdefConfig cdef`. The integrator must add CCSO and LR sections
   (or pass `SequenceFilterConfig`/`CcsoParams`/`LrParams` separately). This
   module does not edit that header.
2. **`m1_headers.cpp` hardcodes `enable_restoration = 0` and
   `enable_ccso = 0`,** and has no `lr_params()`/`ccso_params()` call. The
   current `cdef_params()` placement is correct; `lr_params()` and
   `ccso_params()` must be inserted between it and `tx_mode_select()`.
3. **`df_delta_q_present` bits.** They are already in `m1_headers.cpp` after the
   four `apply_deblocking_filter[]` bits and before `cdef_params()`. Keep that
   order. `write_cdef_params()` assumes the caller has already written them.
4. **`BlockSyntax` has `cdef_idx` but no CCSO/LR fields.** CCSO values can be
   read from the plan's `CcsoBlks` grid at the call site. LR decisions are
   per-unit, not per-leaf, and are passed as a list to `lr_write_block()`.
5. **`m1_syntax.cpp` currently calls `cdef_encode_block_index()` directly.**
   `cdef_write_block()` is the same call bound to the §8.2 coder. Do not call
   both. `m1_syntax.cpp` has no `read_ccso()` or `read_lr()`; the `Writer` needs
   `CcsoTileState` and `LrState` members.
6. **`read_lr()` is once per superblock, before the partition symbol.** The
   restoration README's "invoked from the block walk" wording can be read as
   per coding block; the syntax is explicit (`decode_partition()` calls
   `read_lr(r, c, SbSize)` when `bSize == SbSize`). Calling it per leaf would
   desynchronise the stream.
7. **`restoration::CcsoParams` does not retain `ccso_offset_idx`.** It stores
   only `CcsoFilterOffset`. A writer needs the separate index list; keep
   `CcsoPlaneParams::offset_idx` in the encoder plan (or reconstruct it from
   offsets). This is the one place where the existing restoration API is not
   sufficient for bitstream output.
8. **`restoration.h` says `ccso_offset_idx` is 7 bits.** The syntax is
   `tu(7)` (truncated unary, 0..7), not `f(7)`. This writer follows the spec;
   the comment in `restoration.h` is stale.
9. **The §7.20.4 class map cannot be derived from the spec text.**
   `Pc_Wiener_Sub_Classify2` (the 16-entry `SubclassLookup` per
   `num_filter_classes_idx`) is an attachment header, not printed in the
   specification. `num_filter_classes_idx` is signalled, but the encoder must
   use the same table as the decoder. `LrParams::class_map` carries a
   caller-supplied 16-entry lookup and the per-MI `FilterClass` grid;
   `lr_state_init()` fails loudly when `num_filter_classes > 1` and neither the
   lookup nor the legacy `subclass_grid` is present. This is the same blocker
   documented in `src/cuda/m1/restoration/README.md`.
10. **`disable_cdf_update`.** M1 still pictures set it; the CDFs stay at their
    §9.3 defaults, and the pinned payloads are reproducible. With it cleared,
    encoder and decoder still adapt identically; the tests cover the frozen
    case.

## Tests

`filters_stream_test.cpp` covers:

* pinned bit strings for `sequence_filter_config()` (all off / CDEF only / all
  three), `cdef_params()`, `ccso_params()` (band-offset-only with `tu(7)`
  offsets), and simple `lr_params()`;
* header round-trips through the mirror readers, including coded-lossless and
  non-single-picture suppression;
* pinned §8.2 payload bytes for `read_cdef()` (two 64x64 blocks, context 2),
  `read_ccso()` (two CCSO units, context 2), and `read_lr()` (four units with
  per-unit coefficients);
* `read_lr()` unit-range arithmetic, luma/chroma unit sizes, per-unit Wiener
  coefficients with `uv_sym`, bank merge reuse, and frame-level Wiener filters
  with multiple classes;
* the class-map blocker (`Pc_Wiener_Sub_Classify2` error).

The current status on this host:

```text
filters_stream_test: PASS
ASan/UBSan: PASS
```

No CUDA, no stream gate. Enabling these tools changes the pinned M1 MD5s; the
stream gates are listed in `src/cuda/m1/cdef/README.md` and
`src/cuda/m1/restoration/README.md`.
