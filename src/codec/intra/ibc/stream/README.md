# M1 intra-block-copy stream adapter

Integration layer for the existing IBC core (`src/encoder/ibc/`).  It owns the
syntax, stack, border and buffer-ring glue that the IBC module deliberately does
not: `still_syntax.cpp` and `main.cpp` are **not modified**.

Status: **16,608 checks pass; `dav2d` f9dafbc decodes four M1 IBC stream
variants bit-exactly at `--strict 1`.**

```sh
./src/encoder/m1/ibc_stream/build_and_test.sh
```

The four gate streams are:

| variant | contents | IVF | MD5 |
|---|---|---|---|
| `ibc_stream_gate` | 4 IBC blocks, 2 NEWMV + 2 NEARMV (one via a neighbour BV) | 101 B | `2e370666597348470833d7bd17921a3b` |
| `ibc_stream_gate.near` | 2 NEARMV blocks using the default `(-320,0)` candidate | 99 B | `9c75c923ade047302ce77cea37a3c25b` |
| `ibc_stream_gate.resid` | same 4 IBC blocks with `skip_flag=0` and EOB=0 residuals | 102 B | `f99b9e0fc62561f16b5ad83048632fc6` |
| `ibc_stream_gate.small` | 320x96 DC-only boundary partition, no IBC | 74 B | `bb0683f3889ca976eff0345be189b991` |

All four reconstruct to the same flat-128 Y4M.  The first three are 768x128,
one tile, 64x64 superblocks split HORZ into 64x32 leaves.

## Files

```text
src/encoder/m1/ibc_stream/
  ibc_cdfs.h              generated §9.3 CDFs used by §8.3.2
  generate_ibc_cdfs.py    regenerates ibc_cdfs.h from spec/tables/default_cdf.json
  ibc_syntax.h/.cpp       exact S()/L(1)/shell-MV writer and reader
  ibc_stack.h/.cpp        §7.12.2 candidate stack, §6.19.12 validity, ±1 border
  still_ibc_stream.h/.cpp    minimal M1 still-picture stream + IVF/Y4M
  ibc_stream_test.cpp     syntax, stack, border, partial-block and stream tests
  build_and_test.sh       standalone build, tests, dav2d gate
```

## Sequence / frame flag interactions

`still_headers.cpp` currently hard-codes `allow_screen_content_tools = 0` and
`allow_intrabc = 0`.  The adapter's writer emits the exact bits for:

```text
screen_content_params():
  allow_screen_content_tools = 1
  force_integer_mv           = 1        (only present when screen tools are on)

intrabc_params():
  allow_intrabc              = 1
  allow_global_intrabc       = 1
  allow_local_intrabc        = 0        (M1 stream; local ring is optional)

max_bvp_drl_bits_minus_1 = seq_max_bvp_drl_bits_minus_1   (M1 writes 0)
```

The M1 sequence header (`single_picture_header_flag = 1`) infers
`seq_force_screen_content_tools = SELECT_SCREEN_CONTENT_TOOLS` and
`seq_force_integer_mv = SELECT_INTEGER_MV`, so those two frame bits are coded.
`force_integer_mv = 1` sets `FrameMvPrecision = MV_PRECISION_ONE_PEL` and skips
the quarter/half/eighth-pel frame-header branch.

`allow_intrabc` requires at least one of global/local to be on.  The stream
writer rejects `allow_intrabc` without both screen tools and a usable region.
`allow_local_intrabc = 1` additionally requires the four 64x64 ring and the
`IBCCoded` grid; both are implemented and tested, but the gate stream uses the
global path only.

## Exact syntax

`ibc_syntax.*` uses the real §8.2 range coder and mutates one CDF copy per tile
exactly like the decoder.

| Syntax | CDF / descriptor | Source |
|---|---|---|
| `use_intrabc` | `TileIntrabcCdf[ctx]`, S() | §8.3.2 |
| `skip_flag` | `TileSkipCdf[ctx]`, S() | §8.3.2 |
| `intrabc_mode` | `TileIntrabcModeCdf`, S() | §8.3.2 |
| `intrabc_drl_mode` | **L(1)**, `m = max_bvp_drl_bits_minus_1 + 1` bits | §5.20.5.4 |
| `intrabc_precision` | `TileIntrabcPrecisionCdf`, S() (only NEWMV && !force_integer_mv) | §5.20.5.4 |
| `read_mv` | shell set/class/offset, col-MV unary, NS remainder, S()/L(1) | §5.20.7.20 |
| MV signs | one L(1) per nonzero component (sign derivation off) | §5.20.7.13 |

The shell MVD code covers every reachable `MvPrecision` (0, 1, 3, 4, 5, 6;
2 is unreachable) and the P6 joint-last-two-classes branch.  MVD values are
quantised by `1 << (6 - MvPrecision)`; the reader reconstructs
`clamp(RefStackMv + MVD)` with `mv_clamp_to_integer()`.

Writer/reader round-trips are exercised with frozen and adaptive CDFs, every
symbol value, every DRL index, both `intrabc_mode` values, and many magnitudes
across all shell classes.

## Candidate stack (§7.12.2)

`build_bv_stack()` implements the spec order, not just four named neighbours:

1. the eight `scan_point()` calls from the Find MV stack process;
2. the `scan_col(-3)` bottom and top points with the `MiColBase` check;
3. `add_to_ref_bv()` defaults:
   `(0,-SbHeight)`, `(-SbWidth-INTRABC_DELAY_PIXELS,0)`,
   `(0,-BlockHeight)`, `(-BlockWidth,0)`.

Neighbour entries are pruned by exact BV; defaults are appended without
pruning, like `add_to_ref_bv()`.  The stack is capped at
`max_bvp_drl_bits_minus_1 + 2` (spec maximum is 4 entries for M1's sequence
value 0).  Candidates beyond `NumMvFound` read as `(0,0)`, matching the spec's
zero-initialised `RefStackMv`.

`BvNeighbourMap` records every decoded 4x4 luma MI (`coded`, `ibc`, BV, and the
containing block's base position).  `begin_ibc_block()` advances the
four-region ring; `finish_ibc_block()` marks `IBCCoded` and the neighbour map.
`ibc::IbcBufferState` supplies the local/global validity arithmetic.

## Validity, range and ±1 border

`is_bv_encodable()` combines:

* open MV range `MV_LOW+1 .. MV_UPP-1` in 1/8 samples;
* `ibc::is_bv_valid()` (§6.19.7.12 tile/local/delay/wavefront);
* `is_bv_border_valid()` = `is_offset_mv_valid(-1,-1)` from §6.19.5.4.

The border rule is not optional for `dav2d --strict 1`: even an integer BV whose
source starts at `x=0` or `y=0` is rejected because the extra top-left reference
pixel is outside the tile.  A source at `x=64` passes.  `is_bv_offset_valid()`
exposes the general `(dx,dy)` form.

## Minimal M1 stream

`encode_ibc_stream()` writes one 8-bit 4:2:0 still:

* M1 sequence header from `av2enc::still::make_sequence_header_obu()` (unchanged);
* screen-content + intrabc frame header above;
* one tile with PARTITION_NONE / HORZ / VERT per 64x64 superblock;
* DC intra (`y_mode_set/index = DC`, `uv_mode = DC`, palette flag 0, EOB=0
  residual) or IBC (`use_intrabc`, `skip_flag`, `read_intrabc_info`).

Every IBC BV is validated before the first bit is emitted.  The writer
reconstructs DC/IBC blocks in raster order so later block copies see earlier
reconstruction.  `skip_flag = 1` writes no residual symbols; `skip_flag = 0`
writes an all-zero residual (EOB=0) after the IBC prediction.

This writer is a **minimal integration stream**, not the production M1 encoder:
CfL and 4:2:2/4:4:4 are rejected, and a block must fit the visible frame.
Partial-block copy/residual semantics are tested directly against
`ibc::copy_block()` / `ibc::copy_then_residual()`.

## Tests

`ibc_stream_test.cpp` covers:

* §8.3 CDF copies and defaults;
* shell-MV round-trips (all precisions, signs, classes, large magnitudes,
  frozen and adaptive CDFs, invalid quantum/context);
* `intrabc_info` round-trips (NEARMV/NEWMV, every DRL index, precision bit,
  MVD, malformed NEARMV target);
* `use_intrabc` / `skip_flag` round-trips and context errors;
* stack order, neighbour pruning, capacity, scan positions, exact
  `NPos`/`NPosBuf` context derivation;
* global validity, delay, wavefront, local ring eviction, MV range;
* ±1 border pass/fail;
* partial-width and partial-height copy, zero/nonzero transform-skip residual;
* four stream variants (NEARMV/NEWMV, neighbour BV, skip_flag 0/1, boundary
  partition) and rejection of an invalid BV.

An important integration finding: the existing `scc_stream.cpp` computes the
`use_intrabc` context as above+left.  That is wrong after an IBC block at a
superblock-row top: `NPos` can visit the same wide left block twice
(bottom-left and left), producing context 2 where above+left gives 1.  This
adapter implements the exact `add_neighbor()` lists and is why a stream with an
IBC block before the last superblock column now stays in sync.

## Exact production call sites (not edited)

1. `src/encoder/m1/still_headers.h/.cpp`
   * add screen-content/intrabc fields to `StillConfig`;
   * in `write_frame_header_syntax()`, replace the two `false` bits with
     `allow_screen_content_tools` / `force_integer_mv` and the
     `intrabc_params()` bits shown above.

2. `src/encoder/m1/still_syntax.cpp`, `Writer::write_block()`:
   * before `code_y_mode()`: write `use_intrabc` with
     `ibc_symbol_context()` and, if set, `write_skip_flag()` +
     `write_intrabc_info()`;
   * when `use_intrabc` is set, skip `code_y_mode()`/`code_uv_mode()` and the
     palette path; code residual only when `skip_flag == 0`;
   * maintain one `ibc::IbcBufferState` and `BvNeighbourMap`, calling
     `begin_ibc_block()` before mode info and `finish_ibc_block()` after
     reconstruction;
   * use `ibc::copy_block()` / `copy_then_residual()` for the prediction.

3. `src/main.cpp`: add the user-facing tool flag (e.g. `--still-ibc`) that sets the
   new `StillConfig` fields.  No parser/encoder change is needed for the component
   tests.

## Limits

* 8-bit 4:2:0, 64x64 superblocks, integer BVs for the stream writer.
* `NEWMV`/MVD is exact in the syntax adapter; the stream writer can emit it
  with `force_integer_mv = 1` (one-pel MVD).
* `morph_pred`, `enable_mvd_sign_derive`, 128x128 SBs, 4:2:2/4:4:4 and the
  derived-SMVP/ref-MV-bank stack fills are out of scope.  M1's sequence header
  has all of those disabled.
* No core files were changed.
