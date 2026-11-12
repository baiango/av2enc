# M2 compound / skip-mode inter mode-info syntax

`src/encoder/m2/compound_stream/` is a **spec-only component**: it implements,
writes, independently reads and gates the block-level inter mode information
that the M2 core (`src/encoder/m2/m2_syntax.cpp`) does not emit.  It does not
touch `CMakeLists.txt`, `src/main.cpp`, or the M2 core; the core is reused
read-only.

The audit (`docs/spec-audit/04-tile-block.md` §5/§7) lists these block elements
as having no writer:

| Element | § | Before | Now |
|---|---|---|---|
| `skip_mode S()` | 5.20.5.9 | ABSENT (`skip_mode_present = 0`) | written + read |
| `single_ref S()` loop | 5.20.7.12 | ABSENT (`NumTotalRefs = 1`) | written + read for every RefFrame |
| `compound_mode_same_refs S()` | 5.20.7.6 | ABSENT (gate writer refuses) | written + read |
| `use_optflow S()` | 5.20.7.6 | ABSENT | written + read (+ OPFL application) |
| compound reference selection (`comp_mode`, `comp_ref`, `read_compound_ref`) | 5.20.7.10-.11 | minimal gate writer | written + read |
| `num_same_ref_compound` / `enable_masked_compound` | 5.4.6 | pinned off | plumbed and enforced |

## Files

```text
compound_syntax.h/.cpp   writer: CDFs, §8.3.2 contexts, block grammar
compound_reader.cpp      independent reader (own CDFs, own neighbour grid)
compound_stream.h/.cpp   three-frame M2 stream builder + recon + byte control
compound_gate.cpp        dav2d gate (--strict 1 --threads 1) + control
compound_test.cpp        writer/reader round-trip and negative unit tests
build_and_test.sh        standalone host build (no CMake change)
```

## Grammar emitted per inter leaf

Order is §5.20.7.1 / .6 / .8 / .16:

```text
read_skip_mode()        skip_mode S()                  §5.20.5.9
read_is_inter()         is_inter S() (skipped if skip_mode) §5.20.7.3
read_skip()             skip_flag S()                  §5.20.5.10
inter_block_mode_info():
  read_ref_frames()     comp_mode S()                  §5.20.7.10
                        read_compound_ref()            §5.20.7.11
                        or read_single_ref() loop      §5.20.7.12
  YMode                 compound_mode_same_refs S()    §5.20.7.6
                        or is_joint S() + compound_mode_non_joint S()
                        or single_mode S()
  (use_optflow)         use_optflow S()                §5.20.7.6
  read_drl_idx()        drl_mode S()                   §5.20.7.8
  read_compound_type()  comp_group_idx S()             §5.20.7.16
                        compound_type S() / read_wedge_mode() / mask_type
  cwp_idx S()                                          §5.20.7.6
```

Residual syntax is not coded here (the gate uses `skip_flag = 1`).  NEWMV MVD,
warp, BAWP, TIP, refine-MV and adaptive-MVD are rejected explicitly: those are
already covered by other components and a block that needs them is not
representable by this writer, so it fails instead of desynchronising.

## Exact CDFs and contexts (§9.3, §8.3.2)

| Symbol | CDF | Context |
|---|---|---|
| `skip_mode` | `TileSkipModeCdf[3]` | sum of the 2 neighbours' `SkipModes` (§9.3 p596) |
| `is_inter` | `TileIsInterCdf[4]` | `NIntra` of the 2 neighbours |
| `skip_flag` | `TileSkipCdf[6]` | neighbour `Skips` + **3 when `skip_mode`** |
| `comp_mode` | `TileCompModeCdf[5]` | `NSingle`/`check_backward`/`NIntra` (§9.3 p595) |
| `comp_ref` | `TileCompRef0Cdf[3][6]`, `TileCompRef1Cdf[3][2][6]` | this/next `count_refs(ref)`; `bitType` from `FrameDistance` signs |
| `single_ref` | `TileSingleRefCdf[3][6]` | same `count_refs` context as `comp_ref` (§9.3 p597) |
| `is_joint` | `TileIsJointCdf[2]` | `is_same_side() \|\| firstDist != secondDist \|\| restricted mismatch` |
| `compound_mode_non_joint` | `[5][6]` | `NewMvContext` |
| `compound_mode_same_refs` | `[5][5]` | `NewMvContext`; 4 symbols → YMode `19,20,22,23` |
| `use_optflow` | `TileUseOptflowCdf[2]` | `YMode != NEAR_NEARMV` |
| `single_mode` | `TileSingleModeCdf[5]` | `NewMvContext` |
| `drl_mode` | `TileDrlModeCdf[3][5]` / `TileSkipDrlModeCdf[3]` | `Min(idx,2)`, `NewMvContext`; skip mode uses the SkipDrl table |
| `comp_group_idx` | `TileCompGroupIdxCdf[12]` | `ctxs[0]+ctxs[1]+min>0+offset*6`; single neighbour using **FurthestFuture** contributes 2 |
| `compound_type` | `TileCompoundTypeCdf[3]` | none |
| `wedge_quad/angle/dist1/dist2`, `mask_type`, `wedge_sign` | wedge tables + `L(1)` | `wedge_quad` selects the angle CDF |
| `cwp_idx` | `TileCwpIdxCdf[4]` | loop index |

`NewMvContext` follows §7.11.2-.3: both reference lists are scanned even for a
single-prediction block (a compound neighbour whose list 1 matches
`RefFrame[0]` still sets `found`), and `NewMvCount` only increments for
`has_newmv_for_list(candMode, list)`.

Contexts needed only by `skip_mode`, `use_optflow` and the compound symbols are
local to this directory because `m2_cdf_tables.h` does not generate
`Default_Skip_Drl_Mode_Cdf` or `Default_Use_Optflow_Cdf`.

## Interactions

* **`num_same_ref_compound`** (`§5.4.6`, `f(2)`).  A same-reference pair
  `(r,r)` is only representable when `r < NumSameRefCompound`; the inverse of
  `read_compound_ref()` emits the implied `comp_ref` and the `ref--` re-visit
  exactly as the decoder does.  With `NumSameRefCompound = 0` the pair `(r,r)`
  cannot be signalled and the writer rejects it.  `Min(num_same_ref_compound,
  NumTotalRefs)` is applied by `num_same_ref_compound()`.
* **`enable_masked_compound = 0`** leaves `comp_group_idx` inferred to 0, so
  `compound_type` is inferred `COMPOUND_AVERAGE`; WEDGE and DIFFWTD are not
  signalable.  The writer refuses a masked decision in that configuration.
* **`enable_cwp = 0`** removes `cwp_idx` and pins `CwpIdx = CWP_EQUAL`.
* **`skip_mode`** removes `comp_mode`/`comp_ref`/`single_ref` (the reference
  pair comes from `skip_mode_frames()`), forces `YMode = NEAR_NEARMV`,
  `use_optflow = 0`, `COMPOUND_AVERAGE`, and `interp_filter = EIGHTAP_SHARP`.
  `skip_flag` is still coded independently and its context gains the `+3`
  half.  One DRL index is coded with `TileSkipDrlModeCdf`.
* **`opfl_refine_type`**:
  * `REFINE_NONE` – no symbol, `use_optflow = 0`.
  * `REFINE_SWITCHABLE` – `use_optflow S()` when `opfl_allowed_for_refs()`
    (one past + one future reference), `YMode != GLOBAL_GLOBALMV` and the
    block is at least 8x8.
  * `REFINE_ALL` – inferred **after** `compound_type`/`cwp_idx` and only for a
    compound `COMPOUND_AVERAGE` `NEAR_NEARMV`/`NEAR_NEWMV`/`NEW_NEARMV`/
    `NEW_NEWMV` block with `CwpIdx == CWP_EQUAL`, `!skip_mode` and allowed
    references (§5.20.7.6 p195).  During DRL/compound-type the value is still
    0, exactly like the decoder.
* **OPFL application**: `build_compound_stream()` runs
  `inter::temporal::opfl_compute()` for `use_optflow` blocks.  The gate stream
  is constructed so both predictions are equal at the starting MVs; every
  OPFL delta is therefore zero and the base compound prediction is exact.  A
  non-zero refinement is reported as an error rather than silently diverging
  from the decoder.

## B-frame interaction (future reference)

`use_optflow` (and `is_joint`) need one past and one future reference:
`opfl_allowed_for_refs()` returns `(d0 <= 0) ^ (d1 <= 0)`.  With a single
already-decoded past frame that is never true.  The gate therefore uses:

```text
decode/output order : frame 0 (key, oh 0) -> frame 1 (P, oh 4) -> frame 2 (oh 2)
frame 2 references  : ref 0 = key (oh 0, d = +2), ref 1 = frame 1 (oh 4, d = -2)
```

Frame 1 is decoded before frame 2 but carries the larger order hint, so frame 2
is a B-like frame *for prediction distances* while `monotonic_output_order_flag
= 1` keeps output order equal to decode order.  No display reordering, SEF,
TIP or `immediate_output_frame = 0` path is needed.  This is the smallest legal
stream that exercises `use_optflow S()`.

## M2 core call sites (read-only, not edited)

These are the places a future integration would connect to; the component is
deliberately not wired into them.

| Call site | Current value | Required for this component |
|---|---|---|
| `src/encoder/m2/m2_headers.cpp:154` | `enable_masked_compound = 0` | `f(1)` from config |
| `src/encoder/m2/m2_headers.cpp:168` | `num_same_ref_compound = 0` (`f(2)`) | `Min(config, NumTotalRefs)` |
| `src/encoder/m2/m2_headers.cpp:172-173` | `enable_cwp = 0`, `enable_imp_msk_bld = 0` | `enable_cwp` from config |
| `src/encoder/m2/m2_headers.cpp:175` | `enable_opfl_refine = REFINE_NONE` (`f(2)`) | `opfl_refine_type` |
| `src/encoder/m2/m2_headers.cpp:291` | frame `reference_select = 0` | `1` for a compound frame |
| `src/encoder/m2/m2_headers.cpp:292` | frame `skip_mode_present = 0` | §5.18.8.2 `skip_mode_params()` |
| `src/encoder/m2/m2_headers.h:31` | `M2Config::num_total_refs = 1` | `2` for the compound frame |
| `src/encoder/m2/m2_syntax.cpp:1721` `write_block()` | `is_inter`, `skip_flag`, `write_inter_mode_info()` on a single reference | replace the mode-info half with `ModeInfoWriter::code_block()` |
| `src/encoder/m2/m2_syntax.cpp:1657` `write_inter_mode_info()` | `single_mode`/DRL/MVD only | compound/skip-mode part is this component |
| `src/encoder/m2/m2_frame.cpp` `M2Options` | one reference, `skip_flag = 1` default | per-leaf compound decision + recon |
| `src/encoder/m2/multiref/ref_map.h` | `RefFrameMap` | used unchanged for NumTotalRefs/FrameDistance/FurthestFuture |

`src/encoder/inter/advanced/` (compound blend, wedge tables) and
`src/encoder/inter/temporal/opfl.{h,cpp}` are reused read-only.

## API

```cpp
// Writer: one tile payload, one 64x64 PARTITION_NONE leaf per superblock.
bool write_compound_tile(const std::vector<CompoundBlockPlan>& blocks,
                         int width, int height,
                         const CompoundFrameConfig& config,
                         std::vector<std::uint8_t>& payload,
                         std::string& error);

// Independent reader for the same payload (own CDFs and contexts).
bool read_compound_tile(std::span<const std::uint8_t> payload,
                        int width, int height,
                        const CompoundFrameConfig& config,
                        std::vector<CompoundBlockResult>& blocks,
                        std::string& error);

// Block-level writer/reader (used by the tile functions and tests).
ModeInfoWriter writer(config, mi_cols, mi_rows);
writer.code_block(encoder, plan, &result, error);
ModeInfoReader reader(config, mi_cols, mi_rows);
reader.read_block(decoder, mi_row, mi_col, w4, h4, &result, error);

// Three-frame stream (frames 0/1 come from the M2 core).
bool build_compound_stream(const CompoundStreamInput&, CompoundStreamOutput&,
                           std::string& error);

// "All disabled" byte-identity control against the M2 core writer.
bool check_disabled_byte_identity(int width, int height, uint8_t base_q_idx,
                                  bool* identical, size_t* bytes,
                                  std::string& error);
```

`CompoundFrameConfig` embeds `multiref::RefFrameMap`, so the frame-header
writer and tile writer share `NumTotalRefs`, `NumSameRefCompound`,
`FrameDistance`, `OrderHints` and `reference_select`.

## Gate

```sh
./src/encoder/m2/compound_stream/build_and_test.sh
./src/encoder/m2/compound_stream/build_and_test.sh --720p
SANITIZE=1 ./src/encoder/m2/compound_stream/build_and_test.sh
```

The gate encodes a three-frame 320x128 stream.  Frame 2 has ten superblocks:
compound AVERAGE, `skip_mode`, same-reference compound, `use_optflow`,
`single_ref` selecting reference 0, `single_ref` selecting reference 1, WEDGE,
DIFFWTD, `cwp_idx = 1` and `cwp_idx = 0` (both `NEAR_NEARMV` + `AVERAGE`).  It then runs

```sh
dav2d --strict 1 --threads 1 -i <gate>.ivf -o <gate>.ivf.decoded.y4m
```

and compares every decoded Y/U/V sample with the encoder reconstruction.  It
also prints the "all disabled" control: a single-reference tile written with
`skip_mode_present = 0`, `num_same_ref_compound = 0`,
`enable_masked_compound = 0`, `enable_cwp = 0`, `enable_opfl_refine =
REFINE_NONE`, compared byte-for-byte with
`m2::syntax::assemble_m2_frame()` for the same tree.

Recorded output:

```text
== compound_stream unit tests ==
== 112 checks, 0 failure(s) ==
== compound_stream gate: 320x128 ==
  key:     bytes=817 tile=787
  future:  bytes=22 tile=12
  frame2:  tile=17 bytes
  payload round-trip: 10 block(s) ok
  byte-identity control: 13 bytes, identical=yes
  psnr frame 0: Y=38.271 dB
  psnr frame 1: Y=32.6285 dB
  psnr frame 2: Y=30.8156 dB
  PASS: 3 frame(s) decoded, recon exact
```

720p (cropped to 1280x704) also passes:

```text
  key:     bytes=36225 tile=36194
  future:  bytes=240 tile=229
  frame2:  tile=236 bytes
  byte-identity control: 219 bytes, identical=yes
  PASS: 3 frame(s) decoded, recon exact
```

ASan+UBSan pass with no findings.

## Limitations (deliberate)

* NEWMV MVD shell coding, JOINT_NEWMV, warp, BAWP, TIP, refine-MV,
  adaptive-MVD, inter-intra and residuals are outside this component; the
  writer rejects them.
* `skip_mode`'s derived MV and CWP (`RefStackCwp[RefMvIdx]`) are not recomputed
  here; the caller supplies the stack-derived CWP (the gate uses
  `CWP_EQUAL`).  `skip_mode_frames()` is modelled by
  `derive_skip_mode_frames()`; a DPB with a compound reference would need the
  §5.20.7.10 buffer scan added to the caller.
* The tile layout is one 64x64 `PARTITION_NONE` leaf per superblock; the
  partition writer is intentionally the same minimal tree the compound gate
  used.  Frames must be multiples of 64.
* Only `enable_imp_msk_bld = 0` is gated; implicit-mask blending is not
  exercised (it is a separate §7.13.3.14 component).
