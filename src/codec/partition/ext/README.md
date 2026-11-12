# encoder/partition_ext — AV2 3-way and uneven 4-way partitions

Spec-only CPU reference for the **extended partition family** of AV2 v1.0.0,
plus a complete, testable partition-symbol model for the M1 entropy writer.

- Spec: `spec/AV2_Spec_v1.0.0.pdf`, 2026-05-28, commit `38f28e7`
  (sha256 `e9916f091e4e83446aad6b4601641c5b292e569c144c4163b26a4497573b533f`).
- Clauses: §5.20.3.1 `decode_partition`, §5.20.3.2 `read_partition` +
  `is_partition_allowed` / `is_ext_partition_allowed` /
  `is_uneven_4way_partition_allowed` / `is_chroma_offset_for_partition`,
  §6.19.3.1–2 partition semantics, §8.3.2 partition contexts, §9.2 block
  tables, §9.3 default CDFs.
- No AVM, SVT-AV1/AV2, clave, or av2-turbo encoder code was consulted.
- `src/encoder/partition/`, `src/encoder/m1/`, `src/cuda/m1/`,
  `src/main.cpp`, and `CMakeLists.txt` are **not** modified.

## Files

| File | Purpose |
|---|---|
| `partition_ext.h` | geometry, §9.2 tables, allowed sets, boundary rules, chroma rules, contexts, decision API, validation |
| `partition_ext.cpp` | implementation (solver, validators) |
| `ext_tables_generated.h` | generated from `spec/tables/block_tables.json` + `default_cdf.json` |
| `generate_ext_tables.py` | regenerates the header; never hand-edit tables |
| `ext_syntax.h` | `read_partition` symbol plans, symbol costs, §8.2 encode/decode of a tree |
| `ext_syntax.cpp` | reference encoder + independent decoder over `src/entropy/av2/` |
| `partition_ext_test.cpp` | geometry, allowed sets, chroma, decisions, validation, rect-only parity |
| `ext_syntax_test.cpp` | symbol plans, contexts, golden bytes, §8.2 round trips |
| `test_util.h` | shared test helpers |
| `build_and_test.sh` | standalone C++20 build (warnings, ASan/UBSan, `-Werror`) |
| `gate_dav2d.py` | sequence-header tool-flag gate against the pinned decoder |

```sh
./src/encoder/partition_ext/build_and_test.sh
SANITIZE=1 WERROR=1 ./src/encoder/partition_ext/build_and_test.sh
python3 src/encoder/partition_ext/generate_ext_tables.py   # after spec cache changes
python3 src/encoder/partition_ext/gate_dav2d.py            # dav2d flag gate
```

## What the spec actually says

`PartitionType` (§6.19.3) is a 10-value enum. The numeric value is the CDF
context/table index, so this module uses it verbatim:

| value | name | split |
|---:|---|---|
| 0 | `PARTITION_NONE` | one block |
| 1 | `PARTITION_HORZ` | two rows |
| 2 | `PARTITION_VERT` | two columns |
| 3 | `PARTITION_HORZ_3` | 1:2:1 rows; the middle row is split vertically |
| 4 | `PARTITION_VERT_3` | 1:2:1 columns; the middle column is split horizontally |
| 5 | `PARTITION_HORZ_4A` | 1/8, 1/4, 1/2, 1/8 from the top |
| 6 | `PARTITION_HORZ_4B` | 1/8, 1/2, 1/4, 1/8 from the top |
| 7 | `PARTITION_VERT_4A` | 1/8, 1/4, 1/2, 1/8 from the left |
| 8 | `PARTITION_VERT_4B` | 1/8, 1/2, 1/4, 1/8 from the left |
| 9 | `PARTITION_SPLIT` | four quadrants (128x128 / 256x256 only) |

Facts that trip up implementations:

- **`PARTITION_SPLIT` is still only legal at 128x128 and 256x256.** Extending
  the partition family does not create a square split at 64x64 or below. A
  64x64 superblock therefore uses `NONE/HORZ/VERT` plus the six extended rows,
  never `SPLIT`.
- **No extended partition is defined at 128x128 or 256x256.** A 128x128 root
  can only use `NONE`/`HORZ`/`VERT`/`SPLIT`; its 64x64 children can use the
  extended rows.
- The 3-way partitions are *1:2:1*, not a 4-way. The middle section is then cut
  perpendicularly, which is what `H_Partition_Midsize[]` encodes.
- The uneven 4-way `A` and `B` variants have identical `Partition_Subsize`
  (1/8) but different middle offsets: `A = 0, 1/8, 3/8, 7/8`,
  `B = 0, 1/8, 5/8, 7/8`. A reachable 4-way subsize exists only where the
  1/8 block is still a legal `BLOCK_*` (e.g. `8x32`, `16x32`, `32x32`,
  `32x64`, `64x64`, and the vertical mirrors).
- Thin blocks (`4x32`, `32x4`, `8x64`, `64x8`) and `4x4` still imply
  `PARTITION_NONE` (§5.20.3.2 `partition_implied`). `8x64` is therefore a
  terminal block: an `8x32` node must be reached from a `16x32` vertical split,
  not by splitting `8x64`.
- `is_ext_partition_allowed()` for `TreeType == SHARED_PART` (the M1 case)
  checks only `enable_ext_partitions`. The chroma-only-tree restrictions do not
  apply. This module models `SHARED_PART` only.
- The aspect-ratio guard is evaluated on the **subsize**. It rejects a
  non-`NONE` split only when the subsize ratio reaches 8:
  `bw >= bh*8 || bh >= bw*8`. A reduced `MaxPbAspectRatio` (4 or 2) therefore
  rejects `32x4`-style 4-way subsizes but keeps a 4:1 3-way subsize.

## Geometry and chroma

`enumerate_children()` returns the exact `decode_partition()` calls, including:

- mi offsets and `Partition_Subsize` / `H_Partition_Midsize` sizes;
- `hasChroma` / `chromaOffset` recursion arguments;
- the `ChromaMi*` (row, col, size) in effect for each child.

`ChromaBlock` matters because a leaf does not always code chroma with its own
luma size:

- `HORZ_4A/4B` at 32x32 uses a `32x4` subsize, so `chromaOffset` becomes 1 and
  the **last** 32x4 child codes the whole 32x32 chroma block (16x16 at 4:2:0).
  The first three children have `hasChroma == 0`.
- `HORZ_3` at 8x32 is the spec's explicit special case: the middle pair
  (`4x16 + 4x16`) shares one 8x16 chroma block at `(r+2, c)`, owned by the
  mid-right child. The bottom 8x8 child then codes its own chroma.
- `VERT_3` at 32x8 is the mirror (`16x4 + 16x4` share a 16x8 chroma block), but
  the spec has **no** `is_chroma_offset_for_partition` special case for VERT_3,
  so the surrounding `hasChroma` arguments differ from HORZ_3. The reference
  reproduces the spec exactly rather than "fixing" the asymmetry.

The decision tree stores `PartitionNode::chroma`, and `LeafBlock::chroma`, so
the integration pass can pass the correct chroma block to the chroma/CfL coder.

## Decision API

```cpp
#include "encoder/partition_ext/partition_ext.h"
using namespace av2enc;

partition_ext::Tools tools;          // sequence-header flags (default off)
tools.enable_ext_partitions = true;        // HORZ_3 / VERT_3
tools.enable_uneven_4way_partitions = true;  // *_4A / *_4B
partition_ext::DecisionOptions opt;  // depth, family switches, tie-break
partition_ext::CostFunction costs = [&](const partition_ext::NodeInfo& info) {
  partition_ext::NodeCosts c;
  c.leaf          = leaf_cost(info);          // NONE payload, no partition flag
  c.none_signal   = do_split_zero_bits(info);
  c.split_signal  = split_bits(info);
  c.horz_signal   = horz_bits(info);
  c.vert_signal   = vert_bits(info);
  c.horz3_signal  = horz3_bits(info);
  c.vert3_signal  = vert3_bits(info);
  c.horz4a_signal = horz4a_bits(info);        // etc.
  return c;
};
auto frame = partition_ext::decide_frame(geometry, costs, tools, opt);
```

- Same shape and rules as `partition::decide_frame`: `kInfiniteCost` means
  "unavailable", NaN/negative costs are rejected, only spec-allowed partitions
  are considered, a spec-forced partition wins and costs zero signal bits.
- Deterministic tie-break. With `TieBreakOrder::kSpecOrder` (default) the enum
  order 0..9 wins ties. With `kRectCompatOrder` the order is
  `NONE, SPLIT, HORZ, VERT, HORZ_3, ...`, which reproduces the rect-only
  module's choices when the extended tools are disabled. `partition_ext_test`
  asserts byte-identical tree signatures against `partition::decide_frame` for
  random and all-equal cost models on 64x64, 640x360, 128x128, 520x128 and 8x8.
- `from_partition_costs()` adapts an existing `partition::NodeCosts`; use it to
  replace the rect-only policy incrementally (extended signals default to
  infinite).
- `DecisionOptions` gates `allow_rect`, `allow_split`, `allow_3way`,
  `allow_uneven4`, `max_depth`, `min_block_size_px`. The floor is applied to
  the smallest child of the candidate (the 1/8 block for uneven 4-way).
- `validate_tree()` / `validate_frame()` re-derive the boundary, allowed set,
  forced flag, child geometry, chroma propagation, and exact in-frame coverage.
  A validation failure must abort the encode, not be papered over.

## Symbol model for the M1 writer

`plan_partition_symbols()` is a direct transcription of `read_partition()`:

```
if (implied && allowed[implied]) return implied          // no syntax
if (numAllowed == 1)            return the only allowed  // no syntax
if (allowed[NONE])  S() do_split                  (0 -> NONE)
if (allowed[SPLIT]) S() do_square_split           (1 -> SPLIT)
rectType = rect_type_implied_by_bsize(bSize)
if (RECT_INVALID) {
    allowHorz = HORZ || HORZ_3 || HORZ_4A || HORZ_4B      // extended counts!
    allowVert = VERT || VERT_3 || VERT_4A || VERT_4B
    if (!allowHorz) rectType = VERT
    else if (!allowVert) rectType = HORZ
}
if (RECT_INVALID) S() rect_type
if (nonExtAllowed && (ext3 || ext4)) S() do_ext_partition
                                    else do_ext_partition = ext3 || ext4
if (do_ext_partition) {
    if (ext3 && ext4) S() do_uneven_4way_partition
    else              do_uneven_4way_partition = ext4
    if (do_uneven_4way_partition) L(1) uneven_4way_partition_type
}
return Rect_Part_Table[do_ext][do_uneven][type][rectType]
```

The extended symbols are only coded when the *spec's* implied rules leave a
choice. In particular `do_ext_partition` is implied `1` when `HORZ`/`VERT` is
unavailable but an extended row is available, and `do_uneven_4way_partition`
is implied when only one of the 3-way / uneven-4-way flavours exists.
`uneven_4way_partition_type` is an `L(1)` literal: one raw bit, **no CDF and no
adaptation** (0 = `*_4A`, 1 = `*_4B`).

### Exact CDF requirements

M1 uses `PlaneStart == 0` for all partition syntax (`SHARED_PART`). Add two
mutable per-tile arrays and copy the defaults from
`ext_tables_generated.h` (or `spec/tables/default_cdf.json`):

| array | shape | context |
|---|---|---|
| `TileDoSplitCdf` | `[2][64][3]` | unchanged |
| `TileDoSquareSplitCdf` | `[2][8][3]` | unchanged (not used by a 64 SB) |
| `TileRectTypeCdf` | `[2][64][3]` | unchanged |
| **`TileDoExtPartitionCdf`** | `[2][64][3]` | `adjSize * 4 + ctx1 * 2 + ctx2` |
| **`TileDoUneven4wayPartitionCdf`** | `[2][64][3]` | same context as `do_ext_partition` |

`adjSize = Partition_Size_Adjust[bSize]`. `ext_syntax_test` pins the
do_split / do_square_split / rect_type tables byte-for-byte against the
already-shipping `src/encoder/basic_cdf_tables.h`, so the plane/context
flattening is verified.

### Exact context requirements

`do_split`, `do_square_split`, and `rect_type` keep their §8.3.2 derivations
(delegated to `partition::partition_contexts`). The new context is:

```
RECT_HORZ:
  bsh  = Max(Mi_Height_Log2[bSize] - 1, 1)
  ctx1 = Mi_Height_Log2[ LeftMiSizes[r] ]              < bsh
  ctx2 = Mi_Height_Log2[ LeftMiSizes[r + h4/2] ]       < bsh
RECT_VERT:
  bsw  = Max(Mi_Width_Log2[bSize] - 1, 1)
  ctx1 = Mi_Width_Log2[ AboveMiSizes[c] ]              < bsw
  ctx2 = Mi_Width_Log2[ AboveMiSizes[c + w4/2] ]       < bsw
ctx = Partition_Size_Adjust[bSize] * 4 + ctx1 * 2 + ctx2
```

`LeftMiSizes` is a per-row array, `AboveMiSizes` is a per-column array. The
second probe is at the vertical/horizontal midpoint of the block, so the M1
writer's current `left_mi(r)` / `above_mi(c)` helpers must be called with
`r + h4/2` / `c + w4/2`. Unset/outside cells behave like `BLOCK_256X256`
(log2 6), exactly as `clear_left_context()` / `clear_above_context()` leave
them (`partition::MiSizeGrid` returns `kInvalid`, mapped to 6 here).

`ext_partition_context()` implements this; `node_contexts()` bundles it with
the three base contexts. `SymbolCosts` + `node_signal_cost()` let an RDO stage
turn per-symbol bit costs into the ten `NodeCosts::*_signal` values without
diverging from what the writer actually codes.

## M1 writer integration point (not edited)

A stream that *uses* the extended rows needs these changes in
`src/encoder/m1/` (owner: the M1 agent). Until they land, block-level `dav2d`
gating is impossible; this component does not fake it.

1. `still_headers.cpp` `write_sequence_header()` currently hardcodes
   `writer.put_bit(false); // enable_ext_partitions` and omits the dependent
   bit. Write `enable_ext_partitions` and, when 1, the
   `enable_uneven_4way_partitions` bit (§5.4.3). The patched-header gate below
   proves the pinned decoder accepts both.
2. `still_syntax.h` `BlockSyntax::allowed_mask` is `std::uint8_t` (4 bits) and
   `partition`/`allowed` assume the rect-only enum. Widen the mask to
   `std::uint16_t` and carry the 10-value `PartitionType`.
3. `still_syntax.cpp` `write_node()` must emit `do_ext_partition`,
   `do_uneven_4way_partition`, and the `L(1)` type bit. Reuse
   `partition_ext::plan_partition_symbols()` and
   `partition_ext::node_contexts()` instead of re-deriving the implication
   rules: the current `need_rect` test only looks at `allowed[HORZ]` /
   `allowed[VERT]`, which is wrong once the extended rows count toward
   `allowHorz` / `allowVert`.
4. Add the two CDF arrays to `CdfState`/`init_cdfs()`. `do_square_split` is
   not needed while `use_128x128_superblock == 0`.
5. `BlockSyntax` must carry the leaf's `has_chroma` and its `ChromaBlock`
   (`chroma_bsize` and, for CfL neighbour context, `chroma_mi_row/col`).
   `write_block()` currently derives `chroma_plane_residual_size(size)` from
   the luma leaf and always codes UV. That is wrong for the first three
   children of `HORZ_4A` (no chroma) and for the shared 8x16/16x8 chroma block
   of `HORZ_3@8x32` / `VERT_3@32x8`.
6. The frame/block plan (CPU or GPU) can be produced by
   `partition_ext::decide_frame()`; `collect_leaves()` already returns the
   decode order and the chroma block each leaf owns.

## dav2d f9dafbc status

`tools/bin/dav2d-macos --version` reports `f9dafbc`.

**Sequence-header level: supported and gated.** `gate_dav2d.py` encodes a
64x64 still with `--still-depth 0` (root `PARTITION_NONE`), rewrites the sequence
header to `enable_ext_partitions = 1, enable_uneven_4way_partitions = 1`, and
decodes both streams. Since a `NONE` root reads `do_split = 0` and returns
before any extended symbol, a supporting decoder must produce identical
pixels. Result:

```
PASS: dav2d decoded the ext/uneven sequence flags; 6187 identical Y4M bytes
```

**Block level: not gated.** The M1 writer cannot emit `HORZ_3`/4-way yet, and
patching a decided frame's partition symbols is not possible without
re-encoding the whole tile (the tree shape changes every subsequent mode and
coefficient). The block-level gate therefore waits on items 1–5 above; the
integration owner should re-run `tools/validate_dav2d.sh` on the first stream
that uses `do_ext_partition`.

## Expected quality gain (estimate only)

No bitstream can be decoded with the new block syntax yet, so **this is an
estimate, not a measurement** and must not be quoted as a result (see
`docs/negative-results.md`, "claim intra-mode gains before integration").
Typical AV2/AV1 extended-partition BD-rate results for intra stills are in the
low single digits:

- 3-way (`HORZ_3`/`VERT_3`) is the useful part: **~0.5–2% BD-rate** at the same
  quality, or ~0.1–0.3 dB at fixed QP, on 1080p stills.
- Uneven 4-way is mostly redundant with `SPLIT + rect` on non-128 SBs; expect
  **~0–0.7%** incremental, concentrated in directional content. It also costs
  one extra symbol (`uneven_4way_partition_type`) per use.
- Current M1 uses an SSE-only mode decision with no rate term
  (`docs/negative-results.md`), so the realized gain will be at the low end
  until rate-aware costs are wired; `SymbolCosts`/`node_signal_cost()` exist for
  that step.
- Search cost: allowing all six rows at every node multiplies partition
  candidates by ~2.5x. A cheap allowed-set + edge-energy pre-check keeps the
  encode-time increase nearer **10–25%**.

## Deliberate limitations

- `SHARED_PART` only: no SDP, BRU, chroma-only partition trees, 256x256
  superblocks, 4:2:2, or tile-relative availability (one tile; `is_inside()` is
  frame-relative, as the rect-only reference does).
- No entropy integration: the M1 writer is untouched. The symbol encoder here
  exists to pin order/contexts/bytes, not to replace §8.2.
- No pixel processing, RDO rate model, or GPU kernel. The cost callback is the
  place a GPU planner plugs in.
- `node_signal_cost()` assumes the caller's `SymbolCosts` are already in bits;
  it does not model adaptation.
