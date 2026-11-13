# encoder/sdp — AV2 semi-decoupled partitioning (CPU reference)

Spec-only CPU reference for **SDP** (semi-decoupled partitioning): a 64x64
intra superblock can carry a luma partition tree and a *separate* chroma
partition tree, with `TreeType`, `ChromaFollowsLuma`, `ChromaPartitionKnown`,
`ChromaMiRow/Col/Size`, and plane-selected partition contexts.

- Spec: `spec/AV2_Spec_v1.0.0.pdf`, v1.0.0, 2026-05-28, commit `38f28e7`
  (sha256 `e9916f091e4e83446aad6b4601641c5b292e569c144c4163b26a4497573b533f`).
- Clauses: §5.4.3 `sequence_partition_config`, §5.20.3.1 `decode_partition`,
  §5.20.3.2 `read_partition` + `partition_implied` / `is_partition_allowed` /
  `is_ext_partition_allowed` / `is_uneven_4way_partition_allowed` /
  `is_chroma_offset_for_partition` / `check_chroma`, §6.4.3 / §6.19.3
  partition semantics, §8.3.2 contexts, §9.2 tables, §9.3
  `Default_Region_Type_Cdf`.
- No AVM, SVT-AV1/AV2, clave, or av2-turbo encoder code was consulted.
- `src/encoder/partition/`, `src/encoder/partition_ext/`, `src/encoder/m1/`,
  `src/cuda/`, `src/main.cpp`, and `CMakeLists.txt` are **not** modified.

## Files

| File | Purpose |
|---|---|
| `sdp.h` | public API: tools, implied/allowed, chroma state, tree, decisions, contexts, symbol plan, decoder |
| `sdp.cpp` | geometry rules, SDP-aware allowed sets, ChromaMi bookkeeping, deterministic solver |
| `sdp_validate.cpp` | independent tree/frame validation, exact two-plane coverage, signatures, plane grids |
| `sdp_syntax.cpp` | §8.3.2 contexts, `read_partition` symbol plans, independent symbol decoder, TreeType switch trace |
| `sdp_internal.h` | private inline helpers shared by the three implementation files |
| `sdp_test.cpp` | spec rules, 64/128 SBs, partial SBs, all TreeType paths, chroma offsets, plans, determinism, negatives, parity |
| `build_and_test.sh` | standalone C++20 build (warnings, optional ASan/UBSan, `-Werror`) |

```sh
./src/encoder/sdp/build_and_test.sh
WERROR=1 ./src/encoder/sdp/build_and_test.sh
SANITIZE=1 WERROR=1 ./src/encoder/sdp/build_and_test.sh
CXX=clang++ ./src/encoder/sdp/build_and_test.sh
```

The test binary is the CPU gate. It links
`src/encoder/partition/partition.cpp` and
`src/encoder/partition_ext/partition_ext.cpp` for the shared block tables; it
does not link CUDA, M1 syntax, or the CLI.

## What "SDP" actually changes

### 1. Sequence header

`sequence_partition_config()` (§5.4.3), in bit order:

```text
f(1) use_256x256_superblock
if (!use_256x256_superblock) f(1) use_128x128_superblock
if (Monochrome) enable_sdp = 0 else f(1) enable_sdp
if (enable_sdp && !single_picture_header_flag) f(1) enable_extended_sdp
else enable_extended_sdp = 0
f(1) enable_ext_partitions
if (enable_ext_partitions) f(1) enable_uneven_4way_partitions
else enable_uneven_4way_partitions = 0
f(1) reduce_pb_aspect_ratio
if (reduce_pb_aspect_ratio) f(1) max_pb_aspect_ratio_log2_minus_1
```

M1 writes `single_picture_header_flag = 1`, so `enable_sdp = 1` adds **one
bit** and `enable_extended_sdp` is inferred 0. `Tools::validate_tools()`
rejects `enable_extended_sdp` with `single_picture_header_flag = 1` and
`enable_uneven_4way_partitions` without `enable_ext_partitions`.

### 2. The intra SDP trigger

At every `decode_partition(r, c, BLOCK_64X64, ...)` call with
`enable_sdp && TreeType == SHARED_PART && FrameIsIntra`:

```text
TreeType = LUMA_PART;   PlaneStart = 0; PlaneEnd = 1;
decode_partition(r, c, BLOCK_64X64, parentSize, 0, 1, 0);
TreeType = CHROMA_PART; PlaneStart = 1; PlaneEnd = NumPlanes;
decode_partition(r, c, BLOCK_64X64, parentSize, 0, 1, 0);
TreeType = SHARED_PART; PlaneStart = 0; PlaneEnd = NumPlanes;
return;                       // no partition syntax for the split itself
```

This happens for a 64x64 superblock root **and** for every 64x64 node inside a
128x128 superblock. In the default tree it is a node with
`sdp_split == true`, `partition == kNone`, and two 64x64 children
(`TreeType::kLumaPart`, `TreeType::kChromaPart`).

`extendedSdpAllowed` is reset to 0 for both trees (the SDP branch passes a
literal 0), so intra SDP never carries `region_type`.

### 3. TreeType-specific rules

| Rule | SHARED | LUMA | CHROMA |
|---|---|---|---|
| `read_block_tx_size()` | yes | yes | **no** |
| `read_intra_y_mode()` | yes | yes | **no** (inherits `YModes`) |
| `read_intra_uv_mode()` | if `HasChroma` | no (`HasChroma=0`) | if `HasChroma` |
| gdf / cdef / ccso / delta-q | yes | yes | **no** |
| palette / dip | yes | yes | **no** |
| `is_ext_partition_allowed` chroma size guard | n/a | n/a | height>16&&width>8 (HORZ) / width>16&&height>8 (VERT) |
| `is_uneven_4way_partition_allowed` chroma guard | n/a | n/a | height==64 (HORZ) / width==64 (VERT) |
| `rect_type_implied_by_bsize` extras | — | — | 8x16/8x32→HORZ, 16x8/32x8→VERT |
| `partition_implied` extras | — | — | 8x8→NONE; 64x64 with `ChromaPartitionKnown`→`LumaPartitions[r][c]` |
| `check_chroma(bSize)` | no | `w>=64 && h>=64` | no |
| local `chromaOffset` update in `is_partition_allowed` | yes | yes | **no** |
| `chromaOffset`/`block_coded` rejections | yes | **no** | yes |

Two asymmetries are normative and are reproduced verbatim:

- In `is_partition_allowed()`, the *local* `chromaOffset` update is skipped for
  `CHROMA_PART`, while the decode recursion (`resolve_chroma()`) updates it for
  every tree type.
- `ChromaFollowsLuma` can be **cleared** by a direct child whose partition is
  `NONE` or continues the root's split direction. The spec's derivation is
  reproduced exactly in `chroma_follows_luma()`; do not "fix" it.

### 4. Chroma bookkeeping

`ChromaMiRow/Col/Size` is updated whenever `!chromaOffset && hasChroma`,
regardless of TreeType. `is_chroma_offset_for_partition()` uses the *subsize*
(and the HORZ_3 middle special case). A 4-pixel-wide or 4-pixel-high chroma
subsize stops chroma splitting: the first child of that split gets
`hasChroma = false`, and only the last child owns the shared `ChromaMi` block.

`Node::chroma` is the `ChromaMi*` value **after** the node's update, i.e. the
block children inherit and the block a `codes_chroma` leaf owns.
`LeafBlock::chroma` carries that ownership to the integration pass. The size is
in luma 4x4 units; convert with `partition::chroma_plane_residual_size()` for
the actual chroma sample block.

### 5. Extended SDP (inter only)

`enable_extended_sdp` is only meaningful when `FrameIsIntra == 0`. At a node
with `bSize != SbSize`, `extendedSdpAllowed`, `TreeType == SHARED_PART`,
`is_bsize_allowed_for_extended_sdp(bSize, partition)`, and
`bru_mode == BRU_ACTIVE`, `read_partition()` emits a `region_type` symbol.
`INTRA_REGION` means:

- the node's partition subtree is decoded under `LUMA_PART` (luma-only), then
- **one** `CHROMA_PART` block of the node's own size is decoded, then
- `TreeType`/`RegionType`/`PlaneStart` are restored.

The reference models this as `Node::extended_sdp == true` with the luma children
followed by one synthetic `sdp_chroma_block` child. The solver chooses between
`MIXED_REGION` and `INTRA_REGION` using `NodeCosts::region_type[2]`; the
default `kInfiniteCost` keeps extended SDP off unless a caller opts in.

`is_bsize_allowed_for_extended_sdp()` is exactly:

```text
bw <= 64 && bh <= 64 && bw >= 8 && bh >= 8
&& partition < PARTITION_HORZ_4A && partition != PARTITION_NONE
```

so it excludes `SPLIT`, `NONE`, and all uneven 4-way rows.

## API

```cpp
#include "encoder/sdp/sdp.h"
using namespace av2enc::sdp;

FrameGeometry geometry;        // width/height/sb_size (64 or 128), 4:2:0
Tools tools;
tools.enable_sdp = true;
tools.frame_is_intra = true;   // still picture
tools.enable_ext_partitions = true;          // optional
tools.enable_uneven_4way_partitions = true;  // optional

DecisionOptions options;
options.max_depth = 32;
options.allow_sdp = true;      // policy switch (must agree with tools)

CostFunction costs = [](const NodeInfo& info) {
  NodeCosts c;
  c.leaf = leaf_rdo(info);                 // NONE payload, no partition flag
  c.none_signal = bits(info);              // do_split=0
  c.split_signal = bits(info);             // do_split=1 + do_square_split
  c.horz_signal = c.vert_signal = bits(info);
  c.horz3_signal = c.vert3_signal = bits(info);
  c.horz4a_signal = c.horz4b_signal = bits(info);
  c.vert4a_signal = c.vert4b_signal = bits(info);
  c.region_type[0] = bits(info);           // INTRA_REGION
  c.region_type[1] = bits(info);           // MIXED_REGION
  return c;
};

FrameDecision frame = decide_frame(geometry, tools, costs, options);
if (frame.status != DecisionStatus::kOk) { /* abort: frame.error */ }
```

`NodeInfo` exposes `tree_type`, `boundary`, `allowed`, `has_chroma`,
`chroma_offset`, `chroma`, `extended_sdp_allowed`, `chroma_partition_known`,
and `implied_luma_partition` so a rate model can stay in sync with
`read_partition()`.

`reduce_pb_aspect_ratio` is expressed through
`FrameGeometry::max_pb_aspect_ratio` (2, 4, or 8); `sdp_validate_geometry()`
rejects anything else. `count_nodes()` / `count_leaves()` are provided for the
plan/syntax stream sizes.

Rules (identical in spirit to `partition::decide_frame`):

- `kInfiniteCost` means "unavailable"; NaN/negative costs return
  `kInvalidCost`.
- Only spec-allowed partitions are considered. `PARTITION_SPLIT` is still only
  legal at 128x128/256x256; a 64x64 root never signals it.
- Ties break in the §6.19.3 enum order `NONE, HORZ, VERT, ... SPLIT`.
- A spec-forced partition is mandatory and ignores `DecisionOptions` policy; if
  `PARTITION_NONE` is forbidden at an edge, a second pass relaxes the depth /
  floor limits so the superblock remains codable.
- `allow_sdp == false` is only for staged integration where
  `tools.enable_sdp == false`; `validate_tree()` enforces the spec trigger, so
  a tree with SDP enabled but no split is rejected.
- Intra SDP is solved greedily: the luma tree first, then `ChromaFollowsLuma`,
  then the chroma tree. It is deterministic, not jointly optimal.

### Tree representation

- `node.sdp_split`: intra luma/chroma split; `children = {luma, chroma}`.
- `node.extended_sdp`: inter `region_type == INTRA_REGION`; the last child is a
  synthetic `sdp_chroma_block` leaf.
- `node.codes_chroma`: the leaf's `HasChroma`. A leaf with
  `codes_chroma == false` in a chroma tree codes **nothing** (no mode, no
  residual) — this is the chroma-offset "shared block" case.
- `collect_leaves()` walks decode order: luma subtree, then chroma subtree;
  for extended SDP, luma children then the single chroma block.
- `tree_type_events()` returns the positions where `TreeType` changes
  (`SDP_INTRA_LUMA_START`, `..._CHROMA_START`, `..._END`, and the extended-SDP
  equivalents). A switch carries no bits.

### Contexts and syntax

`node_contexts(geometry, node, plane_sizes)` implements §8.3.2 with the
`PlaneStart` grid:

- `do_split` / `rect_type`: `Partition_Size_Adjust[_Rect_Type] * 4 + ctx1*2 + ctx2`
- `do_square_split`: `(256 ? 4 : 0) + left*2 + above`
- `do_ext` / `do_uneven_4way`: `Partition_Size_Adjust * 4 + ctx1*2 + ctx2`,
  probing `LeftMiSizes[r + h4/2]` for HORZ and `AboveMiSizes[c + w4/2]` for VERT
- `region_type`: sample-count group `<=128 / <=512 / <=1024 / else` → 0..3

All partition symbols code from `TileDo*Cdf[PlaneStart]`: plane 0 for
`SHARED_PART`/`LUMA_PART`, plane 1 for `CHROMA_PART`. The default arrays are
`Default_Do_Split_Cdf[2][64][3]`, `Default_Do_Square_Split_Cdf[2][8][3]`,
`Default_Rect_Type_Cdf[2][64][3]`, `Default_Do_Ext_Partition_Cdf[2][64][3]`,
`Default_Do_Uneven4way_Partition_Cdf[2][64][3]`, and
`Default_Region_Type_Cdf[4][3]`.

`plan_superblock_symbols()` emits the exact `read_partition()` sequence,
including every implied / single-candidate / BRU-inactive no-symbol case, plus
the `region_type` symbol and the TreeType switches. For a 64x64 intra SDP
superblock whose luma root is `NONE`, the whole plan is one symbol:

```text
do_split(value=0, plane=0, ctx=12)   // Partition_Size_Adjust[64x64] = 3
```

The chroma root is forced by `ChromaPartitionKnown`, so it codes nothing.

`decode_superblock_symbols()` is an independent consumer: it re-derives
implied/allowed sets, consumes symbols in order, rebuilds
`ChromaFollowsLuma`/`ChromaPartitionKnown`, and checks the plane/context of
every symbol. The tests require `tree_signature(decoded) == tree_signature(decided)`.

### Validation

`validate_tree()` / `validate_frame()` re-derive:

- the boundary and the 10-value allowed set for the node's TreeType;
- the forced/single-candidate/BRU rule;
- child geometry through `Partition_ext::enumerate_children`;
- `hasChroma`/`chromaOffset` propagation and `ChromaMi*`;
- `ChromaFollowsLuma`, `ChromaPartitionKnown`, `LumaPartitions`, `CflAllowedInSdp`;
- SDP trigger placement and extended-SDP eligibility;
- exact, non-overlapping coverage of **both** planes inside the superblock.

A validation failure must abort the encode, not be papered over.

### Plane grids

`build_plane_grids()` fills two `MiSizeGrid`s:

- `luma`: every `SHARED`/`LUMA` leaf at its own size;
- `chroma`: every `codes_chroma` leaf's `ChromaMi` extent.

These are the grids the integration pass should feed to `node_contexts()` and
the chroma/CfL path. `chroma.at(...)` stores `ChromaMiSize` (luma-equivalent),
not the subsampled sample size.

## Tests (gate)

The standalone gate covers:

- `partition_implied` parity with `encoder/partition` for SHARED across sizes
  and positions, plus thin-block/4x4 NONE, CHROMA 8x8 NONE, and CHROMA 64x64
  `ChromaPartitionKnown`.
- `rect_type_implied_by_bsize` CHROMA cases (8x16/8x32 HORZ, 16x8/32x8 VERT).
- Allowed sets per TreeType, ext/uneven gating, CHROMA size guards,
  `reduce_pb_aspect_ratio`, MIXED_REGION 4x4 rejection.
- Chroma offset rules, HORZ_3@8x32 middle-chroma asymmetry, and a forced
  32x8→32x4 SHARED tree where only the last child owns the 32x8 ChromaMi block.
- 64x64 and 128x128 superblocks, `PARTITION_SPLIT` at 128, SDP trigger placement
  and TreeType switch counts (3 per intra SDP split; 12 for a 128 SPLIT root).
- Partial superblocks: 640x360, 520x128, 1080p, 136x88, 518x322, 8x8, 16x8,
  264x200, 200x264.
- Determinism: two decides with identical costs produce identical signatures.
- Symbol plans and independent decoder round trips for all of the above, plus
  golden prefixes for `NONE`, `HORZ`, `HORZ_3`, `HORZ_4A` (including the
  `L(1)` literal) and `SPLIT` at 128.
- Validation negatives (no SDP split, wrong TreeType, stale
  ChromaFollowsLuma/ChromaPartitionKnown, missing child, stale allowed set,
  forced-partition mismatch, NaN/negative cost, bad tools).
- Extended SDP `region_type` INTRA/MIXED planning, symbol round trip, and
  `is_bsize_allowed_for_extended_sdp` boundaries.
- Rect-only parity with `partition::decide_frame` (identical leaf geometry and
  order) when SDP and extended partitions are off.

## Minimal stream with SDP: not feasible without core edits

> **Update (completion wave):** this section is the historical plan.  The M1
> writer now implements items 1-5 for the forced-NONE luma/chroma trees; see
> "Integrated product path (`--still-sdp`)" above.  The text below documents
> why a header-only patch is not a valid gate.

A header-only patch is **not** a valid SDP stream. With `enable_sdp = 1`,
`dav2d` immediately enters the luma/chroma split at the 64x64 root: it reads a
luma-only block tree, then a chroma-only block tree. Patching the M1 sequence
header therefore desynchronises the tile at the first block, so it cannot be
used as a gate.

A real stream needs these edits, all outside `src/encoder/sdp/`:

1. **`src/encoder/m1/still_headers.cpp`** — `write_sequence_header()` (the
   `enable_sdp`/`enable_ext_partitions` lines around 102–106):
   write `enable_sdp`; when `enable_sdp && !single_picture_header_flag`, write
   `enable_extended_sdp`. M1 currently always sets
   `single_picture_header_flag = 1` (line 81), so only one extra bit is needed
   for intra SDP.
2. **`src/encoder/m1/still_encoder.cpp`** — `encode()` (~262–298): replace
   `partition::decide_frame()` with `sdp::decide_frame()` and build
   `sdp::Tools` from the M1 config. `collect_node()` (~366) must emit the luma
   subtree and the chroma subtree as separate preorder entries (the current
   `collect_node` assumes one shared tree). `encode_block()` (~1033) must split
   into two leaf kinds:
   - `LUMA_PART`: y mode, cdef/ccso/delta-q, tx-size, luma residual; **no UV
     mode and no chroma residual**.
   - `CHROMA_PART`: no y mode, no cdef/ccso/delta-q, no tx-size; UV mode only
     when `HasChroma`; chroma residual sized by `ChromaMiSize`, not the leaf
     size.
3. **`src/encoder/m1/still_syntax.h`** — `BlockSyntax` (~70–95): widen
   `allowed_mask` to `std::uint16_t`, add `tree_type`/`plane_start`,
   `sdp_split`/`extended_sdp`/`region_type_coded`, `has_chroma`, and the
   owned chroma block (`chroma_bsize`, `chroma_mi_row`, `chroma_mi_col`).
4. **`src/encoder/m1/still_syntax.cpp`**:
   - `code_do_split()` / `code_rect_type()` (~516–536): take a plane index and
     index `cdfs_.do_split[plane][ctx]` / `cdfs_.rect_type[plane][ctx]`.
   - `write_node()` (~538): consume `sdp::plan_superblock_symbols()` instead of
     the current `need_rect` logic (which only checks the 4-value base set);
     handle `sdp_split` nodes (no partition symbols, two children) and
     `region_type`.
   - `write_block()` (~1372): gate every TreeType-dependent syntax element and
     use the leaf's owned chroma block. The current code always derives chroma
     from the luma leaf and always codes UV.
5. **CDF state**: `still_syntax.cpp` already has `do_split[2][64][3]` and
   `rect_type[2][64][3]`. Add `do_ext_partition[2][64][3]`,
   `do_uneven_4way_partition[2][64][3]`, and `region_type[4][3]`; copy the
   defaults from `spec/tables/default_cdf.json`.
6. **Filters / CfL**: pass `frame_is_intra` + `enable_sdp` to deblock (the
   `DeblockOptions` fields already exist in
   `src/encoder/filters/deblock/deblock_core.h:149-151`) and pass
   `CflAllowedInSdp` from the chroma root to
   `cfl::cfl_allowed(..., cfl_allowed_in_sdp, ...)`
   (`src/encoder/cfl/cfl.cpp:515`).
7. **Build/CLI (not ours)**: add `src/encoder/sdp/sdp.cpp`,
   `sdp_validate.cpp`, `sdp_syntax.cpp` to `av2enc_core` in `CMakeLists.txt`
   (after `src/encoder/partition_ext/partition_ext.cpp`) and add a CLI switch
   (for example `--sdp` / `--sdp-extended`) in `src/main.cpp`.

Suggested CMake snippet for the integration owner:

```cmake
src/encoder/sdp/sdp.cpp
src/encoder/sdp/sdp_validate.cpp
src/encoder/sdp/sdp_syntax.cpp
```

No `src/encoder/sdp/` file needs a CUDA build; this is host-side reference
code. A future GPU planner can replace the `CostFunction` with a device cost
array and must match `tree_signature()` for identical costs.

## Decoder support / exact `dav2d` flags

- Local validator: `dav2d` **f9dafbc** (`tools/bin/dav2d-macos --version`).
- `dav2d` has **no SDP-specific command-line switch**. SDP is signalled in the
  sequence header and enabled automatically; the relevant flags are the normal
  decode flags:

  ```sh
  tools/bin/dav2d-macos -q -i out.ivf -o out.y4m --threads 1 --strict 1
  ```

  `--strict 1` is the default; `--threads 1` keeps the decode deterministic.
  `tools/validate_dav2d.sh out.ivf source.y4m` wraps the same invocation and
  performs the pixel comparison.

- **Verification status:** block-level SDP decoding **is gated** for the
  integrated product paths: `--still-sdp` (forced NONE luma/chroma trees) and
  the video `--video-sdp` keyframe (searched luma tree, pinned NONE chroma
  tree).  See the two gate sections below.  Chroma sub-trees inside a decoded
  chroma tree are still not decoder-gated.
- The partition-tree layer (`partition_ext`) remains `SHARED_PART`-only. This
  module is the SDP-aware layer; the two are intentionally separate.

## Integrated product path (`--still-sdp`)
The M1 writer now has an SDP branch:

- `StillConfig::enable_sdp` writes the §5.4.3 `enable_sdp` bit
  (`enable_extended_sdp` is inferred 0 for a single-picture header).
- `still_encoder.cpp` builds the tree with `sdp::decide_frame()` and walks it in
  decode order with `collect_sdp_node()`: an intra 64x64 split node carries
  `kFlagSdpSplit` and its two children are the `LUMA_PART` root followed by the
  `CHROMA_PART` root.
- `still_syntax.cpp::write_node_sdp()` codes each node's partition symbol from
  `TileDoSplitCdf[PlaneStart]` (plane 0 for SHARED/LUMA, plane 1 for CHROMA)
  and keeps a separate chroma-plane partition-context grid.  A `CHROMA_PART`
  leaf codes no Y mode, MRL, TX size, CDEF/CCSO/delta-q/segment id or luma
  residual; it codes UV (and the chroma residual) only when `HasChroma`, sized
  by `ChromaMiSize`.

The first decoder-gated configuration **forces `PARTITION_NONE` on both the
luma and chroma tree** of every intra SDP split (`min_block_size = 64`,
`max_depth = 0`, rect/3-way/uneven disabled).  Searching the sub-trees and
`enable_extended_sdp` remain unwired; the writer rejects a non-NONE SDP
sub-tree partition instead of guessing its tree-specific implied rules.

```
av2enc in.y4m -o out.ivf --still --still-sdp [--still-dc] [--no-cfl] \
       [--no-deblock] --recon out.rec.y4m
```

Constraints: CPU-only (rejected with `--still-gpu`), 64x64 superblocks only,
and mutually exclusive with `--still-single`, `--still-sb128`,
`--still-sb256`, `--still-ext-partitions`, `--still-sub8`, CDEF/CCSO and
segmentation/delta-Q.

Gate (64x64, 128x128, 256x192, 640x384 and 640x360; default / `--still-dc` /
`--no-cfl` / `--no-deblock`; decode with pinned `dav2d`, require sample-exact
Y/U/V reconstruction):

```sh
src/codec/partition/sdp/product_gate.sh build-int
```

Local result: PASS for all 20 cases (`dav2d` f9dafbc, `--strict 1`).  Default
`--still` streams are unchanged (pinned baseline still passes).

## Video path (`--video-sdp`)

The CPU `--video` writer signals §5.4.3 `enable_sdp = 1` (and
`enable_extended_sdp = 0`; the video sequence header has
`single_picture_header_flag = 0`) and codes the intra keyframe with the SDP
two-tree walk.  P frames are not intra, so §5.20.3.1 does not trigger there;
they keep the shared rect tree.

- `InterFrameEncoder::decide_sdp()` builds `sdp::Tools{enable_sdp=true,
  frame_is_intra=true}` and `sdp::DecisionOptions` from the normal
  `--video-min-block` / `--video-max-depth` controls, rect-only (`allow_split`,
  `allow_3way`, `allow_uneven4` are off because the sequence header disables
  those syntax families).
- `collect_sdp_node()` emits the no-syntax split node, then the luma root, then
  the chroma root.  The reference planner
  (`sdp::node_contexts` + `sdp::plan_partition_symbols`) produces the exact
  per-node symbols from the authoritative boundary/allowed set and the
  luma/chroma `MiSize` grids the walk maintains; `video_syntax.cpp` stores and
  emits them (`BlockSyntax::sdp_symbols`), so the writer cannot re-derive
  contexts differently from the solver.
- Leaf coding passes `TreeType`, `HasChroma` and the owned `ChromaMi` block
  through `BlockSyntax` (`kFlagHasChromaInfo`).  A `CHROMA_PART` leaf codes no
  Y mode, TX size, CDEF/CCSO/delta-Q or luma residual, and only codes UV when
  it owns chroma; a `LUMA_PART` leaf codes no chroma.

**Decoder-gated policy.**  The luma tree is searched.  The chroma tree is
pinned to one `PARTITION_NONE` block per superblock (its `NodeCosts::leaf` is
0): the pinned `dav2d` f9dafbc and the reference model disagree on the
`PlaneStart == 1` partition contexts once a decoded chroma tree spans more
than a forced level (the still gate only ever exercised `ChromaPartitionKnown`
=> NONE).  The writer supports and signals chroma sub-trees, but the product
decision never produces one.  Residual proof: a stream with a searched luma
tree and a split chroma tree decodes in the first superblock row and
desynchronises at the first chroma node whose context reads a previously
decoded chroma leaf; the model's `sdp_chroma_sizes_` values, the luma grid, a
chroma-coordinate grid, and an all-sentinel context variant all fail the same
way, while `CHROMA_NONE` is exact at every geometry tried (64x64, 320x360,
640x360, 640x384).

Gate: `src/pipeline/video/gate_video_sdp_ddt.sh` (64x64 and 640x360; default
controls, depth 0/64, depth 3/8, `--video-inter-residual`, `--video-tx-search`,
and SDP+DDT together).  It requires `dav2d --strict 1`, the exact picture
count, sample-exact `--video-recon`, and proves the luma search moves the
stream by comparing depth 0/64 against the default depths.

Constraint list: CPU-only 8-bit 4:2:0, one tile, no frame-type wrappers
(B/temporal/random-access/TIP/SVC), no morph/CfL/MHCCP, no
segmentation/delta-Q.  The video sequence bit alone is byte-identical when the
flag is off.

## Deliberate limitations

- 8-bit 4:2:0 (4:4:4 / monochrome geometry accepted where the rules are
  TreeType-independent; 4:2:2 is rejected).
- 64x64 and 128x128 superblocks; 256x256 is rejected.
- One tile; availability is frame-relative. A tile implementation must narrow
  `is_inside()` and the grids.
- No inter prediction, BRU decode, motion field, or pixel reconstruction.
  Extended SDP is modelled structurally (`region_type`, the single chroma
  block, TreeType switches) but no inter encoder drives it.
- The solver is greedy (luma then chroma) and uses caller-supplied costs, not a
  joint luma+chroma RDO. `NodeCosts::region_type` is the hook for the
  extended-SDP decision.
- No entropy integration. `plan_superblock_symbols()` pins order, contexts, and
  CDF planes; it is not a §8.2 writer.
- No CUDA kernel.
