# encoder/partition — AV2 recursive partition tree (CPU reference)

Spec-only CPU reference for the AV2 v1.0.0 partition tree. This module decides
and describes **geometry**, not entropy and not pixels: it does not write bits,
touch CDFs, or reconstruct samples. It is the correctness oracle the GPU
partition search and the integration pass must match.

- Spec: `spec/AV2_Spec_v1.0.0.pdf`, v1.0.0, 2026-05-28, commit `38f28e7`
  (sha256 `e9916f091e4e83446aad6b4601641c5b292e569c144c4163b26a4497573b533f`).
- Clauses: §5.20.3.1 `decode_partition`, §5.20.3.2 `read_partition` +
  `partition_implied`/`is_partition_allowed`, §6.19.3.1 partition semantics,
  §8.3.2 partition contexts, §9.2 block/partition tables.
- No AVM/SVT/clave/av2-turbo code. Implemented from the specification only.

## Files

| File | Purpose |
|---|---|
| `partition.h` | geometry, spec tables, boundary/allowed rules, decision API |
| `partition.cpp` | implementation, deterministic tree solver, validators |
| `partition_test.cpp` | spec validity, edge cases, deterministic-tree tests |
| `build_and_test.sh` | standalone C++20 build (warnings, optional ASan/UBSan) |

```sh
./src/encoder/partition/build_and_test.sh
SANITIZE=1 WERROR=1 ./src/encoder/partition/build_and_test.sh
CXX=clang++ ./src/encoder/partition/build_and_test.sh
```

## Scope

- Partitions: `PARTITION_NONE`, `PARTITION_SPLIT`, `PARTITION_HORZ`,
  `PARTITION_VERT`.
- Block sizes 4x4 through 64x64, plus the rectangular sizes those splits
  produce (4x8, 8x4, ... 64x8).
- Superblock sizes 64x64 and 128x128. 256x256 requires the sequence-header
  flag and is deliberately rejected (`kInvalidGeometry`).
- 8-bit 4:2:0 geometry by default; monochrome / 4:4:4 flags exist, 4:2:2 is
  not modelled.
- One tile. `is_inside()` is frame-relative; a tile implementation narrows it.
  Partition geometry still uses global `MiRows`/`MiCols`, as the spec does.

Not in scope: extended 3/4-way partitions, SDP, BRU, chroma-only partition
trees, transform partitioning, mode decision, entropy, bitstream syntax.

## The AV2 surprise: SPLIT is only legal at 128x128 / 256x256

AV2 v1.0.0 Table 9.x (`Partition_Subsize`) defines `PARTITION_SPLIT` only for
`BLOCK_128X128 -> BLOCK_64X64` and `BLOCK_256X256 -> BLOCK_128X128`. There is
**no** square 4-way split at 64x64, 32x32, 16x16, or 8x8.

Consequences:

- With the M1 64x64 superblock, `PARTITION_SPLIT` is never legal. A 32x32
  quadrant decomposition is coded as `HORZ` then `VERT` (or `VERT` then
  `HORZ`). The M0 reference already does exactly this: 64x64 -> 32x64 -> 32x32.
- With a 128x128 superblock, `PARTITION_SPLIT` at the root yields four 64x64
  children. Below that, rectangular splits take over.
- `partition_subsize(kSplit, k64x64) == kInvalid` and
  `init_allowed_partitions(...).is_allowed(kSplit) == false`. Do not "fix"
  this or the decoder will reject the stream.

Thin blocks also have a special rule: `partition_implied()` returns
`PARTITION_NONE` for `4x4` and for block enum values `>= BLOCK_4X32`
(4x32, 32x4, 8x64, 64x8). Those sizes are terminal when `NONE` is legal.

## Geometry

```cpp
#include "partition.h"
using namespace av2enc::partition;

FrameGeometry geometry;
geometry.width = 640;
geometry.height = 360;
geometry.sb_size = 64;           // 64 or 128
// has_chroma = true, 4:2:0 subsampling, MaxPbAspectRatio = 8 by default

mi_cols(geometry);               // 160
mi_rows(geometry);               // 90
sb_cols(geometry);               // 10
sb_rows(geometry);               // 6
BlockRect rect = block_rect(geometry, /*mi_row=*/80, /*mi_col=*/128,
                            BlockSize::k64x64);
// x=512 y=320 w=64 h=64, visible 64x40: 640x360's bottom edge is partial.
```

`MiCols`/`MiRows` use §5.18.4.4 `2 * ((dimension + 7) >> 3)`: they round up
to an even number of 4x4 units (8-pixel granularity), not simply
`ceil(dimension / 4)`. The two agree for 640x360, 1080p, and any dimension
that is a multiple of 8.

`partition_implied_at_boundary()` implements the spec's half-block test:

```text
has_rows = (mi_row + (numHigh4x4 >> 1)) < MiRows
has_cols = (mi_col + (numWide4x4  >> 1)) < MiCols
```

and returns the implied partition (`HORZ` for a square whose rows are short,
`VERT` for a square whose columns are short, etc.). `partition_implied()` adds
the 4x4 / thin-block `NONE` cases. `init_allowed_partitions()` and
`is_partition_allowed()` mirror §5.20.3.2, including:

- aspect-ratio rejection (`MaxPbAspectRatio`, default 8),
- rect-type implication for 4x8/8x4/4x16/16x4/64x128/128x64,
- 4:2:0 chroma-offset handling (a partition is rejected if it would leave no
  in-frame chroma block),
- `PARTITION_NONE` rejection when the block is outside the half-block bounds,
- the spec fallback that forces `PARTITION_NONE` when nothing else is allowed.

Both `hasChroma` and `chromaOffset` are explicit arguments because they change
as the recursion descends. `struct PartitionNode` records both.

## Decision API

The caller supplies candidate costs; the solver returns a deterministic tree.

```cpp
struct NodeCosts {
  double leaf;          // cost of coding this block as PARTITION_NONE
  double none_signal;   // cost of do_split = 0
  double split_signal;  // cost of PARTITION_SPLIT
  double horz_signal;   // cost of PARTITION_HORZ
  double vert_signal;   // cost of PARTITION_VERT
};

CostFunction costs = [&](const PartitionNodeInfo& info) {
  NodeCosts c;
  c.leaf = sad_or_rdo_cost(info);       // caller's mode/tx/coefficient cost
  c.none_signal = rate_estimate(...);   // caller's partition flag costs
  c.split_signal = ...;
  c.horz_signal = ...;
  c.vert_signal = ...;
  return c;
};

PartitionDecision sb = decide_superblock(geometry, 0, 0, costs, options);
FrameDecision frame = decide_frame(geometry, costs, options);
```

Rules:

- `kInfiniteCost` means "unavailable". `NaN` and negative costs return
  `kInvalidCost`.
- Only spec-allowed partitions are considered. Rectangular splits are skipped
  when `DecisionOptions::allow_rect == false`; `SPLIT` when
  `allow_split == false`; both when they would create a block below
  `min_block_size_px` (default 4, never below 4). These are policy limits for
  **optional** splits only.
- A spec-forced partition is mandatory and therefore ignores the policy
  limits. At a partial superblock where the half-size leaves the frame the
  spec also forbids `PARTITION_NONE`, so the solver relaxes the depth/floor
  limits as needed instead of failing with "no partition candidate". This is
  what lets `--still-depth 0`/`--still-single` encode a right or bottom edge SB.
- Ties are broken in the fixed order `NONE`, `SPLIT`, `HORZ`, `VERT`, so
  identical inputs always produce the identical tree.
- A spec-implied partition wins even if the caller's costs would prefer
  another. If `forced` is true, **no partition syntax is coded** at that node
  (the decoder infers it); its signal cost is treated as zero.
- `DecisionOptions::max_depth` counts partition recursion from the superblock
  root (depth 0) for optional splits; boundary-forced/relaxed nodes may go
  deeper.

Tree representation:

- `PartitionNode::partition` is the chosen partition.
- `PartitionNode::boundary` and `PartitionNode::allowed` are the spec-derived
  values the solver used. The integration pass can read them instead of
  recomputing, and `validate_tree()` rejects stale copies.
- `children` contains **in-frame children only**, in decode order. A child
  whose origin is at/after `MiRows`/`MiCols` is skipped by
  `decode_partition()` and is not represented. A `HORZ` node at the bottom
  edge can therefore have one child.
- `collect_leaves()` walks in decode order: SPLIT visits TL, TR, BL, BR;
  HORZ top then bottom; VERT left then right.
- `validate_tree()` / `validate_frame()` re-derive the spec rules and check
  exact, non-overlapping coverage of every in-frame 4x4 cell.

## Integration pass: what to call

The integration pass (which owns the M0 core and entropy) should:

1. Build `FrameGeometry` from the frame dimensions and the chosen `sb_size`
   (64 for the M0-compatible path, 128 when `use_128x128_superblock=1`).
2. Build a `CostFunction` from the RDO/mode-cost stage. At a node, use
   `info.allowed` and `info.boundary` to know which partition flags would be
   coded at all; only non-forced nodes code `do_split` / `do_square_split` /
   `rect_type`. The same sets are copied onto each `PartitionNode`.
3. Call `decide_frame()` once per frame (or `decide_superblock()` per SB for
   an SB-parallel GPU scheduler).
4. Handle each node in decode order:
   - when `node.forced`, write **no** partition syntax: `read_partition()`
     returns early because the boundary implied it or `numAllowed == 1`;
   - otherwise, if `info.allowed.is_allowed(NONE)`, code `do_split` (`0` for
     `NONE`, `1` for any split); if `NONE` is not allowed, `do_split` is
     implied `1` and is not coded;
   - if the partition is not `NONE` and
     `info.allowed.is_allowed(SPLIT)`, code `do_square_split` (`1` for
     `SPLIT`, `0` otherwise); if `SPLIT` is not allowed, it is implied `0`;
   - for `HORZ`/`VERT`: let `rect = rect_type_implied_by_bsize(node.size)`.
     If `rect.implied`, code nothing for `rect_type`. If it is not implied,
     code `rect_type` only when **both** `HORZ` and `VERT` are allowed;
     otherwise the single allowed orientation is inferred;
   - use `spec_partition_index(node.partition)` when writing the partition
     value. Extended-partition flags (`do_ext_partition`, uneven 4-way) are
     always absent because those tools are out of scope.
5. Fill a `MiSizeGrid` with each finalized leaf (`fill_block(node.mi_row,
   node.mi_col, node.size)`) and call `partition_contexts()` for the
   `do_split` / `do_square_split` / `rect_type` CDF contexts (§8.3.2).
6. Drive mode/transform/reconstruction from `collect_leaves()`, using
   `block_rect()` for the visible pixel rectangle. Leaves may extend past the
   frame at the bottom/right; `visible_width` / `visible_height` are clipped.
7. Use `node.has_chroma` / `node.chroma_offset` exactly as the spec passes
   `hasChroma` / `chromaOffset`. Chroma prediction and chroma-only partition
   trees belong to the intra/CfL components, not here.
8. Run `validate_tree()` / `validate_frame()` in debug builds before entropy.
   A validity failure must abort the encode, not be papered over.

Do **not** emit `PARTITION_VERT` at a bottom edge where the spec implies
`HORZ`, and do not emit `PARTITION_SPLIT` below a 128x128 root. Those are the
two failure modes most likely to reach `dav2d`.

## Edge cases covered by tests

- 64x64 and 128x128 superblocks.
- 640x360: 10x6 SBs at 64, 5x3 at 128; bottom-edge root 64x64 is legal as
  `NONE` (matches the pinned `56f11d4a63ff07f2356ef5064b84fe0e` stream's
  partition mode), while narrower bottom blocks follow the implied
  `HORZ`/`VERT`/`32x16`/`32x8` paths.
- 1080p: 30x17 SBs at 64, 15x9 at 128; the 128x128 bottom row is forced
  `HORZ` at the root.
- Partial right edge: 520x128 with 128 SB forces `VERT` at the last root.
- 8x8 frame forced down to 4x4 leaves; MiCols/MiRows 8-pixel rounding is
  covered separately (a 4x4 picture has a 2x2-mi nominal grid).
- Deterministic trees: identical costs produce byte-identical tree signatures.
- Validation rejects illegal `SPLIT`, missing children, wrong child
  positions, a fake `forced` flag, and a wrong SB layout.

## Deliberate limitations

- CPU scalar only; no SIMD/threading. This is a correctness oracle, not a
  speed target (`AGENTS.md`, `docs/architecture.md`).
- No CUDA kernel. A future GPU partition search must match this reference;
  the cost callback is the natural place to replace with a device-side array
  of node costs.
- No exact rate accounting: `NodeCosts` are caller-supplied estimates. The
  module only guarantees geometry and deterministic choice.
- 256x256 superblocks, extended partitions, SDP, BRU, and 4:2:2 are rejected
  or not modelled rather than guessed.
