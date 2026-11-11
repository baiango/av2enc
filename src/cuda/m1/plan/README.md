# M1 GPU frame plan (`src/cuda/m1/plan/`)

Host-side implementation of workstream 2 in `docs/m1-gpu.md`: run the CPU M1
partition search on the host, then export both the leaf list and the full
partition tree in exact entropy/syntax order for the CUDA stages and the syntax
bridge.

The CPU M1 encoder (`src/encoder/m1/m1_still.cpp`) remains the reference. This
component does not touch the bitstream, does not write syntax, and does not
change partition decisions.

## Files

| File | Purpose |
|---|---|
| `plan.h` / `plan.cpp` | `PlanOptions`, `FramePlan`, `BlockPlan`, `TreeNode`, builder, validator, TSV/JSON dumps |
| `plan_gpu_bridge.*` | Orchestrator PODs (`m1_gpu.h`) and optional `stream::TreeNode` mapping |
| `plan_test.cpp` | CPU tests (geometry, tree/leaf ordering, determinism, gates, partial SBs, CPU trace signatures) |
| `plan_dump.cpp` | Standalone block/tree JSON or TSV dump from a Y4M or synthetic plane |
| `compare_trace.py` | Compares plan leaf or tree order with `AV2ENC_M1_TRACE=1` |
| `verify_trace.sh` | Live CPU-trace parity check for the two `tests/data` fixtures |
| `tests/trace_order_*.txt` | CPU-trace preorder+leaf signatures used by the unit test |
| `build_and_test.sh` | Self-contained CPU test build/run |
| `build_plan_dump.sh` | Builds the dump tool |

## Decision model (copied from the CPU M1 path)

`build_frame_plan()` calls `partition::decide_frame` with the same cost function
as `Encoder::encode()` in `m1_still.cpp`:

* leaf cost = truncated block variance of the visible source luma
  (`block_variance_cost_at`, double accumulation, `int64_t` truncation);
* signal cost = `lambda * 2`, `lambda = 0.85 * 2^((base_q_idx - 12) / 6)`;
* `min_block_size = 8`, `max_partition_depth = 4`, `allow_rect = true`,
  `allow_split = false`;
* `force_single_block` sets `max_depth = 0`, `min_block_size = 64`;
* `AV2ENC_M1_SQUARE_ONLY` / `AV2ENC_M1_LEAF32` add `1e18` to non-square or
  >32 leaves through `apply_debug_env()`.

`validate_frame(geometry, frame)` is run before flattening; structural problems
are reported with the same `partition decision failed:` / `partition tree
invalid:` prefixes as the CPU path.

**Depth vs. leaf size.** AV2 has no `PARTITION_SPLIT` at 64, so a 64x64
superblock is binary split. Depth 4 means four binary splits, so the smallest
reachable area is `64*64 / 2^4 = 256` samples: 16x16, 8x32, or 32x8. It never
reaches 8x8 (area 64) or 8x16 (area 128). Depth 8 with `min_block_size = 4` is
needed to exercise 4x4 leaves. This follows the code, which is authoritative
over the older "depth 4 for 8x8" note.

## Ordering

`FramePlan::blocks` is the leaf-only view in syntax order:

1. superblocks in raster order (`FrameDecision::superblocks`);
2. inside each superblock, depth-first child order (`SPLIT` TL,TR,BL,BR;
   `HORZ` top,bottom; `VERT` left,right).

`FramePlan::tree` is the same walk including internal nodes in **preorder**: a
node is emitted before its children, and children are contiguous in decode
order. Leaves point at their `blocks` entry through `TreeNode::leaf_index`.
`tree.size() == nodes` and leaves reference `blocks` in order 0,1,2,...

For every leaf, `m1_still.cpp` walks luma TX, U TX, V TX. The plan's coefficient
offsets use that per-plane order. A block's above row and left column are always
emitted by earlier blocks (`verify_decode_order` in the tests checks this
invariant).

## Geometry and offsets

* `luma_w/h` and `chroma_w/h` are **coded** block sizes. An edge block can
  extend past `MiCols*4` / `MiRows*4`; `visible_*` holds the clipped extent.
  The CPU M1 encoder transforms/reconstructs the full coded block.
* Chroma size uses `partition::chroma_plane_residual_size()` (§9.2
  `Subsampled_Size` for 4:2:0). `has_chroma`/`chroma_offset` carry the partition
  recursion state; the M1 CPU reference codes U/V for every leaf, so chroma TX
  and coefficient offsets are always populated (only relevant below the
  default 8-pixel floor).
* `PlaneTxPlan::tx_size` is the `enum_tx_size` index (0..24,
  `src/encoder/tx_variants/tx_variants.h`), `adj_w/adj_h` are the adjusted
  transform dimensions used by the M1 transform path, and M1 always uses
  `DCT_DCT`.
* `coeff_offset` is an element offset into a **per-plane** `int32_t` level
  buffer; `coeff_count` elements follow. `FramePlan::luma_coeff_count` /
  `u_coeff_count` / `v_coeff_count` are the buffer sizes. U and V have separate
  buffers with identical layout in 4:2:0.
* `luma_stride = align(MiCols*4, 64)`, `luma_padded_height =
  align(MiRows*4, 64)`, chroma stride/height are half, matching the CPU M1 recon
  planes.

## Tree node layout

`TreeNode` carries everything the syntax writer needs for partition syntax:

| Field | Meaning |
|---|---|
| `index` | preorder position (`tree[i].index == i`) |
| `sb_index` | raster superblock index |
| `mi_row`, `mi_col` | node origin in 4x4 units |
| `block_size` | `partition::BlockSize` enum value |
| `partition` | `partition::PartitionType` enum value (0 = NONE/leaf) |
| `depth` | 0 at a superblock root |
| `allowed_mask` | spec `init_allowed_partitions()` bits, bit *p* = allowed |
| `forced` | no partition syntax is coded at this node |
| `sb_root` | `depth == 0` |
| `child_count` | 1..4 for internal nodes, 0 for leaves |
| `first_child` | `index + 1` for internal nodes, `kNoTreeNode` for leaves |
| `subtree` | number of descendant entries (0 for a leaf) |
| `leaf_index` | `blocks[]` index for leaves, `kNoTreeNode` for internal nodes |

## Availability

`NeighborAvailability` reproduces what the CPU encoder feeds to
`intra::prepare_edges`:

* `above` = origin luma/chroma row > 0, `left` = origin column > 0,
  `above_left` = both;
* `num4_above_right` / `num4_below_left` use the superblock-local
  `clear_block_decoded_flags` state (reset per SB, top border up to two SBs,
  below-left capped at the current SB). The boolean bits are `count > 0`.

Luma and chroma availability are separate because the chroma 4x4 origin is
`luma_origin / 2`.

## API

```cpp
#include "cuda/m1/plan/plan.h"

av2enc::m1::plan::PlanOptions options;   // defaults = M1Options
av2enc::m1::plan::apply_debug_env(options);
av2enc::m1::plan::FramePlan plan;
std::string error;
bool ok = av2enc::m1::plan::build_frame_plan(width, height, luma_span,
                                             options, plan, error);

// plan.blocks : leaves in syntax order (unchanged)
// plan.tree   : full preorder partition tree
std::string blocks_tsv = av2enc::m1::plan::plan_to_tsv(plan);
std::string tree_tsv   = av2enc::m1::plan::plan_tree_to_tsv(plan);
std::string json       = av2enc::m1::plan::plan_to_json(plan);  // blocks + tree
```

`luma_span` is the tightly packed visible luma plane (`width * height`). Only
luma is needed because the M1 partition cost is luma variance.

If `src/cuda/m1/m1_gpu.h` is present, `plan_gpu_bridge.h` maps the plan to
`av2enc::cuda::m1gpu::FramePlan` / `BlockPlan`:

```cpp
std::vector<av2enc::cuda::m1gpu::BlockPlan> blocks;
av2enc::cuda::m1gpu::FramePlan frame;
av2enc::m1::plan::build_gpu_frame_plan(width, height, luma, options, blocks,
                                       frame, error);
```

The bridge resizes `blocks` to the **leaf** count and points `frame.blocks` at
it. It maps `tx_w/h` to the adjusted TX size, `tx_upscale_w/h` to
`coded / adjusted` (2 for 64 -> 32), `plane_offset_*` to the padded plane
sample offset, and `avail` to the orchestrator's `AvailabilityBit` mask.

When `src/encoder/m1/stream/m1_stream_bridge.h` is present, an overload also
returns the preorder tree in the stream bridge's own POD:

```cpp
std::vector<av2enc::m1::stream::TreeNode> tree;   // ready for pack_gpu_frame()
av2enc::m1::plan::build_gpu_frame_plan(width, height, luma, options, blocks,
                                       frame, tree, error);
```

The syntax owner can instead pass `plan` directly to
`av2enc::m1::stream::plan_tree_to_nodes(plan, tree, error)` /
`pack_plan_frame(plan, ...)`; those converters read `plan.tree` and validate
the preorder bookkeeping, leaf order, `allowed_mask`, `forced`, and the
`has_chroma`/`chroma_offset` recursion.

## Dump and trace comparison

```sh
OUT=/tmp/m1_plan_dump ./src/cuda/m1/plan/build_plan_dump.sh
/tmp/m1_plan_dump --print-options
/tmp/m1_plan_dump -i tests/data/m0_640x360_1f.y4m --tsv > /tmp/f.blocks.tsv
/tmp/m1_plan_dump -i tests/data/m0_640x360_1f.y4m --tree --tsv > /tmp/f.tree.tsv
/tmp/m1_plan_dump -i frame.y4m --json -o /tmp/f.plan.json
/tmp/m1_plan_dump --synthetic 640x360 --tsv
```

Block TSV: two `#` header lines, one column header, one row per leaf. Tree TSV:
same shape, one row per node. JSON contains `blocks` and `tree` plus the
resolved `options`.

Against the CPU reference:

```sh
AV2ENC_M1_TRACE=1 ./build-m1/av2enc tests/data/m0_640x360_1f.y4m \
    -o /tmp/out.ivf --m1 --frames 1 2> /tmp/f.trace
python3 src/cuda/m1/plan/compare_trace.py /tmp/f.blocks.tsv /tmp/f.trace
python3 src/cuda/m1/plan/compare_trace.py --nodes /tmp/f.tree.tsv /tmp/f.trace
```

Block mode compares `(mi_row, mi_col, size)` for leaves. Node mode compares
`(mi_row, mi_col, size, partition, forced)` for the preorder tree. The current
CPU trace prints the node list twice (decision pass and syntax-writer pass);
node mode checks both passes against the same plan. `--single`, `--depth N`,
`--min-block-size N`, `--square-only`, `--leaf32` map to the equivalent CPU
flags/environment.

`verify_trace.sh` runs both comparisons for `tests/data/m0_64x64_1f.y4m` and
`tests/data/m0_640x360_1f.y4m`:

```sh
AV2ENC_BIN=/workspace/av2enc/build-plan-m1/av2enc \
  ./src/cuda/m1/plan/verify_trace.sh
```

## Tests

```sh
WERROR=1 ./src/cuda/m1/plan/build_and_test.sh
SANITIZE=1 ./src/cuda/m1/plan/build_and_test.sh   # slower
```

402,197 checks cover: 64x64 (constant and textured), 640x360, 1080p, a frame
with both a partial right and bottom SB (300x300), every GPU-matrix size
(136x88, 1280x720, 416x208, 518x322, 532x368) at depths 0..8 plus
`force_single_block`, square-only and leaf32 gates, min-size-4/depth-8 leaves,
the min-size-8 depth-8 floor, chroma residual-size rules, coefficient offset
contiguity, tree preorder bookkeeping, tree/leaf agreement, availability
invariants, decode-order invariants, TSV/JSON determinism, error paths, and the
orchestrator POD/stream-tree bridges when their headers are present.

`tests/trace_order_m0_64x64_1f.txt` and
`tests/trace_order_m0_640x360_1f.txt` are the CPU `AV2ENC_M1_TRACE` preorder +
leaf signatures for the two fixtures (first decision pass, including the spec
allowed mask). The unit test rebuilds those plans and requires an exact match.

## Verified parity (2026-11-11, remote 3060 box)

Built with `AV2ENC_ENABLE_CUDA=OFF -DAV2ENC_ALLOW_CPU_ONLY=ON` in a private
`build-plan-m1/` and compared against `AV2ENC_M1_TRACE=1`:

| Input | PlanOptions | Leaves | Nodes | Result |
|---|---|---:|---:|---|
| `tests/data/m0_64x64_1f.y4m` | depth 4, min 8, qp 90, no gates | 11 | 21 | blocks + nodes match |
| `tests/data/m0_640x360_1f.y4m` | depth 4, min 8, qp 90, no gates | **218** | **376** | blocks + nodes match |
| 640x360 (`testsrc2`, different content) | same | 247 | — | blocks match |
| 1080p (`testdata/shine_1920x1080_10f.y4m`) | same | 1,399 | — | blocks match |
| 640x360 (`testsrc2`) + `--m1-single` | force one block/SB | 60 | — | blocks match |
| 640x360 (`testsrc2`) + `--m1-depth 2` | depth 2 | 133 | — | blocks match |
| 640x360 (`testsrc2`) + `AV2ENC_M1_SQUARE_ONLY=1` | square only | 354 | — | blocks match |
| 640x360 (`testsrc2`) + `AV2ENC_M1_LEAF32=1` | leaves <= 32 | 408 | — | blocks match |

**Leaf-count reconciliation:** 218 is the correct leaf count for the CLI's
default options on `tests/data/m0_640x360_1f.y4m`; the orchestrator's 218
matches. The earlier 247 came from a different input (`ffmpeg testsrc2`), not
from an option mismatch. The default `PlanOptions` printed by the dump for that
fixture are:

```json
{"base_q_idx":90,"min_block_size":8,"max_partition_depth":4,
 "force_single_block":false,"square_only":false,"leaf32":false}
```

Live `verify_trace.sh` output for the two fixtures:

```
m0_64x64_1f:    compare_trace: OK 11 blocks, order matches
                compare_trace: OK 21 nodes, order matches in 2 passes
m0_640x360_1f:  compare_trace: OK 218 blocks, order matches
                compare_trace: OK 376 nodes, order matches in 2 passes
```

The syntax bridge's own host test (`src/encoder/m1/stream/build_and_test.sh`)
also passes with this plan export: `PASS (10847 checks)`, including real 64x64
and 640x360 plans via `plan_tree_to_nodes()` and `pack_plan_frame()`.

## Integration status

The orchestrator now passes a resizable `std::vector<m1gpu::BlockPlan>&` to
`stage::build_plan` and calls `plan::build_gpu_frame_plan`, so the real plan is
wired into `encode_m1_gpu`. A private CUDA build on the 3060 host succeeds, and
`--m1-gpu` on both fixtures produces the same IVF MD5 as `--m1`:

| Input | `--m1` MD5 | `--m1-gpu` MD5 |
|---|---|---|
| `tests/data/m0_64x64_1f.y4m` | `0ef9825062a25c4d950ee924113c6f89` | same |
| `tests/data/m0_640x360_1f.y4m` | `e61de628b0812b716b0efa7dcafcdd06` | same |

The CLI still reports `frame_bytes_from_cpu=1` and `linked stage hooks: 3`
(plan plus the stage-1 TX/recon adapters); entropy and the remaining block
stages are other workstreams. The plan stage itself ran and validated its tree
and buffers before the CPU reference bytes were written.

The earlier blocker — a superblock-sized span that could not hold 218/1,399
leaves — is resolved by the orchestrator's resizable vector.

Remaining wiring point for the syntax owner: the current device hook calls the
POD overload and discards the host `plan::FramePlan`. The syntax bridge's
`pack_plan_frame()` / `plan_tree_to_nodes()` take `plan::FramePlan`
(`plan.tree`, `plan.blocks`), so the orchestrator must retain that host plan (or
the `std::vector<stream::TreeNode>` overload output) in `GpuResult`/the plan
stage when the entropy path is wired. `src/encoder/m1/stream/build_and_test.sh`
already proves this consumption path with real 64x64 and 640x360 plans.

## Partial-superblock depth policy

`max_partition_depth` and `min_block_size` are search-policy limits for
optional splits. A spec-forced boundary partition is always followed, and a
node whose half-size leaves the frame (so `PARTITION_NONE` is forbidden) may be
split past `max_partition_depth` or below the nominal block floor. Such a leaf
is still structurally valid; `validate_frame_plan()` only enforces the depth
cap inside fully coded superblocks. This fixes the former
`spec-forced partition conflicts with DecisionOptions` / `no partition
candidate has a finite cost` failures on 136x88, 1280x720, 416x208, 518x322,
532x368 and `--m1-depth 0..3`/`--m1-single`.

With `min_block_size = 8`, depth 8 still stops at 8x8 leaves; 4x4 requires an
explicit `min_block_size = 4`.
