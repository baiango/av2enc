# M1 GPU frame plan (`src/cuda/pipeline/still/plan/`)

Host-side implementation of workstream 2 in `docs/m1-gpu.md`: run the CPU M1
partition search on the host, then export both the leaf list and the full
partition tree in exact entropy/syntax order for the CUDA stages and the syntax
bridge.

The CPU M1 encoder (`src/encoder/m1/still_encoder.cpp`) remains the reference. This
component does not touch the bitstream, does not write syntax, and does not
change partition decisions.

## Files

| File | Purpose |
|---|---|
| `plan.h` / `plan.cpp` | `PlanOptions`, `FramePlan`, `BlockPlan`, `TreeNode`, builder, validator, TSV/JSON dumps |
| `plan_gpu_bridge.*` | Orchestrator PODs (`still_gpu.h`) and optional `stream::TreeNode` mapping |
| `plan_test.cpp` | CPU tests (geometry, tree/leaf ordering, determinism, gates, partial SBs, CPU trace signatures, SB128/256, extended partitions, sub-8 ownership, SDP) |
| `plan_dump.cpp` | Standalone block/tree JSON or TSV dump from a Y4M or synthetic plane |
| `plan_cost.h` / `plan_cost.cu` | Device leaf-variance cost kernel (one thread per node, no allocation) |
| `plan_cost_test.cu` | CPU-vs-GPU plan parity over device-computed costs (tree, leaf order, partition fields, costs) |
| `compare_trace.py` | Compares plan leaf or tree order with `AV2ENC_STILL_TRACE=1` |
| `verify_trace.sh` | Live CPU-trace parity check for the two `tests/data` fixtures |
| `tests/trace_order_*.txt` | CPU-trace preorder+leaf signatures used by the unit test |
| `build_and_test.sh` | Self-contained CPU test build/run |
| `build_plan_dump.sh` | Builds the dump tool |

## Feature coverage

`PlanOptions` keeps the M1 default byte-identical and adds opt-in search modes:

| Feature | Options | Behaviour |
|---|---|---|
| 64x64 SB (default) | `sb_size = 64` | unchanged rect-only tree |
| 128x128 SB | `sb_size = 128` | `PARTITION_SPLIT` is legal at the root; leaves stay <= 64x64 |
| 256x256 sequence flag | `sb_size = 256` | §5.18.1 clamps an intra still to 128x128 roots; `FramePlan::sequence_sb_size` records 256, `sb_size` is 128 |
| extended partitions | `enable_ext_partitions`, `enable_uneven_4way_partitions` | `partition_ext::decide_frame` with the CPU product cost thresholds; `HORZ_3`/`VERT_3` and `HORZ_4A/B`/`VERT_4A/B` |
| sub-8 leaves | `sub8` | 4-pixel floor and depth >= 8, matching the CPU rect search |
| SDP | `sdp` | the CPU-gated forced-`PARTITION_NONE` luma/chroma trees of every intra 64x64 split |

Extended and sub-8 leaves carry owned-chroma (`ChromaMi`) data: only the leaf
that codes U/V has `has_chroma = true`, `chroma_mi_*` is the block origin and
size, and non-owning leaves have zero `chroma_w/h`, zero U/V coefficients and
zero U/V coefficient counts. `TreeType`/SDP luma-only leaves similarly have
`luma_w = luma_h = 0`.

`TreeNode` keeps the original four-value `partition`/`allowed_mask` fields for
existing consumers and adds the exact extended view: `ext_partition` (spec
`PartitionType` 0..9), `allowed_mask_ext` (10 bits), `tree_type`,
`sdp_split`, `sdp_chroma_block`, and the decided `cost`. `BlockPlan` adds
`tree_type`, `chroma_mi_row/col/size`.

Cost model deltas from the original planner:

* a coding block larger than 64x64 adds `1e18` (the M1 writer has one
  `TX_MODE_LARGEST` transform per plane), so a 128x128 root never wins as
  `PARTITION_NONE`;
* the extended solver uses the CPU product thresholds: 3-way needs
  `h >= 32 && w >= 8` (or the vertical mirror), uneven 4-way needs `h >= 32`
  and vertical uneven needs `w >= 64` (the pinned decoder still desyncs on a
  4-wide 1/8 vertical child, see `docs/negative-results.md`).

## Decision model (copied from the CPU M1 path)

`build_frame_plan()` calls `partition::decide_frame` with the same cost function
as `Encoder::encode()` in `still_encoder.cpp`:

* leaf cost = truncated block variance of the visible source luma
  (`block_variance_cost_at`, double accumulation, `int64_t` truncation);
* signal cost = `lambda * 2`, `lambda = 0.85 * 2^((base_q_idx - 12) / 6)`;
* `min_block_size = 8`, `max_partition_depth = 4`, `allow_rect = true`,
  `allow_split = false`;
* `force_single_block` sets `max_depth = 0`, `min_block_size = 64`;
* `AV2ENC_STILL_SQUARE_ONLY` / `AV2ENC_STILL_LEAF32` add `1e18` to non-square or
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
   `HORZ` top,bottom; `VERT` left,right; extended rows in spec child order).
   An SDP split node is not a leaf: its `LUMA_PART` tree is emitted first, then
   its `CHROMA_PART` tree (luma leaves then chroma leaves per superblock).

`FramePlan::tree` is the same walk including internal nodes in **preorder**: a
node is emitted before its children, and children are contiguous in decode
order. Leaves point at their `blocks` entry through `TreeNode::leaf_index`.
`tree.size() == nodes` and leaves reference `blocks` in order 0,1,2,...

For every leaf, `still_encoder.cpp` walks luma TX, U TX, V TX. The plan's coefficient
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
#include "cuda/pipeline/still/plan/plan.h"

av2enc::still::plan::PlanOptions options;   // defaults = StillOptions
av2enc::still::plan::apply_debug_env(options);
// opt-in modes
options.sb_size = 128;                       // or 256 (intra clamp to 128)
options.enable_ext_partitions = true;
options.enable_uneven_4way_partitions = true;
options.sub8 = true;
// or: options.sdp = true;
av2enc::still::plan::FramePlan plan;
std::string error;
bool ok = av2enc::still::plan::build_frame_plan(width, height, luma_span,
                                             options, plan, error);

// plan.blocks : leaves in syntax order (unchanged)
// plan.tree   : full preorder partition tree (ext/SDP fields included)
std::string blocks_tsv = av2enc::still::plan::plan_to_tsv(plan);
std::string tree_tsv   = av2enc::still::plan::plan_tree_to_tsv(plan);
std::string json       = av2enc::still::plan::plan_to_json(plan);  // blocks + tree
```

`luma_span` is the tightly packed visible luma plane (`width * height`). Only
luma is needed because the M1 partition cost is luma variance.

A caller that already has device variances can inject them without changing
the search:

```cpp
using still_plan::NodeGeometry;
still_plan::LeafCostFunction provider = [&](const NodeGeometry& node) {
  return cost_table[index(node)];            // int64 truncated SSE
};
still_plan::build_frame_plan_with_leaf_cost(width, height, luma, options,
                                            provider, plan, error);
```

`plan_cost.h` provides the matching device kernel
(`plan_cost_launch_variance`); it allocates nothing and runs on the caller's
stream. `plan_cost_test.cu` proves the produced plan (tree, leaf order,
partition fields, frame/node costs) is identical to the CPU-cost plan.

If `src/cuda/pipeline/still/still_gpu.h` is present, `plan_gpu_bridge.h` maps the plan to
`av2enc::cuda::still_gpu::FramePlan` / `BlockPlan`:

```cpp
std::vector<av2enc::cuda::still_gpu::BlockPlan> blocks;
av2enc::cuda::still_gpu::FramePlan frame;
av2enc::still::plan::build_gpu_frame_plan(width, height, luma, options, blocks,
                                       frame, error);
```

The bridge resizes `blocks` to the **leaf** count and points `frame.blocks` at
it. It maps `tx_w/h` to the adjusted TX size, `tx_upscale_w/h` to
`coded / adjusted` (2 for 64 -> 32), `plane_offset_*` to the padded plane
sample offset, and `avail` to the orchestrator's `AvailabilityBit` mask.
`plane_offset_u/v` uses the owned `ChromaMi` origin rather than the luma leaf
origin. A `CHROMA_PART` leaf has `luma_w = 0`, `tx_* = 0` and upscale 1; a
non-owning sub-8 leaf has `has_chroma = 0` and zero chroma TX fields.

When `src/encoder/m1/stream/still_stream_bridge.h` is present, an overload also
returns the preorder tree in the stream bridge's own POD:

```cpp
std::vector<av2enc::still::stream::TreeNode> tree;   // ready for pack_gpu_frame()
av2enc::still::plan::build_gpu_frame_plan(width, height, luma, options, blocks,
                                       frame, tree, error);
```

The syntax owner can instead pass `plan` directly to
`av2enc::still::stream::plan_tree_to_nodes(plan, tree, error)` /
`pack_plan_frame(plan, ...)`; those converters read `plan.tree` and validate
the preorder bookkeeping, leaf order, `allowed_mask`, `forced`, and the
`has_chroma`/`chroma_offset` recursion.

## Dump and trace comparison

```sh
OUT=/tmp/still_plan_dump ./src/cuda/pipeline/still/plan/build_plan_dump.sh
/tmp/still_plan_dump --print-options
/tmp/still_plan_dump -i tests/data/basic_640x360_1f.y4m --tsv > /tmp/f.blocks.tsv
/tmp/still_plan_dump -i tests/data/basic_640x360_1f.y4m --tree --tsv > /tmp/f.tree.tsv
/tmp/still_plan_dump -i frame.y4m --json -o /tmp/f.plan.json
/tmp/still_plan_dump --synthetic 640x360 --tsv
/tmp/still_plan_dump --synthetic 640x360 --sb-size 128 --tsv
/tmp/still_plan_dump --synthetic 640x360 --ext-partitions --uneven-4way --tree --tsv
/tmp/still_plan_dump --synthetic 64x64 --sub8 --tsv
/tmp/still_plan_dump --synthetic 64x64 --sdp --tree --tsv
```

Block TSV: two `#` header lines, one column header, one row per leaf. Tree TSV:
same shape, one row per node. JSON contains `blocks` and `tree` plus the
resolved `options`.

Against the CPU reference:

```sh
AV2ENC_STILL_TRACE=1 ./build-m1/av2enc tests/data/basic_640x360_1f.y4m \
    -o /tmp/out.ivf --still --frames 1 2> /tmp/f.trace
python3 src/cuda/pipeline/still/plan/compare_trace.py /tmp/f.blocks.tsv /tmp/f.trace
python3 src/cuda/pipeline/still/plan/compare_trace.py --nodes /tmp/f.tree.tsv /tmp/f.trace
```

Block mode compares `(mi_row, mi_col, size)` for leaves. Node mode compares
`(mi_row, mi_col, size, partition, forced)` for the preorder tree. The current
CPU trace prints the node list twice (decision pass and syntax-writer pass);
node mode checks both passes against the same plan. `--single`, `--depth N`,
`--min-block-size N`, `--sb-size N`, `--sub8`, `--ext-partitions`,
`--uneven-4way`, `--sdp`, `--square-only`, `--leaf32` map to the equivalent CPU
flags/environment.

`verify_trace.sh` runs both comparisons for `tests/data/basic_64x64_1f.y4m` and
`tests/data/basic_640x360_1f.y4m`:

```sh
AV2ENC_BIN=/workspace/av2enc/build-plan-m1/av2enc \
  ./src/cuda/pipeline/still/plan/verify_trace.sh
```

## Tests

```sh
WERROR=1 ./src/cuda/pipeline/still/plan/build_and_test.sh
SANITIZE=1 ./src/cuda/pipeline/still/plan/build_and_test.sh   # slower
BUILD=/tmp/plan_cost_build ./src/cuda/pipeline/still/plan/build_plan_cuda_test.sh
```

507,660 checks cover: 64x64 (constant and textured), 640x360, 1080p, a frame
with both a partial right and bottom SB (300x300), every GPU-matrix size
(136x88, 1280x720, 416x208, 518x322, 532x368) at depths 0..8 plus
`force_single_block`, square-only and leaf32 gates, min-size-4/depth-8 leaves,
the min-size-8 depth-8 floor, chroma residual-size rules, owned-chroma
coverage/origins, coefficient offset contiguity, tree preorder bookkeeping,
tree/leaf agreement, availability invariants, decode-order invariants,
TSV/JSON determinism, error paths, 128x128 and 256-flag roots, extended
partitions (including a 128 root whose 64x64 children use them), SDP
luma/chroma trees, and the orchestrator POD/stream-tree bridges when their
headers are present.

`build_plan_cuda_test.sh` is the CPU-vs-GPU parity gate (sm_86). It fills one
device variance per candidate node with `plan_cost.cu`, rebuilds the plan from
that table, and requires exact equality of the tree, leaf order, partition
fields, frame cost and every node cost. Local RTX 3060 result (2026-11-13):

```
plan_cost: 6/6 cases CPU==GPU, 17270 leaves, 33017 nodes, 3 checks
```

cases: base64, sb128, sb256 (intra clamp), extended+uneven, sub-8, SDP.

`tests/trace_order_basic_64x64_1f.txt` and
`tests/trace_order_basic_640x360_1f.txt` are the CPU `AV2ENC_STILL_TRACE` preorder +
leaf signatures for the two fixtures (first decision pass, including the spec
allowed mask). The unit test rebuilds those plans and requires an exact match.

## Verified parity (2026-11-11, remote 3060 box)

Built with `AV2ENC_ENABLE_CUDA=OFF -DAV2ENC_ALLOW_CPU_ONLY=ON` in a private
`build-plan-m1/` and compared against `AV2ENC_STILL_TRACE=1`:

| Input | PlanOptions | Leaves | Nodes | Result |
|---|---|---:|---:|---|
| `tests/data/basic_64x64_1f.y4m` | depth 4, min 8, qp 90, no gates | 11 | 21 | blocks + nodes match |
| `tests/data/basic_640x360_1f.y4m` | depth 4, min 8, qp 90, no gates | **218** | **376** | blocks + nodes match |
| 640x360 (`testsrc2`, different content) | same | 247 | — | blocks match |
| 1080p (`testdata/shine_1920x1080_10f.y4m`) | same | 1,399 | — | blocks match |
| 640x360 (`testsrc2`) + `--still-single` | force one block/SB | 60 | — | blocks match |
| 640x360 (`testsrc2`) + `--still-depth 2` | depth 2 | 133 | — | blocks match |
| 640x360 (`testsrc2`) + `AV2ENC_STILL_SQUARE_ONLY=1` | square only | 354 | — | blocks match |
| 640x360 (`testsrc2`) + `AV2ENC_STILL_LEAF32=1` | leaves <= 32 | 408 | — | blocks match |

**Leaf-count reconciliation:** 218 is the correct leaf count for the CLI's
default options on `tests/data/basic_640x360_1f.y4m`; the orchestrator's 218
matches. The earlier 247 came from a different input (`ffmpeg testsrc2`), not
from an option mismatch. The default `PlanOptions` printed by the dump for that
fixture are:

```json
{"sb_size":64,"base_q_idx":90,"min_block_size":8,"max_partition_depth":4,
 "force_single_block":false,"sub8":false,"enable_ext_partitions":false,
 "enable_uneven_4way_partitions":false,"sdp":false,"square_only":false,
 "leaf32":false}
```

Live `verify_trace.sh` output for the two fixtures:

```
basic_64x64_1f:    compare_trace: OK 11 blocks, order matches
                compare_trace: OK 21 nodes, order matches in 2 passes
basic_640x360_1f:  compare_trace: OK 218 blocks, order matches
                compare_trace: OK 376 nodes, order matches in 2 passes
```

The syntax bridge's own host test (`src/encoder/m1/stream/build_and_test.sh`)
also passes with this plan export: `PASS (10847 checks)`, including real 64x64
and 640x360 plans via `plan_tree_to_nodes()` and `pack_plan_frame()`.

### 2026-11-13 feature/parity sweep

On the remote RTX 3060 (`sm_86`, CUDA 13.0), with the full CUDA build
(`cmake -DCMAKE_CUDA_ARCHITECTURES=86`, `-j8`):

* `WERROR=1 ./src/cuda/pipeline/still/plan/build_and_test.sh` — all 507,660
  checks pass (default 64-SB plan still 11/21 and 218/376 leaves/nodes).
* `build_plan_cuda_test.sh` — 6/6 CPU-vs-GPU cases, 17,270 leaves, 33,017
  nodes, exact tree/leaf/partition/cost equality.
* Default bitstream unchanged and CPU==GPU:
  `basic_64x64_1f` `0ef9825062a25c4d950ee924113c6f89`,
  `basic_640x360_1f` `e61de628b0812b716b0efa7dcafcdd06` for `--still` and
  `--still-gpu`.
* Default 64-SB plan TSV/tree TSV are byte-identical to the pre-change dumps
  (only new trailing columns differ), so the default search is untouched.

## Integration status

The orchestrator now passes a resizable `std::vector<still_gpu::BlockPlan>&` to
`stage::build_plan` and calls `plan::build_gpu_frame_plan`, so the real plan is
wired into `encode_still_gpu`. A private CUDA build on the 3060 host succeeds, and
`--still-gpu` on both fixtures produces the same IVF MD5 as `--still`:

| Input | `--still` MD5 | `--still-gpu` MD5 |
|---|---|---|
| `tests/data/basic_64x64_1f.y4m` | `0ef9825062a25c4d950ee924113c6f89` | same |
| `tests/data/basic_640x360_1f.y4m` | `e61de628b0812b716b0efa7dcafcdd06` | same |

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

### API still needed by the still orchestrator

The planner supports the new modes, but the shared orchestrator contract is
owned elsewhere and has no place for them yet:

| Gap | Owner action |
|---|---|
| `still_gpu::GpuOptions` has no `sb_size`, ext, `sub8` or `sdp` field, and `stage::build_plan` in `still_gpu.cu` fills only the M1 defaults | plumb the CLI/plan flags into `PlanOptions` |
| `still_gpu::FramePlan::sb_size` carries the **effective** frame SbSize (128 for `--still-sb256`); the sequence-level 256 flag must be carried separately to the header writer | add a sequence-sb field or pass the CLI flag through |
| `still::stream::TreeNode` has a 4-bit `allowed_mask` and the four-value base partition only | widen it and consume `TreeNode::ext_partition`, `allowed_mask_ext`, `tree_type`, `sdp_split`, `sdp_chroma_block` for extended/SDP syntax |
| `still_gpu::BlockPlan` has no `tree_type`/`chroma_mi_*`; the bridge currently encodes the owned chroma origin in `plane_offset_u/v` and sets `luma_w = 0` for a `CHROMA_PART` leaf | downstream TX/recon/chroma stages must honour `has_chroma = 0` (no U/V work) and `luma_w = 0` (no luma work), and SDP needs a luma/chroma tree walk |
| `plan_cost.cu` is intentionally not in the root CMake build (no shared-root edits) | add it to the CUDA sources when the device-cost path is wired |

Search policy limits that match the CPU product path, not planner bugs:

* SDP plans the forced-`PARTITION_NONE` luma/chroma trees only; searching SDP
  sub-trees needs the CPU writer to accept non-NONE SDP partitions first.
* A 256x256 sequence flag still plans 128x128 roots (§5.18.1 intra clamp); a
  256 coding block needs an inter stream.
* Vertical uneven 4-way with a 4-wide 1/8 child stays disabled (pinned decoder
  desync at the bottom edge, `docs/negative-results.md`).

## Partial-superblock depth policy

`max_partition_depth` and `min_block_size` are search-policy limits for
optional splits. A spec-forced boundary partition is always followed, and a
node whose half-size leaves the frame (so `PARTITION_NONE` is forbidden) may be
split past `max_partition_depth` or below the nominal block floor. Such a leaf
is still structurally valid; `validate_frame_plan()` only enforces the depth
cap inside fully coded superblocks. This fixes the former
`spec-forced partition conflicts with DecisionOptions` / `no partition
candidate has a finite cost` failures on 136x88, 1280x720, 416x208, 518x322,
532x368 and `--still-depth 0..3`/`--still-single`.

With `min_block_size = 8`, depth 8 still stops at 8x8 leaves; 4x4 requires an
explicit `min_block_size = 4`.
