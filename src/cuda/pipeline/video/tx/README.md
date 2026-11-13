# M2 video TX-selection stage (`src/cuda/pipeline/video/tx/`)

Batched device TX-size / TX-type / TX-partition selection for the M2 video
residual path, with the exact arithmetic of the product CPU encoder.

The M2 coder currently codes every leaf with one `DCT_DCT` transform sized to
the whole leaf.  That is a measured rate loss: AV2 permits a per-block
`TX_SIZE`/`TX_TYPE` and `TX_MODE_SELECT` partitions (§5.20.6, §5.20.8).  This
module is that decision stage.  For each residual block it evaluates the legal
candidates, runs the real forward → quantise → dequantise → inverse path for
every one, scores each with distortion (coded-domain SSE) and an estimated
rate, and returns the RD-best plan with scan-order levels and EOB.

Status: **complete (3 stages: sizes, types, partition)**.  All 25 AV2
transform sizes, the legal §5.20.8.3 TX-type set of every leaf and every
§5.20.6.3 TX partition (including HORZ4/VERT4 where legal) are evaluated.

### TX-size and TX-partition candidates (§5.20.6)

- The block's own max rectangle (`TX_PARTITION_NONE`, `find_tx_size()`), capped
  at 64 in each dimension.
- Every `legal_splits(mi_size, reduced_tx_part_set)` option that the search
  mask allows: `NONE`, `SPLIT`, `HORZ`, `VERT`, `HORZ4`, `VERT4`.  The leaf
  geometry and TX sizes come from `txpartition::tx_partition_geometry()`
  (normative `find_tx_size()`/`set_tx_size()`), not a re-derivation.
- `HORZ4`/`VERT4` appear only where the block shape permits (4x16, 16x4,
  4x32, 32x4); `reduced_tx_part_set = 1` removes them, and the syntax cannot
  signal `HORZ5`/`VERT5` (the module never emits them).
- `TX_PARTITION_NONE` is still one candidate even when the block has no
  partition syntax (4x4, or the non-MiSize 4x64/64x4 transforms).
- Partition signalling cost is estimated from the §9.3 default
  `tx_do_partition` / `tx_partition_type` / `tx_2or3_partition_type` CDFs and
  added once per plan (`partition_bits`).  It is zero when
  `search_partition = 0` (no `TX_MODE_SELECT` syntax in the stream).
- Plan order (and the final tie-break) follows `legal_splits()`: NONE first,
  then SPLIT/HORZ/VERT/HORZ4/VERT4.

### TX-type candidates (§5.20.8.3)

- **Intra**: `txtype::intra_tx_candidates()` with the block size, mapped intra
  mode, `reduced_tx_set`, `dct_only`; symbol order and the mode-dependent
  `Md_Idx_To_Type` table come from the CPU component.  The `eob <= 1 =>
  DCT_DCT` rule is applied at decision time.
- **Inter**: the canonical §5.20.8.2 inverse tables (`Tx_Type_Inter_Inv_Set1/2`,
  `Tx_Type_Inv_Long`, `Tx_Type_Inverse_Set3/4`) give the types the decoder can
  derive from a symbol; type aliases are never emitted.  §7.15.4.1 is applied
  through `inter_tx_type_is_ddt()`: with `enable_inter_ddt`, ADST/FDST types
  are withheld (their 1-D kernels become DDTX/FDDT at the decoder), while
  DCT_DCT/IDTX/V_DCT/H_DCT remain.
- **1-D classes** (`V_*`/`H_*`, `TX_CLASS_VERT/HORIZ`) are legal AV2 candidates
  and the module produces their class-specific scan-order levels.  They are
  opt-in (`include_1d_types`) because the current M2 syntax writer only codes
  `TX_CLASS_2D`; the product path leaves them off until that writer lands.
- `eob == 0` (all-zero levels) is a skip: only `DCT_DCT` is codable, the rate is
  `skip_bits`, and no type syntactically exists.  Intra `eob == 1` also forces
  `DCT_DCT`; inter codes its type at `eob == 1` (§5.20.8.2).
- The type-symbol cost is host-computed per §8.3.2 context (0..2) from the §9.3
  CDFs; the device selects the context from the decoded eob
  (`inter_tx_type_context()`).  For `TX_SET_INTER_2` the offset term uses the
  type context in this bring-up model (the CPU `tx_type_symbol_bits()` uses the
  unclamped offset context; the difference is a fraction of a bit and both
  sides of the parity test use this module's table).

## What the kernel does

One CUDA block per residual block, whole batch in **one kernel launch**, host
entry point synchronises **once**:

```
source + predictor  (or a precomputed coded-size residual)
  -> coded-size leaf residual
  -> §7.15.4 downsample to the adjusted size      (Round2)
  -> forward 2D transform                          (binary64)
  -> quantise                                      (§7.14.2/§7.14.4 delta-Q)
  -> EOB / nonzero / sum|level| over the §5.20.7.30 scan
  -> dequantise
  -> inverse 2D transform                          (int64, §7.15.4)
  -> exact int64 coded-domain SSE
  -> rate estimate, cost = sse + lambda * rate_bits
  -> per-leaf decision, then per-plan decision, then the winner's
     scan-order levels written to the output
```

Numeric reference is the product CPU path, not a stand-in:

| Step | CPU reference |
|---|---|
| downsample | `txv::downsample_residual` (`src/codec/transform/variants/`) |
| forward 2D | `txv::forward_block` (binary64, §7.15.2) |
| quantise | `TileEncoder`'s delta-Q expression: `lround(coeff * (qf*528/q))` |
| dequantise | §7.14.4 `Round2((|level|*q2)&0xFFFFFF, 3) / dqDenom` |
| inverse 2D | `txv::inverse_block` (§7.15.2.1/.3, §7.15.4.1 pre-scale) |
| §7.14.2 quantisers | shared `deltaq::core::get_q` (the delta-Q component) |
| scan orders | `txpartition::get_scan` (§5.20.7.30, TX_CLASS_2D/HORIZ/VERT) |

The coefficient expression order deliberately matches `still_encoder.cpp`
(factor first, then multiply), **not** `deltaq::quantize_job()`'s
`(coeff*(qf*528))/q2` order: the product still path uses the former and one
level can flip at 1080p between the two (recorded in
`docs/negative-results.md`).  `-fmad=false` on the device and
`-ffp-contract=off` on the host keep the binary64 forward transform
bit-identical.

SSE, EOB, nonzero and `sum|level|` are reduced in integer arithmetic
(`unsigned long long` atomics), so the reduction is exact and independent of
warp order.

## Files

| File | Contents |
|---|---|
| `video_tx.h` | POD job/plan/leaf/candidate/result structs, plan builder, arena workspace contract, batch entry point |
| `video_tx.cu` | one fused kernel, host plan builder, arena workspace, one launcher |
| `video_tx_test.cu` | CPU-vs-GPU bit-exact parity, arena, validation and determinism tests |
| `build_video_tx_test.sh` | `nvcc -arch=sm_86 -fmad=false` build + run |
| `README.md` | this file |

Only files in this directory are part of the module.  The M2 product path
consumes it behind `--video-tx-search` (`video_backend_cuda.cu` ->
`tx_search_luma()`); the CPU reference is
`src/pipeline/video/video_tx_search.{h,cpp}`, which mirrors the plan builder and
candidate arithmetic below so both paths stay byte-identical.  The orchestrator
contract is at the end of this file.

## API

```cpp
#include "cuda/pipeline/video/tx/video_tx.h"
namespace vt = av2enc::cuda::videotx;

// ---- one arena workspace ---------------------------------------------------
vt::VideoTxConfig config;          // worst-case batch shape
config.max_jobs = 880;             // residual blocks in one batch
config.max_plans_per_job = 4;
config.max_candidates_per_leaf = 8;
const std::uint64_t need = vt::required_device_bytes(config);

av2enc::cuda::Arena arena;
arena.init(need, Arena::kAlignment, &error);
vt::VideoTxWorkspace ws;
Status s = ws.init(arena, config, &required, &error);   // kArenaExhausted on a short arena

// ---- per-block requests -> host plan --------------------------------------
vt::VideoTxJobRequest req;
req.plane = 0;                       // 0 Y, 1 U, 2 V
req.mode = vt::VideoTxMode::kInter;  // or kIntra
req.coded_w = 32; req.coded_h = 32;  // one of the 25 AV2 TX sizes
req.qindex = 90;
req.delta_dc = 0; req.delta_ac = 0;  // resolved per-plane §7.14.2 deltas
req.residual_offset = 0;             // >=0: use packed int16 residual
// or: req.residual_offset = -1; req.src_packed = 0; ... (source plane)
req.search.lambda = 544.5;           // cost = sse + lambda * rate_bits
req.search.search_partition = 1;     // §5.20.6.3 partition search
req.search.partition_mask = 0x3f;    // NONE|SPLIT|HORZ|VERT|HORZ4|VERT4
req.search.reduced_tx_part_set = 0;  // 1 removes HORZ4/VERT4
req.search.search_tx_type = 1;       // legal §5.20.8.3 type set per leaf
req.search.include_1d_types = 0;     // product: 2-D classes until the writer
                                     // codes TX_CLASS_HORIZ/VERT
req.search.intra_mapped_mode = 0;    // kIntra only
vt::VideoTxPlanData plan;
vt::video_tx_build_plan(requests, count, &plan, &error);

// ---- one batch call, one sync --------------------------------------------
vt::VideoTxBuffers buffers;          // device-resident src / pred / residual
buffers.residual = d_residual;
const Status rc = vt::video_tx_select_batch(plan, buffers, ws, stream);

// ---- results (workspace-owned device arrays; read after the call) --------
ws.d_job_results;    // one per job: winning plan, partition, cost, sse, rate
ws.d_leaf_results;   // job*kMaxLeavesPerPlan + leaf position
ws.d_plan_results;   // one per plan candidate: cost/sse/rate/valid
ws.d_cand_metrics;   // one per (plan, leaf, type): sse/rate/cost/nonzero/eob/codable
ws.d_levels;         // winner levels in §5.20.7.30 scan order per leaf
ws.last_error_count;
```

`VideoTxJobRequest` carries the residual source, quantiser state and search
options; the builder turns it into `VideoTxJobDesc` / `VideoTxPlanDesc` /
`VideoTxLeafDesc` / `VideoTxCandDesc` POD arrays that the kernel consumes.
Explicit offsets everywhere, so a batch can mix all 25 sizes, all planes,
residual and source/prediction inputs.

### Output layout

For job `j`, leaf position `k` (0-based within the winning plan):

- leaf result: `ws.d_leaf_results[j * kMaxLeavesPerPlan + k]`
  (`tx_size`, `tx_type`, `eob`, `nonzero`, `sse`, `rate_bits`, `cost`,
  `level_offset`, `level_elems`);
- levels: `ws.d_levels[leaf_result.level_offset + c]`, `c < level_elems`, in
  `get_scan(tx_size, tx_class_of(tx_type))` order;
- candidate metrics are indexed by the absolute candidate index in the plan
  (`VideoTxCandDesc` order).

### Cost model

```
eob == 0:  rate = skip_bits                    (txb_skip; DCT_DCT is the only
                                                codable candidate)
otherwise: rate = type_bits[ctx] + coefficient_bits * nonzero
cost      = sse + lambda * rate
```

`type_bits` is host-computed per §8.3.2 context so the device and the CPU
reference see the same number (intra: fixed CDF; inter: context from the eob).  Plan cost adds the plan's `partition_bits`
(estimated from the §9.3 `tx_do_partition` / `tx_partition_type` CDFs).
Codability follows §5.20.8.2: intra `eob <= 1` forces `DCT_DCT`; inter codes
the type even at `eob == 1`.  This is a documented bring-up rate model (like
the TX-type component), not the §8.2 range coder.

## Arena rules

Every device byte comes from the caller's single `av2enc::cuda::Arena`.  The
module calls no `cudaMalloc`/`cudaFree` and allocates nothing per launch:
`VideoTxWorkspace::init()` bump-allocates the persistent 75-entry scan table
(25 TX sizes × 3 TX classes × 1024 entries) plus every descriptor and output
array once.

- `required_device_bytes(config)` is exact: each array rounded up to
  `Arena::kAlignment` (256 B).
- A short arena returns `Status::kArenaExhausted` with the exact required size
  and rewinds the arena to its init mark; there is no allocation fallback.
- The launcher writes descriptors into workspace arrays (copies, not
  allocations) and reads results back after the single synchronisation.

Measured requirements:

| Shape | `required_device_bytes` |
|---|---:|
| 880 jobs, 1 plan, 1 type (stage-1 shape) | 294,144 B |
| 138 jobs, 8 plans, 16 types (full partition + type search, test batch) | 12,289,536 B |

The per-leaf output levels (`kMaxLeavesPerPlan * 1024 * int32` per job) and the
per-candidate descriptor/metric arrays dominate.  A full search over a whole
superblock row is tens of MiB; size the config to the batch and split by
superblock row when needed.  The tests size the arena exactly from
`required_device_bytes()` and verify the one-byte-short rejection.

## Tests

```sh
./src/cuda/pipeline/video/tx/build_video_tx_test.sh          # sm_86, build + run
./src/cuda/pipeline/video/tx/build_video_tx_test.sh --quick  # short q sweep
CUDA_ARCH=sm_120 ./src/cuda/pipeline/video/tx/build_video_tx_test.sh
```

`video_tx_test` links the real CPU references (`tx_variants.cpp`,
`tx_partition.cpp`, `tx_scan.cpp`) and compares, per candidate: SSE, rate,
cost, nonzero, EOB, codability; per plan: validity/SSE/rate/cost; per job: the
winning plan index and every cost; and the winner's scan-order levels element
by element.

Stage-3 matrix (RTX 3060, `sm_86`, CUDA 13.0):

```
av2enc CUDA M2 video TX-selection test
device: NVIDIA GeForce RTX 3060 (sm_86, 28 SMs, 11.6 GiB)
  sizes: 400 jobs, 1600 checks
  types: 1050 jobs, 6072 candidates (4506 codable, 1566 not), 16 type values, 9222 checks
  type rules: inter 2d=16 ddt=4, intra 16x32=8
  partitions: 480 legal (size,partition,mode) combos, 480 jobs, 9423 candidates, 11451 checks
  partition-search: 138 jobs, 486 plans, 9492 candidates, 10316 checks
  partition rules: reduced=2/3 options, mask rejection
  partition edges: 21 jobs, 1211 checks
  qindex: 2727 jobs, 10908 checks
  plane edges: 56 jobs, 224 checks
  delta-q: 144 jobs, 576 checks
  residual-equiv: 14 jobs, 56 checks
  determinism: 7 jobs
  arena: required=294144
  validation: rejected malformed requests

43534 checks, 0 failures
RESULT: PASS
```

Coverage:

- all 25 TX sizes × qindex {1, 25, 50, 90, 140, 190, 255, 303} × residual and
  packed source/prediction inputs;
- **every legal TX type of every size**: 25 sizes × intra mapped modes 0..12 and
  inter × qindex {1, 90, 255}, with the 1-D classes enabled — 6,072 candidates
  and all 16 `TX_TYPE` values, both `eob <= 1` (non-codable) and normal blocks;
- type-rule tests: 1-D classes absent without `include_1d_types`, the DDT guard
  removes every ADST/FDST type but keeps the identity types, and the intra
  16x32 long set exposes all 8 types;
- **every legal (block size, TX partition, mode) combination**: 480 isolated
  cross-product cases over all 25 sizes, intra and inter, with every legal type
  of every resulting leaf; leaf geometry is compared field-by-field with
  `tx_partition_geometry()` and the leaves are verified to tile the block
  exactly (no gap, no overlap);
- full partition search (all plans at once): the winner equals the minimum-cost
  valid plan with the CPU tie-break, and 486 plans are compared plan-by-plan;
- partition rules: `reduced_tx_part_set` drops HORZ4/VERT4, an empty mask is
  rejected, and partition edges (odd visible bounds, overhanging origins) run
  the whole search with source clamping;
- qindex 1..303 for 4x4/8x8/16x16/32x32/64x64/32x8/8x32/16x64/4x64;
- odd visible bounds (37×29 source) with overhanging block origins: the plane
  read clamps exactly like `TileEncoder::source_sample`;
- all three planes with non-zero DC/AC delta totals (including negative);
- residual and source/prediction inputs of the same block agree bit for bit;
- two identical batches produce identical results;
- arena exact fit, one-byte-short rejection with the exact required size and an
  unchanged arena, invalid configs and malformed requests rejected.

`compute-sanitizer --tool memcheck` reports `0 errors`;
`tools/check_arena.sh src` is PASS.

## Integration contract for the video orchestrator

- Call once per dependency level / superblock row with all independent leaves
  in the batch.  The block sits where the current M2 path calls
  `txv_forward_transform_multi` for the chosen leaf.
- Provide device-resident source/prediction planes (the same buffers the
  current TX stage reads) or a packed coded-size residual; `qindex` is the
  effective §5.20.1 `CurrentQIndex` for the tile/segment, `delta_dc`/`delta_ac`
  the resolved per-plane §7.14.2 totals.
- Use the returned leaf plan to write `tx_size`, `tx_type`, `eob` and
  scan-order levels into the existing `BlockSyntax`/`av2::Encoder` path.  The
  module writes no bits.
- The orchestrator still owns reconstruction: re-run the inverse for the
  winning leaves (cheap: one of N candidates) or extend the result struct with a
  per-leaf recon output when the recon stage needs it.
- TX_TYPE signalling needs `txtype::tx_type_symbol()` /
  `residual::inter_tx_symbol()` on the result; the current M2 writer only codes
  `TX_CLASS_2D` types, so leave `search.include_1d_types = 0` on the product
  path until that writer lands.
- TX_MODE_SELECT needs the §5.20.6.3 syntax writer (`tx_partition` component)
  and the matching `tx_mode_select = 1` header change; this module chooses the
  partition, it does not code it.

## Gaps / open items

- **Wired** in the M2 CPU and CUDA paths (`--video-tx-search`, default off).
  `video_syntax.cpp` emits `tx_mode_select = 1`, `tx_do_partition` /
  `tx_partition_type` and one `code_tx_view()` per part with the part's own
  4x4 origin; `video_frame.cpp` / `video_pipeline.cpp` fill the per-part
  `BlockSyntax` fields.
- The device module takes a **fixed block predictor**, so the partition search
  is inter-only.  §5.20.7.24 predicts per transform block and feeds each part's
  reconstruction into the next part's prediction; that is equivalent for MC
  with one MV but not for intra.  Intra leaves use `TX_PARTITION_NONE` plus
  the TX-type search.  A full intra partition search needs per-part
  prediction/reconstruction feedback in this module.
- The `--video-deblock` combination is rejected: the §7.17 grid derives its
  transform edges from one transform per leaf and does not model §5.20.6.3
  parts yet.
- The syntax writer must also emit `y_mode_index`-style TX-type symbols from the
  returned per-leaf `tx_type`; `txtype::tx_type_symbol()` /
  `residual::inter_tx_symbol()` provide the symbol and long-side flag.
- No RDOQ/TCQ/IST: the scalar delta-Q path is the product M2 configuration and
  the RDO layer for those is a separate component.
- No reconstruction output (see the orchestrator contract).
- No quantizer matrix: `q2` is the scalar DC/AC quantiser.  `--still-qm` is a
  still-path feature; a QM interplay needs a per-leaf q2 matrix upload.
- 8-bit 4:2:0 is the target; the transform/quantiser arithmetic is depth-neutral
  but no 10-bit result struct exists here.
- Performance is not claimed.  One block per job is a correctness-first shape;
  the candidate loop is serial within a block.  Batching many jobs keeps the
  device busy; a later revision can split the candidate fan-out across blocks
  once the decision is proven.
