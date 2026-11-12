# M1 GPU → syntax bridge

This directory is workstream 7 ("entropy bridge") in `docs/m1-gpu.md`:
it is the only place that maps the GPU pipeline's PODs
(`src/cuda/m1/still_gpu.h`) onto the AV2 syntax writer
(`src/encoder/m1/still_syntax.*`), so the CPU reference and the GPU path emit the
same bitstream from the same block results.

```
GPU stages                         this bridge                  CPU side
----------                         -----------                  --------
BlockPlan[]      (geometry)   ─┐
BlockDecision[]  (modes/EOB)  ─┼─►  pack_gpu_frame()  ─►  BlockSyntax[]  ─►  assemble_still_frame()
coeff planes     (raster)     ─┘                          levels[]           (still_syntax.cpp)
                                                                             │
TreeNode[]       (partition)  ─┘                                             ▼
                                                             TD + seq header + CLK OBUs
```

The CPU path (`still_encoder.cpp`) builds the same `BlockSyntax[]` + `levels[]`
itself and calls `assemble_still_frame()` directly — after the refactor, syntax
writing is no longer inside the decision loop.

| File | Purpose |
|---|---|
| `still_stream_bridge.h` | `TreeNode`, `GpuCoefficients`, `PackedFrame`, `pack_gpu_frame()`, `pack_gpu_raster_frame()`, `assemble_gpu_frame()`, `assemble_gpu_raster_frame()` |
| `still_stream_bridge.cpp` | packing, coefficient re-ordering, validation |
| `still_stream_bridge_test.cpp` | synthetic-block unit test + CPU byte-identity round trip |
| `build_and_test.sh` | host-only build/run of the unit test (no CUDA needed) |

```sh
./src/encoder/m1/stream/build_and_test.sh        # PASS (70 checks)
SANITIZE=1 ./src/encoder/m1/stream/build_and_test.sh
```

---

## 1. What the orchestrator must produce

### 1.1 Block stream

`BlockSyntax` is a **preorder** list of partition nodes:

* superblocks in raster order (row-major over `sb_cols × sb_rows`);
* each node is immediately followed by its children in decode order
  (SPLIT: TL, TR, BL, BR; HORZ: top, bottom; VERT: left, right);
* `subtree` counts the entries of all descendants (0 for a leaf), so the
  writer can skip a subtree without re-deriving subsizes.

There are two producers, and both end up in the same `BlockSyntax` layout:

| Producer | Entry point | Tree source |
|---|---|---|
| host plan module (`src/cuda/m1/plan/`) | `stream::pack_plan_frame()` / `assemble_plan_frame()` | `plan::FramePlan::tree` (preorder) |
| GPU block PODs (`src/cuda/m1/still_gpu.h`) | `stream::pack_gpu_frame()` with an explicit `TreeNode[]`, or `pack_gpu_raster_frame()` for the placeholder plan | supplied by the orchestrator |

`plan::FramePlan::tree` already carries everything the writer needs, so
`pack_plan_frame()` is a 1:1 conversion plus the coefficient interleave.  It
validates the preorder bookkeeping (`index`, `first_child`, `child_count`,
`subtree`), the leaf order (`leaf_index` must be 0,1,2,… into
`plan.blocks`), and the spec-derived `allowed_mask`/`forced` flags against
`partition_implied()` + `init_allowed_partitions()` using the plan's own
`has_chroma`/`chromaOffset` recursion.  A plan whose tree cannot be coded is
rejected before any bit is written.

> The tree **cannot** be recovered from the leaf list alone: four 16x16 leaves
> inside a 32x32 node can come from HORZ→VERT or VERT→HORZ, which code
> different symbols.  That is why the plan exports `tree` and why the bridge
> does not guess.

Every entry must set `kFlagSbRoot` on the first node of a superblock (the
bridge does this for each top-level tree node). `partition == kNone` marks a
leaf; every other value must have children.

`allowed_mask` and `kFlagForced` are **spec** values, not encoder policy:

* `allowed_mask` bit *p* = `init_allowed_partitions()[p]`
  (§5.20.3.2) for that node's position/size;
* `kFlagForced` is set when the spec codes no partition syntax: the boundary
  implies the partition, or only one partition is allowed.

The writer codes `do_split` only when the node is not forced and `NONE` is
allowed, and codes `rect_type` only when the partition is rectangular, not
implied by the block size, and both HORZ and VERT are allowed. Do **not** fold
`DecisionOptions` (`max_depth`, `min_block_size`, `allow_rect`) into these
flags: they change the *choice*, not the syntax.

`pack_gpu_raster_frame()` derives both flags with the same code path as
`partition::solve_node()`, so the fallback raster plan is byte-exact with the
CPU encoder in `force_single_block` mode (verified in the unit test for
64x64 and 640x360, including the forced bottom-edge superblocks).

### 1.2 Leaf payload

Per leaf, in this order (the writer emits it; the orchestrator only has to
supply the fields):

| Field | Source | Notes |
|---|---|---|
| `y_mode`, `y_angle_delta` | `BlockDecision::luma_mode`, `luma_angle_delta` | §6.19.5.5 values; angle −3..3 |
| `is_cfl` flag | `BlockDecision::is_cfl` | M1 always codes `CFL_DERIVED_ALPHA` (cfl_index = 1); `cfl_alpha_u/v` are prediction values and are **not** coded |
| `uv_mode` | `BlockDecision::uv_mode` | `kDcPred` when `is_cfl` |
| `tx_size[3]` | derived from `BlockPlan::luma_w/h`, `chroma_w/h` | TX_MODE_LARGEST: one transform per plane |
| `eob[3]` | `BlockDecision::eob_y/u/v` | scan positions, `0 ≤ eob ≤ adjusted TX area` |
| `levels_offset` | assigned by the bridge | index into the flat level array |

### 1.3 Coefficient levels

The GPU keeps **three independent per-plane buffers** (`coeff_y`, `coeff_u`,
`coeff_v`), each indexed from the *same block's* `coeff_offset_y/u/v`. The
writer consumes **one flat, block-interleaved array**: for each block in visit
order, luma then U then V, each exactly `eob` entries, all in scan order. The
bridge performs that conversion explicitly:

```c++
for c in [0, eob[plane]):
    levels[levels_offset + eob_y*[plane>0] + eob_u*[plane>1] + c]
        = plane_buffer[coeff_offset_plane + scan[plane][c]];
```

```
GPU (three planes, own offsets, raster)      CPU stream levels[] (per block)
coeff_y: [ ... ][block0 Y 256][block1 Y ...]  [Y0..eob_y) [U0..eob_u) [V0..eob_v) | next block ...
coeff_u: [block0 U 256][ ... ][block1 U ...]  \_ one contiguous slice per block _/
coeff_v: ...
```

Consequences:

* `levels_offset` is assigned by the bridge in block visit order; per-plane GPU
  offsets never appear in the packed stream, and there is no per-plane stride.
* Only the non-zero prefix (`eob`) of each plane's scan is stored, so the flat
  array is `Σ_blocks (eob_y + eob_u + eob_v)` entries — much smaller than the
  sum of the three GPU buffers.
* A plane with `eob == 0` never reads its coefficient array, so an all-skip
  frame may pass empty coefficient spans.
* The bridge rejects a plan whose per-plane ranges **overlap** between blocks
  (two blocks coding the same coefficients), or whose range leaves
  `FramePlan::coeff_count_*` / the provided span.

The `test_coefficient_layout_conversion()` unit test covers this with
deliberately unequal, padded per-plane offsets and checks the packed array
entry by entry, including a 64x64 block (coded 64x64 → **adjusted 32x32**) and a
**32x8** luma block (reached through `VERT(64x16)` → `HORZ(32x16)`; AV2 defines
no rectangular partition at 64x8, so 64x8 is a forced `PARTITION_NONE` leaf).

`test_plan_frame_ordering()` then builds real 64x64 and 640x360 plans with
`plan::build_frame_plan()`, synthesizes decisions + coefficients, and checks
that the packed node/leaf order matches `plan_to_tsv()`, `plan.nodes` and
`plan.leaves`.  The same test finally routes the **CPU reference's own
decisions and coefficients** through the plan tree + bridge and requires the
frame bytes to equal `encode_still_picture()` — i.e. plan tree → syntax is
byte-identical to the CPU path for both resolutions.

### 1.4 Restrictions M1 pins today

| Item | Value | Why |
|---|---|---|
| `tx_mode` | TX_MODE_LARGEST (`tx_mode_select = 0`) | one transform per plane; the bridge validates that `tx_size` matches the coded block size |
| transform type | DCT_DCT only | M1 codes no `tx_type` symbol beyond DCT_DCT; a non-zero `BlockDecision::tx_type_*` is rejected |
| CfL alpha | CFL_DERIVED_ALPHA only | explicit `cfl_sign`/`cfl_alpha` syntax is not emitted |
| chroma | always coded (4:2:0) | `has_chroma` must be 1 |
| `base_q_idx` | 90 | coefficient CDF slice 0 |
| tiles / frames | one tile, one keyframe | §8.2 tile payload + OBU assembly |

---

## 2. Ordering the orchestrator must satisfy

The syntax writer is a serial state machine. The requirements are exactly the
decoder's:

1. **Block order = decode order.** Raster superblocks, preorder inside each
   superblock, leaves visited once. The writer validates superblock origins and
   child geometry against `Partition_Subsize`; a wrong order is rejected, not
   silently coded.
2. **Mode info before residual, per block.** Y-mode set/index/offset, then
   `is_cfl`/`cfl_index`, then `uv_mode`, then the three transforms in plane
   order Y, U, V.
3. **Per-plane residual order:** `txb_skip`, `eob_pt` (+ escape bits),
   `eob_extra`/raw EOB bits, luma transform type, base levels from the EOB
   position down to scan 0, `coeff_br` levels in the same reverse pass, then a
   second reverse pass for signs + quant escape starting at the EOB position
   (which is why the luma DC sign is the *last* symbol of a luma block).
4. **Serial dependency.** Do not reorder, batch, or parallelize symbol
   emission. Even with `disable_cdf_update = 1` the arithmetic coder's carry
   propagation makes the byte stream order-dependent; with `--adaptive-cdf` the
   CDF state is updated in symbol order, so a different order produces a
   different stream.
5. **One frame at a time.** A fresh `CdfState` per frame (the writer constructs
   it per call); no CDF state survives between frames or tiles.

---

## 3. CDF contexts the writer derives itself

The producer must **not** send any context state. Everything below is rebuilt
inside `assemble_still_frame()` from the block stream, in coding order:

| Context | Derived from | Reset rule |
|---|---|---|
| `do_split` / `rect_type` ctx | `above_mi[]`/`left_mi[]` = Mi_Width/Height_Log2 of previously coded leaves, sentinel 6 outside | left grid cleared at every superblock row start (`mi_col == 0`), above grid cleared at frame start |
| Y-mode set/index/offset ctx | `IntraJointModes[]` of the below-left / above-right neighbours (`joint_context`) and the §5.20.5.5 `get_intra_y_mode_set` inversion | frame start |
| `is_cfl` ctx | neighbouring chroma `uvCfL` flags (above and left, 4x4 chroma grid) | frame start |
| `txb_skip` ctx | `culLevel` (clipped to 4) and DC-sign category (0/1/2) per plane, plus luma block-size match and the U-plane EOB (`eob_u`) for V | plane-major, reset to 0 on a skipped plane |
| DC sign ctx | luma `above_dc` / `left_dc` categories | frame start |
| coefficient ctx | pre-classified level magnitudes of the already-coded neighbours **within the same block**, recomputed per plane | per plane |
| EOB CDF selection | `coded_w × coded_h` multisize and plane (luma vs chroma) | static |
| coefficient CDF slice | `base_q_idx` (M1 pins 90) | static |

Consequences for the orchestrator:

* Reconstructed/decoded neighbour state is **not** needed for syntax; only the
  block results are. Deblock/CfL/recon work can therefore run out of order as
  long as the *syntax* order above is preserved when the stream is emitted.
* The `BlockSyntax` stream is self-contained: replaying it twice yields the
  same bytes (asserted by the `from_bridge == from_hand` test).
* `AV2ENC_STILL_YMODE_CTX0/CTX2`, `AV2ENC_STILL_DCSIGN_CTX` and `AV2ENC_STILL_TRACE`
  live in the writer now (they are context/syntax debug knobs). They are read
  per symbol, so a stream produced with them set is not spec-decodable — that is
  intentional and matches the pre-refactor CPU behaviour.

---

## 4. Validation performed before any bit is written

Packing (`pack_gpu_frame`) and writing (`assemble_still_frame`) both fail with a
non-empty `error` string instead of emitting a desynchronized stream:

* tree: preorder/subtree consistency, child count per partition, block sizes
  against `Partition_Subsize`, superblock origins and total superblock count;
* leaves: `BlockPlan` position/size vs tree node, 4:2:0 chroma residual size,
  adjusted TX size vs `make_tx_params()`, `tx_type == DCT_DCT`,
  mode/angle range;
* residual: `eob ≤ adjusted area`, coefficient plane bounds,
  `levels_offset + Σeob` inside the flat array;
* frame: even dimensions, `sb_size ∈ {64, 128}`, `StillConfig` size == plan size.

`assemble_still_frame()` also returns the tile payload size (`tile_bytes`) so the
caller can keep `StillResult::tile_bytes` (and the CLI's report) unchanged.

---

## 5. Byte identity

The CPU path is the reference. After the extraction, the same 24-case local
matrix (64x64 / 640x360 / 1920x1080 × default, full intra, no-CfL, no-deblock,
depth 2, depth 0, adaptive CDF, single block) produces **identical IVF MD5s**
before and after the refactor, on macOS/arm64 and on the 3060 box (Linux,
g++ 13), and `tools/still_verify/run_matrix.sh` reports BIT-EXACT recon for every
resolution and mode (45/45 cases).

Plan-path identity, verified by `build_and_test.sh`:

* `plan::FramePlan::tree` ≡ the CPU encoder's partition tree for 64x64
  (14 leaves / 27 nodes) and 640x360 (790 leaves / 1560 nodes);
* CPU decisions + coefficients → `assemble_plan_frame()` → frame bytes ==
  `encode_still_picture()` bytes for both frames.

---

## 6. Integration: what the CLI/build agent must link

`still_syntax.h/.cpp` is the syntax API. `still_encoder.cpp` (CPU) already calls it;
the GPU orchestrator should call `syntax::assemble_still_frame()` **through the
bridge** so the coefficient layout/ordering rules above are applied once.

Public entry points (no CUDA headers, host-only, no globals):

```c++
// src/encoder/m1/still_syntax.h  (namespace av2enc::still::syntax)
bool assemble_still_frame(const FramePlan& plan,
                       std::span<const BlockSyntax> blocks,
                       std::span<const std::int32_t> levels,
                       const StillConfig& config, std::vector<std::uint8_t>& out,
                       std::string& error, std::size_t* tile_bytes = nullptr);

// src/encoder/m1/stream/still_stream_bridge.h  (namespace av2enc::still::stream)
//   host plan module -> bytes (preferred while the plan carries the tree)
bool assemble_plan_frame(const plan::FramePlan& plan,
                         std::span<const plan::BlockPlan> blocks,
                         std::span<const cuda::still_gpu::BlockDecision> decisions,
                         const GpuCoefficients&, const StillConfig&,
                         std::vector<std::uint8_t>&, std::string&,
                         std::size_t* tile_bytes = nullptr);
//   GPU block PODs -> bytes (explicit tree, or the placeholder raster)
bool assemble_gpu_frame(const cuda::still_gpu::FramePlan&, std::span<const TreeNode>,
                        std::span<const cuda::still_gpu::BlockPlan>,
                        std::span<const cuda::still_gpu::BlockDecision>,
                        const GpuCoefficients&, const StillConfig&,
                        std::vector<std::uint8_t>&, std::string&,
                        std::size_t* tile_bytes = nullptr);
bool assemble_gpu_raster_frame(/* placeholder plan, no tree */ ...);
```

Exact `CMakeLists.txt` (`av2enc_core` source list) entries:

```cmake
  src/encoder/m1/still_headers.cpp
  src/encoder/m1/still_syntax.cpp                 # REQUIRED: syntax writer
  src/encoder/m1/stream/still_stream_bridge.cpp   # REQUIRED: plan/PODs -> syntax
  src/encoder/m1/still_encoder.cpp
```

Status in this tree: `still_syntax.cpp` is already listed; the bridge is **not**
yet, so `encode_still_gpu()` cannot call `pack_plan_frame()` until that line is
added. `src/cuda/m1/plan/plan.cpp` comes in through the existing
`src/cuda/m1/**` glob, so nothing else is needed for the plan path.

The unit test needs no CMake target — `src/entropy/av2/build_and_test.sh`
style standalone build. To run it from CTest, add (in `tests/CMakeLists.txt`,
owned by the build workstream):

```cmake
add_test(NAME still_stream_bridge_test
  COMMAND bash "${PROJECT_SOURCE_DIR}/src/encoder/m1/stream/build_and_test.sh")
set_tests_properties(still_stream_bridge_test PROPERTIES TIMEOUT 300)
```

Do **not** add `still_stream_bridge_test.cpp` to a target that also links
`av2enc_core`: the test `#include`s `still_encoder.cpp` to reach the CPU decision
stream, which would duplicate the core symbols.

## 7. Remaining mismatch between `BlockSyntax` and the GPU PODs

Current state of `src/cuda/m1/still_gpu.h` versus what the writer needs. Items
marked **gap** need a header change from the orchestrator (the bridge cannot
invent the information):

| `BlockSyntax` / writer need | GPU POD | Status |
|---|---|---|
| partition tree: `partition`, `allowed_mask`, `forced`, `child_count` | `plan::FramePlan::tree` | **closed** for the host plan — `pack_plan_frame()` consumes it directly (and validates it) |
| same, for `cuda::still_gpu::BlockPlan` | `BlockPlan` has only `tree_depth` | **gap** — the GPU POD path needs a parallel `TreeNode[]` (`pack_gpu_frame`) until `BlockPlan` grows `partition`/`allowed_mask`/`forced`/`child_count` |
| `subtree` | `plan::TreeNode::subtree` | ok — recomputed and compared |
| leaf block size | `luma_w`/`luma_h`, `chroma_w`/`chroma_h`, `block_size` | ok — validated against the tree node's `BlockSize` |
| `tx_size[3]` | derived from coded dims (`TX_MODE_LARGEST`) | ok — `tx_w/h`, `chroma_tx_w/h` are cross-checked against `make_tx_params()` |
| `eob[3]` | `BlockDecision::eob_y/u/v` (`uint16`) | ok — fits the 1024-element 64x64→32x32 luma TX; `BlockSyntax` uses `uint32` for margin |
| `y_mode`, `y_angle_delta`, `uv_mode`, `is_cfl` | `BlockDecision` | ok |
| transform type | `BlockDecision::tx_type_y/u/v` | ok — must be 0 (DCT_DCT), otherwise rejected |
| CfL alpha syntax | `BlockDecision::cfl_alpha_u/v` (Q8) | **ignored** — M1 codes `CFL_DERIVED_ALPHA` only; explicit `cfl_sign`/`cfl_alpha` is not implemented |
| `levels[]` (flat, interleaved) | three per-plane buffers + `coeff_offset_*` / `PlaneTxPlan::coeff_offset` | **handled by the bridge** (§1.3), incl. non-overlap validation |
| `sse`, `plane_offset_*`, `avail`, `tx_upscale_*`, `reserved` | `BlockPlan`/`BlockDecision` | unused by syntax (recon/prediction only) |
| `has_chroma` | `BlockPlan::has_chroma` | not used by syntax — M1 codes chroma for every leaf; validated indirectly by the chroma residual size and the tree recursion check |

## 8. Build note

`src/encoder/m1/still_syntax.cpp` is a normal translation unit in `av2enc_core`
(`CMakeLists.txt`, owned by the CLI/build workstream). This directory's
`build_and_test.sh` compiles the bridge plus the test and links the core pieces
it needs, so it runs on a Mac/CPU-only host without CUDA or CMake changes. The
test `#include`s `still_encoder.cpp` to reach the CPU decision stream, so that file
must not additionally appear on the test compile line.
