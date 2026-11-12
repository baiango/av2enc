# AV2 TIP — CUDA motion field and prediction

GPU counterpart of `src/encoder/inter/temporal/` §7.10 and §7.13.3.2 for the
build-TIP path. It constructs the TIP motion field from a §7.9 field, builds
the per-8x8/16x16 plan, and predicts 8-bit and 10-bit TIP blocks with the same
integer arithmetic as the CPU reference.

No bitstream, no mode decision, no entropy, no root CMake change. The CPU
reference remains authoritative; this component only reproduces its samples.

## Files

```text
src/cuda/tip/
  tip.h                 public API, arena workspace, required_device_bytes()
  tip_device.cuh        one shared host/device arithmetic implementation
  tip.cu                kernels + host launch wrappers
  tip_emulation_test.cpp host parity test (no GPU/nvcc)
  tip_test.cu           real CUDA parity test (arena-backed)
  build_and_test.sh     nvcc -> GPU test, otherwise emulation test
  build_gpu_test.sh     nvcc + device required
  build_emulation_test.sh
  README.md
```

## API

```cpp
#include "cuda/tip/tip.h"

namespace cuda_tip = av2enc::cuda::tip;

// §7.10.2–.5 field work.  Buffers are caller-owned (arena) device memory.
cudaError_t tip_scale_motion_field(const TipField&, const TipMotionFieldConfig&,
                                   cudaStream_t = nullptr);
cudaError_t tip_hole_fill_motion_field(const TipField&,
                                       const TipMotionFieldConfig&,
                                       cudaStream_t = nullptr);
cudaError_t tip_block_average_motion_field(const TipField&,
                                           const TipMotionFieldConfig&,
                                           const TipFieldScratch&,
                                           cudaStream_t = nullptr);
cudaError_t tip_fill_motion_field_sample_gap(const TipField&,
                                             const TipMotionFieldConfig&,
                                             cudaStream_t = nullptr);
cudaError_t tip_setup_motion_field(const TipField&,
                                   const TipMotionFieldConfig&,
                                   const TipFieldScratch&,
                                   int allow_tip_hole_fill,
                                   cudaStream_t = nullptr);

// §7.10.6 plan + per-4x4 MV grids.
cudaError_t tip_build_plan(const TipField&, const TipMotionFieldConfig&,
                           const TipPlanConfig&, TipPlan*,
                           cudaStream_t = nullptr);

// §7.13.3.2 prediction.  One CUDA block per job / plan block.
cudaError_t tip_predict_block(const TipPredictJob&, cudaStream_t = nullptr);
cudaError_t tip_predict_batch(const TipPredictJob* device_jobs, int count,
                              int sample_bytes, cudaStream_t = nullptr);
cudaError_t tip_predict_plan(const TipPlan&, const TipPlanPredictConfig&,
                             cudaStream_t = nullptr);

// Pure host helper matching §7.22 get_tip_offsets().
TipOffsets tip_get_offsets(const TipMotionFieldConfig&);
```

`TipMotionFieldConfig` carries only the fields the §7.10 process reads:
`mi_cols`, `mi_rows`, `proj_step`, `sb_64x64`, `enable_tip`, `order_hint`,
past/future reference counts, and the closest past/future order hints.
`TipPlanConfig` carries `bw4`/`bh4` (2 = 8x8, 4 = 16x16),
`tip_global_wtd_index` (0..7) and `TipGlobalMv`.

`TipPredictJob` supports:

- `num_refs` 1 or 2; a `cwp_weight` of 16 uses list 0 only, exactly like
  `tip_predict_block()`;
- `bit_depth` 8 or 10, with `sample_bytes` 1 (`uint8_t`) or 2 (`uint16_t`,
  LSB-aligned);
- all `InterpFilter` values, including the automatic four-tap downgrade for a
  width or height ≤ 4;
- optional `use_ref_area` with per-list inclusive `[first_x, first_y,
  last_x, last_y]` clamps;
- full 17-bit MVs, `±REFMVS_LIMIT` field clamps, `Div_Mult` projection,
  `ProjStep` even positions, and `Clip1` at the selected bit depth.

`tip_predict_batch` takes a device array of jobs; the caller owns job validity
(device memory cannot be inspected on the host). `tip_predict_plan` derives all
three planes from a device `TipPlan` in one launch.

## Arena memory model (mandatory)

All device memory comes from the single persistent `av2enc::cuda::Arena`.
There is no `cudaMalloc`, `cudaFree`, or `cudaMallocHost` in this component or
its tests.

```cpp
TipWorkspaceConfig cfg;
cfg.mi_cols = 160; cfg.mi_rows = 120;
cfg.bw4 = cfg.bh4 = 2;           // largest plan
cfg.include_plan = 1;

const std::size_t need = tip::required_device_bytes(cfg);
Arena arena;
arena.init(need + slack);
TipWorkspace ws;                  // persistent, allocated once
tip::tip_workspace_init(arena, cfg, &ws, &need);

const std::uint64_t mark = tip::tip_frame_mark(arena);
// ... upload refs, launch TIP, download the frame ...
tip::tip_frame_rewind(arena, mark);   // net-zero per frame
```

`required_device_bytes(config)` is the exact largest-launch need:

| Buffer | Bytes |
|---|---:|
| field `mv[2]` | 8 · w8 · h8 |
| field `valid` | w8 · h8 |
| field `offset` | 4 · w8 · h8 |
| block-average `avg_mv[2]` | 8 · w8 · h8 |
| block-average `avg_valid` | w8 · h8 |
| plan blocks | `sizeof(TipPlanBlock)` · count |
| `mv_grid` + `mv_grid1` | 8 · mi_cols · mi_rows each |

Each array is rounded up to the arena's 256-byte alignment. For 160x120 with a
16x16 plan (`bw4=bh4=2`, so 8x8 blocks), the reported values are:

- **`required_device_bytes` = 585,728 bytes** total;
- field + hole-fill scratch + block-average buffers = **105,728 bytes**;
- plan blocks + two 4x4 MV grids = **480,000 bytes** (the remainder).

`tip_workspace_init` allocates all of them once and returns
`Status::kArenaExhausted` with `*required_bytes` set when the arena is short.
The bump pointer is rolled back, so no partial pointers leak and the caller can
release/re-init the arena with the reported size. `tip_workspace_attach` binds
caller-allocated device buffers instead of the arena.

The launchers allocate nothing, copy nothing, and synchronise nothing. Hole
fill and block average are in-place scratch operations; the block-average
filter uses the scratch copy because a single kernel cannot read and write the
same field cells.

## Bit exactness

`tip_device.cuh` is the only implementation. The kernels and the host-emulation
test call the same functions.

- §4.8 helpers (`round2`, `round2_signed`, `round2_signed64`,
  `get_relative_dist`, `clip3`) are reused from
  `src/encoder/inter/advanced/advanced_common.h`.
- §7.13.3.17 sample locations reuse `adv::motion_vector_scaling()`, including
  its integer overflow/clamp behaviour.
- §7.9.4/.5 projection, `tmvp_*`, sampled positions, block checks,
  `get_tip_offsets`, `calc_avg`, the §7.10.2–.5 loops, the §7.10.6 block
  derivation, and the §7.13.3.18 two-pass filter are transcribed from
  `temporal_math.h`, `motion_field.cpp`, `temporal_predict.h` and `tip.cpp`.
- Device visibility: namespace-scope `inline constexpr` arrays are host
  variables in device code. `Div_Mult`, the five TIP directions and
  `Weight_Div_Mult` are therefore exposed as `__host__ __device__ constexpr`
  accessors (`div_mult`, `tip_dir_row`/`tip_dir_col`, `weight_div_mult`) with
  identical values. The subpel table already has device storage through
  `AV2ENC_MC_SUBPEL_TABLE` in `inter/mc/mc_subpel_tables.h`. Do not reintroduce
  a plain namespace-scope array into these paths.
- 8-bit is compared against the actual CPU reference (`setup_tip_motion_field`,
  `fill_motion_field_sample_gap`, `build_tip_plan`, `tip_predict_block`).
- The CPU reference is 8-bit only. 10-bit is compared against a
  bit-depth-parameterised CPU transcription of `temporal_predict.h` living in
  the test, with `Clip1` to `(1 << bit_depth) - 1`. This is called out
  explicitly rather than claiming an 8-bit-only oracle validates 10-bit.

## Tests

```
src/cuda/tip/build_and_test.sh        # GPU test if nvcc, else emulation
src/cuda/tip/build_gpu_test.sh        # nvcc + CUDA device required
src/cuda/tip/build_emulation_test.sh  # no GPU, no nvcc
```

### `tip_emulation_test.cpp` (run on this host)

Compiles `tip_device.cuh` with CUDA keywords stubbed and compares it to the CPU
reference in `src/encoder/inter/temporal/`.

- every §7.9.4/.5 projection and clamped projection for all 32 frame
  distances, numerators −31..31, and extreme MVs (±65535, ±4096, ±2047, 0);
- 240 randomised `§7.10.2–.5` fields: 8x8 … 96x60 MI, `ProjStep` 1/2,
  `SbSize` 64x64 and smaller, hole fill on/off, sparse/dense validity (30%/65%/
  100%), MVs ±4096, offsets ±31, all past/future/single-side offset modes; each
  case checks the field, the three offsets and the plan blocks/grids.
- 60 standalone `fill_motion_field_sample_gap()` cases (sparse, ProjStep 2).
- 320 prediction variants × (8-bit CPU reference + 10-bit reference), all
  eight `Tip_Weighting_Factor` values (including −4), four filters, 8x8 and
  16x16 sizes, four-tap downgrades, luma/4:2:0, scaled refs, ref areas, and
  full-range MVs.

Result on the macOS dev host (no nvcc):

```
tip_emulation_test: 330632 checks, 0 failures
```

### `tip_test.cu` (run on the Linux/CUDA box)

Same coverage through the kernels, plus:

- arena sized from `required_device_bytes() + 32 MiB`;
- **one-byte-short negative test**: `Status::kArenaExhausted`, reported size,
  no partial pointers, bump pointer unchanged;
- persistent workspace with two repeated frames using `mark()`/`rewind()` and
  an explicit no-net-growth check;
- 48 kernel field cases + plan/grid parity;
- 240 kernel prediction cases (8-bit against the CPU reference, 10-bit against
  the bit-depth reference), through both `tip_predict_batch` and
  `tip_predict_block`;
- a **full-clip** parity case: CPU §7.9 motion-field estimation seeds a real
  field, then CPU and GPU independently run `setup -> plan -> predict` for
  160x120 frames (odd 30-MI height, 4:2:0, edge blocks), 8x8 and 16x16, 8-bit
  and 10-bit, four weights. Field, plan blocks, both 4x4 grids and all three
  predicted planes must match sample-for-sample.

The test prints the check count, `required_device_bytes`, workspace bytes and
the arena peak at the end. During development the launch bodies were also run
through a host harness that executes the same kernels as sequential host loops
(with a malloc-backed arena shim): **5,759 checks, 0 failures**, peak
803,328 bytes. The nvcc build has not been run on the macOS host (no nvcc).

## Remote command

```sh
tools/sync_remote.sh
ssh -p "$AV2ENC_PORT" "$AV2ENC_HOST" \
  'cd /workspace/av2enc/src/cuda/tip && ./build_and_test.sh'
```

For the secondary target use `ARCH=sm_120 ./build_gpu_test.sh`.

## Integration notes

- The GPU currently starts from a host-produced §7.9 motion field. The §7.9
  projection/estimation loop is not in this component; the CPU result is
  uploaded once per frame. A GPU §7.9 kernel can replace that upload without
  changing this API.
- Call `tip_setup_motion_field()` once per TIP frame, then
  `tip_build_plan()`, then `tip_predict_plan()`; the three are asynchronous on
  the caller's stream. One sync at the frame boundary is the integration
  owner's job.
- `TipPlane::width/height` are the MI-aligned plane extents used by
  §7.13.3.17, matching `PlaneView` in the CPU reference; `TipPlanPredictConfig`
  also takes the visible destination extent so the right/bottom partial block
  is clipped exactly as `predict_tip_plan()` does.
- No root `CMakeLists.txt`, `src/main.cpp`, core or other component files were
  changed.
