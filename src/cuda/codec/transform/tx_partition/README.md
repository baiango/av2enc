# CUDA transform partition / `TX_MODE_SELECT` (`src/cuda/tx_partition/`)

GPU support for the transform-partition decision implemented on the CPU in
`src/encoder/m2/tx_partition/`. This is a **component**: no shipped encoder path
calls it, the root `CMakeLists.txt` is untouched, and no bitstream claim is made
here.

What it adds:

* device-side `find_tx_size()` / `set_tx_size()` and the §5.20.6.3 partition
  geometry (`plan_geometry`, `legal_splits`, `partition_signallable`);
* §5.20.7.30 scans for all **25** TX sizes × **3** `TX_CLASS` values, including
  every rectangular size, as a device table plus the shared generator;
* a device `LumaTxSizes` / `LumaTxMiddle` / `LumaTxScanOrder` writer and a host
  `build_luma_grid()` returning the CPU `av2enc::video::txpartition::LumaTxGrid` type;
* plan resolution from one partition decision to one `TxBlockJob` per TX part,
  executed by the existing `src/cuda/tx_variants/` `_multi` kernels (one CUDA
  block per TX part);
* batched forward, inverse, dequantize and dequant→inverse paths, plus in-place
  forms;
* a host wrapper that builds the per-part jobs, uploads them once and launches
  one multi transform per plan batch;
* CPU-vs-device bit-exact tests: emulation on any host, real GPU on `sm_86`.

Only files in this directory were added.

## Files

| File | Contents |
|---|---|
| `tx_partition_device.cuh` | shared host/device core: geometry, scans, grid, job resolution |
| `tx_partition_plan.h/.cpp` | arena workspace, `required_device_bytes`, batch/plan host helpers, `build_luma_grid` |
| `tx_partition_gpu.h/.cu` | kernels (resolve, dequant, scan, grid) and arena-aware launchers |
| `tx_partition_emulation_test.cpp` | no-nvcc parity test of the exact shared core + arena |
| `tx_partition_gpu_test.cu` | real-GPU parity + arena test |
| `build_and_test.sh` | host emulation (no GPU, no nvcc) |
| `build_gpu_test.sh` | `nvcc -arch=sm_86` build and run |

## API

```cpp
#include "cuda/tx_partition/tx_partition_gpu.h"
namespace txp = av2enc::cuda::txpart;

// ---- sizing and persistent tables -----------------------------------------
txp::LaunchConfig config;      // max_plans: largest batch of partition plans
config.max_plans = 1024;
const std::uint64_t need = txp::required_device_bytes(config);

arena.release();               // caller policy: size from the device
arena.init(need);              // + the caller's own buffers

txp::Workspace workspace;      // persistent d_params[16][25] + 25x3 scan table
std::uint64_t required_out = 0;
std::string error;
av2enc::cuda::Status s =
    workspace.init(arena, config, &required_out, &error);

// ---- one plan = one <= 64x64 partition decision ---------------------------
txp::PlanRequest request;
request.mi_size   = 12;              // e.g. BLOCK_64X64
request.partition = txp::kPartSplit; // §5.20.6.3 tx_partition_type
request.mi_row    = 0;
request.mi_col    = 0;
request.tx_type   = 0;               // luma TX_TYPE 0..15

txp::PlanBatch batch;
txp::make_batch(request, &batch);    // prefix offsets, total parts/elems

// ---- batched transforms (one CUDA block per TX part) ----------------------
txp::forward_batch(arena, workspace, batch,
                   d_residual_adjusted, d_coeff, d_levels, stream /*or 0*/);
txp::inverse_batch(arena, workspace, batch,
                   d_levels, d_residual_coded, stream);
txp::dequantize_batch(arena, workspace, batch,
                      d_levels, d_dequant, stream);       // may be in place
txp::inverse_dequant_batch(arena, workspace, batch,
                           d_levels, d_dequant, d_residual_coded, stream);

// ---- device scans / device grid -------------------------------------------
txp::build_scan_tables_device(workspace.scan_view(), stream);
txp::build_grid_device(arena, workspace, batch, device_grid_view, stream);

// ---- host LumaTxGrid builder ---------------------------------------------
av2enc::video::txpartition::LumaTxGrid grid = txp::build_luma_grid(batch, rows, cols);
```

Buffer layout follows `src/cuda/tx_variants/`: each job's adjusted-size arrays
(residual/coeff/levels/dequant) live at `job.in_offset`, and its coded-size
residual lives at `job.out_offset`. Offsets are prefix sums over the parts, so a
mixed batch of sizes needs no uniform stride. `make_batch()` computes them; the
device resolve kernel writes the identical values.

Supported in-place forms, both tested bit-exactly:

* `inverse_batch(..., d_levels, d_residual_coded)` where
  `d_residual_coded == d_residual_adjusted` (the inverse reads only levels);
* `dequantize_batch(..., d_levels, d_levels)` and
  `inverse_dequant_batch(..., d_levels, d_levels, ...)` — each thread reads then
  writes its own element.

## Arena rules (mandatory)

Every device byte comes from one persistent `av2enc::cuda::Arena`.
`src/cuda/tx_partition/*` contains no direct device allocation call at all; both
build scripts grep the component sources and fail the build if one appears.

* `required_device_bytes(config)` is exact: persistent tables plus the largest
  per-launch scratch, every array rounded to `Arena::kAlignment` (256 B).
* `Workspace::init()` allocates the persistent tables exactly once
  (`allocate_array`), uploads the `[16][25]` `TxParams` table and the 25×3 scan
  table, and verifies the configured scratch still fits. On a short arena it
  returns `Status::kArenaExhausted` with `*required_bytes` set to the full
  requirement and leaves the arena unchanged. There is no allocation fallback.
* Each launcher takes `Arena&`, calls `acquire_scratch()` (arena `mark()`),
  allocates the plan/job/part arrays, launches, and `release_scratch()`s
  (`rewind(mark)`). Scratch reuse by the next launch is safe because all scratch
  writes are stream-ordered copies/kernels on the caller's stream; the module
  never synchronises in a launcher.
* Tests size the arena from `required_device_bytes`, run two launches
  back-to-back, and assert `arena.used()` returns exactly to the mark. A second
  arena one byte short is rejected cleanly with `kArenaExhausted` and the exact
  required size.

### Measured sizing (config `max_plans = 1024`, worst case 4 parts/plan)

| Quantity | Bytes |
|---|---:|
| `persistent_device_bytes` (TxParams table + scans) | 176,384 |
| `scratch_device_bytes` (plans + jobs + resolved parts) | 344,064 |
| **`required_device_bytes`** | **520,448** |
| arena peak during the emulation test | 520,448 |
| arena peak during the GPU test (component + test buffers) | reported at run time |

The one-byte-short test uses a 520,447-byte arena and gets
`kArenaExhausted` with `required_bytes == 520448`.

## Parity tests

### Host emulation (no nvcc, no GPU)

```sh
./src/cuda/tx_partition/build_and_test.sh
```

The test compiles the real shared device header with the host compiler and
links the real `av2enc::cuda::Arena` against a tiny host CUDA runtime stub. That
exercises the arena budget, plan/job offsets, geometry, scans and grid without
a GPU. Transforms are the CPU `src/encoder/tx_variants/` reference invoked
through the exact `TxBlockJob` descriptors the GPU multi-kernel consumes.

Result on this host (macOS, Apple clang, `WERROR=1` clean):

```text
av2enc CUDA transform-partition host-emulation test
config: max_plans=1024 (max parts=4096)
  scans: 76800 table entries, fingerprint 0xbfedfede43185b0b
  legal (block size, partition) pairs: 80
  plan layout cases: 80
  single-plan transform cases: 1280
  chunk-plan transform cases: 640
  batch plans: 91 (parts: see workspace)
  required_device_bytes(1024 plans) = 520448
    persistent = 176384, scratch = 344064

12198934 checks, 0 failures
RESULT: PASS
```

Coverage:

* **geometry** — all 29 block sizes × 8 partition values × `reduced_tx_part_set`
  0/1 × three origins, compared field-by-field with the CPU
  `tx_partition_geometry()` / `legal_splits()`;
* **scans** — all 25 sizes × 3 classes, 76,800 entries, compared with the CPU
  `get_scan()` (rectangular 2D, HORIZ and VERT included);
* **plan layout** — all 80 legal (block size, partition) pairs; option coverage
  reaches all 25 TX sizes; chunked >64×64 blocks use one `NONE` plan per 64×64
  chunk; the grid matches a direct `set_tx_size()` walk cell by cell;
* **transforms** — 1,280 single-plan cases (all 16 `TX_TYPE`s for every legal
  partition) and 640 64×64 chunk cases; coefficients, levels, dequant and
  reconstruction are compared element by element; in-place inverse and in-place
  dequant included; a 91-plan mixed batch and a second independent batch run
  back-to-back through the same arena window;
* **arena** — exact fit, uploaded `TxParams`/scan tables, two full-scratch
  launches with `mark()`/`rewind()` and `used()` returning to the mark, peak
  usage, over-max plan rejection, and the one-byte-short negative case.

Emulation cannot execute CUDA kernels. The parts that **must run on the RTX
3060 (sm_86)** are:

1. `resolve_kernel` — device `find_tx_size` / `set_tx_size` / `plan_resolve`
   (validated indirectly by the device grid and by the transform jobs).
2. `scan_table_kernel` — the device 25×3 scan table.
3. `dequant_kernel` — the 528 / 3-bit dequantization per `TxBlockJob`.
4. `grid_kernel` — the device `LumaTxSizes` writer.
5. The reused `txv_forward_transform_multi` / `txv_inverse_transform_multi` /
   `txv_inverse_block_multi` kernels over real device buffers (exact FP64
   forward parity, in-place aliases, mixed-size jobs).
6. The real `av2enc::cuda::Arena` (`cudaMalloc` at init) and the launch-time
   `kArenaExhausted` path.

### Real GPU

```sh
./src/cuda/tx_partition/build_gpu_test.sh
```

`build_gpu_test.sh` compiles the CPU oracles with
`g++ -O3 -ffp-contract=off`, the CUDA objects with
`nvcc -std=c++20 -arch=sm_86 -O3 -lineinfo -fmad=false
--expt-relaxed-constexpr`, and links one binary. It runs the same plan matrix
(80 legal pairs + 40 chunk plans × 4 TX_TYPE families = 480 plans in one
batch), the device scan table, the device grid over every legal plan, forward /
inverse / dequant / in-place paths, two back-to-back full-batch launches, and
the one-byte-short arena case.

Verified on the RTX 3060 (sm_86, CUDA 12.8) box:

```text
av2enc CUDA transform-partition test
device: NVIDIA GeForce RTX 3060 (sm_86, 28 SMs, 11.6 GiB)
config: max_plans=1024
required_device_bytes = 520448 (persistent 176384, scratch 344064)
arena: 11504384 bytes (component 520448 + test buffers 10983680)
arena after workspace init: used=176384 remaining=11328000
  device scans: 75 (size,class) tables
  device grid/geometry cases: 120 (legal + chunked)
  mixed batch: 480 plans, 872 parts, 342016 adj elems, 890880 coded elems
  batch 2: 40 plans, 69 parts, 27776 adj elems, 71552 coded elems
  leak check: 480 plans, 872 parts, 342016 adj elems, 890880 coded elems
  leak check: 480 plans, 872 parts, 342016 adj elems, 890880 coded elems
arena peak_used = 11321344 (test mark 11160064)
short arena: status=arena exhausted required=520448
13456988 checks, 0 failures
RESULT: PASS
```

The geometry tables are device-side copies marked `__device__ constexpr` in the
`__CUDA_ARCH__` pass (host `constexpr` otherwise), because nvcc cannot odr-use a
namespace-scope host `constexpr` array from a kernel. A host-only
`static_assert` pins all nine tables to the shipped CPU §9.2 arrays.

### Exact remote command

From the development host:

```sh
ssh -p <port> <user>@<host> \
  'cd /workspace/av2enc/src/cuda/tx_partition && ./build_gpu_test.sh'
```

Or interactively:

```sh
ssh -p <port> <user>@<host>
cd /workspace/av2enc/src/cuda/tx_partition
./build_gpu_test.sh                            # sm_86 (RTX 3060)
# optional:
CUDA_LAUNCH_BLOCKING=1 ./build_gpu_test.sh     # serialized launches
compute-sanitizer --tool memcheck \
  "${TMPDIR:-/tmp}/av2enc_tx_partition_gpu/tx_partition_gpu_test"
```

## Notes / limits

* The component covers `TX_MODE_SELECT` geometry and transforms. It does not
  emit syntax or CDFs; that stays in `src/encoder/m2/tx_partition/`.
* Chroma is unchanged: AV2 TX partitioning is luma-only.
* `TX_PARTITION_HORZ5/VERT5` are rejected, matching the CPU geometry (no syntax
  path can select them).
* A plan describes one **≤64×64** region. For a coding block larger than 64 in
  either dimension, the caller emits one `kPartNone` plan per 64×64 chunk with
  `mi_size == BLOCK_64X64`, exactly like `read_tx_size()`'s chunk loop.
* Device and host read the same §9.2 arrays from
  `encoder/m2/tx_partition/tx_partition_tables.h`; the GPU build enables
  `--expt-relaxed-constexpr` so device code can use those constexpr tables.
