# AV2 PC-Wiener (§7.20.4) — CUDA

Device implementation of the pixel-classified Wiener filter, matching the CPU
reference in `src/encoder/filters/spec_tables/pc_wiener.{h,cpp}`.

Scope:

- §7.20.2 luma source sampling (CurrFrame outside the 64-line stripe,
  CdefFrame inside, two-line clamp);
- §7.20.4 features: the 6x6 box (`Lead=1`, `Lag=4`) of vertical /
  anti-diagonal / diagonal curvature;
- `get_q` (§7.14.2) and `get_qval_given_tskip` with
  `Pc_Wiener_Mode_Weights` / `Pc_Wiener_Mode_Offsets`;
- the 4096-entry `Pc_Wiener_LutToClass` 256 → 64-class map;
- `Pc_Wiener_Sub_Classify` (class → 25-tap filter index) and the fixed
  `Pc_Wiener_Filters[4][64][13]` bank;
- `Pc_Wiener_Sub_Classify2` (filter set × `num_classes_idx` → subclass);
- `skipFilter = 1` class-only mode and `skipFilter = 0` block filtering;
- `LrTxSkip` with the MI-block, stripe and tile clamps;
- 8, 10 and 12-bit luma, arbitrary frame geometry, all LR unit sizes
  (64 / 128 / 256 / 512 luma samples).

The normative tables are **not duplicated**. `create()` uploads the CPU tables
from `src/encoder/filters/spec_tables/pc_wiener_tables.h` and
`src/encoder/filters/spec_tables/pc_wiener.h` once; the shared device header
only reads a `PcWienerTables` view of them. The emulation harness fills the same
view with host pointers, so the CPU and GPU cannot drift.

## Files

| File | Role |
|---|---|
| `pc_wiener_device.cuh` | the single device implementation (also compiled by the emulation test) |
| `pc_wiener.h` | host API, `Arena` workspace, launchers |
| `pc_wiener.cu` | kernels and launch wrappers |
| `pc_wiener_emulation_test.cpp` | host parity harness, no GPU / no nvcc |
| `pc_wiener_test.cu` | real CPU-vs-GPU parity test |
| `build_and_test.sh` | `nvcc` → GPU test, otherwise emulation |
| `build_gpu_test.sh` | `nvcc` + a CUDA device required |
| `build_emulation_test.sh` | host-only parity |

## API

```cpp
#include "cuda/pc_wiener/pc_wiener.h"

namespace gp = av2enc::cuda::pc_wiener;

gp::PcWienerConfig config{};
config.width = 1920;
config.height = 1080;
config.bit_depth = 10;      // 8, 10 or 12
config.unit_size = 64;      // 64/128/256/512 (§5.18.7 luma_shift 3..0)

const std::size_t bytes = gp::pc_wiener_required_device_bytes(config);

av2enc::cuda::Arena arena;
arena.init(bytes + slack);

gp::PcWienerWorkspace ws;
gp::Status status = ws.create(arena, config, &error);  // uploads tables once

gp::PcWienerSource src{};   // device planes + §7.20.2 extents/stripes
gp::PcWienerParams params{}; // qindex, bit depth, MiColEnd, tile rows, LrTxSkip

// skipFilter = 1: class map only.
gp::pc_wiener_classify(ws, src, params, stream);
// class map is ws.class_grid() (mi_rows * mi_cols bytes)

// skipFilter = 0: LrFrame := CdefFrame, then every 4x4 MI block.
gp::pc_wiener_filter_grid(ws, src, params, ws.class_grid(), lr, lr_stride,
                          stream);

// Per-unit mode A (RESTORE_PC_WIENER units only).
gp::pc_wiener_filter_units(ws, src, params, ws.class_grid(), unit_type, lr,
                           lr_stride, stream);

// Block primitives (no CdefFrame copy).
gp::pc_wiener_filter_block(ws, src, params, x, y, w, h, lr, lr_stride, stream);
gp::pc_wiener_filter_class_block(ws, src, params, x, y, w, h, cls, lr,
                                 lr_stride, stream);

// Pc_Wiener_Sub_Classify / Pc_Wiener_Sub_Classify2, 256 entries each.
gp::pc_wiener_class_filter_lut(ws, filter_set, out256, stream);
gp::pc_wiener_subclass_lookup(ws, filter_set, num_classes_idx, out256, stream);
```

`pc_wiener_classify_block()` is the single-block `skipFilter = 1` form.

Invalid arguments return `Status::kInvalidArgument`; a workspace smaller than
`required_device_bytes()` returns `Status::kArenaExhausted` and
`ws.required_bytes()` reports the size to give the arena. All launchers are
allocation-free and do not synchronise.

## Memory

Every device byte comes from the caller's persistent
`av2enc::cuda::Arena` (`src/cuda/arena.h`). There is no `cudaMalloc`,
`cudaFree`, `cudaMallocHost` or per-launch allocation in this module, and the
test uses the arena for its planes too.

`required_device_bytes(config)` is the exact sum of the 256-byte-aligned
allocations:

- uploaded normative tables: 22,784 B
  (filters 6,656; LUT 4,096; Sub_Classify 1,024; Sub_Classify2 8,192;
  qlookup + Ac_Qlookup + taps + normalizer + mode weights/offsets);
- persistent class map: `ceil(width/4) * ceil(height/4)` B, 256-aligned;
- scratch unit-active bitmap: the same size, 256-aligned.

Examples:

| Frame | `required_device_bytes` |
|---|---|
| 96 x 72 | 23,808 B |
| 1920 x 1080 | 282,368 B |
| 3840 x 2160 | 1,059,584 B |

The 96x72 test lists 47,360 B of plane/input buffers before the workspace and
23,808 B for the workspace, so `arena.peak_used()` is 71,168 B; the test prints
the measured value. `create()` bumps the workspace from the entry mark and
rewinds the arena on every failure path, so the one-byte-short case leaves
`used() == 0`.

## Tests

```sh
src/cuda/pc_wiener/build_and_test.sh      # GPU test if nvcc exists, else emulation
src/cuda/pc_wiener/build_gpu_test.sh      # nvcc + a CUDA device required
src/cuda/pc_wiener/build_emulation_test.sh
```

`pc_wiener_test.cu` covers:

- table uploads byte-identical to the CPU normative tables;
- `Sub_Classify` / `Sub_Classify2` device LUTs for every entry;
- 8/10/12-bit class maps and full-frame filtered planes over every filter set
  boundary (`base_q_idx` 129/130, 189/190, 219/220) and the 64-line stripes;
- every class 0..63 through the class-grid filter path for each filter set;
- `skipFilter = 1` leaves LrFrame untouched and writes exactly one grid byte
  for the block form;
- every LR unit size 64/128/256/512 with random `unit_type` values, including
  inactive units;
- `required_device_bytes()` sizing, a one-byte-short arena (distinct
  `kArenaExhausted` + reported size), and two launches inside
  `mark()`/`rewind()` with no net arena growth.

`pc_wiener_emulation_test.cpp` runs the same device header against the CPU
reference with CUDA keywords stubbed. On this host (no nvcc) it reports:

```
pc_wiener_emulation_test: 36243 cases, 126633 checks, 0 failures
```

The emulation coverage includes an independent 25-tap oracle for all
4 filter sets × 64 classes × 3 bit depths × 3 block shapes, `get_q` /
`get_qval_given_tskip` over the full `qindex` range, `LrTxSkip` tile clamping,
stripe bounds, and `CountUnits`/`UnitIndexForBlock` for all unit sizes.

## Remote run

The GPU test builds and passes on the RTX 3060 sm_86 box (CUDA 12.8):

```
$ ARCH=sm_86 ./build_gpu_test.sh
device: NVIDIA GeForce RTX 3060 (sm_86), 11920 MiB
pc_wiener_test: 158 cases, 716 checks, 0 failures
  required_device_bytes=23808 persistent=23296 scratch=512
  arena peak_used=71168 bytes (capacity=150804)
PC-WIENER CHECKS PASSED
```

Commands:

```sh
AV2ENC_HOST=<user>@<host>
AV2ENC_PORT=<port>
tools/sync_remote.sh
ssh -p "$AV2ENC_PORT" "$AV2ENC_HOST" \
  'cd /workspace/av2enc/src/cuda/pc_wiener && ARCH=sm_86 ./build_and_test.sh'
```

or directly:

```sh
ssh -p "$AV2ENC_PORT" "$AV2ENC_HOST" \
  'cd /workspace/av2enc/src/cuda/pc_wiener && ARCH=sm_86 ./build_gpu_test.sh'
```
