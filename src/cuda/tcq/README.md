# CUDA TCQ / CCTX / parity hiding (`src/cuda/tcq/`)

Status: **component only.** New directory. It is not wired into
`CMakeLists.txt`, `src/main.cpp`, the M1/M2 core, the entropy writer or the
bitstream writer, and nothing is committed. Files outside `src/cuda/tcq/` are
not modified.

This is the GPU implementation of the encoder-side quantisers in
`src/encoder/tcq/`:

| Feature | Reference | What the GPU does |
|---|---|---|
| Trellis-coded quantisation | `tcq::quantize()` | 8-state reverse-scan DP, exact cost model and tie-breaks, one thread per block |
| Level mapping | `tcq::syntax_level_for()` / `syntax_escape_for()` | Parity-dependent `pre_max` / `pre_max - 1` choice, `x = (q - l) / 2` |
| CCTX | `tcq::cctx_*` | Six `Cctx_Mtx` rotations, joint U/V dequant, seven-candidate scalar search |
| Parity hiding | `parity::quantize()` (`kMinDistortion`) | DC/AC parity-repair candidate set, exhaustive or pruned AC search |
| Coverage | — | 8/10-bit coefficient magnitudes, all 25 TX sizes, 2D/HORIZ/VERT |

The CPU (`src/encoder/tcq/`) remains the reference. The GPU runs the *same*
DP and the *same* answer or reports a rejected job.

## Files

| File | Purpose |
|---|---|
| `tcq_types.h` | CUDA-free POD descriptors (`TcqJob`, `CctxJob`, `ParityJob`) and device batch views |
| `tcq_device.cuh` | Shared device core: state map, syntax mapping, level bound, TCQ DP, CCTX search, parity-hiding search |
| `tcq.h` | CUDA API: config, exact `required_device_bytes()`, Arena workspace, slot helpers, launchers |
| `tcq_host.h` / `.cpp` | Host helpers: `tcq_scan_and_pre_max()`, `tcq_fill_rate_tables()` |
| `tcq.cu` | Kernels and launchers (no allocation, no synchronization) |
| `tcq_emulation_test.cpp` | Host-emulation parity test (no nvcc, no GPU) |
| `tcq_test.cu` | Real GPU test: CPU-vs-GPU bit-exactness and the arena contract |
| `build_and_test.sh` | Emulation test, then the GPU test when nvcc is present |
| `build_gpu_test.sh` | Device-visibility audit + nvcc build + GPU test |

## API

```cpp
namespace av2enc::cuda::tcq {

struct TcqConfig {
  int max_jobs = 8;            // blocks per batch
  int max_coefficients = 1024; // adjusted w*h (32x32 covers every TX size)
  int max_levels = 32;         // rate-table stride L
};

std::uint64_t required_device_bytes(const TcqConfig&);   // exact, incl. 256 B rounding
TcqBytes tcq_bytes(const TcqConfig&);                    // tcq/cctx/parity split

struct TcqWorkspace { /* device pointers + required_bytes + persistent_mark */ };

Status workspace_create(Arena& arena, const TcqConfig&, TcqWorkspace*,
                        std::uint64_t* out_required_bytes = nullptr,
                        std::string* error = nullptr);
// kOk | kInvalidArgument | kNotInitialized | kArenaExhausted (+ exact size)

// One slot-strided descriptor per block.
TcqJob  tcq_make_job(config, slot, count, table_levels, quant_step, lambda,
                     level_cap = 0, allow_zero_to_one = false);
CctxJob tcq_make_cctx_job(config, slot, count, step_u, step_v, bitdepth,
                          enabled = true);
ParityJob tcq_make_parity_job(config, slot, count, w, tx_class, plane,
                              quant_step, allow_parity_hiding);

cudaError_t tcq_quantize_batch(TcqWorkspace&, const TcqJob*, int,
                               cudaStream_t = nullptr);
cudaError_t cctx_search_batch(TcqWorkspace&, const CctxJob*, int,
                              cudaStream_t = nullptr);
cudaError_t parity_hide_batch(TcqWorkspace&, const ParityJob*, int,
                              cudaStream_t = nullptr);

}  // namespace av2enc::cuda::tcq
```

Host helpers:

```cpp
// §5.20.7.30 scan + §5.20.7.27 pre_max for one TX size / class / plane,
// derived from parity::make_geometry/build_scan/level_limits.
bool tcq_scan_and_pre_max(int tx_size, int tx_class, int plane,
                          int max_coefficients, int* count, int* w, int* h,
                          std::int32_t* scan, std::int32_t* pre_max,
                          std::string* error = nullptr);

// Calls the CPU tcq::RateModel for every (coefficient, state, level) and both
// is_eob values; writes the tables the device DP consumes.
bool tcq_fill_rate_tables(const TcqJob&, const double* original,
                          const std::int32_t* pre_max,
                          const std::int32_t* level_context,
                          const av2enc::tcq::RateModel&,
                          double* normal_rate, double* eob_coeff_rate,
                          double* block_eob_rate, std::string* error = nullptr);
```

### Workspace layout

Every `slot` owns a fixed region of each buffer (offsets are in the job):

```
coefficients       coefficient_offset + c           (original, pre_max,
                                                     level_context, scan, ...)
normal rate        rate_offset + (c * 8 + s) * L + q
EOB-coefficient    same index in eob_coeff_rate
block EOB rate     eob_table_offset + e
path choice        path_offset + c * 8 + s
tail energy        tail_offset + c
outputs            output_offset + c
CCTX candidates    candidate_offset + t * count + c   (t = 0..6)
```

`L = job.table_levels` must exceed every derived level bound. The kernel
rejects a job whose bound reaches `L`, rather than clamping silently.

The caller writes the coefficient / `pre_max` / scan / seed buffers through the
workspace pointers (the parity-hiding seed is read and updated in place in
`parity_levels`) (usually `cudaMemcpy`) and passes host job descriptors to the
launcher, which copies them into the workspace and launches. Launchers allocate
nothing, synchronize nothing and never move the bump pointer.

### Memory contract

* **One arena.** Every buffer comes from the caller's `av2enc::cuda::Arena`
  through `allocate_checked()`. No `cudaMalloc`/`cudaFree` anywhere in this
  component (`tools/check_arena.sh src/cuda/tcq` passes).
* **`required_device_bytes(config)`** is the exact sum of the 256-byte-rounded
  allocations; `tcq_bytes()` returns the TCQ / CCTX / parity split. The list is
  generated once by `TCQ_WORKSPACE_BUFFERS`, so the byte total, the allocation
  loop and the split report cannot drift.
* **Too small.** If `arena.remaining() < required`, `workspace_create()` returns
  `Status::kArenaExhausted` with the exact size in `*out_required_bytes` and
  leaves the arena untouched (the docs' name for this condition is
  `kArenaTooSmall`; `Status::kArenaExhausted` is the enum this tree actually
  has, and adding an alias would be a core edit).
* **Mark / rewind.** `workspace_create()` marks the arena and rewinds on any
  failure, so a rejected create leaks nothing. `persistent_mark` exposes the
  watermark for callers that take their own scratch. The DP's own scratch
  (path choices, tail energy) is part of the persistent set, so a launch never
  touches the bump pointer and repeated launches cannot grow the arena.

Measured sizes:

| Config (`jobs x coeff x levels`) | Total | TCQ | CCTX | Parity |
|---|---:|---:|---:|---:|
| `8 x 1024 x 32` (defaults) | 35,365,376 B (33.73 MiB) | 34,704,128 | 592,896 | 68,352 |
| `8 x 1024 x 48` (GPU test) | 52,142,592 B (49.73 MiB) | 51,481,344 | 592,896 | 68,352 |

The rate tables dominate: two doubles per `(coefficient, state, level)`.

## Bit-exactness

* **Rate model.** The device DP has no function pointers and no
  transcendentals. `tcq_fill_rate_tables()` calls the caller's exact
  `tcq::RateModel` for each `(c, s, q)` and both `is_eob` values, using the same
  `syntax_level_for` / `next_state` / `state_q0` decomposition as the CPU DP.
  The kernel then reads those doubles. An arbitrary CPU rate model — including
  the CPU suite's deliberately non-monotone one — is therefore reproduced
  exactly, not approximated.
* **Arithmetic order.** `tcq.cu` is compiled with `-fmad=false` and the CPU
  reference with `-ffp-contract=off`; the DP keeps the CPU's operation order
  (`local + lambda*rate + suffix`, `local + tail + lambda*(rate+eob) + suffix`,
  distortion and rate suffix accumulation, `tail[]` in reverse), the same
  `choice_better` tie-break (cost, distortion, rate, |level|, signed quant),
  the same `all-zero` seed and the same shorter-EOB rule.
* **Level bound.** `tcq_level_bound()` is a transcription of `bound_for()`
  including the `+64` margin and `level_cap` clipping; it is compared against
  `tcq::level_bound()` on 4,000 randomized inputs.
* **Normative mapping.** The block's `base_levels` are checked against
  `tcq::decode_levels()` in the emulation test.

## Parity-hiding seam

`parity_hide_batch()` implements `parity::quantize()` with
`QuantizeMode::kMinDistortion`, bit-exactly:

* the seed is kept when the block is not hidable or already satisfies
  `|DC| & 1 == sumAbs1`;
* otherwise the same DC ±1 candidates (both signs for a zero DC, duplicates
  included) and the same AC ±1 candidates that flip `min(level, 6) & 1` or drop
  the fourth non-zero;
* the same stable `(local_delta asc, scan_index desc)` pruning to
  `pruned_ac_candidates` above `exhaustive_max_area`, and the same
  first-candidate-wins tie-break;
* the same `isHidden`/`sumAbs1` recomputation and the same
  `escape_x > (1 << 20)` feasibility rejection.

**Not on device:** `QuantizeMode::kRateDistortion`. It needs the §9.3 static
coefficient CDFs (`m1::qctx`), `derive_syntax()` and `block_rate_bits()`. The
device path consequently reports `rate_bits = 0` and `cost = distortion` for
parity hiding. Per-position `dequant_step` (quantiser matrix) is also host-only;
the batch uses one `quant_step` per job. Both are stated here rather than
silently approximated.

## Device-visibility audit

`build_gpu_test.sh` runs before compiling:

1. every free function in `tcq_device.cuh` starts with `TCQ_HD`
   (`__host__ __device__` under nvcc, `inline` for the host pass):
   **35 definitions, 35 annotated**;
2. no namespace-scope / `constexpr` array exists in the device header — the
   §5.20.7.27 state map and the CCTX matrix are `switch` expressions, so no
   host array is odr-used from a kernel;
3. the device header includes no `cuda_runtime.h` and no `encoder/` header.

The host-emulation test compiles the same header with the host compiler, so any
division between "device code" and "host code" shows up as a build failure or a
parity failure.

## Build and test

```sh
cd src/cuda/tcq
./build_and_test.sh              # emulation always; GPU test if nvcc exists
WERROR=1 ./build_and_test.sh     # warnings are errors
./build_and_test.sh --no-run
ARCH=sm_86 ./build_gpu_test.sh   # nvcc + a visible device
```

### Current results (this machine: Apple clang, no nvcc, no GPU)

```
tcq:        330 blocks, 330 exact comparisons (8/10-bit, 25 sizes, 3 classes,
            two rate models, plane 1/2 spot checks)
tcq uncapped 4x4: 24 cases (level_cap = 0, +64 margin)
cctx:       100 blocks x 7 candidates (8/10-bit, joint dequant checked on a grid)
parity:     900 blocks (150 hidden, 98 repaired, 450 pruned)
arena:      required=35,365,376 B; one-byte-short rejected with that size
1,491,655 checks, 0 failures
```

`tcq_test.cu` (the real kernel test) is built and run by
`build_gpu_test.sh`; it could not be executed locally because this host has no
CUDA toolkit or device. It checks the same 330 TCQ blocks, 100 CCTX blocks and
900 parity blocks with CPU-vs-GPU `==` on every level, quant level, state, EOB,
distortion, rate and cost, plus the one-byte-short and two-launch no-growth
tests (arena `used()` / `peak_used()` / `mark()` unchanged, identical output
across two launches).

### Remote command

The repository's box is the Vast.ai RTX 3060 (`sm_86`, CUDA 12.8):

```sh
# from the Mac
AV2ENC_HOST=<user>@<host> AV2ENC_PORT=<port> tools/sync_remote.sh
ssh -p <port> <user>@<host> \
  'cd /workspace/av2enc/src/cuda/tcq && ./build_and_test.sh'

# or the GPU test only
ssh -p <port> <user>@<host> \
  'cd /workspace/av2enc/src/cuda/tcq && ARCH=sm_86 ./build_gpu_test.sh'
```

`/workspace` is not persistent; sync before building.

## Limitations

* The encoder DP is our design (the spec defines the decoder mapping only).
  What is claimed is parity with `src/encoder/tcq/`, not normativity.
* The GPU test has not been run on hardware here; the remote command above is
  the gate. The emulation test is the locally verified evidence.
* Parity hiding is minimum-distortion only (see the seam above).
* CCTX scoring is distortion-only and returns all seven candidates; the
  `cctx_type` RD decision (rate) stays with the caller, exactly as in
  `cctx_enumerate_scalar()`.
* The rate tables are sized by `max_levels`; a caller that needs the uncapped
  `bound + 64` range must size `max_levels` accordingly (the emulation test
  uses a dedicated 200-level workspace for that case).
